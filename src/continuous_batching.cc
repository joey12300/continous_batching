// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include<thread>
#include <limits>

#include "triton/core/tritonbackend.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "triton/common/triton_json.h"
#include "triton/backend/backend_common.h"
#include "continuous_batcher.pb.h"
#include "continuous_batcher.grpc.pb.h"

using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::Server;
using grpc::Status;
using continuous_batching::ContinuousBatcher;
using continuous_batching::BatchingRequest;
using continuous_batching::BatchingReply;

namespace triton { namespace core { namespace continuous_batching {

//
// Minimal custom  batching strategy that demonstrates the
// TRITONBACKEND_ModelBatch API. This custom batching strategy dynamically
// creates batches up to 1 request.
//

/////////////

std::mutex g_mutex;
std::condition_variable g_cond;
bool g_batcher_is_init_stage = true;
bool g_has_get_need_num = false;

class ContinuousBatcherController final {
public:
  ContinuousBatcherController(uint16_t port, int64_t max_batch_size) {
    server_thread_ = std::thread([port, max_batch_size, this](){
      int64_t init_need_num = 1;
      if (max_batch_size > 0) {
        init_need_num = (std::max)(max_batch_size / 2, init_need_num);
      }
      this->server_.Run(port, init_need_num);
    });
  }

  ~ContinuousBatcherController() {
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
  }

  int GetNeedNum() const {
    return server_.need_num_;
  }
private:
  class ContinuousBatchingServer final {
    private:
    class ContinuousBatchingServiceImpl final : public ContinuousBatcher::Service {
     public:
      ContinuousBatchingServiceImpl(int* need_num) :
        need_num_(need_num) {}
      Status GetNeedNum(ServerContext* context, const BatchingRequest* request,
                      BatchingReply* reply) override {
        {
          std::lock_guard<std::mutex> lock(g_mutex);
          g_has_get_need_num = true;
          *need_num_ = request->need_num();
        }
        g_cond.notify_one();
        reply->set_actual_batch_size(request->need_num());
        return Status::OK;
      }
      private:
        int* need_num_;
    };
    public:
      void Run(uint16_t port, int init_need_num) {
          std::string server_address = "0.0.0.0:" + std::to_string(port);
          ContinuousBatchingServiceImpl service(&need_num_);
          need_num_ = init_need_num;

          grpc::EnableDefaultHealthCheckService(true);
          grpc::reflection::InitProtoReflectionServerBuilderPlugin();
          ServerBuilder builder;
          // Listen on the given address without any authentication mechanism.
          builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
          // Register "service" as the instance through which we'll communicate with
          // clients. In this case it corresponds to an *synchronous* service.
          builder.RegisterService(&service);
          // Finally assemble the server.
          std::unique_ptr<Server> server(builder.BuildAndStart());
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO, (std::string("Server listening on ") + server_address).c_str());
          // Wait for the server to shutdown. Note that some other thread must be
          // responsible for shutting down the server for this call to ever return.
          server->Wait();
      }
      int need_num_;
  };
  std::thread server_thread_;
  ContinuousBatchingServer server_;
};

extern "C" {

/// Check whether a request should be added to the pending model batch.
///
/// \param request The request to be added to the pending batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch. When the callback returns, this should reflect
/// the latest batch information.
/// \param should_include The pointer to be updated on whether the request
/// should be included in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchIncludeRequest(
    TRITONBACKEND_Request* request, void* userp, bool* should_include)
{
  // Get the batch size of current request
  const int64_t* shape;
  TRITONBACKEND_Input* input;
  auto err = TRITONBACKEND_RequestInputByIndex(request, 0, &input);
  if (err)
    return err;
  err = TRITONBACKEND_InputProperties(
      input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
  if (err)
    return err;
  int64_t request_batch_size = *shape;
  // Default should_include to false in case function returns error.
  *should_include = false;

  // Check if the batch size exceed the need number
  // If not exceed, include this request
  unsigned int* need_num = reinterpret_cast<unsigned int *>(userp);
  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("Userp need_num: ") + std::to_string(*need_num))
        .c_str());
  if (*need_num >= request_batch_size) {
    *need_num -= request_batch_size;
    *should_include = true;
  } else {
    *should_include = false;
  }
  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("should_include: ") + std::to_string(*should_include))
        .c_str());
  return nullptr;  // success
}

/// Callback to be invoked when Triton has begun forming a batch.
///
/// \param batcher The read-only placeholder for backend to retrieve
// information about the batching strategy for this model.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchInitialize(
    const TRITONBACKEND_Batcher* batcher, void** userp, bool is_new_payload)
{
  const ContinuousBatcherController* controller =
      reinterpret_cast<const ContinuousBatcherController*>(batcher);
  if (g_batcher_is_init_stage) {
    g_batcher_is_init_stage = false;
  } else {
    // wait for next need_num
    if (is_new_payload) {
      std::unique_lock<std::mutex> lock(g_mutex);
      g_cond.wait(lock, [](){return g_has_get_need_num;});
    }
  }
  // using userp to store need_num
  *userp = new int(controller->GetNeedNum());
  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("Need num from controller: ") + std::to_string(controller->GetNeedNum())).c_str());
  return nullptr;  // success
}

/// Callback to be invoked when Triton has finishing forming a batch.
///
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void* userp)
{
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_has_get_need_num = false;
  }
  delete static_cast<int*>(userp);
  return nullptr;  // success
}

/// Create a new batcher for use with custom batching. This is called during
/// model loading. The batcher will point to a user-defined data structure that
/// holds read-only data used for custom batching.
///
/// \param batcher User-defined placeholder for backend to store and
/// retrieve information about the batching strategy for this model.
/// return a TRITONSERVER_Error indicating success or failure.
/// \param model The backend model for which Triton is forming a batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherInitialize(
    TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model)
{
  // Batcher will point to an unsigned integer representing the maximum
  // volume in bytes for each batch.

  // Read the user-specified bytes from the model config.
  TRITONSERVER_Message* config_message;
  TRITONBACKEND_ModelConfig(model, 1 /* config_version */, &config_message);

  const char* buffer;
  size_t byte_size;

  auto err =
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size);
  if (err)
    return err;

  triton::common::TritonJson::Value model_config, params, port_param, max_batch_size_param;
  int64_t max_batch_size = -1;
  err = model_config.Parse(buffer, byte_size);
  TRITONSERVER_MessageDelete(config_message);

  if (!model_config.Find("parameters", &params)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find parameters in model config.");
  }

  if (model_config.Find("max_batch_size", &max_batch_size_param)) {
    max_batch_size_param.AsInt(&max_batch_size);
    if (max_batch_size == 0 || max_batch_size < -1) {
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "The model config 'max_batch_size' should be larger than 0 or equal to -1.");
    }
  }

  std::string port_str;
  uint64_t port = 50001;
  // The config 'BATCHER_PORT' is optional.
  bool found_port = params.Find("BATCHER_PORT", &port_param);
  if (found_port) {
    err = port_param.MemberAsString("string_value", &port_str);
    if (err)
      return err;
    try {
      port = static_cast<uint64_t>(std::stoul(port_str));
    }
    catch (const std::invalid_argument& ia) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("failed to convert '") + port_str +
          "' to unsigned int64")
              .c_str());
    }
  }

  if (port > std::numeric_limits<uint16_t>::max()) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE,
      "The available port range should be [0, 65535].");
  }

  *batcher = reinterpret_cast<TRITONBACKEND_Batcher*>(
      new ContinuousBatcherController(port, max_batch_size));
  return nullptr;  // success
}

/// Free memory associated with batcher. This is called during model unloading.
///
/// \param batcher User-defined placeholder for backend to store and
/// retrieve information about the batching strategy for this model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherFinalize(TRITONBACKEND_Batcher* batcher)
{
  delete reinterpret_cast<ContinuousBatcherController*>(batcher);
  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::core::volume_batching
