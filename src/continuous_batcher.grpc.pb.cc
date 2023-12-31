// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: continuous_batcher.proto

#include "continuous_batcher.pb.h"
#include "continuous_batcher.grpc.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/channel_interface.h>
#include <grpcpp/impl/codegen/client_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/rpc_service_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/sync_stream.h>
namespace continuous_batching {

static const char* ContinuousBatcher_method_names[] = {
  "/continuous_batching.ContinuousBatcher/GetNeedNum",
};

std::unique_ptr< ContinuousBatcher::Stub> ContinuousBatcher::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< ContinuousBatcher::Stub> stub(new ContinuousBatcher::Stub(channel, options));
  return stub;
}

ContinuousBatcher::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options)
  : channel_(channel), rpcmethod_GetNeedNum_(ContinuousBatcher_method_names[0], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status ContinuousBatcher::Stub::GetNeedNum(::grpc::ClientContext* context, const ::continuous_batching::BatchingRequest& request, ::continuous_batching::BatchingReply* response) {
  return ::grpc::internal::BlockingUnaryCall< ::continuous_batching::BatchingRequest, ::continuous_batching::BatchingReply, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_GetNeedNum_, context, request, response);
}

void ContinuousBatcher::Stub::async::GetNeedNum(::grpc::ClientContext* context, const ::continuous_batching::BatchingRequest* request, ::continuous_batching::BatchingReply* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::continuous_batching::BatchingRequest, ::continuous_batching::BatchingReply, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_GetNeedNum_, context, request, response, std::move(f));
}

void ContinuousBatcher::Stub::async::GetNeedNum(::grpc::ClientContext* context, const ::continuous_batching::BatchingRequest* request, ::continuous_batching::BatchingReply* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_GetNeedNum_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::continuous_batching::BatchingReply>* ContinuousBatcher::Stub::PrepareAsyncGetNeedNumRaw(::grpc::ClientContext* context, const ::continuous_batching::BatchingRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::continuous_batching::BatchingReply, ::continuous_batching::BatchingRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_GetNeedNum_, context, request);
}

::grpc::ClientAsyncResponseReader< ::continuous_batching::BatchingReply>* ContinuousBatcher::Stub::AsyncGetNeedNumRaw(::grpc::ClientContext* context, const ::continuous_batching::BatchingRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncGetNeedNumRaw(context, request, cq);
  result->StartCall();
  return result;
}

ContinuousBatcher::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      ContinuousBatcher_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< ContinuousBatcher::Service, ::continuous_batching::BatchingRequest, ::continuous_batching::BatchingReply, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](ContinuousBatcher::Service* service,
             ::grpc::ServerContext* ctx,
             const ::continuous_batching::BatchingRequest* req,
             ::continuous_batching::BatchingReply* resp) {
               return service->GetNeedNum(ctx, req, resp);
             }, this)));
}

ContinuousBatcher::Service::~Service() {
}

::grpc::Status ContinuousBatcher::Service::GetNeedNum(::grpc::ServerContext* context, const ::continuous_batching::BatchingRequest* request, ::continuous_batching::BatchingReply* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace continuous_batching

