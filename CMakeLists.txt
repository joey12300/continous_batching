# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(continuousbatching LANGUAGES C CXX)

set(TRITON_COMMON_REPO_TAG "r23.06" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r23.06" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r23.06" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_THIRD_PARTY_REPO_TAG "r23.06" CACHE STRING "Tag for triton-inference-server/third_party repo")
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party" CACHE STRING "Location of third-party build")
set(TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party-src" CACHE STRING "Location of third-party source")

set(_REFLECTION grpc++_reflection)
set(_GRPC_GRPCPP grpc++)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/joey12300/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_MakeAvailable(repo-common repo-core repo-backend)

# Some libs are installed to ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib64 instead
# of ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib on Centos
set (LIB_DIR "lib")
# /etc/os-release does not exist on Windows
if(EXISTS "/etc/os-release")
  file(STRINGS /etc/os-release DISTRO REGEX "^NAME=")
  string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" DISTRO "${DISTRO}")
  message(STATUS "Distro Name: ${DISTRO}")
  if(DISTRO MATCHES "CentOS.*")
    set (LIB_DIR "lib64")
  endif()
endif()

set(GRPC_DIRECTORY "" CACHE PATH "The directory of grpc package")
if (NOT EXISTS "${GRPC_DIRECTORY}")
	message(FATAL_ERROR "Can't find grpc package in path ${GRPC_DIRECTORY}")
endif()

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(gRPC_DIR "${GRPC_DIRECTORY}/grpc/cmake")
  set(Protobuf_DIR "${GRPC_DIRECTORY}/protobuf/cmake")
  set(c-ares_DIR "${GRPC_DIRECTORY}/c-ares/cmake")
  set(absl_DIR "${GRPC_DIRECTORY}/absl/cmake")
else()
  set(gRPC_DIR "${GRPC_DIRECTORY}/grpc/${LIB_DIR}/cmake/grpc")
  set(Protobuf_DIR "${GRPC_DIRECTORY}/protobuf/${LIB_DIR}/cmake/protobuf")
  set(c-ares_DIR "${GRPC_DIRECTORY}/c-ares/${LIB_DIR}/cmake/c-ares/")
  set(absl_DIR "${GRPC_DIRECTORY}/absl/${LIB_DIR}/cmake/absl/")
endif()

#-- Could NOT find absl (missing: absl_DIR)
message(STATUS "gRPC_DIR = ${gRPC_DIR}")
message(STATUS "Protobuf_DIR = ${Protobuf_DIR}")
message(STATUS "c-ares_DIR = ${c-ares_DIR}")
message(STATUS "absl_DIR = ${absl_DIR}")

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND_ModelBatch API.
#
configure_file(src/libtriton_continuousbatching.ldscript libtriton_continuousbatching.ldscript COPYONLY)

set(_PROTOBUF_PROTOC "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/bin/protoc")
set(_GRPC_CPP_PLUGIN_EXECUTABLE "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/bin/grpc_cpp_plugin")

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(cb_proto_path "${SOURCE_DIR}/continuous_batcher.proto")
set(cb_proto_srcs "${SOURCE_DIR}/continuous_batcher.pb.cc")
set(cb_proto_hdrs "${SOURCE_DIR}/continuous_batcher.pb.h")
set(cb_grpc_srcs "${SOURCE_DIR}/continuous_batcher.grpc.pb.cc")
set(cb_grpc_hdrs "${SOURCE_DIR}/continuous_batcher.grpc.pb.h")

add_custom_command(
      OUTPUT "${cb_proto_srcs}" "${cb_proto_hdrs}" "${cb_grpc_srcs}" "${cb_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${SOURCE_DIR}"
        --cpp_out "${SOURCE_DIR}"
        -I "${SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${cb_proto_path}"
      DEPENDS "${cb_proto_path}")

add_library(
  triton-continuous-batching SHARED
  src/continuous_batching.cc
  src/continuous_batcher.grpc.pb.cc
  src/continuous_batcher.pb.cc
)


target_include_directories(
  triton-continuous-batching
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/include
    ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/include
)

target_compile_features(triton-continuous-batching PRIVATE cxx_std_11)
target_compile_options(
  triton-continuous-batching PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

target_link_libraries(
  triton-continuous-batching
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    gRPC::${_GRPC_GRPCPP}
    gRPC::${_REFLECTION}
)

if(WIN32)
  set_target_properties(
    triton-continuous-batching PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_continuousbatching
  )
else()
  set_target_properties(
    triton-continuous-batching PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_continuousbatching
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_continuousbatching.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_continuousbatching.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/continuousBatching)

install(
  TARGETS
    triton-continuous-batching
  EXPORT
    triton-continuous-batching-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/batching/continuous_batching
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/batching/continuous_batching
)

install(
  EXPORT
    triton-continuous-batching-targets
  FILE
    ContinuousBatchingTargets.cmake
  NAMESPACE
    triton-continuous-batching
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/triton-continuous-batching.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/triton-continuous-batching.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/triton-continuous-batching.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-continuous-batching-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/triton-continuous-batching.cmake
  NAMESPACE triton-continuous-batching
)

export(PACKAGE triton-continuous-batching)

