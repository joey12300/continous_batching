GRPC_CPP_PLUGIN_PATH=/paddle/triton-inference-server/server/build/tritonserver/build/third-party/grpc/bin/grpc_cpp_plugin
PROTOS_PATH=./
PROTOC=/paddle/triton-inference-server/server/build/tritonserver/build/third-party/protobuf/bin/protoc
PROTO_FILE=${PROTOS_PATH}/continuous_batcher.proto

all: grpc_python grpc_out cpp_out

grpc_python:
	python3 -m grpc_tools.protoc -I${PROTOS_PATH} --python_out=. --pyi_out=. --grpc_python_out=. ${PROTO_FILE}
grpc_out:
	$(PROTOC) -I$(PROTOS_PATH) --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) ${PROTO_FILE}
cpp_out:
	$(PROTOC) -I$(PROTOS_PATH) --cpp_out=. ${PROTO_FILE}
