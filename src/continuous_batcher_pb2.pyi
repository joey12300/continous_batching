from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BatchingRequest(_message.Message):
    __slots__ = ["need_num"]
    NEED_NUM_FIELD_NUMBER: _ClassVar[int]
    need_num: int
    def __init__(self, need_num: _Optional[int] = ...) -> None: ...

class BatchingReply(_message.Message):
    __slots__ = ["actual_batch_size"]
    ACTUAL_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    actual_batch_size: int
    def __init__(self, actual_batch_size: _Optional[int] = ...) -> None: ...
