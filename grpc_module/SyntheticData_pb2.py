# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: SyntheticData.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13SyntheticData.proto\x12\x14syntheticdata.api.v1\"\xd6\x01\n\x15SyntheticModelRequest\x12\x1b\n\x13real_data_file_path\x18\x01 \x01(\t\x12\x17\n\x0fmodel_save_path\x18\x02 \x01(\t\x12\x12\n\ntabel_type\x18\x03 \x01(\t\x12\x12\n\nmodel_type\x18\x04 \x01(\t\x12\x13\n\x0bprimary_key\x18\x05 \x01(\t\x12\x18\n\x10\x61nonymize_fields\x18\x06 \x01(\t\x12\x17\n\x0fsampling_or_not\x18\x07 \x01(\x08\x12\x17\n\x0fsample_num_rows\x18\x08 \x01(\x05\"J\n\x16SyntheticSampleRequest\x12\x17\n\x0fmodel_save_path\x18\x04 \x01(\t\x12\x17\n\x0fsample_num_rows\x18\t \x01(\x05\"p\n\x11SyntheticResponse\x12,\n\x06status\x18\x01 \x01(\x0b\x32\x1c.syntheticdata.api.v1.Status\x12\x16\n\x0esynthetic_data\x18\x02 \x01(\t\x12\x15\n\rprivacy_score\x18\x03 \x01(\x02\"E\n\x06Status\x12.\n\x04\x63ode\x18\x01 \x01(\x0e\x32 .syntheticdata.api.v1.StatusCode\x12\x0b\n\x03msg\x18\x02 \x01(\t*\xc1\x02\n\nStatusCode\x12\x06\n\x02OK\x10\x00\x12\x14\n\x0fPARAMETER_ERROR\x10\x90N\x12\x14\n\x0f\x46ILE_READ_ERROR\x10\x91N\x12\x1c\n\x17PRIMARY_NOT_EXIST_ERROR\x10\x92N\x12\x1b\n\x16\x41nonymize_Fields_ERROR\x10\x93N\x12!\n\x1c\x41nonymize_Fields_FAKER_ERROR\x10\x94N\x12\x1f\n\x1aMODEL_INITIALIZATION_ERROR\x10\x95N\x12\x16\n\x11MODEL_TRAIN_ERROR\x10\x96N\x12\x15\n\x10MODEL_SAVE_ERROR\x10\x97N\x12\x15\n\x10MODEL_LOAD_ERROR\x10\x98N\x12 \n\x1bSAMPLE_DATA_GENERATOR_ERROR\x10\x99N\x12\x18\n\x13PRIVACY_SCORE_ERROR\x10\x9aN2\xe8\x01\n\x10SyntheticService\x12h\n\x0eSyntheticModel\x12+.syntheticdata.api.v1.SyntheticModelRequest\x1a\'.syntheticdata.api.v1.SyntheticResponse\"\x00\x12j\n\x0fSyntheticSample\x12,.syntheticdata.api.v1.SyntheticSampleRequest\x1a\'.syntheticdata.api.v1.SyntheticResponse\"\x00\x42,\n\x14syntheticdata.api.v1B\x12SyntheticDataProtoP\x01\x62\x06proto3')

_STATUSCODE = DESCRIPTOR.enum_types_by_name['StatusCode']
StatusCode = enum_type_wrapper.EnumTypeWrapper(_STATUSCODE)
OK = 0
PARAMETER_ERROR = 10000
FILE_READ_ERROR = 10001
PRIMARY_NOT_EXIST_ERROR = 10002
Anonymize_Fields_ERROR = 10003
Anonymize_Fields_FAKER_ERROR = 10004
MODEL_INITIALIZATION_ERROR = 10005
MODEL_TRAIN_ERROR = 10006
MODEL_SAVE_ERROR = 10007
MODEL_LOAD_ERROR = 10008
SAMPLE_DATA_GENERATOR_ERROR = 10009
PRIVACY_SCORE_ERROR = 10010


_SYNTHETICMODELREQUEST = DESCRIPTOR.message_types_by_name['SyntheticModelRequest']
_SYNTHETICSAMPLEREQUEST = DESCRIPTOR.message_types_by_name['SyntheticSampleRequest']
_SYNTHETICRESPONSE = DESCRIPTOR.message_types_by_name['SyntheticResponse']
_STATUS = DESCRIPTOR.message_types_by_name['Status']
SyntheticModelRequest = _reflection.GeneratedProtocolMessageType('SyntheticModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _SYNTHETICMODELREQUEST,
  '__module__' : 'SyntheticData_pb2'
  # @@protoc_insertion_point(class_scope:syntheticdata.api.v1.SyntheticModelRequest)
  })
_sym_db.RegisterMessage(SyntheticModelRequest)

SyntheticSampleRequest = _reflection.GeneratedProtocolMessageType('SyntheticSampleRequest', (_message.Message,), {
  'DESCRIPTOR' : _SYNTHETICSAMPLEREQUEST,
  '__module__' : 'SyntheticData_pb2'
  # @@protoc_insertion_point(class_scope:syntheticdata.api.v1.SyntheticSampleRequest)
  })
_sym_db.RegisterMessage(SyntheticSampleRequest)

SyntheticResponse = _reflection.GeneratedProtocolMessageType('SyntheticResponse', (_message.Message,), {
  'DESCRIPTOR' : _SYNTHETICRESPONSE,
  '__module__' : 'SyntheticData_pb2'
  # @@protoc_insertion_point(class_scope:syntheticdata.api.v1.SyntheticResponse)
  })
_sym_db.RegisterMessage(SyntheticResponse)

Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'SyntheticData_pb2'
  # @@protoc_insertion_point(class_scope:syntheticdata.api.v1.Status)
  })
_sym_db.RegisterMessage(Status)

_SYNTHETICSERVICE = DESCRIPTOR.services_by_name['SyntheticService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\024syntheticdata.api.v1B\022SyntheticDataProtoP\001'
  _STATUSCODE._serialized_start=524
  _STATUSCODE._serialized_end=845
  _SYNTHETICMODELREQUEST._serialized_start=46
  _SYNTHETICMODELREQUEST._serialized_end=260
  _SYNTHETICSAMPLEREQUEST._serialized_start=262
  _SYNTHETICSAMPLEREQUEST._serialized_end=336
  _SYNTHETICRESPONSE._serialized_start=338
  _SYNTHETICRESPONSE._serialized_end=450
  _STATUS._serialized_start=452
  _STATUS._serialized_end=521
  _SYNTHETICSERVICE._serialized_start=848
  _SYNTHETICSERVICE._serialized_end=1080
# @@protoc_insertion_point(module_scope)
