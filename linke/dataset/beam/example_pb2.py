# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: example.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()



DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rexample.proto\x12\ntensorflow\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\"\x98\x01\n\x07\x46\x65\x61ture\x12+\n\nbytes_list\x18\x01 \x01(\x0b\x32\x15.tensorflow.BytesListH\x00\x12+\n\nfloat_list\x18\x02 \x01(\x0b\x32\x15.tensorflow.FloatListH\x00\x12+\n\nint64_list\x18\x03 \x01(\x0b\x32\x15.tensorflow.Int64ListH\x00\x42\x06\n\x04kind\"\x83\x01\n\x08\x46\x65\x61tures\x12\x32\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32!.tensorflow.Features.FeatureEntry\x1a\x43\n\x0c\x46\x65\x61tureEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x13.tensorflow.Feature:\x02\x38\x01\"3\n\x0b\x46\x65\x61tureList\x12$\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x13.tensorflow.Feature\"\x9c\x01\n\x0c\x46\x65\x61tureLists\x12?\n\x0c\x66\x65\x61ture_list\x18\x01 \x03(\x0b\x32).tensorflow.FeatureLists.FeatureListEntry\x1aK\n\x10\x46\x65\x61tureListEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.tensorflow.FeatureList:\x02\x38\x01\"1\n\x07\x45xample\x12&\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x14.tensorflow.Features\"i\n\x0fSequenceExample\x12%\n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x14.tensorflow.Features\x12/\n\rfeature_lists\x18\x02 \x01(\x0b\x32\x18.tensorflow.FeatureListsB\x81\x01\n\x16org.tensorflow.exampleB\rFeatureProtosP\x01ZSgithub.com/tensorflow/tensorflow/tensorflow/go/core/example/example_protos_go_proto\xf8\x01\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'example_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\026org.tensorflow.exampleB\rFeatureProtosP\001ZSgithub.com/tensorflow/tensorflow/tensorflow/go/core/example/example_protos_go_proto\370\001\001'
  _globals['_FLOATLIST'].fields_by_name['value']._loaded_options = None
  _globals['_FLOATLIST'].fields_by_name['value']._serialized_options = b'\020\001'
  _globals['_INT64LIST'].fields_by_name['value']._loaded_options = None
  _globals['_INT64LIST'].fields_by_name['value']._serialized_options = b'\020\001'
  _globals['_FEATURES_FEATUREENTRY']._loaded_options = None
  _globals['_FEATURES_FEATUREENTRY']._serialized_options = b'8\001'
  _globals['_FEATURELISTS_FEATURELISTENTRY']._loaded_options = None
  _globals['_FEATURELISTS_FEATURELISTENTRY']._serialized_options = b'8\001'
  _globals['_BYTESLIST']._serialized_start=29
  _globals['_BYTESLIST']._serialized_end=55
  _globals['_FLOATLIST']._serialized_start=57
  _globals['_FLOATLIST']._serialized_end=87
  _globals['_INT64LIST']._serialized_start=89
  _globals['_INT64LIST']._serialized_end=119
  _globals['_FEATURE']._serialized_start=122
  _globals['_FEATURE']._serialized_end=274
  _globals['_FEATURES']._serialized_start=277
  _globals['_FEATURES']._serialized_end=408
  _globals['_FEATURES_FEATUREENTRY']._serialized_start=341
  _globals['_FEATURES_FEATUREENTRY']._serialized_end=408
  _globals['_FEATURELIST']._serialized_start=410
  _globals['_FEATURELIST']._serialized_end=461
  _globals['_FEATURELISTS']._serialized_start=464
  _globals['_FEATURELISTS']._serialized_end=620
  _globals['_FEATURELISTS_FEATURELISTENTRY']._serialized_start=545
  _globals['_FEATURELISTS_FEATURELISTENTRY']._serialized_end=620
  _globals['_EXAMPLE']._serialized_start=622
  _globals['_EXAMPLE']._serialized_end=671
  _globals['_SEQUENCEEXAMPLE']._serialized_start=673
  _globals['_SEQUENCEEXAMPLE']._serialized_end=778
# @@protoc_insertion_point(module_scope)