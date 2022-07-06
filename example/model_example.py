import grpc
import pandas as pd

from grpc_module.Model import synthetic_model_pb2, synthetic_model_pb2_grpc
from pprint import pprint


def send_request(to_synthetic_file_path,  model_save_path, tabel_type='tabular', model_type='TVAE', primary_key=None, anonymize_fields=None):
    request = synthetic_model_pb2.SyntheticRequest()
    request.to_synthetic_file_path = to_synthetic_file_path
    request.model_save_path = model_save_path
    request.tabel_type = tabel_type
    request.model_type = model_type
    if primary_key is not None:
        request.primary_key = primary_key
    if anonymize_fields is not None:
        request.anonymize_fields = anonymize_fields

    # ip和端口与synthetic_model_server.py中设置的一致
    with grpc.insecure_channel('127.0.0.1:40051') as channel:
        stub  = synthetic_model_pb2_grpc.SyntheticModelServiceStub(channel)
        response = stub.SyntheticModel(request)
        status = response.status

    return status


to_synthetic_file_path ="example/data/adult.csv"
model_save_path = "example/models/example.pkl"
tabel_type = "tabular"
model_type = 'TVAE'
# primary_key = None
# anonymize_fields = None


status= send_request(to_synthetic_file_path,  model_save_path, tabel_type, model_type)
print('status.code: ', status.code)
print('status.msg: ', status.msg)






