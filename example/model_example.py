import grpc
import pandas as pd

from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from pprint import pprint


def send_request(real_data_file_path,  model_save_path, tabel_type='tabular', model_type='TVAE', primary_key=None, anonymize_fields=None, sampling_or_not=False, sample_num_rows=1000):
    request = SyntheticData_pb2.SyntheticModelRequest()
    request.real_data_file_path = real_data_file_path
    request.model_save_path = model_save_path
    request.tabel_type = tabel_type
    request.model_type = model_type
    request.sampling_or_not = sampling_or_not

    if primary_key is not None:
        request.primary_key = primary_key
    if anonymize_fields is not None:
        request.anonymize_fields = anonymize_fields
    if sampling_or_not and sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows



    # ip和端口与synthetic_model_server.py中设置的一致
    with grpc.insecure_channel('127.0.0.1:40053') as channel:
        stub  = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)

        result = stub.SyntheticModel(request)

    return  result.status, result.synthetic_data, result.privacy_score


real_data_file_path ="example/data/adult.csv"
model_save_path = "example/models/example.pkl"
tabel_type = "tabular"
model_type = 'TVAE'
# primary_key = None
# anonymize_fields = None
sampling_or_not = True
sample_num_rows = 100


status, synthetic_data, privacy_score = send_request(real_data_file_path,  model_save_path, tabel_type, model_type,sampling_or_not=sampling_or_not, sample_num_rows=sample_num_rows)
print(status.code)
print(status.msg)
print(synthetic_data)
print(privacy_score)






