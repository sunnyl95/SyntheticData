import grpc
import pandas as pd

from grpc_module.Sample import synthetic_sample_pb2, synthetic_sample_pb2_grpc
from pprint import pprint

def send_request(real_data_file_path, model_path,  model_type, sample_number,anonymize_fields=None):
    request = synthetic_sample_pb2.SyntheticRequest()
    request.real_data_file_path = real_data_file_path
    request.model_path = model_path
    request.model_type = model_type
    request.sample_number = sample_number
    if anonymize_fields is not None:
        request.anonymize_fields = anonymize_fields

    # ip和端口与synthetic_model_server.py中设置的一致
    with grpc.insecure_channel('127.0.0.1:40052') as channel:
        stub  = synthetic_sample_pb2_grpc.SyntheticSampleServiceStub(channel)
        response = stub.SyntheticSample(request)
        status = response.status
        synthetic_data = response.synthetic_data
        privacy_score = response.privacy_score

    return status, synthetic_data,  privacy_score



real_data_file_path ="example/data/adult.csv"
model_path = "example/models/example.pkl"
model_type = 'TVAE'
sample_number = 100
anonymize_fields = None

status, synthetic_data,  privacy_score= send_request(real_data_file_path, model_path,  model_type, sample_number,anonymize_fields)

#返回的数据标准类型：二进制和json
synthetic_data = pd.read_json(synthetic_data)

print('status.code: ', status.code)
print('status.msg: ', status.msg)
print('synthetic_data: ', synthetic_data)
print('privacy_score: ',privacy_score)




