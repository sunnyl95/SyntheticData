import grpc
import pandas as pd

from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from pprint import pprint


def send_request(model_save_path, sample_num_rows=1000):
    request = SyntheticData_pb2.SyntheticSampleRequest()
    request.model_save_path = model_save_path
    if sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows

    # ip和端口与synthetic_model_server.py中设置的一致
    with grpc.insecure_channel('127.0.0.1:40052') as channel:
        stub  = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)

        result = stub.SyntheticSample(request)

    return  result.status, result.synthetic_data, result.privacy_score



# model_save_path = "example/models/example.pkl"
# sample_num_rows = 100
#
#
# status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path, sample_num_rows=sample_num_rows)
# print(status.code)
# print(status.msg)
# print(synthetic_data)
# print(privacy_score)


from copulas.datasets import sample_trivariate_xyz

data = sample_trivariate_xyz()
data.head()




