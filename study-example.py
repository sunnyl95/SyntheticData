import grpc
import pandas as pd

from grpc_module import  study_pb2_grpc, study_pb2
from pprint import pprint


def send_request(input):
    request = study_pb2.Request()

    request.input.CopyFrom(input)


    # ip和端口与synthetic_model_server.py中设置的一致
    with grpc.insecure_channel('127.0.0.1:40054') as channel:
        stub  = study_pb2_grpc.StudyServiceStub(channel)
        response = stub.Study(request)

        key = response.key
        value = response.value


    return key, value



fake_type = study_pb2.FakerType.Value("UNIVERSAL")

maps_val = study_pb2.maps(key="test", value=fake_type)


study_pb2.maps()
key, value= send_request(input=maps_val)

print(key)
print(value)





