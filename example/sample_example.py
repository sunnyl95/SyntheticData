import grpc
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from syntheticdata.config import config


def send_request(model_save_path, sample_num_rows=100):
    request = SyntheticData_pb2.SyntheticSampleRequest()
    request.model_save_path = model_save_path
    if sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows

    # ip和端口与config文件中设置的一致
    with grpc.insecure_channel('{}:{}'.format(config.IP, config.PORT)) as channel:
        stub = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)

        result = stub.SyntheticSample(request)

    return result.status, result.synthetic_data, result.privacy_score


if __name__ == "__main__":
    model_save_path = "example/models/tvae.pkl"
    sample_num_rows = 100

    status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                         sample_num_rows=sample_num_rows)

    print(f"服务应答状态码：{status.code}")
    print(f"服务应答消息：{status.msg}")
    print(f"生成的仿真数据：{synthetic_data}")
    print(f"生成的仿真数据隐私性得分：{privacy_score}")
