import unittest
import grpc
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from syntheticdata.config import config


def send_request(model_save_path, sample_num_rows=1000):
    request = SyntheticData_pb2.SyntheticSampleRequest()
    if model_save_path is not None:
        request.model_save_path = model_save_path
    if sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows

    # ip和端口与config文件中设置的一致
    with grpc.insecure_channel('{}:{}'.format(config.IP, config.PORT)) as channel:
        stub = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)
        result = stub.SyntheticSample(request)

    return result.status, result.synthetic_data, result.privacy_score


class TestSyntheticDataServiceSamplingErrors(unittest.TestCase):
    def test_model_save_path_empty_error(self):
        # model_save_path参数为空
        model_save_path = ''
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '模型路径参数 model_save_path 不能为空。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)


    def test_sample_num_rows_value_error1(self):
        # sample_num_rows_参数为非正整数
        model_save_path = 'tests/models/tvae.pkl'
        sample_num_rows = 0
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '参数 sample_num_rows 不能为空，且sample_num_rows必须为正整数。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_sample_num_rows_value_error2(self):
        # sample_num_rows_参数为非正整数
        model_save_path = 'tests/models/tvae.pkl'
        sample_num_rows = -1
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '参数 sample_num_rows 不能为空，且sample_num_rows必须为正整数。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)


    def test_model_load_error(self):
        # 模型路径不存在，加载失败
        model_save_path = 'tests/models/noexistpath.pkl'
        sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 10008)
        self.assertEqual(status.msg, '模型加载失败，模型路径{}不存在或其他错误'.format(model_save_path))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

#
class TestSyntheticDataServiceSamplingSuccess(unittest.TestCase):
    def test_tvae_success_sample(self):
        model_save_path = 'tests/models/tvae.pkl'
        sample_num_rows = 200
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_ctgan_success_sample(self):
        model_save_path = 'tests/models/ctgan.pkl'
        sample_num_rows = 50
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_gaussian_copula_success_sample(self):
        model_save_path = 'tests/models/gaussian_copula.pkl'
        sample_num_rows = 10
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_copula_gan_success_sample(self):
        model_save_path = 'tests/models/copula_gan.pkl'
        sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)


if __name__ == '__main__':
    unittest.main()
