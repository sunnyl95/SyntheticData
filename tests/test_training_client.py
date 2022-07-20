import unittest
import grpc
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from syntheticdata.config import config


def send_request(real_data_file_path=None, model_save_path=None, tabel_type=None, model_type=None, primary_key=None,
                 anonymize_fields=None,
                 sampling_or_not=None,
                 sample_num_rows=1000):
    request = SyntheticData_pb2.SyntheticModelRequest()
    if real_data_file_path is not None:
        request.real_data_file_path = real_data_file_path
    if model_save_path is not None:
        request.model_save_path = model_save_path
    if model_save_path is not None:
        request.model_save_path = model_save_path
    if tabel_type is not None:
        request.tabel_type = tabel_type
    if model_type is not None:
        request.model_type = model_type
    if sampling_or_not is not None:
        request.sampling_or_not = sampling_or_not
    if primary_key is not None:
        request.primary_key = primary_key
    if anonymize_fields is not None:
        request.anonymize_fields = anonymize_fields
    if sampling_or_not and sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows

    # ip和端口与config文件中设置的一致
    with grpc.insecure_channel('{}:{}'.format(config.IP, config.PORT)) as channel:
        stub = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)
        result = stub.SyntheticModel(request)

    return result.status, result.synthetic_data, result.privacy_score


class TestSyntheticDataServiceTrainingErrors(unittest.TestCase):

    def test_real_data_file_path_empty_error(self):
        # real_data_file_path参数为空
        real_data_file_path = ''
        status, synthetic_data, privacy_score = send_request(real_data_file_path)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '待仿真数据表文件参数 real_data_file_path 不能为空。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_model_save_path_empty_error(self):
        # model_save_path参数为空
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = ''
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '待模型保存路径参数 model_save_path 不能为空。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    #
    def test_tabel_type_empty_error(self):
        # tabel_type参数为空
        real_data_file_path = './data/adult.csv'
        model_save_path = './models/model.pkl'
        tabel_type = ''
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '数据表类型参数 tabel_type 不能为空。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_model_type_empty_error(self):
        # model_type参数为空
        real_data_file_path = './data/adult.csv'
        model_save_path = './models/model.pkl'
        tabel_type = 'tabular'
        model_type = ''
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '模型类型参数 model_type 不能为空。')
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_file_read_error(self):
        # 文件读取失败检查
        real_data_file_path = './adult.csv'
        model_save_path = './models/tvae.pkl'
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = False
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not)

        self.assertEqual(status.code, 10001)
        self.assertEqual(status.msg, '待识别数据表文件{}读取失败，请检测路径!'.format(real_data_file_path))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_file_is_not_endwith_csv_error(self):
        # 非csv文件
        real_data_file_path = './data/adult.txt'
        model_save_path = 'tests/models/tvae.pkl'
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = True
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, '待仿真数据文件{}不是csv文件，目前仅支持csv文件。'.format(real_data_file_path))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_primary_key_not_exist_error(self):
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = 'tests/models/tvae.pkl'
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = False
        primary_key = "not_exist_column"
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not,
                                                             primary_key=primary_key
                                                             )
        self.assertEqual(status.code, 10002)
        self.assertEqual(status.msg, '数据中不存在这样的主键列--{}--,请检查！'.format(primary_key))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    #
    def test_anonymize_fields_error1(self):
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = 'tests/models/tvae.pkl'
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = False
        anonymize_fields = "{'test': 'name'}"  # 'test'不存在在原始数据表字段中
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not,
                                                             anonymize_fields=anonymize_fields)
        self.assertEqual(status.code, 10003)
        self.assertEqual(status.msg, '{}中含有原始数据表中不存在的字段列，请检查！'.format(anonymize_fields))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    # 0
    def test_anonymize_fields_error2(self):
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = 'tests/models/tvae.pkl'
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = False
        anonymize_fields = "{'age': 'test'}"  # 'test'不存在在fake列表中
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not,
                                                             anonymize_fields=anonymize_fields)
        self.assertEqual(status.code, 10004)
        self.assertEqual(status.msg, '当前不支持该名称 {} 表示的敏感字段匿名，请检查！'.format(anonymize_fields))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_anonymize_fields_error3(self):
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = 'tests/models/tvae.pkl'
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = False
        anonymize_fields = "{'age': 'test}"  # 传入的字典格式不对，test右侧少了个引号
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not,
                                                             anonymize_fields=anonymize_fields)
        self.assertEqual(status.code, 10000)
        self.assertEqual(status.msg, 'anonymize_fields参数{}不符合格式要求，请检查！'.format(anonymize_fields))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_model_initialization_error(self):
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = 'tests/models/tvae.pkl'
        tabel_type = 'timeseries'
        model_type = 'TVAE'
        sampling_or_not = False
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not)
        self.assertEqual(status.code, 10005)
        self.assertEqual(status.msg,
                         '模型初始化失败,数据表类型{}有误（当前仅支持"tabular"），或模型类型{}有误，或者两个参数不匹配！'.format(tabel_type, model_type))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)

    def test_model_save_error(self):
        real_data_file_path = 'tests/data/adult.csv'
        model_save_path = 'tests/notexistpath/tvae.pkl'  # 传入一个不存在的模型保存路径
        tabel_type = 'tabular'
        model_type = 'TVAE'
        sampling_or_not = False
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             sampling_or_not=sampling_or_not)
        self.assertEqual(status.code, 10007)
        self.assertEqual(status.msg, '模型保存失败，请检查路径{}！'.format(model_save_path))
        self.assertEqual(synthetic_data, '')
        self.assertEqual(privacy_score, 0)


class TestSyntheticDataServiceTrainingSuccess(unittest.TestCase):
    def test_train_tvae_success_train_sample(self):
        # 测试TVAE模型，训练完成后生成仿真数据
        real_data_file_path = "tests/data/adult.csv"
        model_save_path = "tests/models/tvae.pkl"
        tabel_type = "tabular"
        model_type = "TVAE"
        anonymize_fields = "{'native-country':'country'}"
        sampling_or_not = True
        sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             anonymize_fields=anonymize_fields,
                                                             sampling_or_not=sampling_or_not,
                                                             sample_num_rows=sample_num_rows)
        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_train_tvae_success_only_train(self):
        # 测试TVAE模型，仅训练，不生成仿真数据
        real_data_file_path = "tests/data/adult.csv"
        model_save_path = "tests/models/tvae.pkl"
        tabel_type = "tabular"
        model_type = "TVAE"
        anonymize_fields = "{'native-country':'country'}"
        sampling_or_not = False
        # sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             anonymize_fields=anonymize_fields,
                                                             sampling_or_not=sampling_or_not,
                                                             )
        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '模型训练成功，并成功保存！（该任务类型是仅训练模型，不生成仿真数据样本，因此隐私性得分为1）')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_train_ctgan_success(self):
        # 测试CTGAN模型，训练完成后生成仿真数据
        real_data_file_path = "tests/data/adult.csv"
        model_save_path = "tests/models/ctgan.pkl"
        tabel_type = "tabular"
        model_type = "CTGAN"
        anonymize_fields = "{'native-country':'country'}"
        sampling_or_not = True
        sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             anonymize_fields=anonymize_fields,
                                                             sampling_or_not=sampling_or_not,
                                                             sample_num_rows=sample_num_rows)

        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_train_gaussiancopula_success(self):
        # 测试GaussianCopula模型，训练完成后生成仿真数据
        real_data_file_path = "tests/data/adult.csv"
        model_save_path = "tests/models/gaussian_copula.pkl"
        tabel_type = "tabular"
        model_type = "GaussianCopula"
        anonymize_fields = "{'native-country':'country'}"
        sampling_or_not = True
        sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             anonymize_fields=anonymize_fields,
                                                             sampling_or_not=sampling_or_not,
                                                             sample_num_rows=sample_num_rows)

        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)

    def test_train_copulagan_success(self):
        # 测试CopulaGAN模型，训练完成后生成仿真数据
        real_data_file_path = "tests/data/adult.csv"
        model_save_path = "tests/models/copula_gan.pkl"
        tabel_type = "tabular"
        model_type = "CopulaGAN"
        anonymize_fields = "{'native-country':'country'}"
        sampling_or_not = True
        sample_num_rows = 100
        status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                             model_save_path=model_save_path,
                                                             tabel_type=tabel_type,
                                                             model_type=model_type,
                                                             anonymize_fields=anonymize_fields,
                                                             sampling_or_not=sampling_or_not,
                                                             sample_num_rows=sample_num_rows)

        self.assertEqual(status.code, 0)
        self.assertEqual(status.msg, '生成仿真数据成功！')
        print("synthetic_data:\n", synthetic_data)
        print("privacy_score:\n", privacy_score)


if __name__ == '__main__':
    unittest.main()
