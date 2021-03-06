# -*- coding: utf-8 -*-

import logging
import pickle
import pandas as pd

from grpc_module import SyntheticData_pb2
from syntheticdata.config import config
from syntheticdata.config import fake
from syntheticdata.config.model_dict import model_dict
from syntheticdata.metrics.single_table import CategoricalGeneralizedCAP, NumericalMLP

LOGGER = logging.getLogger('SyntheticData')


class SyntheticDataModel:
    def __init__(self,
                 status,
                 real_data_file_path=None,
                 model_save_path=None,
                 tabel_type=None,
                 model_type=None,
                 primary_key=None,
                 anonymize_fields=None,
                 sampling_or_not=False,
                 sample_num_rows=1000):
        '''

        @param status:
        @param real_data_file_path:
        @param model_save_path:
        @param tabel_type:
        @param model_type:
        @param primary_key:
        @param anonymize_fields:
        @param sample_num_rows:
        '''
        self.status = status
        self.real_data_file_path = real_data_file_path
        self.model_save_path = model_save_path
        self.table_type = tabel_type
        self.model_type = model_type
        self.primary_key = primary_key
        self.anonymize_fields = anonymize_fields

        self.sampling_or_not = sampling_or_not
        self.sample_num_rows = sample_num_rows
        self.data = None
        self.trained = False
        self._model = None

    def param_check(self):
        try:
            self.data = pd.read_csv(self.real_data_file_path, dtype=str)
            data_columns = self.data.columns
        except Exception as error:
            LOGGER.error(error)
            self.status.code = SyntheticData_pb2.FILE_READ_ERROR
            self.status.msg = '待识别数据表文件{}读取失败，请检测路径!'.format(self.real_data_file_path)

            return

        LOGGER.info('   table columns: {}'.format(len(self.data.columns)))
        LOGGER.info('   table rows: {}'.format(len(self.data)))

        # # 随机获取n条数据作为训练数据,采样的数量和策略会影响仿真效果，因此暂不进行抽样训练，等调研后给出实施方案
        # train_num = min(len(self.data), config.MAX_ROWS)
        # LOGGER.info('   Randomly select {} rows as train data from original {} rows'.format(train_num, len(self.data)))
        # self.data = self.data.sample(train_num)

        # 校验primary_key是否在数据中存在
        if self.primary_key == '':
            LOGGER.info("   not exist primary key!")
            self.primary_key = None
        else:
            try:
                t = self.data[self.primary_key].dtype
            except Exception as error:
                self.status.code = SyntheticData_pb2.PRIMARY_NOT_EXIST_ERROR
                self.status.msg = '数据中不存在这样的主键列--{}--,请检查！'.format(self.primary_key)
                LOGGER.error(error)
                return

        # 校验anonymize_fields是否在数据表中存在

        if self.anonymize_fields == '':
            LOGGER.info("   not exist anonymize_fields!")
            self.anonymize_fields = None
        else:
            try:
                self.anonymize_fields = eval(self.anonymize_fields)
                for col, fake_col in self.anonymize_fields.items():
                    if col not in data_columns:
                        self.status.code = SyntheticData_pb2.Anonymize_Fields_ERROR
                        self.status.msg = '{}中含有原始数据表中不存在的字段列，请检查！'.format(self.anonymize_fields)
                        LOGGER.error(TypeError(f'column name \"{col}\" not in data.columns, Please check！'))
                        return

                    if fake_col not in fake.FAKE_LIST:
                        self.status.code = SyntheticData_pb2.Anonymize_Fields_FAKER_ERROR
                        self.status.msg = '当前不支持该名称 {} 表示的敏感字段匿名，请检查！'.format(self.anonymize_fields)
                        LOGGER.error(TypeError(f'fake column name \"{col}\" not in supported list, Please check！'))
                        return
            except Exception as error:
                self.status.code = SyntheticData_pb2.PARAMETER_ERROR
                self.status.msg = 'anonymize_fields参数{}不符合格式要求，请检查！'.format(self.anonymize_fields)
                LOGGER.error(error)
                return

    def get_model(self):
        LOGGER.info("   get model......")

        if self.status.code == SyntheticData_pb2.OK:
            try:
                self._model = model_dict[self.table_type][self.model_type]
                self._model = self._model(primary_key=self.primary_key, anonymize_fields=self.anonymize_fields,
                                          verbose=True)  # 默认打印训练过程  #epochs=300

            except Exception as error:
                self.status.code = SyntheticData_pb2.MODEL_INITIALIZATION_ERROR
                self.status.msg = '模型初始化失败,数据表类型{}有误（当前仅支持"tabular"），或模型类型{}有误，或者两个参数不匹配！'.format(self.table_type,
                                                                                                  self.model_type)

    def fit_model(self):
        LOGGER.info("   train starting......")
        try:
            self._model.fit(self.data)
        except Exception as error:
            self.status.code = SyntheticData_pb2.MODEL_TRAIN_ERROR
            self.status.msg = '模型训练失败'
            LOGGER.error(error)

    def save_model(self):
        LOGGER.info("   save model......")
        try:
            self.data = None  # 删除数据后保存模型

            # 由于状态是基于proto3的变量，pickle无法保存，因此这里先置为空，重新定义两个变量存储下来
            tmp_status = self.status
            self.status_code = self.status.code
            self.status_msg = self.status.msg
            self.status = None

            with open(self.model_save_path, 'wb') as output:
                pickle.dump(self, output)

            self.status = tmp_status
            LOGGER.info('   model save successfully!')
        except Exception as error:
            self.status = tmp_status
            self.status.code = SyntheticData_pb2.MODEL_SAVE_ERROR
            self.status.msg = '模型保存失败，请检查路径{}！'.format(self.model_save_path)
            LOGGER.error(error)

    def get_params(self):
        params = {}
        params['real_data_file_path'] = self.real_data_file_path
        params['model_save_path'] = self.model_save_path
        params['table_type '] = self.table_type
        params['model_type '] = self.model_type
        params['primary_key'] = self.primary_key
        params['anonymize_fields '] = self.anonymize_fields
        params['sampling_or_not'] = self.sampling_or_not
        params['sample_num_rows'] = self.sample_num_rows
        params['trained'] = self.trained

        return params

    def load_model(self):
        model = None
        try:
            with open(self.model_save_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as error:
            self.status.code = SyntheticData_pb2.MODEL_LOAD_ERROR
            self.status.msg = '模型加载失败，模型路径{}不存在或其他错误'.format(self.model_save_path)
            LOGGER.error(error)
        return model

    def sample(self, sample_num_rows):
        if sample_num_rows is None:
            sample_num_rows = self.sample_num_rows

        synthetic_data = pd.DataFrame()
        if self._model is not None and self.trained:
            try:
                synthetic_data = self._model.sample(num_rows=sample_num_rows)
            except Exception as error:
                self.status.code = SyntheticData_pb2.SAMPLE_DATA_GENERATOR_ERROR
                self.status.msg = '生成仿真样本数据失败'
                LOGGER.error(error)
                return synthetic_data
        return synthetic_data

    def get_privacy_score(self, synthetic_data):
        '''
        1、计算合成数据的隐私性
        通过合成数据，攻击者是否可以预测真实数据集中的敏感数据列。
        模型通过合成数据的非敏感字段学习拟合敏感数据列，然后用模型预测真实数据的敏感字段，评估预测值在真实数据上的准确性。
        计算公式：1-match_count/total_count,因此得分越高隐私性保护的越好，得分为0表示全部预测准确，得分为1表示全部预测错误

        2、字段类型要求
        若敏感数据列是Categorical类型，则只能用Categorical类型数据训练模型
        若敏感数据列是Numerical类型，则只能用Numerical类型数据训练模型
        当前不支持两种类型的数据混合训练模型，这也是后期需优化的点。（目前只能用同类型的字段预测同类型的敏感字段，然而正常情况会利用所有字段预测目标字段，因此该方式从理论上降低了隐私性评估的标准，需进一步完善）
        '''

        LOGGER.info("   compute privacy score starting......")
        if self.anonymize_fields is None:
            LOGGER.info("   The Data don't have sensitivate columns,so its privacy_score is 1")
            return 1

        privacy_score = 0

        # 空值处理
        def fill_null(df):
            # bool(2), datetime64[ns](2), float64(6), int64(2), object(5)
            for column in df.columns:
                try:
                    df[column].astype("float64")
                except:
                    pass
                dtype = df[column].dtype.name
                if dtype.startswith("f") or dtype.startswith("i"):
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode().values[0], inplace=True)

            return df

        try:
            if self.data is None:
                try:
                    real_data = pd.read_csv(self.real_data_file_path, dtype=str)
                except Exception as error:
                    LOGGER.error(error)
                    self.status.code = SyntheticData_pb2.FILE_READ_ERROR
                    self.status.msg = '待识别数据表文件{}读取失败，请检测路径!'.format(self.real_data_file_path)
                    return None

            else:
                real_data = fill_null(self.data)

            real_data = fill_null(real_data)

            synthetic_data = fill_null(synthetic_data)

            numberical_columns = []
            numberical_privacy = 0
            categorical_columns = []
            categorical_privacy = 0

            for column in self.anonymize_fields.keys():
                dtype = real_data[column].dtype.name
                # 1、Numberical Type
                if dtype.startswith('i') or dtype.startswith('f'):
                    numberical_columns.append(column)

                # 2、Categorical Type
                elif dtype.startswith('b') or dtype.startswith('o'):
                    categorical_columns.append(column)
            if len(numberical_columns) > 0:
                key_fields = real_data.select_dtypes(include='int').columns + real_data.select_dtypes(
                    include='float').columns
                key_fields = list(set(key_fields) - set(numberical_columns))

                numberical_privacy = NumericalMLP.compute(real_data, synthetic_data, key_fields=key_fields,
                                                          sensitive_fields=numberical_columns)

            if len(categorical_columns) > 0:
                key_fields = list(real_data.select_dtypes(include='bool').columns) + list(
                    real_data.select_dtypes(include='object').columns)
                key_fields = list(set(key_fields) - set(categorical_columns))

                categorical_privacy = CategoricalGeneralizedCAP.compute(real_data, synthetic_data,
                                                                        key_fields=key_fields,
                                                                        sensitive_fields=categorical_columns)

            LOGGER.info("   compute privacy score Finished!")

            if isinstance(categorical_privacy, float) or isinstance(categorical_privacy, int):
                privacy_score += categorical_privacy
            if isinstance(numberical_privacy, float) or isinstance(numberical_privacy, int):
                privacy_score += numberical_privacy

            return privacy_score

        except Exception as error:

            self.status.code = SyntheticData_pb2.PRIVACY_SCORE_ERROR
            self.status.msg = '计算仿真数据的隐私性失败'
            LOGGER.error(error)
            return privacy_score

    def training_task(self):

        # 1.参数校验
        self.param_check()

        # 2.模型实例化
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        self.get_model()

        # 3.模型训练
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        self.fit_model()

        # 4.模型保存
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        LOGGER.info('   training finished.')
        self.trained = True
        self.save_model()

        # 5.生成仿真数据样本
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        if self.sampling_or_not:
            return self.sampling_task()

        self.status.msg = '模型训练成功，并成功保存！（该任务类型是仅训练模型，不生成仿真数据样本，因此隐私性得分为1）'
        #若没有进行仿真数据生成样本数据服务，则隐私性得分为1
        return self.status, None, 1

    def sampling_task(self):
        synthetic_data = pd.DataFrame()
        privacy_score = None
        # 1.加载模型
        model = self.load_model()

        # 2.生成仿真数据
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        synthetic_data = model.sample(self.sample_num_rows)
        self.status.code = model.status_code
        self.status.msg = model.status_msg

        # 3.计算隐私性保护得分
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, synthetic_data.to_json(), privacy_score
        privacy_score = model.get_privacy_score(synthetic_data=synthetic_data)

        self.status.code = model.status_code
        self.status.msg = model.status_msg

        # 4.返回结果
        if self.status.code == SyntheticData_pb2.OK:
            self.status.msg = '生成仿真数据成功！'
        return self.status, synthetic_data.to_json(), privacy_score
