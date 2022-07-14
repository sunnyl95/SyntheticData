# -*- coding: utf-8 -*-

import logging
import pickle
import warnings
import pandas as pd

from grpc_module import SyntheticData_pb2
from grpc_module import SyntheticData_pb2_grpc
from syntheticdata.conf import config
from syntheticdata.conf import fake
from syntheticdata.MODEL import all_model
from syntheticdata.metrics.tabular import NumericalLR, CategoricalEnsemble, CategoricalKNN
from syntheticdata.tabular.model_tvae import TVAE

LOGGER = logging.getLogger('SyntheticData')


class SyntheticDataModel:
    def __init__(self,
                 status,
                 task_type = ""
                 real_data_file_path="",
                 model_save_path="",
                 tabel_type='tabular',
                 model_type='TVAE',
                 primary_key=None,
                 anonymize_fields=None,
                 sampling_or_not=True,
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
            self.msg = '待识别数据表文件{}读取失败。'.format(self.real_data_file_path)

        LOGGER.info('   table columns: {}'.format(len(self.data.columns)))
        LOGGER.info('   table rows: {}'.format(len(self.data)))

        # 随机获取n条数据作为训练数据
        train_num = min(len(self.data), config.MAX_ROWS)
        LOGGER.info('Randomly select {} rows as train data from original {} rows'.format(train_num, len(self.data)))
        self.data = self.data.sample(train_num)

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
                pass

        # 校验anonymize_fields是否在数据表中存在

        if self.anonymize_fields == '':
            LOGGER.info("   not exist anonymize_fields!")
            self.anonymize_fields = None
        else:
            try:
                anonymize_fields = eval(self.anonymize_fields)
                for col, fake_col in anonymize_fields.items():
                    if col not in data_columns:
                        self.status.code = SyntheticData_pb2.Anonymize_Fields_ERROR
                        self.status.msg = '{anonymize_fields}中含有原始数据表中不存在的字段列，请检查！'.format(self.anonymize_fields)
                        LOGGER.error(TypeError(f'column name \"{col}\" not in data.columns, Please check！'))
                        pass

                    if fake_col not in fake.FAKE_LIST:
                        self.status.code = SyntheticData_pb2.Anonymize_Fields_FAKER_ERROR
                        self.status.msg = '当前不支持该名称 {} 表示的敏感字段匿名，请检查！'.format(fake_col)
                        LOGGER.error(TypeError(f'fake column name \"{col}\" not in supported list, Please check！'))
                        pass
            except Exception as error:
                self.status.code = SyntheticData_pb2.Anonymize_Fields_ERROR
                self.status.msg = '{anonymize_fields}不符合格式要求，请检查！'.format(self.anonymize_fields)
                LOGGER.error(error)
                pass

    def get_model(self):
        if self.status.code == SyntheticData_pb2.OK:
            try:
                self._model = all_model[self.table_type][self.model_type]
                print(self._model)
                self._model = self._model(primary_key=self.primary_key, anonymize_fields=self.anonymize_fields,
                                          verbose=True, epochs=10)  # 默认打印训练过程

            except Exception as error:
                self.status.code = SyntheticData_pb2.MODEL_INITIALIZATION_ERROR
                self.status.msg = '模型初始化失败,数据表类型{}有误（当前仅支持"tabular"），或模型类型{}有误，或者两个参数不匹配！'.format(self.table_type,
                                                                                                  self.model_type)

    def fit_model(self):
        try:
            self._model.fit(self.data)
        except Exception as error:
            self.status.code = SyntheticData_pb2.MODEL_TRAIN_ERROR
            self.status.msg = '模型训练失败'
            LOGGER.error(error)

    def save_model(self):
        try:
            self.data = None  # 删除数据后保存模型
            tmp_status = self.status
            self.status = None  # 无需保存状态
            with open(self.model_save_path, 'wb') as output:
                pickle.dump(self, output)
            self.status = tmp_status
        except Exception as error:
            self.status.code = SyntheticData_pb2.MODEL_SAVE_ERROR
            self.status.msg = '模型保存失败，请检查路径{}！'.format(self.model_save_path)
            LOGGER.error(error)

        LOGGER.info('模型保存成功！')

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

    def sample(self):
        synthetic_data = pd.DataFrame()
        if self._model is not None and self.trained:
            try:
                synthetic_data = self._model.sample(num_rows=self.sample_num_rows)
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

        2、字段类型要求
        若敏感数据列是Categorical类型，则只能用Categorical类型数据训练模型
        若敏感数据列是Numerical类型，则只能用Numerical类型数据训练模型
        当前不支持两种类型的数据混合训练模型，这也是后期需优化的点。（目前只能用同类型的字段预测同类型的敏感字段，然而正常情况会利用所有字段预测目标字段，因此该方式从理论上降低了隐私性评估的标准，需进一步完善）


        @param real_data(dataframe):
            真实数据
        @param synthetic_data(dataframe):
            合成数据
        @param anonymize_fields(list):
            敏感数据列, default=None
        @return: score(float)
        '''

        if self.anonymize_fields is None:
            LOGGER.info("The Data don't have sensitivate columns,so it don't have privacy_score")
            return None

        privacy_score = None

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
            if self.data is not None:
                try:
                    real_data = pd.read_csv(self.real_data_file_path, dtype=str)
                except Exception as error:
                    LOGGER.error(error)
                    self.status.code = SyntheticData_pb2.FILE_READ_ERROR
                    self.msg = '待识别数据表文件{}读取失败。'.format(self.real_data_file_path)
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

                numberical_privacy = CategoricalKNN.compute(real_data, synthetic_data, key_fields=key_fields,
                                                            sensitive_fields=numberical_columns)

            if len(categorical_columns) > 0:
                key_fields = list(real_data.select_dtypes(include='bool').columns) + list(
                    real_data.select_dtypes(include='object').columns)
                key_fields = list(set(key_fields) - set(categorical_columns))

                categorical_privacy = CategoricalKNN.compute(real_data, synthetic_data, key_fields=key_fields,
                                                             sensitive_fields=categorical_columns)

            return (numberical_privacy + categorical_privacy) / 2

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
        LOGGER.info('training finished.')
        self.trained = True
        self.save_model()

        if self.status.code == SyntheticData_pb2.OK:
            self.status.msg = '模型训练成功，并成功保存！'

        # 5.生成仿真数据样本
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        if self.sampling_or_not:
            return self.sampling_task()

        return self.status, None, None

    def sampling_task(self):
        synthetic_data = pd.DataFrame()
        privacy_score = None
        # 1.加载模型
        model = self.load_model()

        print(model)

        # 2.生成仿真数据
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, None, None
        synthetic_data = model.sample()

        # 3.计算隐私性保护得分
        if self.status.code != SyntheticData_pb2.OK:
            return self.status, synthetic_data.to_json(), privacy_score
        privacy_score = model.get_privacy_score(synthetic_data=synthetic_data)

        # 4.返回结果
        if self.status.code == SyntheticData_pb2.OK:
            self.status.msg = '生成仿真数据成功！'
        return self.status, synthetic_data.to_json(), privacy_score
