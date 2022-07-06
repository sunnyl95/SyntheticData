# -*- coding: utf-8 -*-

import logging
import pickle
import warnings
import pandas as pd
from pandas import DataFrame

from syntheticdata.errors import NotFittedError
from syntheticdata.utils import get_package_versions, throw_version_mismatch_warning

from syntheticdata.tabular.model_tvae import TVAE
from syntheticdata.tabular.model_ctgan import CTGAN

from syntheticdata.metrics.tabular import NumericalLR,CategoricalEnsemble,CategoricalKNN

from grpc_module.Sample import  synthetic_sample_pb2
from syntheticdata.conf import config
from syntheticdata.conf import fake

LOGGER = logging.getLogger('SyntheticData')

def get_privacy_score(real_data,  synthetic_data, anonymize_fileds):
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
    @param anonymize_fileds(list):
        敏感数据列, default=None
    @return: score(float)
    '''

    if anonymize_fileds is None:
        LOGGER.INFO("The Data don't have sensitivate columns,so it don't have privacy_score")
        return  None

    #空值处理
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

    real_data = fill_null(real_data)
    synthetic_data = fill_null(synthetic_data)


    numberical_columns = []
    numberical_privacy = 0
    categorical_columns = []
    categorical_privacy = 0
    for column in anonymize_fileds:
        dtype = real_data[column].dtype.name
        #1、Numberical Type
        if dtype.startswith('i') or dtype.startswith('f') :
            numberical_columns.append(column)

        # 2、Categorical Type
        elif dtype.startswith('b') or  dtype.startswith('o') :
            categorical_columns.append(column)
    if len(numberical_columns) > 0:
        key_fields = real_data.select_dtypes(include='int').columns +  real_data.select_dtypes(include='float').columns
        key_fields =  list(set(key_fields)-set(numberical_columns))

        numberical_privacy = CategoricalKNN.compute(real_data, synthetic_data, key_fields= key_fields ,
                               sensitive_fields= numberical_columns)

    if  len(categorical_columns) > 0:
        key_fields = list(real_data.select_dtypes(include='bool').columns) + list(real_data.select_dtypes(include='object').columns)
        key_fields = list(set(key_fields) - set(categorical_columns))

        categorical_privacy = CategoricalKNN.compute(real_data, synthetic_data, key_fields=key_fields,
                                                    sensitive_fields=categorical_columns)

    return (numberical_privacy + categorical_privacy) / 2



def generate_synthetic_sample(status,
                              real_data_file_path,
                              model_path,
                              model_type='TVAE',
                              sample_number=1000,  # defalut:1000
                              anonymize_fields=None
                              ):

    """
    @param real_data_file_path(string):
        原始真实数据集文件路径，用于对比仿真数据与真实数据的差距，目前仅支持csv文件
    @param model_path(string):
        用于生成仿真数据的模型（已经训练好的模型）
    @param model_type(string):
        模型类型,["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"], 要与model_path提供的数据集保持一致
    @param sample_number(int):
        欲生成的仿真数据条数， default=1000
    @param anonymize_fields(list):
        敏感数据列，用于衡量合成数据的隐私性，即通过合成数据，攻击者是否可以预测真实数据集中的敏感数据列。
        模型通过合成数据的非敏感字段学习拟合敏感数据列，然后用模型预测真实数据的敏感字段，评估预测值在真实数据上的准确性。

    """
    LOGGER.info('start sampling......')

    synthetic_data = None
    privacy_score = None
    try:
        data = pd.read_csv(real_data_file_path, dtype=str)
        data_columns = data.columns
    except Exception as error:
        LOGGER.error(error)
        status.code = synthetic_sample_pb2.FILE_READ_ERROR
        status.msg = '待识别数据表文件{}读取失败。'.format(real_data_file_path)
        return status, synthetic_data, privacy_score

        # 校验anonymize_fields是否在数据表中存在

        if anonymize_fields == '':
            LOGGER.info("not exist anonymize_fields!")
            anonymize_fields = None
        else:
            try:
                anonymize_fields = eval(anonymize_fields)
                for col in anonymize_fields:
                    if col not in data_columns:
                        status.code = synthetic_model_pb2.Anonymize_Fields_ERROR
                        status.msg = '{anonymize_fields}中含有原始数据表中不存在的字段列，请检查！'.format(anonymize_fields)
                        LOGGER.error(TypeError(f'column name \"{col}\" not in data.columns, Please check！'))
                        return status, synthetic_data, privacy_score
            except Exception as error:
                status.code = synthetic_model_pb2.Anonymize_Fields_ERROR
                status.msg = '{anonymize_fields}不符合格式要求，请检查！'.format(anonymize_fields)
                LOGGER.error(error)
                return status, synthetic_data, privacy_score

    if model_type == "TVAE":
        try:
            model = TVAE.load(model_path)
        except Exception as error:
            status.code = synthetic_sample_pb2.MODEL_LOAD_ERROR
            status.msg = '模型加载失败，model_path传入的模型有问题，或者model_path模型与model_type模型不匹配'.format(model_type)
            LOGGER.error(error)
            return status, synthetic_data, privacy_score

        try:
            synthetic_data = model.sample(num_rows=sample_number)
        except Exception as error:
            status.code = synthetic_sample_pb2.SAMPLE_DATA_GENERATOR_ERROR
            status.msg = '生成仿真样本数据失败'.format(model_type)
            LOGGER.error(error)
            return status, synthetic_data, privacy_score

        try:
            privacy_score = get_privacy_score(data, synthetic_data, anonymize_fileds=anonymize_fields)

        except Exception as error:
            status.code = synthetic_sample_pb2.PRIVACY_SCORE_ERROR
            status.msg = '计算仿真数据的隐私性失败'.format(model_type)
            LOGGER.error(error)
            return status, synthetic_data, privacy_score

        LOGGER.info('sampling finished.')

        if status.code == synthetic_sample_pb2.OK:
            status.msg = '成功生成{}条仿真数据'.format(sample_number)

        synthetic_data = synthetic_data.to_json()  #dataframe转string

        return status, synthetic_data, privacy_score

    elif model_type == "CTGAN":
        pass

    elif model_type == "GaussianCopula":
        pass

    elif model_type == "CopulaGAN":
        pass

    else:
        status.code = synthetic_model_pb2.MODEL_TYPE_ERROR
        status.msg = '当前不支持该类模型{}，请从["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"]选择一种！'.format(model_type)
        LOGGER.error(
            'Not exist this model type {},   must be  one of the list ["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"] '.format(
                model_type))
        return status, synthetic_data, privacy_score
