# -*- coding: utf-8 -*-

"""Main SDV module."""
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


def generate_synthetic_model(data,
                             model_save_path,
                             tabel_type='tabular',
                             model_type='TVAE',
                             primary_key=None,
                             anonymize_fields=None):
    """

    @param data:(dataframe) :
        原始仿真数据
    @param model_save_path( string ):
        模型保存路径
    @param tabel_type(string):
        数据表类型，分为三类：["tabular":"单表"， "relational":"关系表", "timeseries":"时间序列表"],default="tabular"
    @param model_type(string):
        模型类型，针对单表数据提供四种模型["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"], default="TVAE"
    @param primary_key(string):
        主键列名,若没有主键，则为默认值 default= None
    @param anonymize_fields():
        敏感列名，以list格式传递，要求list长度大于等于1。若没有敏感数据列，则为默认值 default= None
    """
    # 校验数据类型是否是dataframe
    if isinstance(data, pd.DataFrame):
        data_columns = data.columns
    else:
        raise TypeError('``data`` should be either pd.DataFrame')

    # 校验primary_key是否在数据中存在
    if (primary_key is not None) and (primary_key not in data_columns):
        raise TypeError(f'{primary_key} not in data.columns, Please check！')

    # 校验anonymize_fields是否在数据表中存在
    if anonymize_fields is not None:
        if not isinstance(anonymize_fields, dict):
            raise TypeError('``anonymize_fields`` should be either dict')
        else:
            for  col in anonymize_fields:
                if col not in data.columns:
                    raise TypeError(f'column name \"{col}\" not in data.columns, Please check！')

    if tabel_type == "tabular":
        if model_type == "TVAE":
            model = TVAE(anonymize_fields=anonymize_fields, primary_key=primary_key, batch_size=500, epochs=10)

            try:
                model.fit(data)
                model.save(model_save_path)
                return "Success"
            except Exception as e:
                print(e.__class__.__name__, e)
                return "Fail"
        if model_type == "CTGAN":
            model = CTGAN(anonymize_fields=anonymize_fields, primary_key=primary_key, batch_size=500, epochs=300)

            try:
                model.fit(data)
                model.save(model_save_path)
                return "Success"
            except:
                return "Fail"

        if model_type == "GaussianCopula":
            pass

        if model_type == "CopulaGAN":
            pass
    if tabel_type == 'relational':
        pass

    if tabel_type == "timeseries":
        pass


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
        logging.INFO("The Data don't have sensitivate columns,so it don't have privacy_score")
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



def generate_synthetic_sample(real_data,
                              model_path,
                              model_type='TVAE',
                              sample_number=1000,  # defalut:1000
                              anonymize_fields=None
                              ):

    """
    @param real_data(pd.DataFrame):
        原始真实数据集，用于对比仿真数据与真实数据的差距
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
    # 校验数据类型是否是dataframe
    if isinstance(real_data, pd.DataFrame):
        data_columns = real_data.columns
    else:
        raise TypeError('``real_data`` should be either pd.DataFrame')


    # 校验anonymize_fields是否在数据表中存在
    if anonymize_fields is not None:
        if not isinstance(anonymize_fields, list):
            raise TypeError('``anonymize_fields`` should be either list')
        else:
            for col in anonymize_fields:
                if col not in real_data.columns:
                    raise TypeError(f'column name \"{col}\" not in data.columns, Please check！')

    if model_type == "TVAE":
        try:
            model = TVAE.load(model_path)

            synthetic_data = model.sample(num_rows=sample_number)

            privacy_score = get_privacy_score(real_data, synthetic_data, anonymize_fileds=anonymize_fields)


            return synthetic_data, privacy_score
        except Exception as e:
            print(e.__class__.__name__, e)
            return "Fail"

    if model_type == "CTGAN":
        try:
            model = CTGAN.load(model_path)
            synthetic_data = model.sample(num_rows=sample_number)

            privacy_score = get_privacy_score(real_data, synthetic_data, anonymize_fileds=anonymize_fields)

            return synthetic_data, privacy_score

            return "Success"
        except Exception as e:
            print(e.__class__.__name__, e)
            return "Fail"


data = pd.read_csv("../example/data/adult.csv")
data = data.sample(5000)
model_path="./model.pkl"
generate_synthetic_model(data=data, model_save_path = model_path, anonymize_fields={"occupation":"job"})
syn_data, privacy_score = generate_synthetic_sample(real_data = data,
                                                    model_path= model_path,
                                                    model_type="TVAE",
                                                    sample_number=1000,
                                                    anonymize_fields=["income"])
print(syn_data)
print(f"隐私保护得分：{privacy_score}")
