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

#from grpc_module.Model import synthetic_model_pb2

from  grpc_module.Model import synthetic_model_pb2
from syntheticdata.conf import config
from syntheticdata.conf import fake

LOGGER = logging.getLogger('SyntheticData')


def generate_synthetic_model(status,
                             to_synthetic_file_path,
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

    LOGGER.info('start training......')

    try:
        data = pd.read_csv(to_synthetic_file_path, dtype=str)
        data_columns = data.columns
    except Exception as error:
        LOGGER.error(error)
        status.code = synthetic_model_pb2.FILE_READ_ERROR
        status.msg = '待识别数据表文件{}读取失败。'.format(to_synthetic_file_path)
        return status

    LOGGER.info('table columns: {}'.format(len(data.columns)))
    LOGGER.info('table rows: {}'.format(len(data)))

    #随机获取n条数据作为训练数据
    train_num = min(len(data), config.MAX_ROWS)
    data_table = data.sample(train_num)
    LOGGER.info('Randomly select {} rows as train data from original {} rows'.format(train_num, len(data)))


    # 校验primary_key是否在数据中存在
    if primary_key == '':
        LOGGER.info("not exist primary key!")
        primary_key = None
    else:
        try:
            t = data[primary_key].dtype
        except Exception as error:
            status.code = synthetic_model_pb2.PRIMARY_NOT_EXIST_ERROR
            status.msg = '数据中不存在这样的主键列--{}--,请检查！'.format(primary_key)
            LOGGER.error(error)
            return status

    # 校验anonymize_fields是否在数据表中存在

    if anonymize_fields == '':
        LOGGER.info("not exist anonymize_fields!")
        anonymize_fields = None

    else:
        try:
            anonymize_fields = eval(anonymize_fields)
            for col, fake_col in anonymize_fields.items():
                if col not in data_columns:
                    status.code = synthetic_model_pb2.Anonymize_Fields_ERROR
                    status.msg = '{anonymize_fields}中含有原始数据表中不存在的字段列，请检查！'.format(anonymize_fields)
                    LOGGER.error(TypeError(f'column name \"{col}\" not in data.columns, Please check！'))
                    return status

                if fake_col not in fake.FAKE_LIST:
                    status.code = synthetic_model_pb2.Anonymize_Fields_FAKER_ERROR
                    status.msg = '当前不支持该名称 {} 表示的敏感字段匿名，请检查！'.format(fake_col)
                    LOGGER.error(TypeError(f'fake column name \"{col}\" not in supported list, Please check！'))
                    return status

        except Exception as error:
            status.code = synthetic_model_pb2.Anonymize_Fields_ERROR
            status.msg = '{anonymize_fields}不符合格式要求，请检查！'.format(anonymize_fields)
            LOGGER.error(error)
            return status

    # tabel_type、 model_type 变成类

    if tabel_type == "tabular":
        if model_type == "TVAE":
            model = TVAE(anonymize_fields=anonymize_fields, primary_key=primary_key, batch_size=500, epochs=10)

            try:
                model.fit(data_table)

                #

            except Exception as error:
                status.code = synthetic_model_pb2.MODEL_TRAIN_ERROR
                status.msg = '模型训练失败'.format(model_type)
                LOGGER.error(error)
                return status

            try:
                model.save(model_save_path)

            except Exception as error:
                #print(e.__class__.__name__, e)

                status.code = synthetic_model_pb2.MODEL_SAVE_ERROR
                status.msg = '模型保存失败，请检查路径{}！'.format(model_save_path)
                LOGGER.error(error)
                return status

            LOGGER.info('training finished.')

            if status.code == synthetic_model_pb2.OK:
                status.msg = '模型训练成功，并成功保存！'
            return status


        elif model_type == "CTGAN":
            model = CTGAN(anonymize_fields=anonymize_fields, primary_key=primary_key, batch_size=500, epochs=300)

            try:
                model.fit(data)

            except Exception as error:
                # print(e.__class__.__name__, e)
                status.code = synthetic_model_pb2.MODEL_TYPE_ERROR
                status.msg = '当前不支持该类模型{}，请从["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"]选择一种！'.format(model_type)
                LOGGER.error(error)
                return status

            try:
                model.save(model_save_path)

            except Exception as error:
                # print(e.__class__.__name__, e)

                status.code = synthetic_model_pb2.MODEL_SAVE_ERROR
                status.msg = '模型保存失败，请检查路径{}！'.format(model_save_path)
                LOGGER.error(error)
                return status

            LOGGER.info('training finished.')

            if status.code == synthetic_model_pb2.OK:
                status.msg = '模型训练成功，并成功保存！'
            return status


        elif model_type == "GaussianCopula":
            pass

        elif model_type == "CopulaGAN":
            pass

        else:
            status.code = synthetic_model_pb2.MODEL_TYPE_ERROR
            status.msg = '当前不支持该类模型{}，请从["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"]选择一种！'.format(model_type)
            LOGGER.error('Not exist this model type {},   must be  one of the list ["GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"] '.format(model_type))
            return status



    elif tabel_type == 'relational':
        pass

    elif tabel_type == "timeseries":
        pass

    else:
        status.code = synthetic_model_pb2.TABLE_TYPE_ERROR
        status.msg = '当前不支持该类型的数据表{}，请从["tabular",  "relational",  "timeseries"]选择一种！'.format(model_type)
        LOGGER.error(
            'Not exist this table type {},   must be  one of the list ["tabular",  "relational",  "timeseries"] '.format(
                tabel_type))
        return status





