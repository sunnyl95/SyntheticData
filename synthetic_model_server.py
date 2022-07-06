import os
import json
from pathlib import Path
import logging
from logging.config import fileConfig
import grpc
from concurrent import futures
from grpc_module.Model import synthetic_model_pb2
from grpc_module.Model import synthetic_model_pb2_grpc
from syntheticdata import synthetic_model

LOGGING_CONF_FILE = 'synthetic_model_logging.ini'
fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
LOGGER = logging.getLogger('SyntheticData')


def check_param(status, to_synthetic_file_path, model_save_path, tabel_type, model_type, primary_key, anonymize_fields):
    if to_synthetic_file_path == '':
        status.code = synthetic_model_pb2.PARAMETER_ERROR
        status.msg = '待仿真数据表文件参数 to_synthetic_file_path 不能为空。'
        return status

    if model_save_path == '':
        status.code = synthetic_model_pb2.PARAMETER_ERROR
        status.msg = '待模型保存路径参数 model_save_path 不能为空。'
        return status

    if tabel_type == '':
        status.code = synthetic_model_pb2.PARAMETER_ERROR
        status.msg = '数据表类型参数 tabel_type 不能为空。'
        return status

    if model_type == '':
        status.code = synthetic_model_pb2.PARAMETER_ERROR
        status.msg = '模型类型参数 model_type 不能为空。'
        return status

    if not to_synthetic_file_path.endswith('.csv'):
        status.code = synthetic_model_pb2.PARAMETER_ERROR
        status.msg = '待仿真数据文件{}不是csv文件，目前仅支持csv文件。'.format(to_synthetic_file_path)
        return status

    status.code = synthetic_model_pb2.OK
    return status


class ModelGenerator(synthetic_model_pb2_grpc.SyntheticModelServiceServicer):
    def SyntheticModel(self, request, context):
        LOGGER.info("------ 接收仿真数据生成模型训练参数:")
        to_synthetic_file_path = request.to_synthetic_file_path
        model_save_path = request.model_save_path
        tabel_type = request.tabel_type
        model_type = request.model_type
        primary_key = request.primary_key
        anonymize_fields = request.anonymize_fields

        LOGGER.info("   to_synthetic_file_path: {}".format(to_synthetic_file_path))
        LOGGER.info("   model_save_path: {}".format(model_save_path))
        LOGGER.info("   tabel_type: {}".format(tabel_type))
        LOGGER.info("   model_type: {}".format(model_type))
        LOGGER.info("   primary_key: {}".format(primary_key))
        LOGGER.info("   anonymize_fields: {}".format(anonymize_fields))

        status = synthetic_model_pb2.Status()
        status.code = synthetic_model_pb2.OK
        status = check_param(status, to_synthetic_file_path, model_save_path, tabel_type, model_type, primary_key,
                             anonymize_fields)
        if status.code == synthetic_model_pb2.OK:
            status = synthetic_model.generate_synthetic_model(status,
                                                                  to_synthetic_file_path,
                                                                  model_save_path,
                                                                  tabel_type,
                                                                  model_type,
                                                                  primary_key,
                                                                  anonymize_fields)

        return synthetic_model_pb2.SyntheticResponse(status=status)

def main():
    LOGGER.info('启动服务，服务监听端口为:40051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    synthetic_model_pb2_grpc.add_SyntheticModelServiceServicer_to_server(ModelGenerator(), server)
    server.add_insecure_port('[::]:40051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
