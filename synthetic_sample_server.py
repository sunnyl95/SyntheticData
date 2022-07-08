import os
import json
from pathlib import Path
import logging
from logging.config import fileConfig
import grpc
from concurrent import futures
from grpc_module import SyntheticData_pb2_grpc
from grpc_module import SyntheticData_pb2
from syntheticdata import SyntheticDataModel

LOGGING_CONF_FILE = 'logging.ini'
fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
LOGGER = logging.getLogger('SyntheticData')


def check_param(status, model_save_path, sample_num_rows):
    if model_save_path == '':
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '待模型保存路径参数 model_save_path 不能为空。'
        return status

    if (sample_num_rows == '') or (not str(sample_num_rows).isdecimal()):
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = 'sampling_or_not为True时，参数 sample_num_rows 不能为空，且sample_num_rows必须为正整数。'
        return status

    status.code = SyntheticData_pb2.OK
    return status


class Generator(SyntheticData_pb2_grpc.SyntheticServiceServicer):
    def SyntheticSample(self, request, context):
        LOGGER.info("------ 接收仿真数据训练模型服务参数:")

        model_save_path = request.model_save_path
        sample_num_rows = request.sample_num_rows

        LOGGER.info("   model_save_path: {}".format(model_save_path))
        LOGGER.info("   sample_num_rows: {}".format(sample_num_rows))



        status = SyntheticData_pb2.Status()
        status.code = SyntheticData_pb2.OK
        status = check_param(status, model_save_path, sample_num_rows)

        synthetic_data = None
        privacy_score = None
        if status.code == SyntheticData_pb2.OK:
            status, synthetic_data, privacy_score = SyntheticDataModel.SyntheticDataModel(status,model_save_path=model_save_path,
                                                                                          sample_num_rows=sample_num_rows).sampling_task()

        return SyntheticData_pb2.SyntheticResponse(status=status, synthetic_data=synthetic_data,
                                                   privacy_score=privacy_score)


def main():
    LOGGER.info('启动服务，服务监听端口为:40052')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    SyntheticData_pb2_grpc.add_SyntheticServiceServicer_to_server(Generator(), server)
    server.add_insecure_port('[::]:40052')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
