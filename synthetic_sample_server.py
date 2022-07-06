import os
import json
from pathlib import Path
import logging
from logging.config import fileConfig
import grpc
from concurrent import futures
from grpc_module.Sample import synthetic_sample_pb2
from grpc_module.Sample import synthetic_sample_pb2_grpc
from syntheticdata import synthetic_sample

LOGGING_CONF_FILE = 'synthetic_model_logging.ini'
fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
LOGGER = logging.getLogger('SyntheticData')



def check_param(status, real_data_file_path, model_path, model_type, sample_number, anonymize_fields):
    if real_data_file_path == '':
        status.code = synthetic_sample_pb2.PARAMETER_ERROR
        status.msg = '真实数据的文件参数 real_data_file_path 不能为空。'
        return status

    if model_path == '':
        status.code = synthetic_sample_pb2.PARAMETER_ERROR
        status.msg = '模型路径参数 model_path 不能为空。'
        return status

    if model_type == '':
        status.code = synthetic_sample_pb2.PARAMETER_ERROR
        status.msg = '模型类型参数 model_type 不能为空。'
        return status

    if not real_data_file_path.endswith('.csv'):
        status.code = synthetic_sample_pb2.PARAMETER_ERROR
        status.msg = '真实数据的文件{}不是csv文件，目前仅支持csv文件。'.format(real_data_file_path )
        return status

    if (not isinstance(sample_number, int)) or (sample_number <= 0):
        status.code = synthetic_sample_pb2.PARAMETER_ERROR
        status.msg = '生成仿真数据的数目参数 {} 不符合要求，数据要求是大于0的正整数 '.format(sample_number )
        return status

    status.code = synthetic_sample_pb2.OK
    return status


class SampleGenerator(synthetic_sample_pb2_grpc.SyntheticSampleServiceServicer):
    def SyntheticSample(self, request, context):
        LOGGER.info("------ 接收仿真数据生成样本数据参数:")
        real_data_file_path = request.real_data_file_path
        model_path = request.model_path
        model_type = request.model_type
        sample_number = request.sample_number
        anonymize_fields = request.anonymize_fields

        LOGGER.info("   real_data_file_path: {}".format(real_data_file_path))
        LOGGER.info("   model_type: {}".format(model_path))
        LOGGER.info("   model_type: {}".format(model_type))
        LOGGER.info("   sample_number: {}".format(sample_number))
        LOGGER.info("   anonymize_fields: {}".format(anonymize_fields))

        status = synthetic_sample_pb2.Status()
        status.code = synthetic_sample_pb2.OK
        status = check_param(status, real_data_file_path, model_path,  model_type, sample_number,anonymize_fields)
        if status.code == synthetic_sample_pb2.OK:
            status, synthetic_data, privacy_score = synthetic_sample.generate_synthetic_sample(status, real_data_file_path, model_path,  model_type, sample_number,anonymize_fields)

        return synthetic_sample_pb2.SyntheticResponse(status=status, synthetic_data=synthetic_data, privacy_score=privacy_score)

def main():
    LOGGER.info('启动服务，服务监听端口为:40052')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    synthetic_sample_pb2_grpc.add_SyntheticSampleServiceServicer_to_server(SampleGenerator(), server)
    server.add_insecure_port('[::]:40052')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
