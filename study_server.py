import os
import json
from pathlib import Path
import logging
from logging.config import fileConfig
import grpc
from concurrent import futures
from grpc_module import study_pb2_grpc
from grpc_module import study_pb2
from syntheticdata import synthetic_sample


LOGGING_CONF_FILE = 'logging-study.ini'
fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
LOGGER = logging.getLogger('Study')

def run(input):

    return  input.key, input.value

class Generator(study_pb2_grpc.StudyServiceServicer):
    def Study(self, request, context):
        LOGGER.info("------ 接收仿真数据生成样本数据参数:")
        input = request.input

        key, value = run(input)

        return study_pb2.Response(key=key, value=value)

def main():
    LOGGER.info('启动服务，服务监听端口为:40054')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    study_pb2_grpc.add_StudyServiceServicer_to_server(Generator(), server)
    server.add_insecure_port('[::]:40054')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
