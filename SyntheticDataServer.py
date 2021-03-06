from pathlib import Path
import logging
from logging.config import fileConfig
import grpc
from concurrent import futures
from grpc_module import SyntheticData_pb2_grpc,SyntheticData_pb2, SyntheticDataModel
from syntheticdata.config import config

LOGGING_CONF_FILE = 'log/logging.ini'
fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
LOGGER = logging.getLogger('SyntheticData')

_IP =  config.IP
_PORT = config.PORT

def model_check_param(status, real_data_file_path, model_save_path, tabel_type, model_type, primary_key, anonymize_fields, sampling_or_not,
                sample_num_rows):
    if real_data_file_path == '':
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '待仿真数据表文件参数 real_data_file_path 不能为空。'
        return status

    if model_save_path == '':
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '待模型保存路径参数 model_save_path 不能为空。'
        return status

    if tabel_type == '':
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '数据表类型参数 tabel_type 不能为空。'
        return status

    if model_type == '':
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '模型类型参数 model_type 不能为空。'
        return status

    if not isinstance(sampling_or_not, bool):
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '是否生成仿真数据参数 sampling_or_not 不能为空,且必须为bool类型参数。'
        return status

    if sampling_or_not and (sample_num_rows == '' or (not str(sample_num_rows).isdecimal()) or (sample_num_rows < 1)):
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = 'sampling_or_not为True时，参数 sample_num_rows 不能为空，且sample_num_rows必须为正整数。'
        return status

    if not real_data_file_path.endswith('.csv'):
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '待仿真数据文件{}不是csv文件，目前仅支持csv文件。'.format(real_data_file_path)
        return status


    status.code = SyntheticData_pb2.OK
    return status

def sample_check_param(status, model_save_path, sample_num_rows):
    if model_save_path == '':
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '模型路径参数 model_save_path 不能为空。'
        return status

    if (sample_num_rows == '') or (not str(sample_num_rows).isdecimal()) or (sample_num_rows < 1):
        status.code = SyntheticData_pb2.PARAMETER_ERROR
        status.msg = '参数 sample_num_rows 不能为空，且sample_num_rows必须为正整数。'
        return status

    status.code = SyntheticData_pb2.OK
    return status


class Generator(SyntheticData_pb2_grpc.SyntheticServiceServicer):
    def SyntheticModel(self, request, context):
        LOGGER.info("------ 接收仿真数据训练模型服务参数:")
        real_data_file_path = request.real_data_file_path
        model_save_path = request.model_save_path
        tabel_type = request.tabel_type
        model_type = request.model_type
        primary_key = request.primary_key
        anonymize_fields = request.anonymize_fields
        sampling_or_not = request.sampling_or_not
        sample_num_rows = request.sample_num_rows

        LOGGER.info("   real_data_file_path: {}".format(real_data_file_path))
        LOGGER.info("   model_save_path: {}".format(model_save_path))
        LOGGER.info("   tabel_type: {}".format(tabel_type))
        LOGGER.info("   model_type: {}".format(model_type))
        LOGGER.info("   primary_key: {}".format(primary_key))
        LOGGER.info("   anonymize_fields: {}".format(anonymize_fields))
        LOGGER.info("   sampling_or_not: {}".format(sampling_or_not))
        if sampling_or_not:
            LOGGER.info("   sample_num_rows: {}".format(sample_num_rows))
            LOGGER.info("   Traing and Sampling")
        else:
            LOGGER.info("   Just Traing, Nor Sampling")

        status = SyntheticData_pb2.Status()
        status.code = SyntheticData_pb2.OK
        status = model_check_param(status, real_data_file_path, model_save_path, tabel_type,
                             model_type, primary_key, anonymize_fields, sampling_or_not, sample_num_rows)

        synthetic_data = None
        privacy_score = None
        if status.code == SyntheticData_pb2.OK:
            status, synthetic_data, privacy_score = SyntheticDataModel.SyntheticDataModel(status,
                                                                                          real_data_file_path,
                                                                                          model_save_path,
                                                                                          tabel_type,
                                                                                          model_type,
                                                                                          primary_key,
                                                                                          anonymize_fields,
                                                                                          sampling_or_not,
                                                                                          sample_num_rows).training_task()



        return SyntheticData_pb2.SyntheticResponse(status=status, synthetic_data=synthetic_data,
                                                   privacy_score=privacy_score)

    def SyntheticSample(self, request, context):
        LOGGER.info("------ 接收仿真数据生成样本数据服务参数:")

        model_save_path = request.model_save_path
        sample_num_rows = request.sample_num_rows

        LOGGER.info("   model_save_path: {}".format(model_save_path))
        LOGGER.info("   sample_num_rows: {}".format(sample_num_rows))

        status = SyntheticData_pb2.Status()
        status.code = SyntheticData_pb2.OK
        status = sample_check_param(status, model_save_path, sample_num_rows)

        synthetic_data = None
        privacy_score = None
        if status.code == SyntheticData_pb2.OK:
            status, synthetic_data, privacy_score = SyntheticDataModel.SyntheticDataModel(status, model_save_path=model_save_path,
                                                                                          sample_num_rows=sample_num_rows).sampling_task()

        return SyntheticData_pb2.SyntheticResponse(status=status, synthetic_data=synthetic_data,
                                                   privacy_score=privacy_score)


def start_service():
    LOGGER.info('启动服务，服务主机地址为:{}，监听端口为:{}'.format(_IP, _PORT))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    SyntheticData_pb2_grpc.add_SyntheticServiceServicer_to_server(Generator(), server)
    server.add_insecure_port('[::]:{}'.format(_PORT))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    start_service()