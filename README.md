# SyntheticData

## 1.环境

python版本：Python3.8、其他Python版本未经测试
依赖包：

+ 仅CPU：执行如下命令安装即可。
```shell
 pip install -r requirements-cpu.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple -f https://download.pytorch.org/whl/torch_stable.html
 ```


+ GPU：执行如下命令安装。这里安装的GPU版torch对应的驱动是CUDA11.1，若自己的设备非CUDA11.1，则执行该行命令时会报错，请删除 requirements-gpu.txt文件最后两行代码，再次执行该命令。然后根据自己的设备情况，自行安装torch和torchvision（可参考Pytorch官方[安装文档](https://pytorch.org/get-started/previous-versions/)）

```shell
pip install -r requirements-gpu.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple  -f https://download.pytorch.org/whl/torch_stable.html
```

## 2.使用

### 2.1配置IP地址和端口

本项目默认设置IP=127.0.0.1，PORT=50050，可根据自己的需求在SyntheticData/syntheticdata/config/config.py中修改配置。

### 2.2启动服务

python SyntheticDataServer.py 启动仿真数据生成服务

### 2.3请求服务

#### 2.3.1第一种类型请求：训练仿真数据生成模型


SyntheticModelRequest接收以下八个参数：

+ real_data_file_path ：待进行仿真的真实数据表格文件路径，目前仅支持csv文件，且需要有表头。（必需参数）
  
  <p>
+ model_save_path：训练好的模型保存路径（必填参数）

<p>

+ tabel_type：待进行仿真的真实数据表类型，<font color=red>参数只能来自列表
  ["tabular"， "relational"， "timeseries"]，**当前仅支持"tabular"**，表示（一张）单表类型（必填参数）</font>
  
  <p>
+ model_type ：进行仿真数据训练的模型类型， <font color=red>参数只能来自列表[ "TVAE"，"CTGAN",, "CopulaGAN","GaussianCopula" ]</font>，多数情况下，"TVAE"模型效果最佳。（必填参数）

<p>

+ primary_key ：主键列名，只能传入一个主键字段名称（可选参数）

<p>

+ anonymize_fields：敏感字段映射字典，具体指的是【真实数据表中的敏感列名】对应的【Faker匿名化字段名称】的字典对，  <font color=red>例如传入的参数为"{'addr':'address'}"（接口里设计的是string类型，因此需要在字典的外面加上双引号转成字符串，方可作为参数传入成功）, key表示真实数据表中敏感字段的名称，value表示将使用Faker中哪种类型生成假数据，替换掉真实敏感数据值。另外支持的Faker类型见SyntheticData/syntheticdata/config/fake.py列表。（可选参数）</font>

<p>

+ sampling_or_not：是否需要在训练完成后紧接着生成一份仿真数据样本（必填参数）

<p>

+ sample_num_rows ：  欲生成仿真数据样本的数量，当sampling_or_not=True时，该参数为必填参数，且必须为正整数。

接口请求调用**示例1**【SyntheticData/example/model_example.py】如下，**训练仿真数据生成模型，紧接着生成对应的仿真数据样本**（更多示例见tests/test_training_client.py）：

```python
import grpc
from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from syntheticdata.config import config


def send_request(real_data_file_path, model_save_path, tabel_type, model_type, primary_key=None, anonymize_fields=None,
                 sampling_or_not=False, sample_num_rows=1000):
    request = SyntheticData_pb2.SyntheticModelRequest()
    request.real_data_file_path = real_data_file_path
    request.model_save_path = model_save_path
    request.tabel_type = tabel_type
    request.model_type = model_type
    request.sampling_or_not = sampling_or_not

    if primary_key is not None:
        request.primary_key = primary_key
    if anonymize_fields is not None:
        request.anonymize_fields = anonymize_fields
    if sampling_or_not and sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows

    # ip和端口与config文件中设置的一致
    with grpc.insecure_channel('{}:{}'.format(config.IP, config.PORT)) as channel:
        stub = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)
        result = stub.SyntheticModel(request)

    return result.status, result.synthetic_data, result.privacy_score


if __name__ == "__main__":
    real_data_file_path = "example/data/adult.csv"
    model_save_path = "example/models/tvae.pkl"
    tabel_type = "tabular"
    model_type = "TVAE"
    anonymize_fields = "{'native-country':'country'}"
    sampling_or_not = True
    sample_num_rows = 10
    status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                         model_save_path=model_save_path,
                                                         tabel_type=tabel_type,
                                                         model_type=model_type,
                                                         anonymize_fields=anonymize_fields,
                                                         sampling_or_not=sampling_or_not,
                                                         sample_num_rows=sample_num_rows)
    print(f"服务应答状态码：{status.code}")
    print(f"服务应答消息：{status.msg}")
    print(f"生成的仿真数据：{synthetic_data}")
    print(f"生成的仿真数据隐私性得分：{privacy_score}")
```

服务请求结果如下，包含：

+ 服务的应答情况（状态码+消息）<p>
+ 生成的仿真数据（json串，在python中，可利用pandas.DataFrame(eval(synthetic_data))将 json格式转成dataframe格式）<p>
+ 仿真数据样本的隐私性保护得分（<font color=red>分值越大表示隐私性保护程度越高，最高为1，最低为0）</font><p>

```shell
服务应答状态码：0
服务应答消息：生成仿真数据成功！
生成的仿真数据：{"age":{"0":"31","1":"30","2":"30","3":"43","4":"31","5":"30","6":"43","7":"32","8":"44","9":"30"},"workclass":{"0":" Private","1":" Private","2":" Private","3":" Private","4":" Private","5":" Private","6":" Private","7":" Private","8":" Private","9":" Private"},"fnlwgt":{"0":"163003","1":"163003","2":"163003","3":"163003","4":"194636","5":"163003","6":"149640","7":"116632","8":"163003","9":"163003"},"education":{"0":" HS-grad","1":" HS-grad","2":" HS-grad","3":" HS-grad","4":" HS-grad","5":" HS-grad","6":" HS-grad","7":" HS-grad","8":" HS-grad","9":" HS-grad"},"education-num":{"0":"9","1":"9","2":"9","3":"9","4":"9","5":"9","6":"9","7":"9","8":"9","9":"9"},"marital-status":{"0":" Married-civ-spouse","1":" Married-civ-spouse","2":" Married-civ-spouse","3":" Married-civ-spouse","4":" Married-civ-spouse","5":" Married-civ-spouse","6":" Married-civ-spouse","7":" Married-civ-spouse","8":" Married-civ-spouse","9":" Married-civ-spouse"},"occupation":{"0":" Craft-repair","1":" Craft-repair","2":" Other-service","3":" Exec-managerial","4":" Sales","5":" Craft-repair","6":" Exec-managerial","7":" Craft-repair","8":" Craft-repair","9":" Prof-specialty"},"relationship":{"0":" Husband","1":" Husband","2":" Husband","3":" Husband","4":" Husband","5":" Husband","6":" Husband","7":" Husband","8":" Husband","9":" Husband"},"race":{"0":" White","1":" White","2":" White","3":" White","4":" White","5":" White","6":" White","7":" White","8":" White","9":" White"},"sex":{"0":" Male","1":" Male","2":" Male","3":" Male","4":" Male","5":" Male","6":" Male","7":" Male","8":" Male","9":" Male"},"capital-gain":{"0":"0","1":"0","2":"0","3":"0","4":"0","5":"0","6":"0","7":"0","8":"0","9":"0"},"capital-loss":{"0":"0","1":"0","2":"0","3":"0","4":"0","5":"0","6":"0","7":"0","8":"0","9":"0"},"hours-per-week":{"0":"40","1":"40","2":"40","3":"40","4":"40","5":"40","6":"40","7":"40","8":"40","9":"40"},"native-country":{"0":"\u79d8\u9c81","1":"\u8461\u8404\u7259","2":"\u5384\u7acb\u7279\u91cc\u4e9a","3":"\u5723\u8d6b\u52d2\u62ff","4":"\u897f\u73ed\u7259","5":"\u5df4\u62c9\u572d","6":"\u54e5\u65af\u8fbe\u9ece\u52a0","7":"\u5e03\u9686\u8fea","8":"\u7f05\u7538","9":"\u6bdb\u91cc\u5854\u5c3c\u4e9a"},"income":{"0":" <=50K","1":" <=50K","2":" <=50K","3":" <=50K","4":" <=50K","5":" <=50K","6":" <=50K","7":" <=50K","8":" <=50K","9":" <=50K"}}
生成的仿真数据隐私性得分：1.0
```

接口请求调用**示例2**【SyntheticData/example/model_example.py】如下，**仅仅训练仿真数据生成模型，不生成对应的仿真数据样本**（更多示例见tests/test_training_client.py）：

```python
import grpc
from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from syntheticdata.config import config


def send_request(real_data_file_path, model_save_path, tabel_type, model_type, primary_key=None, anonymize_fields=None,
                 sampling_or_not=False, sample_num_rows=1000):
    request = SyntheticData_pb2.SyntheticModelRequest()
    request.real_data_file_path = real_data_file_path
    request.model_save_path = model_save_path
    request.tabel_type = tabel_type
    request.model_type = model_type
    request.sampling_or_not = sampling_or_not

    if primary_key is not None:
        request.primary_key = primary_key
    if anonymize_fields is not None:
        request.anonymize_fields = anonymize_fields
    if sampling_or_not and sample_num_rows is not None:
        request.sample_num_rows = sample_num_rows

    # ip和端口与config文件中设置的一致
    with grpc.insecure_channel('{}:{}'.format(config.IP, config.PORT)) as channel:
        stub = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)
        result = stub.SyntheticModel(request)

    return result.status, result.synthetic_data, result.privacy_score


if __name__ == "__main__":
    real_data_file_path = "example/data/adult.csv"
    model_save_path = "example/models/tvae.pkl"
    tabel_type = "tabular"
    model_type = "TVAE"
    anonymize_fields = "{'native-country':'country'}"
    sampling_or_not = False     #仅仅训练仿真数据生成模型，不生成对应的仿真数据样本
    sample_num_rows = 10
    status, synthetic_data, privacy_score = send_request(real_data_file_path=real_data_file_path,
                                                         model_save_path=model_save_path,
                                                         tabel_type=tabel_type,
                                                         model_type=model_type,
                                                         anonymize_fields=anonymize_fields,
                                                         sampling_or_not=sampling_or_not,
                                                         sample_num_rows=sample_num_rows)
    print(f"服务应答状态码：{status.code}")
    print(f"服务应答消息：{status.msg}")
    print(f"生成的仿真数据：{synthetic_data}")
    print(f"生成的仿真数据隐私性得分：{privacy_score}")
```

服务请求结果如下，包含：

+ 服务的应答情况（状态码+消息）<p>
+ 生成的仿真数据（json串，在python中，可利用pandas.DataFrame(eval(synthetic_data))将 json格式转成dataframe格式）<p>
+ 仿真数据样本的隐私性保护得分（<font color=red>分值越大表示隐私性保护程度越高，最高为1，最低为0）</font><p>

```shell
服务应答状态码：0
服务应答消息：模型训练成功，并成功保存！（该任务类型是仅训练模型，不生成仿真数据样本，因此隐私性得分为1）
生成的仿真数据：
生成的仿真数据隐私性得分：1.0
```

#### 2.3.2第二种类型请求：生成仿真数据样本

```
SyntheticSampleRequest接收以下两个参数：
```

+ model_save_path：欲加载的仿真数据生成模型路径（必填参数）

<p>

+ sample_num_rows ：  欲生成仿真数据样本的数量，该参数必须为正整数。（必填参数）

接口请求调用**示例**【SyntheticData/example/sample_example.py】如下，**生成100条仿真数据**（更多示例见tests/test_sampling_client.py）：

```python
import grpc
from grpc_module import SyntheticData_pb2, SyntheticData_pb2_grpc
from syntheticdata.config import config

def send_request(model_save_path, sample_num_rows=1000):
request = SyntheticData_pb2.SyntheticSampleRequest()
request.model_save_path = model_save_path
if sample_num_rows is not None:
request.sample_num_rows = sample_num_rows

# ip和端口与config文件中设置的一致

with grpc.insecure_channel('{}:{}'.format(config.IP, config.PORT)) as channel:
stub = SyntheticData_pb2_grpc.SyntheticServiceStub(channel)

result = stub.SyntheticSample(request)

return result.status, result.synthetic_data, result.privacy_score

if __name__ == "__main__":
model_save_path = "example/models/tvae.pkl"
sample_num_rows = 100

status, synthetic_data, privacy_score = send_request(model_save_path=model_save_path,
sample_num_rows=sample_num_rows)

print(f"服务应答状态码：{status.code}")
print(f"服务应答消息：{status.msg}")
print(f"生成的仿真数据：{synthetic_data}")
print(f"生成的仿真数据隐私性得分：{privacy_score}")
```

服务请求结果如下，包含：

+ 服务的应答情况（状态码+消息）<p>
+ 生成的仿真数据（json串，在python中，可利用pandas.DataFrame(eval(synthetic_data))将 json格式转成dataframe格式）<p>
+ 仿真数据样本的隐私性保护得分（<font color=red>分值越大表示隐私性保护程度越高，最高为1，最低为0）</font><p>

```shell
服务应答状态码：0
服务应答消息：生成仿真数据成功！
生成的仿真数据：{"age":{"0":"43","1":"44","2":"31","3":"37","4":"43","5":"37","6":"31","7":"43","8":"26","9":"37","10":"44","11":"41","12":"34","13":"43","14":"31","15":"43","16":"44","17":"32","18":"44","19":"31","20":"31","21":"44","22":"31","23":"31","24":"31","25":"32","26":"31","27":"37","28":"41","29":"41","30":"37","31":"44","32":"44","33":"32","34":"44","35":"44","36":"41","37":"24","38":"38","39":"31","40":"31","41":"41","42":"38","43":"32","44":"31","45":"31","46":"44","47":"37","48":"41","49":"31","50":"44","51":"31","52":"44","53":"37","54":"37","55":"43","56":"44","57":"31","58":"44","59":"44","60":"43","61":"31","62":"37","63":"43","64":"41","65":"31","66":"31","67":"37","68":"31","69":"37","70":"31","71":"31","72":"24","73":"44","74":"44","75":"44","76":"31","77":"44","78":"44","79":"31","80":"44","81":"44","82":"31","83":"31","84":"31","85":"31","86":"44","87":"37","88":"43","89":"31","90":"44","91":"37","92":"30","93":"31","94":"31","95":"31","96":"31","97":"31","98":"44","99":"37"},"workclass":{"0":" Private","1":" Private","2":" Private","3":" Private","4":" Private","5":" Private","6":" Private","7":" Private","8":" Private","9":" Private","10":" Private","11":" Private","12":" Private","13":" Private","14":" Private","15":" Private","16":" Private","17":" Private","18":" Private","19":" Private","20":" Private","21":" Private","22":" Private","23":" Private","24":" Private","25":" Private","26":" Private","27":" Private","28":" Private","29":" Private","30":" Private","31":" Private","32":" Private","33":" Private","34":" Private","35":" Private","36":" Private","37":" Private","38":" Private","39":" Private","40":" Private","41":" Private","42":" Private","43":" Private","44":" Private","45":" Private","46":" Private","47":" Private","48":" Private","49":" Private","50":" Private","51":" Private","52":" Private","53":" Private","54":" Private","55":" Private","56":" Private","57":" Private","58":" Private","59":" Private","60":" Private","61":" Private","62":" Private","63":" Private","64":" Private","65":" Private","66":" Private","67":" Private","68":" Private","69":" Private","70":" Private","71":" Private","72":" Private","73":" Private","74":" Private","75":" Private","76":" Private","77":" Private","78":" Private","79":" Private","80":" Private","81":" Private","82":" Private","83":" Private","84":" Private","85":" Private","86":" Private","87":" Private","88":" Private","89":" Private","90":" Private","91":" Private","92":" Private","93":" Private","94":" Private","95":" Private","96":" Private","97":" Private","98":" Private","99":" Private"},"fnlwgt":{"0":"116632","1":"116632","2":"116632","3":"116632","4":"86551","5":"116632","6":"116632","7":"120939","8":"116632","9":"116632","10":"119272","11":"116632","12":"116632","13":"116632","14":"116632","15":"119272","16":"116632","17":"167350","18":"116632","19":"116632","20":"116632","21":"116632","22":"116632","23":"116632","24":"116632","25":"116632","26":"116632","27":"116632","28":"116632","29":"167350","30":"116632","31":"116632","32":"116632","33":"116632","34":"116632","35":"116632","36":"116632","37":"116632","38":"116632","39":"163003","40":"116632","41":"235271","42":"163003","43":"116632","44":"167350","45":"116632","46":"163003","47":"116632","48":"116632","49":"116632","50":"116632","51":"116632","52":"212448","53":"116632","54":"116632","55":"116632","56":"116632","57":"116632","58":"116632","59":"116632","60":"163003","61":"116632","62":"116632","63":"116632","64":"116632","65":"167350","66":"116632","67":"116632","68":"116632","69":"116632","70":"167350","71":"116632","72":"163003","73":"116632","74":"116632","75":"116632","76":"235271","77":"116632","78":"116632","79":"116632","80":"116632","81":"116632","82":"280093","83":"35633","84":"116632","85":"116632","86":"112847","87":"116632","88":"116632","89":"116632","90":"116632","91":"116632","92":"117789","93":"117312","94":"116632","95":"116632","96":"116632","97":"116632","98":"116632","99":"116632"},"education":{"0":" HS-grad","1":" HS-grad","2":" HS-grad","3":" HS-grad","4":" HS-grad","5":" HS-grad","6":" HS-grad","7":" HS-grad","8":" HS-grad","9":" HS-grad","10":" HS-grad","11":" HS-grad","12":" HS-grad","13":" HS-grad","14":" HS-grad","15":" HS-grad","16":" HS-grad","17":" HS-grad","18":" HS-grad","19":" HS-grad","20":" HS-grad","21":" HS-grad","22":" HS-grad","23":" HS-grad","24":" HS-grad","25":" HS-grad","26":" HS-grad","27":" HS-grad","28":" HS-grad","29":" HS-grad","30":" HS-grad","31":" HS-grad","32":" HS-grad","33":" HS-grad","34":" HS-grad","35":" HS-grad","36":" HS-grad","37":" HS-grad","38":" HS-grad","39":" HS-grad","40":" HS-grad","41":" HS-grad","42":" HS-grad","43":" HS-grad","44":" HS-grad","45":" HS-grad","46":" HS-grad","47":" HS-grad","48":" HS-grad","49":" HS-grad","50":" HS-grad","51":" HS-grad","52":" HS-grad","53":" HS-grad","54":" HS-grad","55":" HS-grad","56":" HS-grad","57":" HS-grad","58":" HS-grad","59":" HS-grad","60":" HS-grad","61":" HS-grad","62":" HS-grad","63":" HS-grad","64":" HS-grad","65":" HS-grad","66":" HS-grad","67":" HS-grad","68":" HS-grad","69":" HS-grad","70":" HS-grad","71":" HS-grad","72":" HS-grad","73":" HS-grad","74":" HS-grad","75":" HS-grad","76":" HS-grad","77":" HS-grad","78":" HS-grad","79":" HS-grad","80":" HS-grad","81":" HS-grad","82":" HS-grad","83":" HS-grad","84":" HS-grad","85":" HS-grad","86":" HS-grad","87":" HS-grad","88":" HS-grad","89":" HS-grad","90":" HS-grad","91":" HS-grad","92":" HS-grad","93":" HS-grad","94":" HS-grad","95":" HS-grad","96":" HS-grad","97":" HS-grad","98":" HS-grad","99":" HS-grad"},"education-num":{"0":"9","1":"9","2":"9","3":"9","4":"9","5":"9","6":"9","7":"9","8":"9","9":"9","10":"9","11":"9","12":"9","13":"9","14":"9","15":"9","16":"9","17":"9","18":"9","19":"9","20":"9","21":"9","22":"9","23":"9","24":"9","25":"9","26":"9","27":"9","28":"9","29":"9","30":"9","31":"9","32":"9","33":"9","34":"9","35":"9","36":"9","37":"9","38":"9","39":"9","40":"9","41":"9","42":"9","43":"9","44":"9","45":"9","46":"9","47":"9","48":"9","49":"9","50":"9","51":"9","52":"9","53":"9","54":"9","55":"9","56":"9","57":"9","58":"9","59":"9","60":"9","61":"9","62":"9","63":"9","64":"9","65":"9","66":"9","67":"9","68":"9","69":"9","70":"9","71":"9","72":"9","73":"9","74":"9","75":"9","76":"9","77":"9","78":"9","79":"9","80":"9","81":"9","82":"9","83":"9","84":"9","85":"9","86":"9","87":"9","88":"9","89":"9","90":"9","91":"9","92":"9","93":"9","94":"9","95":"9","96":"9","97":"9","98":"9","99":"9"},"marital-status":{"0":" Married-civ-spouse","1":" Married-civ-spouse","2":" Married-civ-spouse","3":" Married-civ-spouse","4":" Married-civ-spouse","5":" Married-civ-spouse","6":" Married-civ-spouse","7":" Married-civ-spouse","8":" Married-civ-spouse","9":" Married-civ-spouse","10":" Married-civ-spouse","11":" Married-civ-spouse","12":" Married-civ-spouse","13":" Married-civ-spouse","14":" Married-civ-spouse","15":" Married-civ-spouse","16":" Married-civ-spouse","17":" Married-civ-spouse","18":" Married-civ-spouse","19":" Married-civ-spouse","20":" Married-civ-spouse","21":" Married-civ-spouse","22":" Married-civ-spouse","23":" Married-civ-spouse","24":" Married-civ-spouse","25":" Married-civ-spouse","26":" Married-civ-spouse","27":" Married-civ-spouse","28":" Never-married","29":" Married-civ-spouse","30":" Married-civ-spouse","31":" Married-civ-spouse","32":" Married-civ-spouse","33":" Married-civ-spouse","34":" Married-civ-spouse","35":" Married-civ-spouse","36":" Married-civ-spouse","37":" Married-civ-spouse","38":" Married-civ-spouse","39":" Married-civ-spouse","40":" Married-civ-spouse","41":" Married-civ-spouse","42":" Married-civ-spouse","43":" Married-civ-spouse","44":" Married-civ-spouse","45":" Married-civ-spouse","46":" Married-civ-spouse","47":" Married-civ-spouse","48":" Married-civ-spouse","49":" Married-civ-spouse","50":" Married-civ-spouse","51":" Never-married","52":" Married-civ-spouse","53":" Married-civ-spouse","54":" Married-civ-spouse","55":" Married-civ-spouse","56":" Married-civ-spouse","57":" Married-civ-spouse","58":" Married-civ-spouse","59":" Married-civ-spouse","60":" Married-civ-spouse","61":" Married-civ-spouse","62":" Married-civ-spouse","63":" Married-civ-spouse","64":" Married-civ-spouse","65":" Married-civ-spouse","66":" Married-civ-spouse","67":" Married-civ-spouse","68":" Never-married","69":" Married-civ-spouse","70":" Married-civ-spouse","71":" Married-civ-spouse","72":" Married-civ-spouse","73":" Married-civ-spouse","74":" Married-civ-spouse","75":" Never-married","76":" Never-married","77":" Married-civ-spouse","78":" Married-civ-spouse","79":" Married-civ-spouse","80":" Married-civ-spouse","81":" Married-civ-spouse","82":" Married-civ-spouse","83":" Married-civ-spouse","84":" Married-civ-spouse","85":" Never-married","86":" Married-civ-spouse","87":" Married-civ-spouse","88":" Married-civ-spouse","89":" Married-civ-spouse","90":" Married-civ-spouse","91":" Married-civ-spouse","92":" Married-civ-spouse","93":" Married-civ-spouse","94":" Married-civ-spouse","95":" Married-civ-spouse","96":" Married-civ-spouse","97":" Married-civ-spouse","98":" Married-civ-spouse","99":" Married-civ-spouse"},"occupation":{"0":" Prof-specialty","1":" Other-service","2":" Exec-managerial","3":" Other-service","4":" Prof-specialty","5":" Exec-managerial","6":" Exec-managerial","7":" Adm-clerical","8":" Adm-clerical","9":" Adm-clerical","10":" Adm-clerical","11":" Exec-managerial","12":" Exec-managerial","13":" Adm-clerical","14":" Adm-clerical","15":" Other-service","16":" Exec-managerial","17":" Adm-clerical","18":" Exec-managerial","19":" Exec-managerial","20":" Sales","21":" Exec-managerial","22":" Prof-specialty","23":" Exec-managerial","24":" Exec-managerial","25":" Exec-managerial","26":" Adm-clerical","27":" Exec-managerial","28":" Exec-managerial","29":" Prof-specialty","30":" Other-service","31":" Exec-managerial","32":" Adm-clerical","33":" Prof-specialty","34":" Adm-clerical","35":" Exec-managerial","36":" Exec-managerial","37":" Prof-specialty","38":" Prof-specialty","39":" Exec-managerial","40":" Exec-managerial","41":" Other-service","42":" Prof-specialty","43":" Exec-managerial","44":" Exec-managerial","45":" Other-service","46":" Adm-clerical","47":" Exec-managerial","48":" Exec-managerial","49":" Adm-clerical","50":" Exec-managerial","51":" Prof-specialty","52":" Adm-clerical","53":" Adm-clerical","54":" Other-service","55":" Adm-clerical","56":" Exec-managerial","57":" Exec-managerial","58":" Prof-specialty","59":" Exec-managerial","60":" Exec-managerial","61":" Adm-clerical","62":" Adm-clerical","63":" Exec-managerial","64":" Adm-clerical","65":" Exec-managerial","66":" Exec-managerial","67":" Other-service","68":" Exec-managerial","69":" Sales","70":" Exec-managerial","71":" Adm-clerical","72":" Exec-managerial","73":" Exec-managerial","74":" Prof-specialty","75":" Exec-managerial","76":" Prof-specialty","77":" Craft-repair","78":" Adm-clerical","79":" Exec-managerial","80":" Exec-managerial","81":" Adm-clerical","82":" Exec-managerial","83":" Sales","84":" Adm-clerical","85":" Exec-managerial","86":" Exec-managerial","87":" Exec-managerial","88":" Exec-managerial","89":" Exec-managerial","90":" Exec-managerial","91":" Adm-clerical","92":" Adm-clerical","93":" Exec-managerial","94":" Exec-managerial","95":" Prof-specialty","96":" Prof-specialty","97":" Adm-clerical","98":" Other-service","99":" Sales"},"relationship":{"0":" Husband","1":" Husband","2":" Husband","3":" Husband","4":" Husband","5":" Husband","6":" Not-in-family","7":" Husband","8":" Husband","9":" Husband","10":" Husband","11":" Husband","12":" Husband","13":" Husband","14":" Husband","15":" Husband","16":" Husband","17":" Husband","18":" Husband","19":" Husband","20":" Husband","21":" Husband","22":" Husband","23":" Husband","24":" Husband","25":" Husband","26":" Not-in-family","27":" Husband","28":" Husband","29":" Husband","30":" Husband","31":" Husband","32":" Husband","33":" Husband","34":" Husband","35":" Husband","36":" Not-in-family","37":" Husband","38":" Husband","39":" Husband","40":" Husband","41":" Husband","42":" Husband","43":" Husband","44":" Husband","45":" Husband","46":" Husband","47":" Husband","48":" Husband","49":" Husband","50":" Husband","51":" Husband","52":" Husband","53":" Husband","54":" Husband","55":" Husband","56":" Husband","57":" Husband","58":" Husband","59":" Husband","60":" Husband","61":" Husband","62":" Husband","63":" Husband","64":" Husband","65":" Husband","66":" Not-in-family","67":" Husband","68":" Husband","69":" Husband","70":" Husband","71":" Husband","72":" Husband","73":" Husband","74":" Husband","75":" Husband","76":" Husband","77":" Husband","78":" Husband","79":" Husband","80":" Husband","81":" Husband","82":" Husband","83":" Husband","84":" Husband","85":" Husband","86":" Husband","87":" Husband","88":" Husband","89":" Husband","90":" Husband","91":" Husband","92":" Husband","93":" Husband","94":" Husband","95":" Husband","96":" Husband","97":" Husband","98":" Husband","99":" Not-in-family"},"race":{"0":" White","1":" White","2":" White","3":" White","4":" White","5":" White","6":" White","7":" White","8":" White","9":" White","10":" White","11":" White","12":" White","13":" White","14":" White","15":" White","16":" White","17":" White","18":" White","19":" White","20":" White","21":" White","22":" White","23":" White","24":" White","25":" White","26":" White","27":" White","28":" White","29":" White","30":" White","31":" White","32":" White","33":" White","34":" White","35":" White","36":" White","37":" White","38":" White","39":" White","40":" White","41":" White","42":" White","43":" White","44":" White","45":" White","46":" White","47":" White","48":" White","49":" White","50":" White","51":" White","52":" White","53":" White","54":" White","55":" White","56":" White","57":" White","58":" White","59":" White","60":" White","61":" White","62":" White","63":" White","64":" White","65":" White","66":" White","67":" White","68":" White","69":" White","70":" White","71":" White","72":" White","73":" White","74":" White","75":" White","76":" White","77":" White","78":" White","79":" White","80":" White","81":" White","82":" White","83":" White","84":" White","85":" White","86":" White","87":" White","88":" White","89":" White","90":" White","91":" White","92":" White","93":" White","94":" White","95":" White","96":" White","97":" White","98":" White","99":" White"},"sex":{"0":" Male","1":" Male","2":" Male","3":" Male","4":" Male","5":" Male","6":" Male","7":" Male","8":" Male","9":" Male","10":" Male","11":" Male","12":" Male","13":" Male","14":" Male","15":" Male","16":" Male","17":" Male","18":" Male","19":" Male","20":" Male","21":" Male","22":" Male","23":" Male","24":" Male","25":" Male","26":" Male","27":" Male","28":" Male","29":" Male","30":" Male","31":" Male","32":" Male","33":" Male","34":" Male","35":" Male","36":" Male","37":" Male","38":" Male","39":" Male","40":" Male","41":" Male","42":" Male","43":" Male","44":" Male","45":" Male","46":" Male","47":" Male","48":" Male","49":" Male","50":" Male","51":" Male","52":" Male","53":" Male","54":" Male","55":" Male","56":" Male","57":" Male","58":" Male","59":" Male","60":" Male","61":" Male","62":" Male","63":" Male","64":" Male","65":" Male","66":" Male","67":" Male","68":" Male","69":" Male","70":" Male","71":" Male","72":" Male","73":" Male","74":" Male","75":" Male","76":" Male","77":" Male","78":" Male","79":" Male","80":" Male","81":" Male","82":" Male","83":" Male","84":" Male","85":" Male","86":" Male","87":" Male","88":" Male","89":" Male","90":" Male","91":" Male","92":" Male","93":" Male","94":" Male","95":" Male","96":" Male","97":" Male","98":" Male","99":" Male"},"capital-gain":{"0":"0","1":"0","2":"0","3":"0","4":"0","5":"0","6":"0","7":"0","8":"0","9":"0","10":"0","11":"0","12":"0","13":"0","14":"0","15":"0","16":"0","17":"0","18":"0","19":"0","20":"0","21":"0","22":"0","23":"0","24":"0","25":"0","26":"0","27":"0","28":"0","29":"0","30":"0","31":"0","32":"0","33":"0","34":"0","35":"0","36":"0","37":"0","38":"0","39":"0","40":"0","41":"0","42":"0","43":"0","44":"0","45":"0","46":"0","47":"0","48":"0","49":"0","50":"0","51":"0","52":"0","53":"0","54":"0","55":"0","56":"0","57":"0","58":"0","59":"0","60":"0","61":"0","62":"0","63":"0","64":"0","65":"0","66":"0","67":"0","68":"0","69":"0","70":"0","71":"0","72":"0","73":"0","74":"0","75":"0","76":"0","77":"0","78":"0","79":"0","80":"0","81":"0","82":"0","83":"0","84":"0","85":"0","86":"0","87":"0","88":"0","89":"0","90":"0","91":"0","92":"0","93":"0","94":"0","95":"0","96":"0","97":"0","98":"0","99":"0"},"capital-loss":{"0":"0","1":"0","2":"0","3":"0","4":"0","5":"0","6":"0","7":"0","8":"0","9":"0","10":"0","11":"0","12":"0","13":"0","14":"0","15":"0","16":"0","17":"0","18":"0","19":"0","20":"0","21":"0","22":"0","23":"0","24":"0","25":"0","26":"0","27":"0","28":"0","29":"0","30":"0","31":"0","32":"0","33":"0","34":"0","35":"0","36":"0","37":"0","38":"0","39":"0","40":"0","41":"0","42":"0","43":"0","44":"0","45":"0","46":"0","47":"0","48":"0","49":"0","50":"0","51":"0","52":"0","53":"0","54":"0","55":"0","56":"0","57":"0","58":"0","59":"0","60":"0","61":"0","62":"0","63":"0","64":"0","65":"0","66":"0","67":"0","68":"0","69":"0","70":"0","71":"0","72":"0","73":"0","74":"0","75":"0","76":"0","77":"0","78":"0","79":"0","80":"0","81":"0","82":"0","83":"0","84":"0","85":"0","86":"0","87":"0","88":"0","89":"0","90":"0","91":"0","92":"0","93":"0","94":"0","95":"0","96":"0","97":"0","98":"0","99":"0"},"hours-per-week":{"0":"40","1":"40","2":"40","3":"40","4":"40","5":"40","6":"40","7":"40","8":"40","9":"40","10":"40","11":"40","12":"40","13":"40","14":"40","15":"40","16":"40","17":"40","18":"40","19":"40","20":"40","21":"40","22":"40","23":"40","24":"40","25":"40","26":"40","27":"40","28":"40","29":"40","30":"40","31":"40","32":"40","33":"40","34":"40","35":"40","36":"40","37":"40","38":"40","39":"40","40":"40","41":"40","42":"40","43":"40","44":"40","45":"40","46":"40","47":"40","48":"40","49":"40","50":"40","51":"40","52":"40","53":"40","54":"40","55":"40","56":"40","57":"40","58":"40","59":"40","60":"40","61":"40","62":"40","63":"40","64":"40","65":"40","66":"40","67":"40","68":"40","69":"40","70":"40","71":"40","72":"40","73":"40","74":"40","75":"40","76":"40","77":"40","78":"40","79":"40","80":"40","81":"40","82":"40","83":"40","84":"40","85":"40","86":"40","87":"40","88":"40","89":"40","90":"40","91":"40","92":"40","93":"40","94":"40","95":"40","96":"40","97":"40","98":"40","99":"40"},"native-country":{"0":"\u9a6c\u91cc","1":"\u5229\u6bd4\u91cc\u4e9a","2":"\u683c\u9675\u5170\u5c9b","3":"\u5188\u6bd4\u4e9a","4":"\u5723\u6587\u68ee\u7279\u5c9b","5":"\u83f2\u5f8b\u5bbe","6":"\u82cf\u91cc\u5357","7":"\u65af\u6d1b\u4f10\u514b","8":"\u5df4\u6797","9":"\u897f\u8428\u6469\u4e9a","10":"\u7eb3\u7c73\u6bd4\u4e9a","11":"\u8d1d\u5b81","12":"\u7acb\u9676\u5b9b","13":"\u963f\u62c9\u65af\u52a0","14":"\u4e39\u9ea6","15":"\u5e03\u9686\u8fea","16":"\u7a81\u5c3c\u65af","17":"\u5f00\u66fc\u7fa4\u5c9b","18":"\u9a6c\u62c9\u7ef4","19":"\u65b0\u5580\u91cc\u591a\u5c3c\u4e9a\u7fa4\u5c9b","20":"\u610f\u5927\u5229","21":"\u683c\u9c81\u5409\u4e9a","22":"\u745e\u58eb","23":"\u5e03\u9686\u8fea","24":"\u6cd5\u5c5e\u6ce2\u91cc\u5c3c\u897f\u4e9a","25":"\u56fe\u74e6\u5362","26":"\u79d1\u79d1\u65af\u5c9b","27":"\u7ebd\u57c3\u5c9b","28":"\u65af\u6d1b\u4f10\u514b","29":"\u5c3c\u65e5\u5c14","30":"\u65af\u91cc\u5170\u5361","31":"\u65af\u91cc\u5170\u5361","32":"\u54e5\u4f26\u6bd4\u4e9a","33":"\u5df4\u62c9\u572d","34":"\u5173\u5c9b","35":"\u963f\u585e\u62dc\u7586","36":"\u683c\u9c81\u5409\u4e9a","37":"\u83b1\u7d22\u6258","38":"\u6469\u5c14\u591a\u74e6","39":"\u4e2d\u975e","40":"\u5c3c\u52a0\u62c9\u74dc","41":"\u6851\u7ed9\u5df4\u5c14","42":"\u6469\u5c14\u591a\u74e6","43":"\u5723\u76ae\u57c3\u5c14\u5c9b\u53ca\u5bc6\u514b\u9686\u5c9b","44":"\u6fb3\u5927\u5229\u4e9a","45":"\u5723\u6587\u68ee\u7279\u5c9b","46":"\u5df4\u897f","47":"\u5b89\u54e5\u62c9","48":"\u4f2f\u5229\u5179","49":"\u65af\u5a01\u58eb\u5170","50":"\u5173\u5c9b","51":"\u7ef4\u5c14\u4eac\u7fa4\u5c9b\u548c\u5723\u7f57\u514b\u4f0a","52":"\u521a\u679c","53":"\u6ce2\u65af\u5c3c\u4e9a\u548c\u9ed1\u585e\u54e5\u7ef4\u90a3","54":"\u83f2\u5f8b\u5bbe","55":"\u7f8e\u56fd","56":"\u7259\u4e70\u52a0","57":"\u5723\u76ae\u57c3\u5c14\u5c9b\u53ca\u5bc6\u514b\u9686\u5c9b","58":"\u6bd4\u5229\u65f6","59":"\u7f57\u9a6c\u5c3c\u4e9a","60":"\u7ea6\u65e6","61":"\u5fb7\u56fd","62":"\u7279\u7acb\u5c3c\u8fbe\u548c\u591a\u5df4\u54e5","63":"\u767e\u6155\u5927\u7fa4\u5c9b","64":"\u624e\u4f0a\u5c14","65":"\u535a\u8328\u74e6\u7eb3","66":"\u74dc\u5fb7\u7f57\u666e\u5c9b","67":"\u8d64\u9053\u51e0\u5185\u4e9a","68":"\u52a0\u90a3\u5229\u7fa4\u5c9b","69":"\u5723\u514b\u91cc\u65af\u6258\u5f17\u548c\u5c3c\u7ef4\u65af","70":"\u80af\u5c3c\u4e9a","71":"\u4f0a\u62c9\u514b","72":"\u82f1\u56fd","73":"\u6240\u7f57\u95e8\u7fa4\u5c9b","74":"\u6ce2\u5170","75":"\u963f\u585e\u62dc\u7586","76":"\u5fb7\u56fd","77":"\u5854\u5409\u514b\u65af\u5766","78":"\u6bdb\u91cc\u5854\u5c3c\u4e9a","79":"\u4e2d\u56fd","80":"\u5e15\u52b3","81":"\u8001\u631d","82":"\u6469\u5c14\u591a\u74e6","83":"\u571f\u5e93\u66fc\u65af\u5766","84":"\u79d1\u79d1\u65af\u5c9b","85":"\u5361\u5854\u5c14","86":"\u6469\u7eb3\u54e5","87":"\u963f\u9c81\u5df4\u5c9b","88":"\u8d5e\u6bd4\u4e9a","89":"\u4e39\u9ea6","90":"\u5371\u5730\u9a6c\u62c9","91":"\u798f\u514b\u5170\u7fa4\u5c9b","92":"\u6c99\u7279\u963f\u62c9\u4f2f","93":"\u521a\u679c","94":"\u4e2d\u56fd","95":"\u7d22\u9a6c\u91cc","96":"\u5e03\u9686\u8fea","97":"\u74dc\u5fb7\u7f57\u666e\u5c9b","98":"\u4e1c\u8428\u6469\u4e9a","99":"\u76f4\u5e03\u7f57\u9640"},"income":{"0":" <=50K","1":" <=50K","2":" <=50K","3":" <=50K","4":" <=50K","5":" <=50K","6":" <=50K","7":" <=50K","8":" <=50K","9":" <=50K","10":" <=50K","11":" <=50K","12":" <=50K","13":" <=50K","14":" <=50K","15":" <=50K","16":" <=50K","17":" <=50K","18":" <=50K","19":" <=50K","20":" <=50K","21":" <=50K","22":" <=50K","23":" <=50K","24":" <=50K","25":" <=50K","26":" <=50K","27":" <=50K","28":" <=50K","29":" <=50K","30":" <=50K","31":" <=50K","32":" <=50K","33":" <=50K","34":" <=50K","35":" <=50K","36":" <=50K","37":" <=50K","38":" <=50K","39":" <=50K","40":" <=50K","41":" <=50K","42":" <=50K","43":" <=50K","44":" <=50K","45":" <=50K","46":" <=50K","47":" <=50K","48":" <=50K","49":" <=50K","50":" <=50K","51":" <=50K","52":" <=50K","53":" <=50K","54":" <=50K","55":" <=50K","56":" <=50K","57":" <=50K","58":" <=50K","59":" <=50K","60":" <=50K","61":" <=50K","62":" <=50K","63":" <=50K","64":" <=50K","65":" <=50K","66":" <=50K","67":" <=50K","68":" <=50K","69":" <=50K","70":" <=50K","71":" <=50K","72":" <=50K","73":" <=50K","74":" <=50K","75":" <=50K","76":" <=50K","77":" <=50K","78":" <=50K","79":" <=50K","80":" <=50K","81":" <=50K","82":" <=50K","83":" <=50K","84":" <=50K","85":" <=50K","86":" <=50K","87":" <=50K","88":" <=50K","89":" <=50K","90":" <=50K","91":" <=50K","92":" <=50K","93":" <=50K","94":" <=50K","95":" <=50K","96":" <=50K","97":" <=50K","98":" <=50K","99":" <=50K"}}
生成的仿真数据隐私性得分：1.0
```

## 3.性能测试报告

训练集：10w行 \* 30列credit数据集（23.7MB）

模型参数设置batch\_size=500,  epoch=300(论文默认参数),四种模型的测试汇总表如下(更多测试详情见[链接](https://wpu0sccir8.feishu.cn/wiki/wikcnhI7i4dtpTqqPDjMN0K2iag))：

|环境|模型|训练耗时(min)|模型大小(KB)|生成10w条仿真数据耗时（s）|顶峰CPU占用资源（核）|占用内存资源（GB)|顶峰占用GPU资源（MB）|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|x86-GPU|GaussianCopulas|-|-|-|-|-|不支持GPU|
||TVAE|30|306|2.7|28|5|1800左右|
||CTGAN|46|138MB|4.4|30|5|1800左右|
||CopulaGAN|1h|143MB|-|30|6|1800左右|
|x86-CPU|GaussianCopulas|5|34|2.4|1|1|不支持GPU|
||TVAE|43|306|2.7|30|5|-|
||CTGAN|1h37min|139MB|6.2|30|5|-|
||CopulaGAN|2h6min|143MB|-|30|9|-|
|ARM-CPU（AI靶场）|GaussianCopulas|16|26|40|1|1|不支持GPU|
||TVAE|3h52min|286|4.3|16|2|-|
||CTGAN|13h35min|138MB|17|16|2|-|
||CopulaGAN|17h26min|140MB|-|16|3|-|

- x86-GPU是x86-CPU的1.5倍， x86-CPU是ARM-CPU的3-8.5倍
- 训练时长跟训练数据的列数和行数均有关，列数越多耗时越长，行数越多耗时越长；
- GaussianCopula训练耗时最短、训练资源占用最少，模型30kb左右
- TVAE生成的模型大小只跟列数和列的数据类型、缺失程度有关，跟数据行数无关，列数越多、数据类型越负责、缺失列越多，模型就越大，最前测试最大列数785列，模型大小为2M左右。
- CTGAN和CopulaGAN训练耗时最长，模型最大，生成的模型大小与列数和行数均有关：
- 列数越多、数据类型复杂、缺失列越多，模型就越大，最前测试最大列数785列，模型大小为184M左右。
- 随着训练数据量增加，模型大小大概呈现线性增加，原因是在condition-sample环节模型将训练数据作为成员变量，保存模型时也将数据保存在了模型中。（劣势，后期需要优化）
- 生成仿真数据耗时跟欲生成的数据行数有关，行数越多耗时越久，生成数据时，GPU与CPU耗时基本一致。
- 训练模型对资源的要求较高，生成仿真数据对资源要求低，且效率较高（0.4ms每行数据）
