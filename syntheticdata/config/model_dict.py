from syntheticdata.tabular.model_tvae import TVAE
from syntheticdata.tabular.model_ctgan import CTGAN
from syntheticdata.tabular.model_GaussianCopula import GaussianCopula
from syntheticdata.tabular.model_copulagan import CopulaGAN

# 目前仅支持单表仿真，含有以下四种模型
# 未来添加更多模型，需要在此处增加字典内容
model_dict = {
    'tabular': {'TVAE': TVAE,
                'CTGAN': CTGAN,
                'GaussianCopula': GaussianCopula,
                'CopulaGAN': CopulaGAN,
                }
}
