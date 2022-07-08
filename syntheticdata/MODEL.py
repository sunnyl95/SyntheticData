from syntheticdata.tabular.model_tvae import TVAE
from syntheticdata.tabular.model_ctgan import CTGAN
from syntheticdata.tabular.model_GaussianCopula import  GaussianCopula

all_model= {
        'tabular':{'TVAE':TVAE,
                   'CTGAN': CTGAN,
                   'GaussianCopula' :GaussianCopula,
                   # 'CopulaGAN':,
                   }
        }
