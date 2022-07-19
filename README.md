### 目录结构
SyntheticData
│  Readme.txt
│  requirements.txt
│  SyntheticDataServer.py
│  
├─.idea
│  │  .gitignore
│  │  deployment.xml
│  │  misc.xml
│  │  modules.xml
│  │  SyntheticData.iml
│  │  vcs.xml
│  │  workspace.xml
│  │  
│  └─inspectionProfiles
│          profiles_settings.xml
│          Project_Default.xml
│          
├─example
│  │  model_example.py
│  │  sample_example.py
│  │  
│  ├─data
│  │      adult.csv
│  │      
│  └─models
│          tvae.pkl
│          
├─grpc_module
│  │  readme
│  │  SyntheticDataModel.py
│  │  SyntheticData_pb2.py
│  │  SyntheticData_pb2_grpc.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          SyntheticDataModel.cpython-38.pyc
│          SyntheticData_pb2.cpython-38.pyc
│          SyntheticData_pb2_grpc.cpython-38.pyc
│          __init__.cpython-38.pyc
│          
├─log
│      logging.ini
│      SyntheticData.log
│      
├─proto
│      SyntheticData.proto
│      
├─syntheticdata
│  │  demo.py
│  │  errors.py
│  │  utils.py
│  │  __init__.py
│  │  
│  ├─config
│  │  │  config.py
│  │  │  fake.py
│  │  │  model_dict.py
│  │  │  __init__.py
│  │  │  
│  │  └─__pycache__
│  │          config.cpython-38.pyc
│  │          fake.cpython-38.pyc
│  │          model_dict.cpython-38.pyc
│  │          __init__.cpython-38.pyc
│  │          
│  ├─constraints
│  │  │  base.py
│  │  │  errors.py
│  │  │  tabular.py
│  │  │  utils.py
│  │  │  __init__.py
│  │  │  
│  │  └─__pycache__
│  │          base.cpython-38.pyc
│  │          errors.cpython-38.pyc
│  │          tabular.cpython-38.pyc
│  │          utils.cpython-38.pyc
│  │          __init__.cpython-38.pyc
│  │          
│  ├─ctgan
│  │  │  data.py
│  │  │  data_sampler.py
│  │  │  data_transformer.py
│  │  │  __init__.py
│  │  │  __main__.py
│  │  │  
│  │  ├─synthesizers
│  │  │  │  base.py
│  │  │  │  ctgan.py
│  │  │  │  tvae.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  │  └─__pycache__
│  │  │          base.cpython-38.pyc
│  │  │          ctgan.cpython-38.pyc
│  │  │          tvae.cpython-38.pyc
│  │  │          __init__.cpython-38.pyc
│  │  │          
│  │  └─__pycache__
│  │          data_sampler.cpython-38.pyc
│  │          data_transformer.cpython-38.pyc
│  │          __init__.cpython-38.pyc
│  │          
│  ├─metadata
│  │  │  dataset.py
│  │  │  errors.py
│  │  │  table.py
│  │  │  utils.py
│  │  │  visualization.py
│  │  │  __init__.py
│  │  │  
│  │  └─__pycache__
│  │          dataset.cpython-38.pyc
│  │          errors.cpython-38.pyc
│  │          table.cpython-38.pyc
│  │          utils.cpython-38.pyc
│  │          visualization.cpython-38.pyc
│  │          __init__.cpython-38.pyc
│  │          
│  ├─metrics
│  │  │  base.py
│  │  │  demos.py
│  │  │  errors.py
│  │  │  goal.py
│  │  │  utils.py
│  │  │  __init__.py
│  │  │  
│  │  ├─single_table
│  │  │  │  base.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  │  ├─privacy
│  │  │  │  │  base.py
│  │  │  │  │  cap.py
│  │  │  │  │  categorical_sklearn.py
│  │  │  │  │  ensemble.py
│  │  │  │  │  loss.py
│  │  │  │  │  numerical_sklearn.py
│  │  │  │  │  radius_nearest_neighbor.py
│  │  │  │  │  util.py
│  │  │  │  │  __init__.py
│  │  │  │  │  
│  │  │  │  └─__pycache__
│  │  │  │          base.cpython-38.pyc
│  │  │  │          cap.cpython-38.pyc
│  │  │  │          categorical_sklearn.cpython-38.pyc
│  │  │  │          ensemble.cpython-38.pyc
│  │  │  │          loss.cpython-38.pyc
│  │  │  │          numerical_sklearn.cpython-38.pyc
│  │  │  │          radius_nearest_neighbor.cpython-38.pyc
│  │  │  │          util.cpython-38.pyc
│  │  │  │          __init__.cpython-38.pyc
│  │  │  │          
│  │  │  └─__pycache__
│  │  │          base.cpython-38.pyc
│  │  │          __init__.cpython-38.pyc
│  │  │          
│  │  └─__pycache__
│  │          base.cpython-38.pyc
│  │          demos.cpython-38.pyc
│  │          errors.cpython-38.pyc
│  │          goal.cpython-38.pyc
│  │          __init__.cpython-38.pyc
│  │          
│  ├─tabular
│  │  │  base.py
│  │  │  model_copulagan.py
│  │  │  model_ctgan.py
│  │  │  model_GaussianCopula.py
│  │  │  model_tvae.py
│  │  │  utils.py
│  │  │  __init__.py
│  │  │  
│  │  └─__pycache__
│  │          base.cpython-37.pyc
│  │          base.cpython-38.pyc
│  │          model_copulagan.cpython-38.pyc
│  │          model_ctgan.cpython-37.pyc
│  │          model_ctgan.cpython-38.pyc
│  │          model_GaussianCopula.cpython-38.pyc
│  │          model_tvae.cpython-38.pyc
│  │          utils.cpython-38.pyc
│  │          __init__.cpython-37.pyc
│  │          __init__.cpython-38.pyc
│  │          
│  └─__pycache__
│          demo.cpython-37.pyc
│          demo.cpython-38.pyc
│          errors.cpython-38.pyc
│          utils.cpython-38.pyc
│          __init__.cpython-37.pyc
│          __init__.cpython-38.pyc
│          
└─tests
    │  test_sampling_client.py
    │  test_training_client.py
    │  
    ├─data
    │      adult.csv
    │      adult.txt
    │      
    └─models
            copula_gan.pkl
            ctgan.pkl
            gaussian_copula.pkl
            tvae.pkl
            
