import os
os.chdir("../../../2.0 myCode_edge_index")


# 获取配置
from mydynalearn.config import ExperimentConfig
config = ExperimentConfig.default(
    "dynamics-sis-sc",
    "sis",
    "sc",
    seed=0
)
# T总时间步
config.train_details.num_samples = 500
config.train_details.epochs = 1 # 10
config.set_path()

from DynamicExperiment import DynamicExperiment
exp = DynamicExperiment(config)
exp.run()