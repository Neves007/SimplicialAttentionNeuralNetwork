import os
os.chdir("../../2.0 myCode_edge_index")


# 获取配置
from mydynalearn.config import ExperimentConfig
from mydynalearn.experiments import Experiment


def fix_config(config):
    # T总时间步
    config.train_details.num_samples = num_samples
    config.train_details.testSet_timestep = testSet_timestep
    config.train_details.epochs = epochs  # 10
    # 检查点
    config.train_details.checkFirstEpoch = checkFirstEpoch  # 10
    config.train_details.checkFirstEpoch_max_time = checkFirstEpoch_max_time
    config.train_details.checkFirstEpoch_timestep = checkFirstEpoch_timestep
    config.set_path()

def sis_SC_notWeight():
    config = ExperimentConfig.default(
        "dynamicLearning-edgeindex-sis-sc",
        "sis_sc",
        "sc",
        is_weight=False,
        seed=0
    )
    fix_config(config)
    exp = Experiment(config)
    exp.run()



num_samples = 10000
testSet_timestep = 10
epochs = 30 # 10
checkFirstEpoch = False # 10
checkFirstEpoch_max_time = 1000
checkFirstEpoch_timestep = 100

sis_SC_notWeight()