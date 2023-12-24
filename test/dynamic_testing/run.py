import os
from mydynalearn.config import ExperimentTrainConfig
from dynamic_experiment import DynamicExperiment

def fix_config(config):
    # T总时间步
    config.dataset.NUM_SAMPLES = NUM_SAMPLES
    config.dataset.NUM_TEST = TESTSET_TIMESTEP
    config.dataset.EPOCHS = EPOCHS  # 10
    # 检查点
    config.dataset.check_first_epoch = check_first_epoch  # 10
    config.dataset.check_first_epoch_maxtime = check_first_epoch_max_time
    config.dataset.check_first_epoch_timestep = check_first_epoch_timestep
    config.set_path()


def get_experiment(NAME, network, dynamics, dataset, MODEL_NAME, weight):
    config = ExperimentTrainConfig.default(
        NAME=NAME,
        network=network,
        dynamics=dynamics,
        dataset=dataset,
        MODEL_NAME=MODEL_NAME,
        IS_WEIGHT=weight,
        seed=0
    )
    fix_config(config)
    exp = DynamicExperiment(config)
    return exp

# 获取配置

# T总时间步
NUM_SAMPLES = 1000
TESTSET_TIMESTEP = 10
EPOCHS = 1 # 10
check_first_epoch = False # 10
check_first_epoch_max_time = 1000
check_first_epoch_timestep = 100

'''
network = ["ER","SCER","ToySCER"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU","ToySCCompUAU"]
dataset = ["graph","simplicial"]
'''
exp = get_experiment(NAME="dynamicLearning-SCER-SCCompUAU-SAT",
                     network="ToySCER",
                     dynamics="ToySCCompUAU",
                     dataset="simplicial",
                     MODEL_NAME="simplicial",
                     weight=False)

exp.run()