import os
from mydynalearn.config import ExperimentConfig
from dynamic_experiment import DynamicExperiment

def fix_config(config):
    # T总时间步
    config.dataset.num_samples = num_samples
    config.dataset.num_test = testSet_timestep
    config.dataset.epochs = epochs  # 10
    # 检查点
    config.dataset.check_first_epoch = check_first_epoch  # 10
    config.dataset.check_first_epoch_maxtime = check_first_epoch_max_time
    config.dataset.check_first_epoch_timestep = check_first_epoch_timestep
    config.set_path()


def get_experiment(NAME, network, dynamics, dataset, nn_type, weight):
    config = ExperimentConfig.default(
        NAME=NAME,
        network=network,
        dynamics=dynamics,
        dataset=dataset,
        nn_type=nn_type,
        is_weight=weight,
        seed=0
    )
    fix_config(config)
    exp = DynamicExperiment(config)
    return exp

# 获取配置

# T总时间步
num_samples = 1000
testSet_timestep = 10
epochs = 1 # 10
check_first_epoch = False # 10
check_first_epoch_max_time = 1000
check_first_epoch_timestep = 100

'''
network = ["ER","SCER"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU"]
dataset = ["graph","simplicial"]
'''
exp = get_experiment(NAME="dynamicLearning-SCER-SCCompUAU-SAT",
                     network="ToySCER",
                     dynamics="SCCompUAU",
                     dataset="simplicial",
                     nn_type="simplicial",
                     weight=False)

exp.run()