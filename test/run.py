import os

# 获取配置
from mydynalearn.config import ExperimentConfig
from mydynalearn.experiments import ExperimentTrain


def fix_config(config):
    # T总时间步
    config.network.NUM_NODES = num_nodes
    config.dataset.num_samples = num_samples
    config.dataset.num_test = testset_timestep
    config.dataset.epochs = epochs  # 10
    # 检查点
    config.dataset.check_first_epoch = check_first_epoch  # 10
    config.dataset.check_first_epoch_maxtime = check_first_epoch_maxtime
    config.dataset.check_first_epoch_timestep = check_first_epoch_timestep
    config.set_path()


def get_experiment(**kwags):
    config = ExperimentConfig.default(**kwags)
    fix_config(config)
    exp = ExperimentTrain(config)
    return exp


num_samples = 50
num_nodes = 1000
testset_timestep = 10
epochs = 2  # 10
check_first_epoch = False  # 10
check_first_epoch_maxtime = 1000
check_first_epoch_timestep = 100


'''
network = ["ER","SCER","CONFERENCE","HIGHSCHOOL","HOSPITAL","WORKPLACE"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU"]
dataset = ["GAT","SAT","DiffSAT"]
'''
kwags = {
    "NAME": "dynamicLearning-HOSPITAL-SCCompUAU-GAT",
    "network": "HOSPITAL",
    "dynamics": "SCCompUAU",
    "nn_type": "GAT",
    "is_weight": False,
    "seed": 0
}
exp = get_experiment(**kwags)
exp.run()





