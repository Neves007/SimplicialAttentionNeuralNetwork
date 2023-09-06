
import torch

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


num_samples = 10
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
    "NAME": "dynamicLearning-ER-UAU-GAT",
    "network": "ER",
    "dynamics": "UAU",
    "nn_type": "GAT",
    "is_weight": False,
    "seed": 0
}
model_exp = get_experiment(**kwags)
model_exp.generate_data()
model_exp.partition_dataSet()
file = os.path.join(model_exp.config.datapath_to_model_state_dict, "model_state_dict.pth")
model_exp.model.load_state_dict(torch.load(file))

kwags = {
    "NAME": "dynamicLearning-SCER-SCUAU-SAT",
    "network": "SCER",
    "dynamics": "SCCompUAU",
    "nn_type": "SAT",
    "is_weight": False,
    "seed": 0
}
data_exp = get_experiment(**kwags)
data_exp.generate_data()
data_exp.partition_dataSet()
test_result_curepoch = model_exp.model.get_test_result(0, data_exp.test_loader)
