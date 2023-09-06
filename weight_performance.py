import matplotlib.pyplot as plt
import numpy as np

import os
import pickle

# 获取配置
from mydynalearn.config import ExperimentConfig
from mydynalearn.experiments import ExperimentTrain
import itertools
from mydynalearn.drawer.matplot_drawer.drawer_matplot_maxR import DrawerMatplotMaxR


def fix_config(config):
    # T总时间步
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


num_samples = 10000
testset_timestep = 10
epochs = 30  # 10
check_first_epoch = False  # 10
check_first_epoch_maxtime = 1000
check_first_epoch_timestep = 100


'''
network = ["ER","SCER"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU"]
dataset = ["graph","simplicial"]
'''
grpah_network = ["ER"]
grpah_dynamics = ["UAU","CompUAU"]
simplicial_network = ["SCER"]
simplicial_dynamics = ["SCUAU","SCCompUAU"]
model = ["GAT","SAT","DiffSAT"]
is_weight = [True,False]

graph_network_dynamics_dataset_config_list = list(itertools.product(grpah_network,grpah_dynamics, model,is_weight))
simplicial_network_dynamics_dataset_config_list = list(itertools.product(simplicial_network,simplicial_dynamics, model,is_weight))
network_dynamics_dataset_config_list = graph_network_dynamics_dataset_config_list + simplicial_network_dynamics_dataset_config_list

DrawerMatplotMaxR.create_fig()
for network_dynamics_dataset_config in network_dynamics_dataset_config_list:
    network,dynamics,model,is_weight = network_dynamics_dataset_config
    exp_name = "dynamicLearning-" + network + "-" + dynamics + "-"+ model
    kwags = {
        "NAME" : exp_name,
        "network" : network,
        "dynamics" : dynamics,
        "nn_type" : model,
        "is_weight" : is_weight,
        "seed" : 0
    }
    exp = get_experiment(**kwags)
    drawer_matplot_maxR = DrawerMatplotMaxR(exp.config, exp.dynamics)
    drawer_matplot_maxR.draw()
drawer_matplot_maxR.save_fig()
