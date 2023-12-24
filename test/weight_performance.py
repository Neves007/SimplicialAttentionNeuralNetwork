import matplotlib.pyplot as plt
import numpy as np

import os
import pickle

# 获取配置
from mydynalearn.config import ExperimentTrainConfig
from mydynalearn.experiments import ExperimentTrain
import itertools


def fix_config(config):
    # T总时间步
    config.dataset.NUM_SAMPLES = NUM_SAMPLES
    config.dataset.NUM_TEST = TESTSET_TIMESTEP
    config.dataset.EPOCHS = EPOCHS  # 10
    # 检查点
    config.dataset.check_first_epoch = check_first_epoch  # 10
    config.dataset.check_first_epoch_maxtime = check_first_epoch_maxtime
    config.dataset.check_first_epoch_timestep = check_first_epoch_timestep
    config.set_path()


def get_experiment(**kwargs):
    config = ExperimentTrainConfig().default(**kwargs)
    fix_config(config)
    exp = ExperimentTrain(config)
    return exp


NUM_SAMPLES = 10000
TESTSET_TIMESTEP = 10
EPOCHS = 30  # 10
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
IS_WEIGHT = ["IS_WEIGHT","not_weight","DiffSAT"]

graph_network_dynamics_dataset_config_list = list(itertools.product(grpah_network,grpah_dynamics, model))
simplicial_network_dynamics_dataset_config_list = list(itertools.product(simplicial_network,simplicial_dynamics, model))
network_dynamics_dataset_config_list = graph_network_dynamics_dataset_config_list + simplicial_network_dynamics_dataset_config_list

for network_dynamics_dataset_config in network_dynamics_dataset_config_list:
    network,dynamics,model = network_dynamics_dataset_config
    exp_name = "dynamicLearning-" + network + "-" + dynamics + "-"+ model
    kwargs = {
        "NAME" : exp_name,
        "network" : network,
        "dynamics" : dynamics,
        "MODEL_NAME" : model,
        "IS_WEIGHT" : False,
        "seed" : 0
    }
    config = ExperimentTrainConfig().default(**kwargs)
    datapath_to_epochdata = config.datapath_to_epochdata
    for epoch_index in range(EPOCHS):
        fileName = datapath_to_epochdata + "\\epoch{:d}Data.pkl".format(epoch_index)
        file_path = "../data/dynamicLearning-ER-UAU-GAT/not_weight/modelResult/epoch_data/epoch0Data.pkl"
        with open(fileName, "rb") as file:
            data = pickle.load(file)
        file.close()



