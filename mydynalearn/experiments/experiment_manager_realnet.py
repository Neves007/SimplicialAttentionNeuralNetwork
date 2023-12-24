import os

# 获取配置
from mydynalearn.config import ExperimentTrainConfig, ExperimentRealConfig
from mydynalearn.experiments import ExperimentTrain, ExperimentRealnet
import torch
import itertools
from mydynalearn.logger.logger import *


class ExperimentManagerRealnet():
    def __init__(self, NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.TESTSET_TIMESTEP = TESTSET_TIMESTEP
        self.EPOCHS = EPOCHS
        self.rootpath = r"./output/data/realnet"
        self.params = params

    def get_loaded_real_exp(self,network_dynamics_dataset_config):
        real_exp = self.get_realnet_exp(*network_dynamics_dataset_config)
        real_exp.generate_data()
        real_exp.partition_dataSet()
        return real_exp

    def fix_config(self, config):
        # T总时间步
        config.dataset.NUM_SAMPLES = self.NUM_SAMPLES
        config.dataset.NUM_TEST = self.TESTSET_TIMESTEP
        config.dataset.FOR_REALNET = True  # 用于区是否是真实网络

    def set_realnet_params(self,net_params =None ,dynamics_params=None):
        if net_params == None:
            net_params = self.params["real_network"]
        else:
            net_params = net_params
        if dynamics_params == None:
            dynamics_params = self.params["simplicial_dynamics"]
        else:
            dynamics_params = dynamics_params

        real_network_dynamics_dataset_config_list = list(
            itertools.product(
                net_params,
                dynamics_params))
        return real_network_dynamics_dataset_config_list

    def get_realnet_exp(self, network, dynamics):
        exp_name = "realnetwork-" + network + "-" + dynamics
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "seed": 0,
            "rootpath": self.rootpath
        }
        config = ExperimentRealConfig().default(**kwargs)
        self.fix_config(config)
        exp = ExperimentRealnet(config)
        return exp
    def get_available_realnet_dynamics(self,train_dynamics):
        dynamic_map = {
            "UAU": ["SCUAU"],
            "CompUAU": ["SCCompUAU"],
            "SCUAU": ["SCUAU"],
            "SCCompUAU": ["SCCompUAU"],
        }
        return dynamic_map[train_dynamics]
    def run(self):
        '''
        生成真实数据
        '''
        realnet_params = self.set_realnet_params()
        print("*"*10+" REALNET DATASET BUILDING PROCESS "+"*"*10)
        for realnet_param in realnet_params:
            log_realnet_begin(realnet_param)
            exp = self.get_realnet_exp(*realnet_param)
            exp.run()
        print("PROCESS COMPLETED!\n\n")


