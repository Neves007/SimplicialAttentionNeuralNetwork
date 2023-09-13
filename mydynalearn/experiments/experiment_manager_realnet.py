import os

# 获取配置
from mydynalearn.config import ExperimentTrainConfig, ExperimentRealConfig
from mydynalearn.experiments import ExperimentTrain, ExperimentRealnet
import torch
import itertools


class ExperimentManagerRealnet():
    def __init__(self, num_samples, testset_timestep, epochs, params):
        self.num_samples = num_samples
        self.testset_timestep = testset_timestep
        self.epochs = epochs
        self.rootpath = r"./output/data/realnet"
        self.params = params

    def get_loaded_real_exp(self,network_dynamics_dataset_config):
        real_exp = self.get_realnet_exp(*network_dynamics_dataset_config)
        real_exp.generate_data()
        real_exp.partition_dataSet()
        return real_exp
    def fix_config(self, config):
        # T总时间步
        config.dataset.num_samples = self.num_samples
        config.dataset.num_test = self.testset_timestep
        config.dataset.epochs = self.epochs  # 10

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
        config = ExperimentRealConfig.default(**kwargs)
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
    def create_realnet_dynamics(self):
        network_dynamics_dataset_config_list = self.set_realnet_params()
        for network_dynamics_dataset_config in network_dynamics_dataset_config_list:
            network, dynamics = network_dynamics_dataset_config
            exp = self.get_realnet_exp(network, dynamics)
            exp.run()



