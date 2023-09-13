import os

# 获取配置
from mydynalearn.config import ExperimentTrainConfig,ExperimentRealConfig
from mydynalearn.experiments import ExperimentTrain,ExperimentRealnet
import torch
import itertools

class ExperimentManagerTrain():
    def __init__(self,num_samples, testset_timestep, epochs,params):
        self.num_samples = num_samples
        self.testset_timestep = testset_timestep
        self.epochs = epochs
        self.rootpath = r"./output/data/train_model"
        self.params = params

    def fix_config(self,config):
        # T总时间步
        config.dataset.num_samples = self.num_samples
        config.dataset.num_test = self.testset_timestep
        config.dataset.epochs = self.epochs  # 10

    def get_loaded_model_exp(self,train_args):
        model_exp = self.get_train_exp(*train_args)
        model_file = os.path.join(model_exp.config.datapath_to_model_state_dict, "model_state_dict.pth")
        model_exp.model.load_state_dict(torch.load(model_file))
        return model_exp

    def set_train_params(self):
        network_dynamics_dataset_config_list = []
        if "grpah_network" in self.params:
            assert "grpah_dynamics" in self.params
            network_dynamics_dataset_config_list += list(
                itertools.product(
                    self.params["grpah_network"],
                    self.params["grpah_dynamics"],
                    self.params["model"],
                    self.params["is_weight"]))
        if "simplicial_network" in self.params:
            assert "simplicial_dynamics" in self.params
            network_dynamics_dataset_config_list += list(
                itertools.product(
                    self.params["simplicial_network"],
                    self.params["simplicial_dynamics"],
                    self.params["model"],
                    self.params["is_weight"]))
        return network_dynamics_dataset_config_list
    def get_train_exp(self,network, dynamics, model, is_weight):
        exp_name = "dynamicLearning-" + network + "-" + dynamics + "-" + model
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "nn_type": model,
            "is_weight": is_weight,
            "seed": 0,
            "rootpath": self.rootpath
        }
        config = ExperimentTrainConfig.default(**kwargs)
        self.fix_config(config)
        exp = ExperimentTrain(config)
        return exp

    def train_model(self):
        train_params = self.set_train_params()
        for train_param in train_params:
            args = train_param
            exp = self.get_train_exp(*args)
            exp.run()

