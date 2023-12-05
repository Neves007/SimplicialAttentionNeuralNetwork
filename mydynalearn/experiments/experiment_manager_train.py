import os

# 获取配置
from mydynalearn.config import ExperimentTrainConfig,ExperimentRealConfig
from mydynalearn.experiments import ExperimentTrain,ExperimentRealnet
import torch
import itertools
from mydynalearn.logger.logger import *

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

    def get_loaded_model_exp(self, train_args, epoch_index):
        model_exp = self.get_train_exp(*train_args)
        model_exp.model.load_model(epoch_index)
        model_exp.generate_data()
        model_exp.partition_dataSet()
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
        config = ExperimentTrainConfig().default(**kwargs)
        self.fix_config(config)
        exp = ExperimentTrain(config)
        return exp

    def run(self):
        '''
        训练模型
        输出：模型的参数 model_state_dict
        '''
        print("*"*10+" TRAINING PROCESS "+"*"*10)
        train_params = self.set_train_params()
        for train_param in train_params:
            log_train_begin(train_param)
            exp = self.get_train_exp(*train_param)
            exp.run()
            torch.cuda.empty_cache()
        print("PROCESS COMPLETED!\n\n")

