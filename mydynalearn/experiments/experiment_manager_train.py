import os

# 获取配置
from mydynalearn.config import ExperimentTrainConfig,ExperimentRealConfig
from mydynalearn.experiments import ExperimentTrain,ExperimentRealnet
import torch
import itertools
from mydynalearn.logger.logger import *

class ExperimentManagerTrain():
    def __init__(self,NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS,params):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.TESTSET_TIMESTEP = TESTSET_TIMESTEP
        self.EPOCHS = EPOCHS
        self.rootpath = r"./output/data/train_model"
        self.params = params

    def fix_config(self,config):
        # T总时间步
        config.dataset.NUM_SAMPLES = self.NUM_SAMPLES
        config.dataset.NUM_TEST = self.TESTSET_TIMESTEP
        config.model.EPOCHS = self.EPOCHS  # 10

    def get_loaded_model_exp(self, train_args, epoch_index):
        model_exp = self.get_train_exp(*train_args)
        model_exp.set_model()
        model_exp.model.load_model(epoch_index)
        model_exp.generate_data()
        model_exp.partition_dataSet()
        return model_exp

    def get_train_params(self, only_higher_order=False):
        train_params_graph = []
        train_params_simplex = []

        if "grpah_network" in self.params:
            assert "grpah_dynamics" in self.params
            train_params_graph = list(
                itertools.product(
                    self.params["grpah_network"],
                    self.params["grpah_dynamics"],
                    self.params["model"],
                    self.params["IS_WEIGHT"]))
        if "simplicial_network" in self.params:
            assert "simplicial_dynamics" in self.params
            train_params_simplex = list(
                itertools.product(
                    self.params["simplicial_network"],
                    self.params["simplicial_dynamics"],
                    self.params["model"],
                    self.params["IS_WEIGHT"]))
        if only_higher_order:
            train_params = train_params_simplex
        else:
            train_params = train_params_graph + train_params_simplex
        return train_params



    def get_train_exp(self,network, dynamics, model, IS_WEIGHT):
        exp_name = "dynamicLearning-" + network + "-" + dynamics + "-" + model
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "MODEL_NAME": model,
            "IS_WEIGHT": IS_WEIGHT,
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
        train_params = self.get_train_params()
        for train_param in train_params:
            log_train_begin(train_param)
            exp = self.get_train_exp(*train_param)
            exp.run()
            torch.cuda.empty_cache()
        print("PROCESS COMPLETED!\n\n")

