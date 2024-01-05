import os

# 获取配置
from mydynalearn.config import ExperimentTrainConfig
from mydynalearn.experiments import ExperimentTrain
import torch
import itertools
from mydynalearn.logger.logger import *
from mydynalearn.util.params_dealer import PasramsDealer

class ExperimentManager():
    def __init__(self,NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS,params):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.TESTSET_TIMESTEP = TESTSET_TIMESTEP
        self.EPOCHS = EPOCHS
        self.root_dir = r"./output/data/train_model"
        self.train_params = PasramsDealer.assemble_train_params(params)

    def fix_config(self,config):
        '''调整配置
        '''
        # T总时间步
        config.dataset.NUM_SAMPLES = self.NUM_SAMPLES
        config.dataset.NUM_TEST = self.TESTSET_TIMESTEP
        config.model.EPOCHS = self.EPOCHS  # 10

    def get_loaded_model_exp(self, train_args, epoch_index):
        '''加载指定模型指定epoch_index的训练模型
        :param train_args: (network, dynamics, model, IS_WEIGHT)
        :param epoch_index: int
        :return: model_exp
        '''
        # todo：这个函数不应该在这里
        model_exp = self.get_train_exp(*train_args)
        model_exp.model.load_model(epoch_index)
        model_exp.create_dataset()
        return model_exp


    def get_train_exp(self,network, dynamics, model, IS_WEIGHT):
        '''通过参数获得实验对象

        :param network:
        :param dynamics:
        :param model:
        :param IS_WEIGHT:
        :return: exp
        '''
        exp_name = "dynamicLearning-" + network + "-" + dynamics + "-" + model
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "MODEL_NAME": model,
            "IS_WEIGHT": IS_WEIGHT,
            "seed": 0,
            "root_dir": self.root_dir
        }
        config = ExperimentTrainConfig().default(**kwargs)
        self.fix_config(config)
        exp = ExperimentTrain(config)
        return exp

    def get_exp_iter(self):
        for train_param in self.train_params:
            exp = self.get_train_exp(*train_param)
            yield exp

    def run(self):
        '''
        训练模型
        输出：模型的参数 model_state_dict
        '''
        print("*"*10+" TRAINING PROCESS "+"*"*10)
        exp_iter = self.get_exp_iter()
        for exp in exp_iter:
            exp.run()
            # torch.cuda.empty_cache()
        print("TRAINING PROCESS COMPLETED!\n\n")


