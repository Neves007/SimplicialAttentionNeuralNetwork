import os
import torch.nn as nn
from mydynalearn.model.optimizer import get as get_optimizer

from mydynalearn.model.epoch_tasks import EpochTasks

from torch.utils.data import DataLoader
import torch
class Model():
    def __init__(self, config):
        """Dense version of GAT."""
        # config
        self.config = config
        self.NAME = config.model.NAME
        self.epoch_tasks = EpochTasks(config)
        self.need_to_train = self.epoch_tasks.need_to_train

    # 放进数据集类里面

    # 定义模型
    def run(self,network, dynamics, train_set, val_set, test_set):
        self.epoch_tasks.run_all(network, dynamics, train_set, val_set, test_set)