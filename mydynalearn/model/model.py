import os
import torch.nn as nn
from mydynalearn.model.optimizer import get as get_optimizer
from mydynalearn.drawer import VisdomController

from mydynalearn.model.epoch_task import EpochTask
from mydynalearn.model.getter import get as get_attmodel

class Model():
    def __init__(self, config):
        """Dense version of GAT."""
        # config
        self.config = config
        self.model_config = config.model
        # dataset

        # model params
        self.IS_WEIGHT = config.IS_WEIGHT
        self.criterion = nn.MSELoss()
        self.EPOCHS = self.model_config.EPOCHS

        self.untrained_epoch_tasks = [epoch_task for epoch_task in self.untrained_epoch_task_iter()]
        self.need_to_train = self.is_need_to_train() # 判断是否需要训练

        #
        self.attention_model = get_attmodel(self.config)
        self.get_optimizer = get_optimizer(self.model_config.optimizer)
        self.optimizer = self.get_optimizer(self.attention_model.parameters())

    def set_dataset(self,dataset):
        self.dataset = dataset
    def set_dynamics(self):
        self.dynamics = self.dataset.dynamics
    def set_network(self):
        self.network = self.dataset.network

    def get_epoch_task_iter(self):
        for epoch_index in range(self.EPOCHS):
            epochtask = EpochTask(self.config, epoch_index)
            yield epochtask
    def untrained_epoch_task_iter(self):
        for epoch_index in range(self.EPOCHS):
            epochtask = EpochTask(self.config, epoch_index)
            if not os.path.exists(epochtask.model_state_dict_file):
                yield epochtask
    def is_need_to_train(self):
        '''判断是否需要训练
        如果所有epoch都训练了，那么不需要再训练了
        否则重新训练
        :return: bool
        '''
        if len(self.untrained_epoch_tasks) > 0:
            tag = True
        else:
            tag = False
        return tag

    # 定义模型
    def run(self,dataset):
        epoch_tasks_iter = self.get_epoch_task_iter()
        for epoch_task in epoch_tasks_iter:
            epoch_task.run(dataset,
                           self.attention_model,
                           self.optimizer)

