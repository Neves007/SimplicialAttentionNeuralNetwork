from mydynalearn.model.nn.nnlayers import *
import os
import pickle
import torch.nn as nn
import torch
from .optimizer import get as get_optimizer
from mydynalearn.dataset import graphDataSetLoader
from mydynalearn.drawer import VisdomController
from .util import *
import copy

from tqdm import tqdm
from mydynalearn.dataset.getter import get as dataset_getter

class Model(nn.Module):
    def __init__(self, config,network,dynamics):
        """Dense version of GAT."""
        super(Model, self).__init__()
        self.model_config = config.model
        self.dataset_config = config.dataset
        self.config = config
        self.network = network
        self.dynamics = dynamics
        self.device = config.device
        self.path_to_model_params = config.datapath_to_model
        self.out_layers = get_out_layers(self.model_config)
        self.is_weight = config.is_weight
        self.VisdomDrawer = VisdomController(config, self.dynamics)
        self.criterion = nn.MSELoss()
        self.epochs = self.dataset_config.epochs
        self.get_optimizer = get_optimizer(self.model_config.optimizer)
        self.optimizer = self.get_optimizer(self.parameters())


    def weighted_cross_entropy(self,y_true, y_pred, weights=None):
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
        # loss = weights * self.criterion(y_true,y_pred)
        return loss.sum()

    def save_epoch_data(self, epoch_index, testResult):
        fileName_testreselt = self.config.datapath_to_epochdata + "/epoch{:d}Data.pkl".format(epoch_index)
        with open(fileName_testreselt, "wb") as file:
            pickle.dump(testResult,file)
    def save_model_state_dict(self,epoch_index):
        fileName_model_state_dict = self.config.datapath_to_model_state_dict + "/epoch{:d}_model_state_dict.pth".format(epoch_index)
        if not os.path.exists(fileName_model_state_dict):
            torch.save(self.state_dict(), fileName_model_state_dict)

    def low_the_lr(self, epoch_i):
        if (epoch_i>0) and (epoch_i % 5 == 0):
            lr = self.optimizer.param_groups[0]['lr'] * 0.5
            self.optimizer.param_groups[0]['lr'] = lr

    # 定义模型
    def fit(
            self,
            batch_size=1,
            learning_rate=1e-3,
            train_loader=None,
            val_loader=None,
            test_loader=None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train()
        for epoch_index in range(self.epochs):
            fileName_testreselt = self.config.datapath_to_epochdata + "/epoch{:d}Data.pkl".format(epoch_index)
            if not os.path.exists(fileName_testreselt):
                test_result_curepoch = self.get_test_result(epoch_index,self.test_loader)
                # self.VisdomDrawer.visdomDrawEpoch(epoch_index, test_result_curepoch)
                self._do_epoch_(epoch_index, batch_size=batch_size)  # 训练
                self.low_the_lr(epoch_index)
                self.save_epoch_data(epoch_index, test_result_curepoch)
                self.save_model_state_dict(epoch_index)
        self.eval()

    def pack_batch_data(self, epoch_idx, time_idx, loss, x, y_pred, y_true, y_ob, w):
        data = {'epoch_idx': epoch_idx,
                'time_idx': time_idx,
                'loss': loss.cpu(),
                'acc': get_acc(y_ob, y_pred).cpu(),
                'x': x.cpu(),
                'y_pred': y_pred.cpu(),
                'y_true': y_true.cpu(),
                'y_ob': y_ob.cpu(),
                'w': w.cpu(),
                }
        return data

    def get_test_result(self, epoch_index,test_loader):
        self.eval()
        process_bar = tqdm(
            enumerate(test_loader),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=test_loader.data_set['x0_T'].shape[0],
        )
        test_result_curepoch = []
        for time_idx, test_dataset_per_time in process_bar:
            test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w = self._do_batch_(test_dataset_per_time)
            test_data = self.pack_batch_data(epoch_index, time_idx, test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w)
            test_result_curepoch.append(test_data)
        return test_result_curepoch


    def _do_epoch_(self, epoch_idx, batch_size=1,):
        self.train()
        process_bar = tqdm(
            enumerate(zip(self.train_loader, self.val_loader)),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=self.train_loader.data_set['x0_T'].shape[0],
        )
        for time_idx, (train_dataset_per_time,val_dataset_per_time) in process_bar:
            self.optimizer.zero_grad()
            train_loss,train_x,train_y_pred,train_y_true,train_y_ob, train_w = self._do_batch_(train_dataset_per_time)
            train_loss.backward()
            self.optimizer.step()
            val_loss, val_x, val_y_pred, val_y_true, val_y_ob, val_w = self._do_batch_(val_dataset_per_time)
            train_data = self.pack_batch_data(epoch_idx,
                                              time_idx,
                                              train_loss,
                                              train_x,
                                              train_y_pred,
                                              train_y_true,
                                              train_y_ob,
                                              train_w)
            val_data = self.pack_batch_data(epoch_idx,
                                            time_idx,
                                            val_loss,
                                            val_x,
                                            val_y_pred,
                                            val_y_true,
                                            val_y_ob,
                                            val_w)
            # self.VisdomDrawer.visdomDrawBatch(train_data, val_data)
        self.eval()

    def _do_batch_(self, train_dataset_per_time):
        x,y_pred,y_true,y_ob, w = self.prepare_output(train_dataset_per_time)
        loss = self.weighted_cross_entropy(y_true, y_pred, w)
        return loss,x,y_pred,y_true,y_ob, w
    # save
    def prepare_output(self, data):
        x0,y_pred,y_true,y_ob, weight = self.forward(**data)
        if self.is_weight==False:
            weight = torch.ones(x0.shape[0]).to(self.device)
        return x0,y_pred,y_true,y_ob, weight
