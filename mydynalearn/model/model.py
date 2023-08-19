from mydynalearn.model.nn.nnlayers import *
import os
import pickle
import torch.nn as nn
import torch
from .optimizer import get as get_optimizer
from mydynalearn.dataset import graph_DataSetLoader
from mydynalearn.drawer import VisdomController
from .util import *
import copy

from tqdm import tqdm

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
        self.path_to_model_params = config.path_to_model
        self.in_layers_node_feature = get_node_in_layers(self.model_config)
        self.in_layers_edge_feature = get_edge_in_layers(self.model_config)
        self.in_layers_simplex_feature_1D = get_node_in_layers(self.model_config)
        self.gat_layer = get_gat_layer(self.model_config)
        self.out_layers = get_out_layers(self.model_config)
        self.first_epoch_checkpoints = []
        self.check_first_epoch = self.dataset_config.check_first_epoch
        self.is_weight = config.is_weight
        self.check_first_epoch_max_time = self.dataset_config.check_first_epoch_maxtime
        self.check_first_epoch_timestep = self.dataset_config.check_first_epoch_timestep
        self.modelDir_first_epoch_checkpoints = self.path_to_model_params + "/first_epoch_checkpoints/"
        self.modelFile_name_first_epoch_checkpoints = self.modelDir_first_epoch_checkpoints + "/{:d}_{:d}model_first_epoch_checkpoints.pkl".format(self.check_first_epoch_max_time,self.check_first_epoch_timestep)
        self.VisdomDrawer = VisdomController(config, self.dynamics)
        if self.check_first_epoch:
            self.epochs = 1
        else:
            self.epochs = self.dataset_config.epochs
        self.get_optimizer = get_optimizer(self.model_config.optimizer)
        self.optimizer = self.get_optimizer(self.parameters())


    def weighted_cross_entropy(self,y_true, y_pred, weights=None):
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
        return loss.sum()

    def save_epoch_data(self, epoch_index, testResult):
        self.fileName = self.config.path_to_epochdata + "/epoch{:d}Data.pkl".format(epoch_index)
        with open(self.fileName, "wb") as file:
            pickle.dump(testResult,file)

    def set_dataset_loader(self,test_dataset, train_dataset, val_dataset):
        pass
    def fit(
            self,
            train_dataset,
            batch_size=1,
            learning_rate=1e-3,
            val_dataset=None,
            test_dataset=None,
    ):
        self.set_dataset_loader(test_dataset, train_dataset, val_dataset)
        if self.check_first_epoch:
            if os.path.exists(self.modelFile_name_first_epoch_checkpoints):
                self.loadfirst_epoch_checkpoints()
            else:
                self.train()
                for epoch_index in range(self.epochs):
                    test_result_curepoch = self._do_epoch_(epoch_index, batch_size=batch_size)
                self.eval()
                self.savefirst_epoch_checkpoints()
        else:
            self.train()
            for epoch_index in range(self.epochs):
                test_result_curepoch = self.get_test_result(epoch_index)
                self.VisdomDrawer.visdomDrawEpoch(epoch_index, test_result_curepoch)
                self._do_epoch_(epoch_index, batch_size=batch_size)  # 训练
                self.low_the_lr(epoch_index)
                self.save_epoch_data(epoch_index, test_result_curepoch)
            self.eval()

    # 定义模型
    def low_the_lr(self, epoch_i):
        if (epoch_i>0) and (epoch_i % 5 == 0):
            lr = self.optimizer.param_groups[0]['lr'] * 0.5
            self.optimizer.param_groups[0]['lr'] = lr

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

    def get_test_result(self, epoch_index):
        self.eval()
        process_bar = tqdm(
            enumerate(self.test_loader),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=self.test_loader.data_set['x0_T'].shape[0],
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
            if self.check_first_epoch:
                if (time_idx > self.check_first_epoch_max_time) and (self.check_first_epoch_timestep):
                    break
                if time_idx % self.check_first_epoch_timestep == 0:
                    self.setfirst_epoch_checkpoints(epoch_idx, time_idx)
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
            self.VisdomDrawer.visdomDrawBatch(train_data, val_data)
        self.eval()

    def _do_batch_(self, train_dataset_per_time):
        x,y_pred,y_true,y_ob, w = self.prepare_output(train_dataset_per_time)
        loss = self.weighted_cross_entropy(y_true, y_pred, w)
        return loss,x,y_pred,y_true,y_ob, w


    # save
    def loadfirst_epoch_checkpoints(self):
        with open(self.modelFile_name_first_epoch_checkpoints, "rb") as file:
            self.first_epoch_checkpoints = pickle.load(file)
    def savefirst_epoch_checkpoints(self):
        if not os.path.exists(self.modelDir_first_epoch_checkpoints):
            os.makedirs(self.modelDir_first_epoch_checkpoints)
        with open(self.modelFile_name_first_epoch_checkpoints, "wb") as file:
            pickle.dump(self.first_epoch_checkpoints,file)

    def setfirst_epoch_checkpoints(self, epoch_idx, time_idx):
        checkpoint = {
            'model_state_dict': copy.deepcopy(self.state_dict()),
            'epoch': epoch_idx,
            'time': time_idx
        }
        self.first_epoch_checkpoints.append(checkpoint)
