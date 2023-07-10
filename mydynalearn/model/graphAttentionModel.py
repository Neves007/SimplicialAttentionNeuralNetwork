from mydynalearn.model.nn.GraphLayers import *
import os
import pickle
import torch.nn as nn
import torch
from .optimizer import get as get_optimizer
from mydynalearn.dataset import graph_DataSetLoader
from mydynalearn.drawer import FigureDrawer
from .util import *

import copy

from tqdm import tqdm

class graphAttentionModel(nn.Module):
    def __init__(self, config):
        """Dense version of GAT."""
        super(graphAttentionModel, self).__init__()
        model_config = config.model
        train_details_config = config.train_details
        self.device = config.device
        self.path_to_modelParams = config.path_to_model
        self.in_layers_nodeFeature = get_node_in_layers(model_config)
        self.in_layers_edgeFeature = get_edge_in_layers(model_config)
        self.in_layers_simplexFeature_1D = get_node_in_layers(model_config)
        self.gat_layers = get_gat_layers(model_config)
        self.out_layers = get_out_layers(model_config)
        self.get_optimizer = get_optimizer(model_config.optimizer)
        self.firstEpochCheckpoints = []
        self.checkFirstEpoch = train_details_config.checkFirstEpoch
        self.is_weight = config.is_weight
        self.checkFirstEpoch_max_time = train_details_config.checkFirstEpoch_max_time
        self.checkFirstEpoch_timestep = train_details_config.checkFirstEpoch_timestep
        self.modelDir_firstEpochCheckpoints = self.path_to_modelParams + "/firstEpochCheckpoints/"
        self.modelFile_name_firstEpochCheckpoints = self.modelDir_firstEpochCheckpoints + "/{:d}_{:d}model_firstEpochCheckpoints.pkl".format(self.checkFirstEpoch_max_time,self.checkFirstEpoch_timestep)
        self.figureDrawer = FigureDrawer(config)
        if self.checkFirstEpoch:
            self.epochs = 1
        else:
            self.epochs = train_details_config.epochs
        self.optimizer = self.get_optimizer(self.parameters())

    def forward(self, x0,x1,network):
        # 只考虑edge_index
        # todo:修改
        x0 = self.in_layers_nodeFeature(x0)
        x1 = self.in_layers_edgeFeature(x1)
        x = self.gat_layers(x0,x1,network)
        out = self.out_layers(x)
        return out

    def weighted_cross_entropy(self,y_true, y_pred, weights=None):
        weights /= weights.sum()
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
        return loss.sum()


    def fit(
            self,
            train_dataset,
            batch_size=1,
            learning_rate=1e-3,
            val_dataset=None,
            test_dataset=None,
    ):
        self.test_loader = graph_DataSetLoader(test_dataset)
        self.train_loader = graph_DataSetLoader(train_dataset)
        self.val_loader = graph_DataSetLoader(val_dataset)
        if self.checkFirstEpoch:
            if os.path.exists(self.modelFile_name_firstEpochCheckpoints):
                self.loadFirstEpochCheckpoints()
            else:
                self.train()
                for epoch_index in range(self.epochs):
                    testResult_curEpoch = self._do_epoch_(epoch_index, batch_size=batch_size)
                self.eval()
                self.saveFirstEpochCheckpoints()
        else:
            self.train()
            for epoch_index in range(self.epochs):
                testResult_curEpoch = self.getTestResult(epoch_index)
                self.figureDrawer.visdomDrawEpoch(epoch_index, testResult_curEpoch)
                self._do_epoch_(epoch_index, batch_size=batch_size)  # 训练
                self.low_the_lr(epoch_index)
                self.figureDrawer.matplot_epochPerformance.saveEpochData(epoch_index,testResult_curEpoch)
            self.eval()

    # 定义模型
    def low_the_lr(self, epoch_i):
        if (epoch_i>0) and (epoch_i % 5 == 0):
            lr = self.optimizer.param_groups[0]['lr'] * 0.5
            self.optimizer.param_groups[0]['lr'] = lr

    def packBatchData(self, epoch_idx, time_idx, loss, x, y_pred, y_true, y_ob, w):
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

    def getTestResult(self,epoch_index):
        self.eval()
        process_bar = tqdm(
            enumerate(self.test_loader),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=self.test_loader.data_set['x0_T'].shape[0],
        )
        testResult_curEpoch = []
        for time_idx, test_dataset_per_time in process_bar:
            test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w = self._do_batch_(test_dataset_per_time)
            test_data = self.packBatchData(epoch_index,time_idx,test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w)
            testResult_curEpoch.append(test_data)
        return testResult_curEpoch


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
            if self.checkFirstEpoch:
                if (time_idx > self.checkFirstEpoch_max_time) and (self.checkFirstEpoch_timestep):
                    break
                if time_idx % self.checkFirstEpoch_timestep == 0:
                    self.setFirstEpochCheckpoints(epoch_idx, time_idx)
            self.optimizer.zero_grad()
            train_loss,train_x,train_y_pred,train_y_true,train_y_ob, train_w = self._do_batch_(train_dataset_per_time)
            train_loss.backward()
            self.optimizer.step()
            val_loss, val_x, val_y_pred, val_y_true, val_y_ob, val_w = self._do_batch_(val_dataset_per_time)
            train_data = self.packBatchData(epoch_idx,
                                            time_idx,
                                            train_loss,
                                            train_x,
                                            train_y_pred,
                                            train_y_true,
                                            train_y_ob,
                                            train_w)
            val_data = self.packBatchData(epoch_idx,
                                          time_idx,
                                          val_loss,
                                          val_x,
                                          val_y_pred,
                                          val_y_true,
                                          val_y_ob,
                                          val_w)
            self.figureDrawer.visdomDrawBatch(train_data, val_data)
        self.eval()

    def _do_batch_(self, train_dataset_per_time):
        x,y_pred,y_true,y_ob, w = self.prepare_output(train_dataset_per_time)
        loss = self.weighted_cross_entropy(y_true, y_pred, w)
        return loss,x,y_pred,y_true,y_ob, w

    def prepare_output(self, data):
        network, x0, x1, y_ob, y_true, adjActEdges, weight = data
        if self.is_weight==False:
            w = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)]).to(self.device)
        y_true = y_true
        y_pred = self.forward(x0,x1,network)
        return x0,y_pred,y_true,y_ob, weight


    # save
    def loadFirstEpochCheckpoints(self):
        with open(self.modelFile_name_firstEpochCheckpoints, "rb") as file:
            self.firstEpochCheckpoints = pickle.load(file)
    def saveFirstEpochCheckpoints(self):
        if not os.path.exists(self.modelDir_firstEpochCheckpoints):
            os.makedirs(self.modelDir_firstEpochCheckpoints)
        with open(self.modelFile_name_firstEpochCheckpoints, "wb") as file:
            pickle.dump(self.firstEpochCheckpoints,file)

    def setFirstEpochCheckpoints(self, epoch_idx, time_idx):
        checkpoint = {
            'model_state_dict': copy.deepcopy(self.state_dict()),
            'epoch': epoch_idx,
            'time': time_idx
        }
        self.firstEpochCheckpoints.append(checkpoint)
