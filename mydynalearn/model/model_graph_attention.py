from mydynalearn.model.nn.nnlayers import *
import os
import pickle
import torch.nn as nn
import torch
from .optimizer import get as get_optimizer
from mydynalearn.dataset import graph_DataSetLoader
from .util import *
from .model import Model
import copy

from tqdm import tqdm

class graphAttentionModel(Model):
    def __init__(self, config,network,dynamics):
        """Dense version of GAT."""
        super().__init__(config,network,dynamics)
        self.in_layers_node_feature = get_node_in_layers(self.model_config)
        self.in_layers_edge_feature = get_edge_in_layers(self.model_config)
        self.gat_layer_1 = get_gat_layer(self.model_config)
        self.gat_layer_2 = get_gat_layer(self.model_config)

    def set_dataset_loader(self,test_dataset, train_dataset, val_dataset):
        self.test_loader = graph_DataSetLoader(test_dataset)
        self.train_loader = graph_DataSetLoader(train_dataset)
        self.val_loader = graph_DataSetLoader(val_dataset)
    def prepare_output(self, data,is_weight):
        network, x0, x1, y_ob, y_true, weight = data
        if is_weight==False:
            weight = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)]).to(self.device)
        y_true = y_true
        y_pred = self.forward(x0,x1,network)
        return x0,y_pred,y_true,y_ob, weight
    def forward(self, x0,x1,network):
        # 只考虑edge_index
        x0 = self.in_layers_node_feature(x0)
        x1 = self.in_layers_edge_feature(x1)

        x = self.gat_layer_1(x0, network)
        # x = self.gat_layer_2(x, network)
        out = self.out_layers(x)
        return out





