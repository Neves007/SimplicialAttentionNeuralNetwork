from mydynalearn.model.nn.nnlayers import *
import os
import pickle
import torch.nn as nn
import torch
from .optimizer import get as get_optimizer
from mydynalearn.dataset import graphDataSetLoader
from .util import *
from .model import Model
import copy

from tqdm import tqdm

class GraphAttentionModel(Model):
    def __init__(self, config,dataset):
        """Dense version of GAT."""
        super().__init__(config,dataset)
        self.in_layer = get_node_in_layers(self.model_config)
        self.gat_layer_1 = get_gnn_layer(self.model_config)
        self.out_layers = get_out_layers(self.model_config)

    def forward(self, x0, y_ob, y_true, weight, networkid):
        # 数据预处理
        x0 = x0.squeeze()
        y_ob = y_ob.squeeze()
        y_true = y_true.squeeze()
        weight = weight.squeeze()
        network = self.networks[networkid]  # 网络数据
        # attention
        x0_in = self.in_layer(x0)
        x = self.gat_layer_1(x0_in, network)
        out = self.out_layers(x)
        return x0,out,y_true,y_ob, weight

