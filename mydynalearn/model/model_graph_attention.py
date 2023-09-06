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
    def __init__(self, config,network,dynamics):
        """Dense version of GAT."""
        super().__init__(config,network,dynamics)
        self.in_layer = get_node_in_layers(self.model_config)
        self.gat_layer_1 = get_gnn_layer(self.model_config)

    def forward(self, x0,x1,y_true,y_ob,weight,network,**kwargs):
        # 只考虑edge_index
        x0_in = self.in_layer(x0)
        x = self.gat_layer_1(x0_in, network)
        out = self.out_layers(x)
        return x0,out,y_true,y_ob, weight






