import random
from abc import abstractmethod

import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb
from easydict import EasyDict as edict
from mydynalearn.networks.util.util import nodeToEdge_matrix

class Network():
    def __init__(self, net_config):
        # toy_network = True
        self.net_config = net_config
        self.NAME = self.net_config.NAME
        self.device = self.net_config.device
        self.MAX_DIMENSION = self.net_config.MAX_DIMENSION

        self.net_info = self._create_network()  # 网络信息
        self._set_net_info()
        self.inc_matrix_adj_info = self._get_adj()  # 关联矩阵
        self.set_inc_matrix_adj_info()
        self._to_device()
        pass

    @abstractmethod
    def set_inc_matrix_adj_info(self):
        pass
    @abstractmethod
    def _set_net_info(self):
        pass
    @abstractmethod
    def _create_network(self):
        pass

    @abstractmethod
    def _get_adj(self):
        pass

    @abstractmethod
    def _to_device(self):
        pass

    @abstractmethod
    def _unpack_net_info(self):
        pass
