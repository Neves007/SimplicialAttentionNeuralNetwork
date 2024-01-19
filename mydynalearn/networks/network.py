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
    def __init__(self, config):
        # toy_network = True
        self.config = config
        self.net_config = config.network
        self.NAME = self.net_config.NAME
        self.DEVICE = self.config.DEVICE
        self.MAX_DIMENSION = self.net_config.MAX_DIMENSION
        pass

    def create_net(self):
        self.net_info = self.get_net_info()
        self._set_net_info()
        self.inc_matrix_adj_info = self._get_adj()  # 关联矩阵
        self.set_inc_matrix_adj_info()
        self.to_device(self.DEVICE)

    def print_log(self,num_indentation=0):
        ''' 对齐输出

        :param num_indentation: 程度
        :return:
        '''
        # 缩进
        num_indentation += 1
        indentation = num_indentation*"\t"
        # 输出内容
        log_items_list = (("network name:",self.NAME),
                    ("network dimension:", self.MAX_DIMENSION),
                    ("number of nodes:", self.NUM_NODES),
                    ("average degree:", self.AVG_K),
                    )

        # 对齐字段宽度
        field_width = int(max([len(log_items[0]) for log_items in log_items_list])) + 2
        # 输出
        for log_items in log_items_list:
            print("{}{:<{}}{}".format(indentation,log_items[0],field_width,log_items[1]))


    @abstractmethod
    def set_inc_matrix_adj_info(self):
        pass

    @abstractmethod
    def _set_net_info(self):
        pass

    @abstractmethod
    def get_net_info(self):
        pass

    @abstractmethod
    def _get_adj(self):
        pass

    @abstractmethod
    def to_device(self,device):
        pass

    @abstractmethod
    def _unpack_net_info(self):
        pass
