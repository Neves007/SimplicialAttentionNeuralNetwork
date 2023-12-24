'''

'''
import random
from abc import abstractmethod

import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb
from easydict import EasyDict as edict
from mydynalearn.networks.util.util import nodeToEdge_matrix
from mydynalearn.networks.getter import get as get_network

class NetworkManager():
    '''分配网络
    配置不同平均度的网络用于训练

    netwrok_manager = NetworkManager(config)
    networks_generator = netwrok_manager.networks_generator()
    networks = [net for net in networks_generator]
    '''
    def __init__(self, config):
        self.dataset_config = config.dataset
        self.network_config = config.network
        self.config = config
        # 网络的类型
        self.NAME = self.network_config.NAME
        self.NUM_NODES = self.network_config.NUM_NODES
        self.MAX_DIMENSION = self.network_config.MAX_DIMENSION

        # 网络的种类
        self.AVG_K_MIN = self.dataset_config.AVG_K_MIN
        self.AVG_K_MAX = self.dataset_config.AVG_K_MAX
        self.NUM_K = self.dataset_config.NUM_K
        self.AVG_K_LIST =iter(self.dataset_config.AVG_K_LIST)
        self.NUM_NET = self.NUM_K ** self.MAX_DIMENSION

    def create_network(self, AVG_K, AVG_K_DELTA):
        '''
        :param NETWORK_NAME: str
        :param AVG_K: int
        :param AVG_K_DELTA: int
        :return:
        '''
        # get network
        network = get_network(self.config)
        network.create_net(AVG_K, AVG_K_DELTA)
        return network

    def networks_generator(self):
        '''网络生成器
        根据度来生成网络
        '''
        # 低阶网络
        if self.MAX_DIMENSION==1:
            for AVG_K in self.AVG_K_LIST:
                network = self.create_network(AVG_K,AVG_K_DELTA=0)
                yield network
        # 高阶网络
        elif self.MAX_DIMENSION==2:
            for AVG_K in self.AVG_K_LIST:
                AVG_K_DELTA_MIN = 1
                AVG_K_DELTA_MAX = AVG_K / 2  # 根据公式 最大二阶度是AVG_K/2
                NUM_K_DELTA = self.NUM_K  # 网络的个数
                # 每一个一阶平均度
                AVG_K_DELTA_LIST = iter(torch.linspace(AVG_K_DELTA_MIN, AVG_K_DELTA_MAX, NUM_K_DELTA))
                for AVG_K_DELTA in AVG_K_DELTA_LIST:
                    # 通过一阶和二阶平均度生成网络
                    network = self.create_network(AVG_K, AVG_K_DELTA)
                    yield network




