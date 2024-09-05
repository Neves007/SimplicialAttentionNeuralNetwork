import os.path
import pickle
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
        self.file_path = self.get_file_path()
        pass

    def get_file_path(self):
        network_dir_path = self.config.network_dir_path
        file_name = self.NAME+'.pkl'
        file_path = os.path.join(network_dir_path,file_name)
        return file_path

    def save(self):
        """
        Save the current network object to a file using pickle.
        """
        try:
            with open(self.file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Network saved successfully to {self.file_path}")
        except Exception as e:
            print(f"An error occurred while saving the network: {e}")

    def load(self):
        """
        Load a network object from a file using pickle.
        """
        with open(self.file_path, 'rb') as f:
            loaded_network = pickle.load(f)
        self.__dict__.update(loaded_network.__dict__)
        print(f"Network loaded successfully from {self.file_path}")

    def create_net(self):
        try:
            self.load()
        except FileNotFoundError:
            self.net_info = self.get_net_info()
            self._set_net_info()
            self.inc_matrix_adj_info = self._get_adj()  # 关联矩阵
            self.set_inc_matrix_adj_info()
            self.to_device(self.DEVICE)
            self.save()
        return self

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
