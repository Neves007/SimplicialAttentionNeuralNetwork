import os.path
import pickle
import random
from abc import abstractmethod
from mydynalearn.logger import Log

import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb
from easydict import EasyDict as edict
from mydynalearn.networks.util.util import nodeToEdge_matrix
from mydynalearn.util.lazy_loader import PickleLazyLoader
class Network(PickleLazyLoader):
    def __init__(self, config):
        # toy_network = True
        self.logger = Log("Network")
        self.config = config
        self.DEVICE = self.config.DEVICE
        self.net_config = config.network
        self.set_attr(self.net_config)
        self.file_path = self.get_file_path()
        super().__init__(self.file_path)
        pass
    def set_attr(self, attributes):
        '''
        批量设置属性
        :param attributes:
        :return:
        '''
        for key, value in attributes.items():
            setattr(self, key, value)

    def get_file_path(self):
        network_dir_path = self.config.network_dir_path
        file_name = self.NAME+'.pkl'
        file_path = os.path.join(network_dir_path,file_name)
        return file_path

    def save(self):
        """
        Save the current network object to a file using pickle.
        """
        with open(self.file_path, 'wb') as f:
            pickle.dump(self, f)
        self.logger.log(f"Network saved successfully to: {self.file_path}")

    def load(self):
        """
        Load a network object from a file using pickle.
        """
        with open(self.file_path, 'rb') as f:
            loaded_network = pickle.load(f)
        self.__dict__.update(loaded_network.__dict__)
        self.logger.log(f"Network loaded successfully from {self.file_path}")
    
    def _create_data(self):
        """创建新的数据。

        子类必须实现该方法以定义如何生成新的数据。

        Returns:
            数据：新创建的数据。
        """
        self.build()
        self.to_device(self.DEVICE)
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
    def _update_adj(self):
        pass

    @abstractmethod
    def to_device(self,device):
        pass