import numpy as np
import torch
from enum import Enum
from easydict import EasyDict as edict
from math import log, exp
from .config import Config
class NetworkConfig(Config):
    def __init__(self):
        super().__init__()
    @classmethod
    def er(cls, num_nodes=2000):
        cls = cls()
        cls.name = 'er'
        cls.num_nodes = num_nodes
        cls.avg_k = torch.tensor([20])
        return cls
    @classmethod
    def ba(cls, num_nodes=2000):
        cls = cls()
        cls.name = 'ba'
        cls.num_nodes = num_nodes
        cls.avg_k = torch.tensor([20])  # 每次加入2条边的无标度网络
        cls.__BE_eadge_mu = 0.8
        cls.BE_eadge_p = 1 - log(1 - cls.__BE_eadge_mu + exp(1) * cls.__BE_eadge_mu) # 断边概率
        return cls
    @classmethod
    def sc(cls,num_nodes=2000,):
        cls = cls()
        cls.name = 'sc'
        cls.num_nodes = num_nodes
        cls.avg_k = torch.tensor([20, 6])
        cls.maxDimension = cls.avg_k.shape[0]
        return cls


