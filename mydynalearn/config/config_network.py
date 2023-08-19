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
    def er(cls, NUM_NODES=1000):
        cls = cls()
        cls.NAME = 'ER'
        cls.NUM_NODES = NUM_NODES
        cls.AVG_K = torch.tensor([20])
        cls.MAX_DIMENSION = 1
        return cls
    @classmethod
    def ba(cls, NUM_NODES=1000):
        cls = cls()
        cls.NAME = 'ba'
        cls.NUM_NODES = NUM_NODES
        cls.AVG_K = torch.tensor([20])  # 每次加入2条边的无标度网络
        cls.MAX_DIMENSION = 1
        cls.__BE_eadge_mu = 0.8
        cls.BE_eadge_p = 1 - log(1 - cls.__BE_eadge_mu + exp(1) * cls.__BE_eadge_mu) # 断边概率
        return cls
    @classmethod
    def sc_er(cls,NUM_NODES=1000,):
        cls = cls()
        cls.NAME = 'SCER'
        cls.NUM_NODES = NUM_NODES
        cls.AVG_K = torch.tensor([20, 6])
        cls.MAX_DIMENSION = 2
        return cls
    @classmethod
    def toy_sc_er(cls,NUM_NODES=8,):
        cls = cls()
        cls.NAME = 'ToySCER'
        cls.NUM_NODES = NUM_NODES
        cls.MAX_DIMENSION = 2
        return cls

    @classmethod
    def sc_ba(cls,NUM_NODES=1000,):
        cls = cls()
        cls.NAME = 'sc_ba'
        cls.NUM_NODES = NUM_NODES
        cls.AVG_K = torch.tensor([20, 6])
        cls.MAX_DIMENSION = 2
        return cls
