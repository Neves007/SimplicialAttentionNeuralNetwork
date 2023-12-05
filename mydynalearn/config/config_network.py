import numpy as np
import torch
from enum import Enum
from easydict import EasyDict as edict
from math import log, exp
from .config import Config
class NetworkConfig(Config):
    def __init__(self):
        super().__init__()

    def er(self, NUM_NODES=1000):
        
        self.NAME = 'ER'
        self.NUM_NODES = NUM_NODES
        self.AVG_K = torch.tensor([20])
        self.MAX_DIMENSION = 1
        return self

    def ba(self, NUM_NODES=1000):
        
        self.NAME = 'ba'
        self.NUM_NODES = NUM_NODES
        self.AVG_K = torch.tensor([20])  # 每次加入2条边的无标度网络
        self.MAX_DIMENSION = 1
        self.__BE_eadge_mu = 0.8
        self.BE_eadge_p = 1 - log(1 - self.__BE_eadge_mu + exp(1) * self.__BE_eadge_mu) # 断边概率
        return self

    def sc_er(self,NUM_NODES=1000,):
        
        self.NAME = 'SCER'
        self.NUM_NODES = NUM_NODES
        self.AVG_K = torch.tensor([40, 15])
        self.MAX_DIMENSION = 2
        return self

    def toy_sc_er(self,NUM_NODES=8,):
        
        self.NAME = 'ToySCER'
        self.NUM_NODES = NUM_NODES
        self.MAX_DIMENSION = 2
        return self


    def sc_ba(self,NUM_NODES=1000,):
        
        self.NAME = 'sc_ba'
        self.NUM_NODES = NUM_NODES
        self.AVG_K = torch.tensor([20, 6])
        self.MAX_DIMENSION = 2
        return self


    def real_scnet_conference(self,NUM_NODES=1000,):
        
        self.NAME = 'CONFERENCE'
        self.REALNET_DATA_PATH = r"mydynalearn/networks/realnet_source"
        self.REALNET_SOURCEDATA_FILENAME = r"conference.dat"
        self.REALNET_NETDATA_FILENAME = r"conference.pkl"
        self.MAX_DIMENSION = 2
        return self

    def real_scnet_high_school(self,NUM_NODES=1000,):
        
        self.NAME = 'HIGHSCHOOL'
        self.REALNET_DATA_PATH = r"mydynalearn/networks/realnet_source"
        self.REALNET_SOURCEDATA_FILENAME = r"high_school.csv"
        self.REALNET_NETDATA_FILENAME = r"high_school.pkl"
        self.MAX_DIMENSION = 2
        return self

    def real_scnet_hospital(self,NUM_NODES=1000,):
        
        self.NAME = 'HOSPITAL'
        self.REALNET_DATA_PATH = r"mydynalearn/networks/realnet_source"
        self.REALNET_SOURCEDATA_FILENAME = r"hospital.dat"
        self.REALNET_NETDATA_FILENAME = r"hospital.pkl"
        self.MAX_DIMENSION = 2
        return self

    def real_scnet_workplace(self,NUM_NODES=1000,):
        
        self.NAME = 'WORKPLACE'
        self.REALNET_DATA_PATH = r"mydynalearn/networks/realnet_source"
        self.REALNET_SOURCEDATA_FILENAME = r"workplace.dat"
        self.REALNET_NETDATA_FILENAME = r"workplace.pkl"
        self.MAX_DIMENSION = 2
        return self