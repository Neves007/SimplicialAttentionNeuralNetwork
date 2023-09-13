import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os
import random
import torch
import numpy as np

network_config = {
    "ER": NetworkConfig.er(),
    "SCER": NetworkConfig.sc_er(),
    "CONFERENCE": NetworkConfig.real_scnet_conference(),
    "HIGHSCHOOL": NetworkConfig.real_scnet_high_school(),
    "HOSPITAL": NetworkConfig.real_scnet_hospital(),
    "WORKPLACE": NetworkConfig.real_scnet_workplace(),
}

dynamics_config = {
    "UAU": DynamicConfig.UAU(),
    "CompUAU": DynamicConfig.comp_UAU(),
    "SCUAU": DynamicConfig.sc_UAU(),
    "SCCompUAU": DynamicConfig.sc_comp_UAU(),
    "ToySCCompUAU": DynamicConfig.toy_sc_comp_UAU(),
}
nn_config = {
    "GAT": TrainableConfig.graph_attention_model,
    "SAT": TrainableConfig.simplicial_attention_model,
    "DiffSAT": TrainableConfig.simplicial_diff_attention_model
}
dataset_config = DatasetConfig.graph_DynamicDataset()


class ExperimentRealConfig(Config):
    '''
    实验类基类用于初始化参数
    '''

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def set_path(self, rootpath="./output"):
        # path
        self.data_path_1 = os.path.join(rootpath, self.NAME)

        # 数据集文件
        self.datapath_to_datasets = os.path.join(self.data_path_1, "datasets")
        self.make_dir(self.datapath_to_datasets)

    @classmethod
    def default(
            cls,
            NAME,
            network,
            dynamics,
            rootpath,
            seed=None,
            **kwargs
    ):
        print("network: ", network)
        print("dynamics: ", dynamics)
        cls = cls()

        cls.NAME = NAME
        cls.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # data
        if dynamics not in dynamics_config:
            raise ValueError(
                f"{dynamics} is invalid, valid entries are {list(dynamics_config.keys())}"
            )
        if network not in network_config:
            raise ValueError(
                f"{network} is invalid, valid entries are {list(network_config.keys())}"
            )
        cls.set_path(rootpath)
        cls.network = network_config[network]
        cls.dynamics = dynamics_config[dynamics]
        cls.dataset = dataset_config
        return cls