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
    "ToySCER": NetworkConfig.toy_sc_er(),
}

dynamics_config = {
    "UAU": DynamicConfig.UAU(),
    "CompUAU": DynamicConfig.comp_UAU(),
    "SCUAU": DynamicConfig.sc_UAU(),
    "SCCompUAU": DynamicConfig.sc_comp_UAU(),
}
nn_config = {
    "graph": TrainableConfig.graphAttentionModel,
    "simplicial": TrainableConfig.simplicialAttentionModel
}
dataset_config = DatasetConfig.graph_DynamicDataset()


class ExperimentConfig(Config):
    '''
    实验类基类用于初始化参数
    '''

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def set_path(self, homepath="./data"):
        # path
        self.homepath_dataset = os.path.join(homepath, self.NAME)
        if self.is_weight:
            self.homepath = os.path.join(homepath, self.NAME, 'is_weight')
        else:
            self.homepath = os.path.join(homepath, self.NAME, 'not_weight')

        # 模型文件
        self.path_to_model = os.path.join(self.homepath, "modelResult")
        self.make_dir(self.path_to_model)
        # 数据集文件
        self.path_to_datasets = os.path.join(self.homepath_dataset, "datasets")
        self.make_dir(self.path_to_datasets)
        # 图片文件
        self.path_to_fig = os.path.join(self.homepath, "fig")
        self.make_dir(self.path_to_fig)
        # 模型结果文件
        self.path_to_checkpoints_data = os.path.join(self.path_to_model, "checkpoints_data")
        self.make_dir(self.path_to_checkpoints_data)
        self.path_to_epochdata = os.path.join(self.path_to_model, "epoch_data")
        self.make_dir(self.path_to_epochdata)

    @classmethod
    def default(
            cls,
            NAME,
            network,
            dynamics,
            dataset,
            nn_type,
            path_to_best="./",
            path_to_summary="./",
            weight_type="state",
            is_weight=False,
            seed=None,
    ):
        cls = cls()

        cls.NAME = NAME
        cls.topology = nn_type
        cls.is_weight = is_weight
        cls.set_path()
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
        if nn_type not in nn_config:
            raise ValueError(
                f"{nn_type} is invalid, valid entries are {list(nn_config.keys())}"
            )

        cls.network = network_config[network]
        cls.dynamics = dynamics_config[dynamics]
        cls.dataset = dataset_config
        cls.model = nn_config[nn_type](cls.dynamics.NUM_STATES)
        return cls