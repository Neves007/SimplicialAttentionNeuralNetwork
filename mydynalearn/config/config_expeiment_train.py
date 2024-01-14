import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os
import random
import torch
import numpy as np

network_config = {
    "ER": NetworkConfig().er(),
    "SCER": NetworkConfig().sc_er(),
    "CONFERENCE": NetworkConfig().real_scnet_conference(),
    "HIGHSCHOOL": NetworkConfig().real_scnet_high_school(),
    "HOSPITAL": NetworkConfig().real_scnet_hospital(),
    "WORKPLACE": NetworkConfig().real_scnet_workplace(),
}

dynamics_config = {
    "UAU": DynamicConfig().UAU(),
    "CompUAU": DynamicConfig().comp_UAU(),
    "SCUAU": DynamicConfig().sc_UAU(),
    "SCCompUAU": DynamicConfig().sc_comp_UAU(),
    "ToySCCompUAU": DynamicConfig().toy_sc_comp_UAU(),
}

dataset_config = DatasetConfig().dataset()


class ExperimentTrainConfig(Config):
    '''
    实验类基类用于初始化参数
    '''

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def set_path(self, root_dir="./output"):
        dataset_dir_path = os.path.join(root_dir, "dataset")
        self.modelparams_dir_path = os.path.join(root_dir, "modelparams", self.NAME)
        self.dataset_dir_path = os.path.join(dataset_dir_path, self.NAME)

        self.make_dir(self.dataset_dir_path)
        self.make_dir(self.modelparams_dir_path)



    def default(
            self,
            NAME,
            network,
            dynamics,
            MODEL_NAME,
            root_dir,
            path_to_best="./",
            path_to_summary="./",
            weight_type="state",
            IS_WEIGHT=False,
            seed=None,
    ):
        self.NAME = NAME
        self.IS_WEIGHT = IS_WEIGHT
        self.seed = seed
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
        self.set_path(root_dir)
        self.network = network_config[network]
        self.dynamics = dynamics_config[dynamics]
        self.dataset = dataset_config
        self.model = ModelConfig(MODEL_NAME, self.dynamics.NUM_STATES)
        return self