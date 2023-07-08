import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os
import random
import numpy as np

network_config = {
    "ba": NetworkConfig.ba(),
    "er": NetworkConfig.er(),
    "sc": NetworkConfig.sc(),
}

dynamics_config = {
    "sis": DynamicConfig.sis(),
    "sir": DynamicConfig.sir(),
    "sis_sc": DynamicConfig.sis_sc(),
}
gnn_config = {
    "sis": TrainableConfig.sis(),
    "sir": TrainableConfig.sis(),
    "sis_sc": TrainableConfig.sis(),
}

class ExperimentConfig(Config):
    '''
    实验类基类用于初始化参数
    '''

    def makeDir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def set_path(self, homePath="./data"):
        # path
        self.homePath_dataset = os.path.join(homePath, self.name)
        if self.is_weight:
            self.homePath = os.path.join(homePath, self.name,'isWeight')
        else:
            self.homePath = os.path.join(homePath, self.name, 'notWeight')

        # 模型文件
        self.path_to_model = os.path.join(self.homePath, "modelResult")
        self.makeDir(self.path_to_model)
        # 数据集文件
        self.path_to_datasets = os.path.join(self.homePath_dataset, "datasets")
        self.makeDir(self.path_to_datasets)
        # 图片文件
        self.path_to_fig = os.path.join(self.homePath, "fig")
        self.makeDir(self.path_to_fig)
        # 模型结果文件
        self.path_to_checkpointsData = os.path.join(self.path_to_model, "checkpointsData")
        self.makeDir(self.path_to_checkpointsData)
        self.path_to_epochData = os.path.join(self.path_to_model, "epochData")
        self.makeDir(self.path_to_epochData)

    @classmethod
    def default(
            cls,
            name,
            dynamics,
            network,
            path_to_best="./",
            path_to_summary="./",
            weight_type="state",
            is_weight=False,
            seed=None,
    ):
        cls = cls()
        if dynamics not in dynamics_config:
            raise ValueError(
                f"{dynamics} is invalid, valid entries are {list(dynamics_config.keys())}"
            )
        if network not in network_config:
            raise ValueError(
                f"{network} is invalid, valid entries are {list(network_config.keys())}"
            )

        cls.name = name
        cls.is_weight = is_weight
        cls.set_path()
        cls.train_details = TrainingConfig.discrete()

        # data
        cls.network = network_config[network]
        cls.dynamics = dynamics_config[dynamics]
        cls.model = gnn_config[dynamics]
        cls.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return cls