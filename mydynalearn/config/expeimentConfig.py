import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os
import random
import numpy as np

graph_network_config = {
    "ba": NetworkConfig.ba(),
    "er": NetworkConfig.er(),
}

graph_dynamics_config = {
    "sis": DynamicConfig.sis(),
    "sir": DynamicConfig.sir(),
}
graph_gnn_config = TrainableConfig.graphAttentionModel()

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
    def set_graphConfig(self,network,dynamics):
        if dynamics not in graph_dynamics_config:
            raise ValueError(
                f"{dynamics} is invalid, valid entries are {list(graph_dynamics_config.keys())}"
            )
        if network not in graph_network_config:
            raise ValueError(
                f"{network} is invalid, valid entries are {list(graph_network_config.keys())}"
            )
        self.network = graph_network_config[network]
        self.dynamics = graph_dynamics_config[dynamics]
        self.model = graph_gnn_config
    @classmethod
    def default(
            cls,
            name,
            dynamics,
            network,
            topology,
            path_to_best="./",
            path_to_summary="./",
            weight_type="state",
            is_weight=False,
            seed=None,
    ):
        cls = cls()

        cls.name = name
        cls.is_weight = is_weight
        cls.set_path()
        cls.train_details = TrainingConfig.discrete()
        cls.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # data
        if topology=='graph':
            cls.set_graphConfig(network,dynamics)

        return cls