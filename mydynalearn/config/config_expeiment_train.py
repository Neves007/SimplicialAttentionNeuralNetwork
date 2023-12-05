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
nn_config = {
    "GAT": TrainableConfig().graph_attention_model,
    "SAT": TrainableConfig().simplicial_attention_model,
    "DiffSAT": TrainableConfig().simplicial_diff_attention_model
}
dataset_config = DatasetConfig().graph_DynamicDataset()


class ExperimentTrainConfig(Config):
    '''
    实验类基类用于初始化参数
    '''

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def set_path(self, rootpath="./output"):
        # path
        self.data_path_1 = os.path.join(rootpath, self.NAME)
        if self.is_weight:
            self.data_path_2 = os.path.join(self.data_path_1, 'is_weight')
        else:
            self.data_path_2 = os.path.join(self.data_path_1, 'not_weight')

        # 模型文件
        self.datapath_to_model = os.path.join(self.data_path_2, "modelResult")
        self.make_dir(self.datapath_to_model)
        # 数据集文件
        self.datapath_to_datasets = os.path.join(self.data_path_1, "datasets")
        self.make_dir(self.datapath_to_datasets)
        self.datapath_to_epochdata = os.path.join(self.datapath_to_model, "epoch_data")
        self.make_dir(self.datapath_to_epochdata)
        self.datapath_to_model_state_dict = os.path.join(self.datapath_to_model, "model_state_dict")
        self.make_dir(self.datapath_to_model_state_dict)


        # # root path
        # self.rootpath_to_fig = os.path.join(rootpath, "fig")
        # # 图片文件
        # ## epoch_performance_fig_ytrure_ypred
        # self.figpath_to_epoch_performance_fig_ytrure_ypred_1 = os.path.join(self.rootpath_to_fig, "epoch_performance_fig_ytrure_ypred",self.NAME)
        # if self.is_weight:
        #     self.figpath_to_epoch_performance_fig_ytrure_ypred_2 = os.path.join(self.figpath_to_epoch_performance_fig_ytrure_ypred_1, 'is_weight')
        # else:
        #     self.figpath_to_epoch_performance_fig_ytrure_ypred_2 = os.path.join(self.figpath_to_epoch_performance_fig_ytrure_ypred_1, 'not_weight')
        #
        # self.make_dir(self.figpath_to_epoch_performance_fig_ytrure_ypred_2)
        # ## maxR
        # self.figpath_to_max_R = os.path.join(self.rootpath_to_fig, "maxR")
        # self.make_dir(self.figpath_to_max_R)



    def default(
            self,
            NAME,
            network,
            dynamics,
            nn_type,
            rootpath,
            path_to_best="./",
            path_to_summary="./",
            weight_type="state",
            is_weight=False,
            seed=None,
    ):
        self.NAME = NAME
        self.is_weight = is_weight
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
        if nn_type not in nn_config:
            raise ValueError(
                f"{nn_type} is invalid, valid entries are {list(nn_config.keys())}"
            )
        self.set_path(rootpath)
        self.network = network_config[network]
        self.dynamics = dynamics_config[dynamics]
        self.dataset = dataset_config
        self.model = nn_config[nn_type](self.dynamics.NUM_STATES)
        return self