import os
import random
import torch
import numpy as np
from mydynalearn.config.yaml_config.config import Config



class ConfigDynamicTestingExp:
    def __init__(self,
                 NAME,
                 network,
                 dynamics,
                 root_dir,
                 seed=None,):
        self.NAME = NAME
        self.seed = seed
        self.set_path(root_dir)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        config_network = Config.get_config_network()
        config_dynamic = Config.get_config_dynamic()
        config_dataset = Config.get_config_dataset()

        self.network = config_network[network]
        self.dynamics = config_dynamic[dynamics]
        self.dataset = config_dataset['default']

        self.BETA_LIST = torch.linspace(0,2,11)
        self.ROUND = 10
        self.DEVICE = torch.device('cuda')
        # self.DEVICE = torch.device('cpu')

    def set_path(self, root_dir="./output"):
        dataset_dir_path = os.path.join(root_dir, "test_dynamic_dataset")
        fig_dir_path = os.path.join(root_dir, "fig/test_dynamic")
        self.dataset_dir_path = os.path.join(dataset_dir_path)
        self.fig_dir_path = fig_dir_path
        self.make_dir(self.dataset_dir_path)
        self.make_dir(self.fig_dir_path)

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)