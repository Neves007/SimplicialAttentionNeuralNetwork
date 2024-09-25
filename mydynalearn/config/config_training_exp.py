import yaml
import os
import random
import torch
import numpy as np
from mydynalearn.config.yaml_config.config import Config


class ConfigTrainingExp:
    def __init__(self,
                 NAME,
                 network,
                 dynamics,
                 root_dir,
                 IS_WEIGHT=False,
                 seed=None,
                 ):
        self.NAME = NAME
        self.IS_WEIGHT = IS_WEIGHT
        self.seed = seed
        self.set_path(root_dir)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        config_network = Config.get_config_network()
        config_dynamic = Config.get_config_dynamic()
        config_dataset = Config.get_config_dataset()
        config_optimizer = Config.get_config_optimizer()
        config_model = Config.get_config_model()


        self.network = config_network[network]
        self.dynamics = config_dynamic[dynamics]
        self.dataset = config_dataset['default']
        self.optimizer = config_optimizer['radam']
        self.model = config_model['default']

        self.DEVICE = torch.device('cpu')

    def set_path(self, root_dir="./output"):
        self.network_dir_path = os.path.join(root_dir, "network")
        self.dataset_dir_path = os.path.join(root_dir, "dataset")
        self.time_evolution_dataset_dir_path = os.path.join(root_dir, "time_evolution_dataset")
        self.ori_time_evolution_dataset_dir_path = os.path.join(self.time_evolution_dataset_dir_path, "ori")
        self.ml_time_evolution_dataset_dir_path = os.path.join(self.time_evolution_dataset_dir_path, "ml")
        self.modelparams_dir_path = os.path.join(root_dir, "modelparams", self.NAME)

        self.make_dir(self.network_dir_path)
        self.make_dir(self.dataset_dir_path)
        self.make_dir(self.time_evolution_dataset_dir_path)
        self.make_dir(self.ori_time_evolution_dataset_dir_path)
        self.make_dir(self.ml_time_evolution_dataset_dir_path)
        self.make_dir(self.modelparams_dir_path)

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)