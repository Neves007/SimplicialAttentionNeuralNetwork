import yaml
import os
import random
import torch
import numpy as np

config_analyze_file_name = 'config_analyze.yaml'
config_dataset_file_name = 'config_dataset.yaml'
config_drawer_file_name = 'config_drawer.yaml'
config_dynamic_file_name = 'config_dynamic.yaml'
config_model_file_name = 'config_model.yaml'
config_network_file_name = 'config_network.yaml'


with open(config_analyze_file_name, 'r') as file:
    config_analyze = yaml.safe_load(file)
with open(config_dataset_file_name, 'r') as file:
    config_dataset = yaml.safe_load(file)
with open(config_drawer_file_name, 'r') as file:
    config_drawer = yaml.safe_load(file)
with open(config_dynamic_file_name, 'r') as file:
    config_dynamic = yaml.safe_load(file)
with open(config_model_file_name, 'r') as file:
    config_model = yaml.safe_load(file)
with open(config_network_file_name, 'r') as file:
    config_network = yaml.safe_load(file)



class config:
    def __init__(self,
                 NAME,
                 network,
                 dynamics,
                 MODEL_NAME,
                 root_dir,
                 IS_WEIGHT=False,
                 seed=None,):
        self.NAME = NAME
        self.IS_WEIGHT = IS_WEIGHT
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.network = config_network[network]
        self.dynamics = config_dynamic[dynamics]
        self.dataset = config_dataset['default']
        self.model = config_model['default']
        self.DEVICE = torch.device('cpu')

    def set_path(self, root_dir="./output"):
        dataset_dir_path = os.path.join(root_dir, "dataset")
        self.modelparams_dir_path = os.path.join(root_dir, "modelparams", self.NAME)
        self.dataset_dir_path = os.path.join(dataset_dir_path, self.NAME)

        self.make_dir(self.dataset_dir_path)
        self.make_dir(self.modelparams_dir_path)

    def make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)