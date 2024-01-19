import os.path
import pickle
import torch
from tqdm import *
from abc import abstractmethod
from mydynalearn.dynamics.simple_dynamic_weight.getter import get as weight_getter
from mydynalearn.networks import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class TestDynamicDataset(Dataset):
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''
    def __init__(self, config) -> None:
        self.config = config
        self.dataset_config = config.dataset
        self.NUM_SAMPLES = self.dataset_config.NUM_SAMPLES
        self.DEVICE = self.config.DEVICE
        self.EFF_BETA_LIST = self.config.EFF_BETA_LIST
        self.stady_rho_list = 1. * torch.ones(len(self.EFF_BETA_LIST))
        self.dataset_file_path = self.dataset_config.dataset_file_path

    def __len__(self) -> int:
        return self.NUM_SAMPLES

    def __getitem__(self, index):
        x0 = self.x0_T[index]
        y_ob = self.y_ob_T[index]
        y_true = self.y_true_T[index]
        return x0, y_ob, y_true

    def is_dataset_exist(self):
        return os.path.exists(self.dataset_file_path)

    def save_dataset(self,*data):
        file_name = self.dataset_file_path
        with open(file_name, "wb") as file:
            pickle.dump(data,file)
        file.close()

    def load_dataset(self):
        file_name = self.dataset_file_path
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        file.close()
        return data


    def set_dataset_network(self):
        network = get_network(self.config)
        network.create_net()
        self.network = network

    def set_dataset_dynamics(self):
        dynamics = get_dynamics(self.config)
        dynamics.set_network(self.network)
        dynamics.init_stateof_network()
        self.dynamics = dynamics

    def init_dataset(self):
        self.set_dataset_network()
        self.set_dataset_dynamics()
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.NUM_SAMPLES
        self.x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)


    def save_onesample_dataset(self, t, old_x0, new_x0, true_tp, **kwargs):
        self.x0_T[t] = old_x0
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp

    def set_beta(self,eff_beta):
        self.dynamics.EFF_AWARE_A1[0] = eff_beta


    def get_stady_rho(self):
        '''动力学实验
            简介
                - 运行一段连续时间演化的动力学过程
                - 用于验证动力学是否正确
        '''

        self.init_dataset()  # 设置
        self.dynamics.init_stateof_network()
        print("create dynamics")
        for t in tqdm(range(self.NUM_SAMPLES)):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self.save_onesample_dataset(t, **onestep_spread_result)

        node_timeEvolution = self.y_ob_T
        stady_nodeState = node_timeEvolution[-50:]
        stady_rho = stady_nodeState.sum(dim=-2)[:,1].mean()
        return stady_rho



    def run(self):
        for index, eff_beta in enumerate(self.EFF_BETA_LIST):
            self.set_beta(eff_beta)
            stady_rho = self.get_stady_rho()
            self.stady_rho_list[index] = stady_rho / self.network.NUM_NODES