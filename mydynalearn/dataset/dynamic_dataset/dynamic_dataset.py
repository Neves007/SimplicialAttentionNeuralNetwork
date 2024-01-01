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
class DynamicDataset(Dataset):
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
        self.T_INIT = self.dataset_config.T_INIT
        self.DEVICE = self.dataset_config.DEVICE
        self.dataset_file = self.config.path_to_datasets + "/dataset.pkl"
    def __len__(self) -> int:
        return self.NUM_SAMPLES
    def __getitem__(self, index):
        x0 = self.x0_T[index]
        y_ob = self.y_ob_T[index]
        y_true = self.y_true_T[index]
        weight = self.weight_T[index]
        return x0, y_ob, y_true, weight

    def is_dataset_exist(self):
        return os.path.exists(self.dataset_file)

    def save_dataset(self):
        data = self
        file_name = self.dataset_file
        with open(file_name, "wb") as file:
            pickle.dump(data,file)
        file.close()

    def load_dataset(self):
        file_name = self.dataset_file
        with open(file_name, "rb") as file:
            info = pickle.load(file)
        file.close()
        self.network = info.network
        self.dynamics = info.dynamics
        self.x0_T = info.x0_T
        self.y_ob_T = info.y_ob_T
        self.y_true_T = info.y_true_T
        self.weight_T = info.weight_T


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
        self.weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

    def save_onesample_dataset(self, t, old_x0, old_x1, new_x0, true_tp, weight, **kwargs):
        self.x0_T[t] = old_x0
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp
        self.weight_T[t] = weight

    def buid_dataset(self):
        '''生成动力学数据
            简介
                - 在T_INIT时间后重置初始节点，从而增加传播动力学异质性。
        '''
        # 获取动力学数据
        t = 0
        self.init_dataset()
        # 生成数据集
        for sample in range(self.NUM_SAMPLES):
            # 动力学初始化
            if t % self.T_INIT == 0:
                self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
            # 生成并存储一个样本数据集
            onestep_spread_result = self.dynamics._run_onestep()
            self.save_onesample_dataset(t, **onestep_spread_result)
            t += 1

    def run_dynamic_process(self):
        '''动力学实验
            简介
                - 运行一段连续时间演化的动力学过程
                - 用于验证动力学是否正确
        '''
        self.init_dataset()  # 设置
        self.dynamics.init_stateof_network()
        print("create dynamics")
        for t in tqdm(range(self.NUM_SAMPLES)):
            self.dynamics._run_onestep()
            result_dict = self.dynamics.get_spread_result()
            self.dynamics.set_features(**result_dict)
            self.save_onesample_dataset(t, **result_dict)

    def print_log(self,num_indentation=0):
        num_indentation += 1
        indentation = num_indentation*"\t"
        print(indentation+"network info:")
        self.network.print_log(num_indentation)
        print(indentation+"dynamics info:")
        self.dynamics.print_log(num_indentation)

    def run(self):
        if self.is_dataset_exist():
            print("load dataset...")
            self.load_dataset()
        else:
            print("build dataset...")
            self.buid_dataset()
            self.save_dataset()
        self.print_log()
        print("The data has been loaded completely!")