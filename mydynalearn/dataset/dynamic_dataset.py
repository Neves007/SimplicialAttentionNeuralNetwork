import os.path
import pickle
import torch
from tqdm import *
from abc import abstractmethod
from mydynalearn.networks import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
        self.TIME_EVOLUTION_STEPS = self.dataset_config.TIME_EVOLUTION_STEPS
        self.T_INIT = self.dataset_config.T_INIT
        self.DEVICE = self.config.DEVICE
        self.network = self._get_dataset_network()
        self.dynamics = self._get_dataset_dynamics()
        self.dataset_file_path = self._get_dataset_file_path(self.network.NAME,self.dynamics.NAME)
        self.need_to_run = not os.path.exists(self.dataset_file_path)


    def _get_dataset_file_path(self,network_name,dynamics_name):
        dataset_file_name = f"DATASET_{network_name}_{dynamics_name}.pkl"
        return os.path.join(self.config.dataset_dir_path, dataset_file_name)



    def __len__(self) -> int:
        return self.NUM_SAMPLES
    def __getitem__(self, index):
        x0 = self.x0_T[index]
        y_ob = self.y_ob_T[index]
        y_true = self.y_true_T[index]
        weight = self.weight_T[index]
        return x0, y_ob, y_true, weight

    def _get_dataset_network(self):
        network = get_network(self.config)
        return network

    def _get_dataset_dynamics(self):
        dynamics = get_dynamics(self.config)
        return dynamics



    def is_dataset_exist(self):
        return os.path.exists(self.dataset_file_path)
    def _partition_dataSet(self):
        test_size = self.config.dataset.NUM_TEST
        val_size = int((len(self) - test_size) / 2)
        train_size = len(self) - test_size - val_size
        train_set, val_set, test_set = torch.utils.data.random_split(self, [train_size, val_size, test_size])
        return train_set, val_set, test_set

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.NUM_SAMPLES
        self.x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

    def _save_onesample_dataset(self, t, old_x0, new_x0, true_tp, weight, **kwargs):
        self.x0_T[t] = old_x0
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp
        self.weight_T[t] = weight

    def _buid_dataset(self):
        '''生成动力学数据
            简介
                - 在T_INIT时间后重置初始节点，从而增加传播动力学异质性。
        '''
        # 获取动力学数据
        self.network.create_net()  # 构造网络
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self._init_dataset()
        # 生成数据集
        for t in tqdm(range(self.NUM_SAMPLES)):
            # 动力学初始化
            if t % self.T_INIT == 0:
                self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
            # 生成并存储一个样本数据集
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self._save_onesample_dataset(t, **onestep_spread_result)

    def _save_dataset(self, *data):
        file_name = self.dataset_file_path
        with open(file_name, "wb") as file:
            pickle.dump(data, file)
        file.close()

    def _load_dataset(self):
        file_name = self.dataset_file_path
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        file.close()
        return data

    def _buid_dynamic_process_dataset(self):
        '''动力学实验
            简介
                - 运行一段连续时间演化的动力学过程
                - 用于验证动力学是否正确
        '''
        self._init_dataset()  # 设置
        self.dynamics.init_stateof_network()
        print("create dynamics")
        for t in tqdm(range(self.NUM_SAMPLES)):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self._save_onesample_dataset(t, **onestep_spread_result)



    def run(self):
        if self.is_dataset_exist():
            print("load dataset...")
            network, dynamics, train_set, val_set, test_set = self._load_dataset()

        else:
            print("build dataset...")
            self._buid_dataset()
            train_set, val_set, test_set = self._partition_dataSet()
            network = self.network
            dynamics = self.dynamics
            self._save_dataset(network,
                               dynamics,
                               train_set,
                               val_set,
                               test_set)
        print("dataset.output dataset_file: ",self.dataset_file_path)
        print("The data has been loaded completely!")
        return network, dynamics, train_set, val_set, test_set