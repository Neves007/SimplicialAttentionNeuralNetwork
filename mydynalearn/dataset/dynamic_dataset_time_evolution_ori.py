import os.path
import pickle
import torch
from tqdm import *
from abc import abstractmethod
from mydynalearn.networks import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset, Subset
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import *

class DynamicDatasetTimeEvolutionOrigion(TorchLazyLoader):
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''
    def __init__(self, exp, ml_model, network, dynamics) -> None:
        self.config = exp.config
        self.logger = Log("DynamicDatasetTimeEvolution")
        self.exp = exp
        self.ml_model = ml_model
        self.DEVICE = self.config.DEVICE
        self.network = network
        self.dynamics = dynamics
        self.TIME_EVOLUTION_STEPS = self.exp.config.dataset.TIME_EVOLUTION_STEPS
        self.dataset_file_path = self._get_dataset_file_path()
        super().__init__(self.dataset_file_path)
        self._init_dataset()


    def _get_dataset_file_path(self):
        dataset_file_name = f"ORIGION_TIME_EVOLUTION_DATASET_{self.network.NAME}_{self.dynamics.NAME}.pth"
        return os.path.join(self.config.ori_time_evolution_dataset_dir_path, dataset_file_name)

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.TIME_EVOLUTION_STEPS
        self.ori_x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ori_y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ori_y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ori_weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

    def _save_ori_onesample_dataset(self, t, old_x0, new_x0, true_tp, weight, **kwargs):
        self.ori_x0_T[t] = old_x0
        self.ori_y_ob_T[t] = new_x0
        self.ori_y_true_T[t] = true_tp
        self.ori_weight_T[t] = weight

    def _create_data(self):
        '''原动力学模型产生的时间序列数据
        '''
        # 获取动力学数据
        self.logger.increase_indent()
        self.logger.log("Build original time evolution dataset")
        self.network.create_net()  # 构造网络
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
        for t in range(self.TIME_EVOLUTION_STEPS):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self._save_ori_onesample_dataset(t, **onestep_spread_result)
        data = self.ori_x0_T
        self.logger.decrease_indent()
        return data