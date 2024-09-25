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

class DynamicDatasetTimeEvolutionML(TorchLazyLoader):
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
        dataset_file_name = f"ML_EVOLUTION_DATASET_{self.network.NAME}_{self.dynamics.NAME}_{self.exp.model.NAME}_epoch{self.ml_model.epoch_index}.pth"
        return os.path.join(self.config.ml_time_evolution_dataset_dir_path, dataset_file_name)

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.TIME_EVOLUTION_STEPS
        self.ml_x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

    def _save_ml_onesample_dataset(self, t, old_x0, new_x0, true_tp, weight, **kwargs):
        self.ml_x0_T[t] = old_x0
        self.ml_y_ob_T[t] = new_x0
        self.ml_y_true_T[t] = true_tp
        self.ml_weight_T[t] = weight

    def _create_data(self):
        '''机器学习动力学模型产生的时间序列数据
        '''
        # 获取动力学数据
        self.logger.increase_indent()
        self.logger.log("Build machine learning time evolution dataset")
        self.network = self.network.get_data()  # 构造网络
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
        for t in range(self.TIME_EVOLUTION_STEPS):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.spread_result_to_float(onestep_spread_result)
            _, x0, y_pred, y_true, y_ob, w = self.ml_model.batch_task._do_batch_(self.ml_model.attention_model,self.network,self.dynamics, tuple(onestep_spread_result.values()))
            new_x0 = self.dynamics.get_transition_state(y_pred.clone().detach())
            ml_onestep_spread_result = {
                "old_x0": x0.clone().detach(),
                "new_x0": new_x0.clone().detach(),
                "true_tp": y_true.clone().detach(),
                "weight": w.clone().detach()
            }
            self.dynamics.set_features(**ml_onestep_spread_result)
            self._save_ml_onesample_dataset(t, **ml_onestep_spread_result)
        data = self.ml_x0_T
        self.logger.decrease_indent()
        return data
