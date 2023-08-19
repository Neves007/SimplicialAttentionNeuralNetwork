import pickle
import torch
from abc import abstractmethod
from mydynalearn.dynamics.simple_dynamic_weight.getter import get as weight_getter

class DynamicDataset():
    def __init__(self, config,network,dynamics) -> None:
        self.config = config
        self.dataset_config = config.dataset
        self.num_samples = self.dataset_config.num_samples
        self.resampling = 2
        self.device = self.dataset_config.device
        self.network = network
        self.dynamics = dynamics
        self.SimpleDynamicWeight = weight_getter(self.dynamics.NAME)
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION

    def save_dataset(self):
        data = self
        file_name = self.config.path_to_datasets+"/dataset.pkl"
        with open(file_name, "wb") as file:
            pickle.dump(data,file)

    def load_dataset(self):
        file_name = self.config.path_to_datasets + "/dataset.pkl"
        with open(file_name, "rb") as file:
            info = pickle.load(file)
        return info

    def split_dataset(self, num_test):

        num_train = int((self.num_samples-num_test)/2)
        num_val = num_train
        sample_index = torch.randperm(self.num_samples)
        start = 0
        end = num_train
        train_index = sample_index[start:end]
        start = end
        end += num_val
        val_index = sample_index[start:end]
        start = end
        end += num_test
        test_index = sample_index[start:end]
        # 获取数据集
        train_set = self.get_dataset_from_index(train_index)
        val_set = self.get_dataset_from_index(val_index)
        test_set = self.get_dataset_from_index(test_index)
        return train_set, val_set, test_set
    def run_dynamic_process(self):
        self.set_dynamic_info()  # 设置
        self.dynamics.init_net_features()
        for t in range(self.num_samples):
            self.dynamics._run_onestep()
            result_dict = self.dynamics.get_spread_result()
            self.dynamics.set_features(**result_dict)
            self.save_dnamic_info(t, **result_dict)
    def run(self):
        self.set_dynamic_info() # 设置
        for t in range(self.num_samples):
            self.dynamics.init_net_features()  # 初始化单纯型状态
            self.dynamics._run_onestep()
            result_dict = self.dynamics.get_spread_result()
            self.dynamics.set_features(**result_dict)
            self.save_dnamic_info(t, **result_dict)

    @abstractmethod
    def save_dnamic_info(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_dynamic_info(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_dataset_from_index(self, *args, **kwargs):
        pass