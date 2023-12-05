import pickle
import torch
from abc import abstractmethod
from mydynalearn.dynamics.simple_dynamic_weight.getter import get as weight_getter
from tqdm import *
class DynamicDataset():
    def __init__(self, config) -> None:
        self.config = config
        self.dataset_config = config.dataset
        self.num_samples = self.dataset_config.num_samples
        self.t_ini = self.dataset_config.t_ini
        self.device = self.dataset_config.device

    def save_dataset(self):
        data = self
        file_name = self.config.datapath_to_datasets+"/dataset.pkl"
        with open(file_name, "wb") as file:
            pickle.dump(data,file)
        file.close()

    def load_dataset(self):
        file_name = self.config.datapath_to_datasets + "/dataset.pkl"
        with open(file_name, "rb") as file:
            info = pickle.load(file)
        file.close()
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
        '''动力学实验
            简介
                - 运行一段连续时间演化的动力学过程
                - 用于验证动力学是否正确
        '''
        self.set_dynamic_info()  # 设置
        self.dynamics.init_net_features()
        print("create dynamics")
        for t in tqdm(range(self.num_samples)):
            self.dynamics._run_onestep()
            result_dict = self.dynamics.get_spread_result()
            self.dynamics.set_features(**result_dict)
            self.save_dnamic_info(t, **result_dict)
    def run(self,network,dynamics):
        '''生成动力学数据
            简介
                - 在t_ini时间后重置初始节点，从而增加传播动力学异质性。
        '''
        # 确认网络和动力学，并保证其是一个维度
        self.network = network
        self.dynamics = dynamics
        self.SimpleDynamicWeight = weight_getter(self.dynamics.NAME)
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION

        # 初始化动力学数据
        print("create dynamics")
        self.set_dynamic_info() # 设置
        self.dynamics.init_net_features(network)

        # 运行动力学
        for t in tqdm(range(self.num_samples)):
            if t % self.t_ini==0:
                self.dynamics.init_net_features(network)  # 在t_ini时间后重置初始节点
            self.dynamics._run_onestep() # 运行一步动力学
            # 获取结果并保存
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