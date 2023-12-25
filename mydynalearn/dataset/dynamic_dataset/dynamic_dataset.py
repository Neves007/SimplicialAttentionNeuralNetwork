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
    def __init__(self, config) -> None:
        self.config = config
        self.dataset_config = config.dataset
        self.NUM_SAMPLES = self.dataset_config.NUM_SAMPLES
        if self.dataset_config.FOR_REALNET is False:
            # 获取网络列表
            network_manager = NetworkManager(config)
            networks_generator = network_manager.networks_generator()
            self.networks = [net for net in networks_generator]

        else:
            real_network = get_network(self.config)
            real_network.create_net()
            self.networks = [real_network]
        self.dynamics = get_dynamics(config)
        self.T_INIT = self.dataset_config.T_INIT
        self.DEVICE = self.dataset_config.DEVICE
    def __len__(self) -> int:
        return self.NUM_SAMPLES
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

    def run_dynamic_process(self):
        '''动力学实验
            简介
                - 运行一段连续时间演化的动力学过程
                - 用于验证动力学是否正确
        '''
        self.init_dataset()  # 设置
        self.dynamics.preparation_before_dynamics()
        print("create dynamics")
        for t in tqdm(range(self.NUM_SAMPLES)):
            self.dynamics._run_onestep()
            result_dict = self.dynamics.get_spread_result()
            self.dynamics.set_features(**result_dict)
            self.save_onesample_dataset(t, **result_dict)

    def run(self):
        '''生成动力学数据
            简介
                - 在T_INIT时间后重置初始节点，从而增加传播动力学异质性。
        '''
        # 获取动力学数据
        t = 0
        NUM_SAMPLES_for_a_network = self.NUM_SAMPLES // len(self.networks)
        self.init_dataset()  # 设置
        for net_index,network in enumerate(self.networks):
            # 初始化动力学数据
            assert network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
            print("start to create dynamic data...")
            network.show_info()
            self.dynamics.show_info()
            # 动力学初始化
            self.dynamics.preparation_before_dynamics(network)
            # 生成数据集
            for sample in range(NUM_SAMPLES_for_a_network):
                # 动力学初始化
                if t % self.T_INIT == 0:
                    self.dynamics.preparation_before_dynamics(network)  # 在T_INIT时间后重置初始节点
                # 生成并存储一个样本数据集
                result_dict = self.get_onesample_dataset()
                self.save_onesample_dataset(t, net_index, **result_dict)
                t += 1
            print("Finish!\n")

    def get_onesample_dataset(self) -> 'result_dict for one sample of dataset':
        # 运行一步动力学
        self.dynamics._run_onestep()
        # 获取结果并保存
        result_dict = self.dynamics.get_spread_result()
        self.dynamics.set_features(**result_dict)
        return result_dict

    def save_onesample_dataset(self, t, net_index, old_x0, old_x1, new_x0, true_tp, weight, **kwargs):
        self.networkid_T[t] = net_index
        self.x0_T[t] = old_x0
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp
        self.weight_T[t] = weight

    def init_dataset(self):
        NUM_NODES = self.networks[0].NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.NUM_SAMPLES
        self.networkid_T = torch.zeros(NUM_SAMPLES).to(self.config.DEVICE, dtype=torch.int8)
        self.x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

    def __getitem__(self, index):
        networkid = self.networkid_T[index]
        # item list
        x0 = self.x0_T[index]
        y_ob = self.y_ob_T[index]
        y_true = self.y_true_T[index]
        weight = self.weight_T[index]
        return x0, y_ob, y_true, weight, networkid