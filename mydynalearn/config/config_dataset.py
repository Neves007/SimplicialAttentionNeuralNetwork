from .config import Config
import torch
class DatasetConfig(Config):
    def __init__(self):
        super().__init__()

    def dataset(
        self,
    ):
        # 数据集分配
        self.FOR_REALNET = False
        # 网络
        self.AVG_K_MIN = 4
        self.AVG_K_MAX = 20
        self.NUM_K = 5
        self.AVG_K_LIST = torch.linspace(self.AVG_K_MIN,self.AVG_K_MAX,self.NUM_K)
        # 数据集参数
        self.NUM_SAMPLES = 10000
        self.NUM_TEST = 100 # 测试集的时间步
        self.T_INIT = 4 # 重新初始化的时间
        self.IS_WEIGHT = False
        return self