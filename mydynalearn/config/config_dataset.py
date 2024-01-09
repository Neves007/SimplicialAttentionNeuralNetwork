from .config import Config
import torch
class DatasetConfig(Config):
    def __init__(self):
        super().__init__()

    def dataset(
        self,
    ):
        # 数据集参数
        self.NUM_SAMPLES = 10000
        self.NUM_TEST = 100 # 测试集的时间步
        self.T_INIT = 5 # 重新初始化的时间
        self.IS_WEIGHT = False
        return self
