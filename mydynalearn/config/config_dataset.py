from .config import Config
class DatasetConfig(Config):
    def __init__(self):
        super().__init__()

    def graph_DynamicDataset(
        self,
    ):
        
        # 数据集分配
        self.NAME = "DynamicDataset"
        self.val_fraction = 0.01
        self.val_bias = 0.8
        # 训练参数
        self.epochs = 30
        self.batch_size = 1
        self.num_networks = 1
        self.num_samples = 10000
        self.num_test = 100 # 测试集的时间步
        self.t_ini = 4 # 重新初始化的时间
        self.maxlag = 1
        self.is_weight = False
        self.resample_when_dead = True
        # first_epoch_checkpoints
        self.check_first_epoch = True
        self.check_first_epoch_max_time = 100
        self.check_first_epoch_timestep = 2
        return self

    # def simplicial_DynamicDataset(
    #     self,
    # ):
    #     
    #     # 数据集分配
    #     self.NAME = "DynamicDatasetSimplicial"
    #     self.val_fraction = 0.01
    #     self.val_bias = 0.8
    #     # 训练参数
    #     self.epochs = 30
    #     self.batch_size = 1
    #     self.num_networks = 1
    #     self.num_samples = 10000
    #     self.num_test = 100 # 测试集的时间步
    #     self.t_ini = 2 # 重新初始化的时间
    #     self.maxlag = 1
    #     self.is_weight = False
    #     self.resample_when_dead = True
    #     # first_epoch_checkpoints
    #     self.check_first_epoch = True
    #     self.check_first_epoch_max_time = 100
    #     self.check_first_epoch_timestep = 2
    #     return self
