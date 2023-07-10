from .config import Config
class TrainingConfig(Config):
    def __init__(self):
        super().__init__()
    @classmethod
    def discrete(
        cls,
    ):
        cls = cls()
        # 数据集分配
        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        # 训练参数
        cls.epochs = 30
        cls.batch_size = 1
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.num_test = 100 # 测试集的时间步
        cls.resampling = 2 # 重新初始化的时间
        cls.maxlag = 1
        cls.is_weight = False
        cls.resample_when_dead = True
        # firstEpochCheckpoints
        cls.checkFirstEpoch = True
        cls.checkFirstEpoch_max_time = 100
        cls.checkFirstEpoch_timestep = 2
        return cls
