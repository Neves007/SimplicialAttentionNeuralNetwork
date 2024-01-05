from .util import OptimizerConfig
from .config import Config

class ModelConfig(Config):
    def __init__(self, MODEL_NAME, NUM_STATES):
        super().__init__()
        self.NAME = MODEL_NAME
        # 训练参数
        self.NUM_STATES = NUM_STATES
        self.EPOCHS = 30
        self.batch_size = 1
        self.optimizer = OptimizerConfig().default()
        # 神经网络
        self.heads = 2
        self.in_channels = [self.NUM_STATES, 32, 32]
        self.gnn_channels = 32
        self.out_channels = [32, 32, 32, self.NUM_STATES]
        self.concat = False
        self.bias = True
        self.self_attention = True