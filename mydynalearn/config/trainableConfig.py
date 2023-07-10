from .util import OptimizerConfig
from .config import Config

class TrainableConfig(Config):
    def __init__(self):
        super().__init__()

    @classmethod
    def graphAttentionModel(cls):
        cls = cls()
        cls.name = "graphAttentionModel"
        cls.gnn_name = "graphAttention"
        cls.type = "linear"

        cls.num_states = 2
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.heads = 2
        cls.in_channels = [2,32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32,2]
        cls.concat = False
        cls.bias = True
        cls.self_attention = True
        return cls
