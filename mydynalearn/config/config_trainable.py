from .util import OptimizerConfig
from .config import Config

class TrainableConfig(Config):
    def __init__(self):
        super().__init__()

    @classmethod
    def graph_attention_model(cls, NUM_STATES):
        cls = cls()
        cls.NAME = "GraphAttentionModel"

        cls.NUM_STATES = NUM_STATES
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.heads = 2
        cls.in_channels = [cls.NUM_STATES,32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32, 32,cls.NUM_STATES]
        cls.concat = False
        cls.bias = True
        cls.self_attention = True
        return cls
    @classmethod
    def simplicial_attention_model(cls, NUM_STATES):
        cls = cls()
        cls.NAME = "SimplicialAttentionModel"

        cls.NUM_STATES = NUM_STATES
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.heads = 2
        cls.in_channels = [cls.NUM_STATES,32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32, 32, cls.NUM_STATES]
        cls.concat = False
        cls.bias = True
        cls.self_attention = True
        return cls
    @classmethod
    def simplicial_diff_attention_model(cls, NUM_STATES):
        cls = cls()
        cls.NAME = "SimplicialDiffAttentionModel"

        cls.NUM_STATES = NUM_STATES
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.heads = 2
        cls.in_channels = [cls.NUM_STATES,32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32, 32, cls.NUM_STATES]
        cls.concat = False
        cls.bias = True
        cls.self_attention = True
        return cls