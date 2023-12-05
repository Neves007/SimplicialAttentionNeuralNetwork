from .util import OptimizerConfig
from .config import Config

class TrainableConfig(Config):
    def __init__(self):
        super().__init__()


    def graph_attention_model(self, NUM_STATES):
        
        self.NAME = "GraphAttentionModel"

        self.NUM_STATES = NUM_STATES
        self.lag = 1
        self.lagstep = 1

        self.optimizer = OptimizerConfig().default()

        self.in_activation = "relu"
        self.gnn_activation = "relu"
        self.out_activation = "relu"

        self.heads = 2
        self.in_channels = [self.NUM_STATES,32, 32]
        self.gnn_channels = 32
        self.out_channels = [32, 32, 32,self.NUM_STATES]
        self.concat = False
        self.bias = True
        self.self_attention = True
        return self

    def simplicial_attention_model(self, NUM_STATES):
        
        self.NAME = "SimplicialAttentionModel"

        self.NUM_STATES = NUM_STATES
        self.lag = 1
        self.lagstep = 1

        self.optimizer = OptimizerConfig().default()

        self.in_activation = "relu"
        self.gnn_activation = "relu"
        self.out_activation = "relu"

        self.heads = 2
        self.in_channels = [self.NUM_STATES,32, 32]
        self.gnn_channels = 32
        self.out_channels = [32, 32, 32, self.NUM_STATES]
        self.concat = False
        self.bias = True
        self.self_attention = True
        return self

    def simplicial_diff_attention_model(self, NUM_STATES):
        
        self.NAME = "SimplicialDiffAttentionModel"

        self.NUM_STATES = NUM_STATES
        self.lag = 1
        self.lagstep = 1

        self.optimizer = OptimizerConfig().default()

        self.in_activation = "relu"
        self.gnn_activation = "relu"
        self.out_activation = "relu"

        self.heads = 2
        self.in_channels = [self.NUM_STATES,32, 32]
        self.gnn_channels = 32
        self.out_channels = [32, 32, 32, self.NUM_STATES]
        self.concat = False
        self.bias = True
        self.self_attention = True
        return self