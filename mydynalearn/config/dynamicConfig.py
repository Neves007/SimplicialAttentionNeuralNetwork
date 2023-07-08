from .config import Config
import torch
class DynamicConfig(Config):
    def __init__(self):
        super().__init__()
    @classmethod
    def sis(cls, num_nodes=1000, p=0.004, weights=None):
        cls = cls()
        cls.name = "sis"
        cls.num_state = 2
        cls.initSeedFraction = 0.1
        cls.eff_infection = torch.tensor([8])
        cls.maxDimension = cls.eff_infection.shape[0]
        cls.recovery = 0.2
        return cls

    @classmethod
    def sis_sc(cls, num_nodes=1000, p=0.004, weights=None):
        cls = cls()
        cls.name = "sis_sc"
        cls.num_state = 2
        cls.initSeedFraction = 0.1
        cls.eff_infection = torch.tensor([8,2])
        cls.maxDimension = cls.eff_infection.shape[0]
        cls.recovery = 0.2
        return cls

    @classmethod
    def sir(cls, num_nodes=1000, p=0.004, weights=None):
        cls = cls()
        cls.name = "sir"
        cls.num_state = 3
        cls.initSeedFraction = 0.1
        cls.eff_infection = torch.tensor([2])
        cls.maxDimension = cls.eff_infection.shape[0]
        cls.recovery = 0.2
        return cls