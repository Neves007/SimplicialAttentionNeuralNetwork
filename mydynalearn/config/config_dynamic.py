from .config import Config
import torch
class DynamicConfig(Config):
    def __init__(self):
        super().__init__()
    @classmethod
    def UAU(cls, NUM_NODES=1000, p=0.004, weights=None):
        cls = cls()
        cls.NAME = "UAU"
        cls.NUM_STATES = 2
        cls.SEED_FREC = 0.2
        cls.EFF_AWARE = torch.tensor([8.])
        cls.MAX_DIMENSION = 1
        cls.RECOVERY = 0.2
        return cls
    @classmethod
    def comp_UAU(cls, NUM_NODES=1000, p=0.004, weights=None):
        cls = cls()
        cls.NAME = "CompUAU"
        cls.NUM_STATES = 3
        cls.SEED_FREC_A1 = 0.2
        cls.SEED_FREC_A2 = 0.2
        cls.EFF_AWARE_A1 = torch.tensor([8.])
        cls.EFF_AWARE_A2 = torch.tensor([8.])
        cls.MAX_DIMENSION = 1
        cls.RECOVERY = 0.2
        return cls
    @classmethod
    def sc_UAU(cls, NUM_NODES=1000, p=0.004, weights=None):
        cls = cls()
        cls.NAME = "SCUAU"
        cls.NUM_STATES = 2
        cls.SEED_FREC = 0.2
        cls.EFF_AWARE = torch.tensor([8., 4])
        cls.MAX_DIMENSION = 2
        cls.RECOVERY = 0.2
        return cls
    
    @classmethod
    def sc_comp_UAU(cls, NUM_NODES=1000, p=0.004, weights=None):
        cls = cls()
        cls.NAME = "SCCompUAU"
        cls.NUM_STATES = 3
        cls.SEED_FREC_A1 = 0.2
        cls.SEED_FREC_A2 = 0.2
        cls.EFF_AWARE_A1 = torch.tensor([8., 4])
        cls.EFF_AWARE_A2 = torch.tensor([8., 4])
        cls.MAX_DIMENSION = 2
        cls.RECOVERY = 0.2
        return cls
    @classmethod
    def toy_sc_comp_UAU(cls):
        cls = cls()
        cls.NAME = "ToySCCompUAU"
        cls.NUM_STATES = 3
        cls.SEED_FREC_A1 = 0.2
        cls.SEED_FREC_A2 = 0.2
        cls.EFF_AWARE_A1 = torch.tensor([8., 4])
        cls.EFF_AWARE_A2 = torch.tensor([8., 4])
        cls.MAX_DIMENSION = 2
        cls.RECOVERY = 0.2
        return cls