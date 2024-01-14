from .config import Config
import torch
class DynamicConfig(Config):
    def __init__(self):
        super().__init__()

    def UAU(self, NUM_NODES=1000, p=0.004, weights=None):
        
        self.NAME = "UAU"
        self.NUM_STATES = 2
        self.SEED_FREC = 0.1
        self.EFF_AWARE = torch.tensor([8.])
        self.MAX_DIMENSION = 1
        self.RECOVERY = 0.2
        return self

    def comp_UAU(self, NUM_NODES=1000, p=0.004, weights=None):
        
        self.NAME = "CompUAU"
        self.NUM_STATES = 3
        self.SEED_FREC_A1 = 0.1
        self.SEED_FREC_A2 = 0.1
        self.EFF_AWARE_A1 = torch.tensor([8.])
        self.EFF_AWARE_A2 = torch.tensor([8.])
        self.MAX_DIMENSION = 1
        self.RECOVERY = 0.2
        return self

    def sc_UAU(self, NUM_NODES=1000, p=0.004, weights=None):
        
        self.NAME = "SCUAU"
        self.NUM_STATES = 2
        self.SEED_FREC = 0.1
        self.EFF_AWARE = torch.tensor([8., 8])
        self.MAX_DIMENSION = 2
        self.RECOVERY = 0.2
        return self
    

    def sc_comp_UAU(self, NUM_NODES=1000, p=0.004, weights=None):
        
        self.NAME = "SCCompUAU"
        self.NUM_STATES = 3
        self.SEED_FREC_A1 = 0.1
        self.SEED_FREC_A2 = 0.1
        self.EFF_AWARE_A1 = torch.tensor([8., 8])
        self.EFF_AWARE_A2 = torch.tensor([8., 8])
        self.MAX_DIMENSION = 2
        self.RECOVERY = 0.2
        return self

    def toy_sc_comp_UAU(self):
        
        self.NAME = "ToySCCompUAU"
        self.NUM_STATES = 3
        self.SEED_FREC_A1 = 0.1
        self.SEED_FREC_A2 = 0.1
        self.EFF_AWARE_A1 = torch.tensor([8., 8])
        self.EFF_AWARE_A2 = torch.tensor([8., 8])
        self.MAX_DIMENSION = 2
        self.RECOVERY = 0.2
        return self