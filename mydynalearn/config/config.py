
import torch

class Config():
    def __init__(self):
        self.DEVICE = torch.device('cuda')
        # self.DEVICE = torch.device('cpu')