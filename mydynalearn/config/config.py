
import torch

class Config():
    def __init__(self):
        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')