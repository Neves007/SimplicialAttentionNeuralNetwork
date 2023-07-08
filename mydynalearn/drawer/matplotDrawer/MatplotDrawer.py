import torch
from .fig import *
from mydynalearn.transformer import data_curEpoch_2_data_T

class MatplotDrawer:
    def __init__(self,config):
        self.path_to_fig = config.path_to_fig
        self.path_to_epochData = config.path_to_epochData
        self.figName = None

