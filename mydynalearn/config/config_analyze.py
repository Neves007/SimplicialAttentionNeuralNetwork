import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os



class AnalyzeConfig(Config):
    '''
    实验类基类用于初始化参数
    '''


    @classmethod
    def analyze_trained_model_to_realnet(
            cls,
    ):
        cls.NAME = "AnalyzeTrainedModelToRealnet"
        cls.rootpath = r"./output/data/analyze/trained_model_to_realnet/"
        if not os.path.exists(cls.rootpath):
            os.makedirs(cls.rootpath)
        return cls