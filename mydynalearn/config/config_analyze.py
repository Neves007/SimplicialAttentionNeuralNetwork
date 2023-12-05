import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os



class AnalyzeConfig(Config):
    '''
    实验类基类用于初始化参数
    '''


    def analyze_trained_model_to_realnet(
            self,
    ):
        self.NAME = "AnalyzeTrainedModelToRealnet"
        self.rootpath = r"./output/data/analyze/trained_model_to_realnet/"
        self.test_result_dir = 'test_result'
        return self


    def analyze_trained_model(
            self,
    ):
        self.NAME = "AnalyzeTrainedModel"
        self.rootpath = r"./output/data/analyze/trained_model/"
        self.test_result_dir = 'test_result'
        self.test_maxR_dir = 'maxR'
        return self