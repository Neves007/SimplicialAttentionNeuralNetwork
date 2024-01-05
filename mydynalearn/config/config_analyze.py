import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os



class AnalyzeConfig(Config):
    '''
    实验类基类用于初始化参数
    '''


    def analyze_model(
            self,
    ):
        self.NAME = "AnalyzeModel"
        self.root_dir_path = r"./output/data/analyze/"
        self.analyze_result_dir_name = 'analyze_result'
        self.r_value_dir_name = 'r_value'
        self.r_value_dataframe_file_name = "r_value_dataframe.csv"
        self.stable_r_value_dataframe_file_name = "stable_r_value_dataframe.csv"
        return self