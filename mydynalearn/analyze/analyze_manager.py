from .exp_models_analyzer import ExpModelsAnalyzer
from mydynalearn.config import Config
import os
import pandas as pd
import numpy as np
class AnalyzeManager:
    # todo: 添加save和load 
    def __init__(self, exp_generator):
        """
        初始化 AnalyzeManager
        :param exp_generator: 提供实验生成器的管理对象
        """
        config_analyze = Config.get_config_analyze()
        self.config = config_analyze['default']
        self.file_path = self.__get_best_epoch_dataframe_file_path()
        self.exp_generator = exp_generator
        self.best_epoch_dataframe = pd.DataFrame()

    def __get_best_epoch_dataframe_file_path(self):
        root_dir_path = self.config.root_dir_path
        dataframe_dir_name = self.config.dataframe_dir_name
        # 创建文件夹
        dataframe_dir_path = os.path.join(root_dir_path, dataframe_dir_name)
        if not os.path.exists(dataframe_dir_path):
            os.makedirs(dataframe_dir_path)
        # 返回文件路径
        best_epoch_dataframe_file_name = "BestEpochDataframe.csv"
        best_epoch_dataframe_file_path = os.path.join(dataframe_dir_path, best_epoch_dataframe_file_name)
        return best_epoch_dataframe_file_path
    
    def analyze_exp(self, exp):
        """
        对单个实验中的所有epoch的模型进行分析
        :param exp: 实验对象
        """
        print(f"Analyzing exp: {exp.NAME}")
        exp_models_analyzer = ExpModelsAnalyzer(self.config, exp)
        exp_models_analyzer.run()
        print(f"Analysis completed for exp: {exp.NAME}")
        try:
            # 实验模型分析器：用于分析一个实验中的所有epoch的模型
            exp_models_analyzer = ExpModelsAnalyzer(self.config, exp)
            exp_models_analyzer.run()
            best_epoch_exp_item, best_epoch_index = exp_models_analyzer.find_best_epoch()
            self.__add_best_epoch_result_item(best_epoch_exp_item)
            print(f"Analysis completed for exp: {exp.NAME}")
        except Exception as e:
            print(f"Error during analysis of exp {exp.NAME}: {e}")
            raise e
    def __add_best_epoch_result_item(self,best_epoch_exp_item):
        self.best_epoch_dataframe = pd.concat([self.best_epoch_dataframe, best_epoch_exp_item])

    def save_best_epoch_dataframe(self):
        self.best_epoch_dataframe.to_csv(self.file_path, index=False)

    def load_best_epoch_dataframe(self):
        self.best_epoch_dataframe = pd.read_csv(self.file_path)
        return self.best_epoch_dataframe


        
    def run(self):
        """
        对所有实验进行分析
        """
        print("Starting analysis of all experiments...")
        for exp in self.exp_generator:
            self.analyze_exp(exp)
        self.save_best_epoch_dataframe()
        print("Analysis of all experiments completed.")