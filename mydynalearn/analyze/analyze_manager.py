import pickle

import numpy as np
import pandas as pd
import torch
import os
from mydynalearn.config import AnalyzeConfig
from mydynalearn.analyze.utils.performance_data.getter import get as performance_data_getter
from mydynalearn.analyze.utils.data_handler import DynamicDataHandler
from mydynalearn.analyze.utils.utils import epochdata_datacur_2_dataT
from mydynalearn.logger.logger import *
from mydynalearn.analyze.analyzer import *

class AnalyzeManager():
    def __init__(self,experiment_manager):
        self.config = AnalyzeConfig().analyze_model()
        self.experiment_manager = experiment_manager
        self.r_value_analyzer = RValueAnalyzer(self.config)
    def init_maxR_DF(self):
        df = pd.DataFrame(columns=["network",
                                   "dynamic",
                                   "model",
                                   "maxR_epoch_index",
                                   "maxR_value"])
        return df

    def buid_series(self,R_list,model_exp):
        network_name = model_exp.config.network.NAME
        dynamic_name = model_exp.config.dynamics.NAME
        model_name = model_exp.config.model.NAME
        maxR_index = R_list.argmax().item()
        maxR_value = R_list.max().item()
        info = {"network": network_name,
                "dynamic": dynamic_name,
                "model": model_name,
                "maxR_epoch_index":maxR_index,
                "maxR_value":maxR_value,}
        return pd.Series(info)

    def save_maxR_DF(self,maxR_DF):
        max_R_filepath = self.get_max_R_filepath()
        maxR_DF.to_csv(max_R_filepath,index=False)

    def get_max_R_filepath(self):
        maxR_path = os.path.join(self.config.root_dir, self.config.test_maxR_dir)
        if not os.path.exists(maxR_path):
            os.makedirs(maxR_path)
        file_path = os.path.join(maxR_path, "maxR.csv")
        return file_path

    def analyze_trained_model(self):
        '''
        将所有实验的数据集引入到自己的模型中，输出analyze_result
        '''
        # 把这个testresult
        print("*"*10+" ANALYZE TRAINED MODEL "+"*"*10)
        exp_iter = self.experiment_manager.get_exp_iter()

        if not os.path.exists(self.r_value_analyzer.r_value_dataframe_file_path):
            for exp in exp_iter:
                network, dynamics, train_loader, val_loader, test_loader = exp.create_dataset()
                epoch_tasks = exp.model.epoch_tasks
                EPOCHS = epoch_tasks.EPOCHS
                for model_exp_epoch_index in range(EPOCHS):
                    # 将数据集带入模型执行结果
                    model_executor = runModelOnTestData(self.config,
                                                 network,
                                                 dynamics,
                                                 test_loader,
                                                 model_exp_epoch_index,
                                                 model_exp=exp,
                                                 dataset_exp=exp,
                                                 )
                    analyze_result = model_executor.run()
                    # 添加结果的r值
                    self.r_value_analyzer.add_r_value(analyze_result)
            # 保存结果
            self.r_value_analyzer.save_r_value_dataframe()

        else:
            self.r_value_analyzer.load_r_value_dataframe()
        if not os.path.exists(self.r_value_analyzer.stable_r_value_dataframe_file_path):
            self.r_value_analyzer.analyze_stable_r_value()
        else:
            self.r_value_analyzer.load_stable_r_value_dataframe()

    def run(self):
        '''
        分析训练数据，为画图做准备
        输出：
        '''
        self.analyze_trained_model()


