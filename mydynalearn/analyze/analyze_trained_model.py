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
class AnalyzeTrainedModel():
    def __init__(self,experiment_manager_train):
        self.experiment_manager_train = experiment_manager_train
        self.config = AnalyzeConfig().analyze_trained_model()

    def init_maxR_DF(self):
        df = pd.DataFrame(columns=["ModelName","maxR_epoch_index","maxR_value"])
        return df

    def buid_series(self,R_list,model_exp):
        model_name = model_exp.NAME
        maxR_index = R_list.argmax().item()
        maxR_value = R_list.max().item()
        info = {"ModelName":model_name,
                "maxR_epoch_index":maxR_index,
                "maxR_value":maxR_value,}
        return pd.Series(info)

    def save_maxR_DF(self,maxR_DF):
        max_R_filepath = self.get_max_R_filepath()
        maxR_DF.to_csv(max_R_filepath,index=False)
    def compute_R(self,test_result_curepoch):
        y_pred = torch.cat([data["y_pred"] for data in test_result_curepoch],dim=0)
        y_true = torch.cat([data["y_true"] for data in test_result_curepoch],dim=0)
        y_ob = torch.cat([data["y_ob"] for data in test_result_curepoch],dim=0)

        R_input_y_pred = y_pred[torch.where(y_ob==1)].detach().numpy()
        R_input_y_true = y_true[torch.where(y_ob==1)].detach().numpy()
        R = np.corrcoef(R_input_y_pred,R_input_y_true)[0,1]
        return R



    def put_testdata_in_trained_model(self, model_exp, epoch_index):
        test_result_curepoch = model_exp.model.get_test_result(epoch_index, model_exp.test_loader)
        kwargs = epochdata_datacur_2_dataT(model_exp,test_result_curepoch)
        dynamic_data_handler = DynamicDataHandler(**kwargs)
        get_performance_index, get_performance_data = performance_data_getter(model_exp.config)
        performance_index = get_performance_index(dynamic_data_handler)
        performance_data = get_performance_data(dynamic_data_handler)
        R = self.compute_R(test_result_curepoch)
        result = {
            "model_name": model_exp.NAME,
            "epoch_index": epoch_index,
            "dynamics": model_exp.dataset.dynamics,
            "performance_index": performance_index,
            "performance_data": performance_data,
            "w_T": kwargs['w_T'],
            "R": R
        }
        return result
    def get_test_result_filepath(self,model_exp,epoch_index):
        test_result_path = os.path.join(self.config.rootpath, self.config.test_result_dir, model_exp.NAME)
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)
        file_path = os.path.join(test_result_path, "epoch{:d}_test_result.pkl".format(epoch_index))
        return file_path
    def get_max_R_filepath(self):
        maxR_path = os.path.join(self.config.rootpath, self.config.test_maxR_dir)
        if not os.path.exists(maxR_path):
            os.makedirs(maxR_path)
        file_path = os.path.join(maxR_path, "maxR.csv")
        return file_path

    def save_test_result(self,model_exp,epoch_index,result):
        file_path = self.get_test_result_filepath(model_exp,epoch_index)
        with open(file_path,'wb') as file:
            pickle.dump(result,file)
        file.close()
        torch.cuda.empty_cache()

    def load_test_result(self,model_exp,epoch_index):
        file_path = self.get_test_result_filepath(model_exp,epoch_index)
        with open(file_path,'rb') as file:
            result = pickle.load(file)
        file.close()
        return result

    def analyze_trained_model(self):
        print("*"*10+" ANALYZE TRAINED MODEL "+"*"*10)
        train_params = self.experiment_manager_train.get_train_params()
        for train_param in train_params:
            log_analyze_trained_model(train_param)
            train_param_keys = ["ModelNet","ModelDynamics","ModelGnn","ModelIsWeight"]
            train_param_dict = {k:v for k,v in zip(train_param_keys,train_param)}
            for epoch_index in range(self.experiment_manager_train.EPOCHS):
                # 获取模型路径，查看模型保存文件是否存在
                model_exp = self.experiment_manager_train.get_train_exp(*train_param)
                file_path = self.get_test_result_filepath(model_exp, epoch_index)
                if not os.path.exists(file_path):
                    model_exp = self.experiment_manager_train.get_loaded_model_exp(train_param, epoch_index)
                    result = self.put_testdata_in_trained_model(model_exp, epoch_index)
                    self.save_test_result(model_exp,epoch_index,result)
                torch.cuda.empty_cache()
        print("PROCESS COMPLETED!\n\n")

    def analyze_maxR(self):
        print("*"*10+" ANALYZE MAX R "+"*"*10)
        train_params = self.experiment_manager_train.get_train_params()
        maxR_DF = self.init_maxR_DF()
        for train_param in train_params:
            R_list = torch.zeros(self.experiment_manager_train.EPOCHS,dtype=torch.float)
            for epoch_index in range(self.experiment_manager_train.EPOCHS):
                model_exp = self.experiment_manager_train.get_loaded_model_exp(train_param, epoch_index)
                test_result = self.load_test_result(model_exp,epoch_index)
                R = test_result['R']
                R_list[epoch_index] = R
            maxR_series = self.buid_series(R_list,model_exp)
            maxR_DF = maxR_DF.append(maxR_series,ignore_index=True)
        self.save_maxR_DF(maxR_DF)
        print("table of max R saved!")
        print("PROCESS COMPLETED!\n\n")

    def run(self):
        '''
        分析训练数据，为画图做准备
        输出：
        '''
        self.analyze_trained_model()
        self.analyze_maxR()



