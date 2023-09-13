import numpy as np
import pandas as pd
import torch
import os
from mydynalearn.config import AnalyzeConfig
class AnalyzeTrainedModelToRealnet():
    def __init__(self,experiment_manager_train, experiment_manager_realnet):
        self.experiment_manager_train = experiment_manager_train
        self.experiment_manager_realnet = experiment_manager_realnet
        self.config = AnalyzeConfig.analyze_trained_model_to_realnet()
        self.df_file_name = r"analyze_trained_model_to_realnet.csv"
        self.R_df = self.init_DF()

    def init_DF(self):
        file_name = os.path.join(self.config.rootpath, self.df_file_name)
        if os.path.exists(file_name):
            os.remove(file_name)
        df = pd.DataFrame(columns=["RealNet","RealDynamics","ModelNet","ModelDynamics","ModelGnn","ModelIsWeight","R"])
        return df
    def save_DF(self):
        file_name = os.path.join(self.config.rootpath,self.df_file_name)
        self.R_df.to_csv(file_name,index=False)
    def compute_R(self,test_result_curepoch):
        y_pred = torch.cat([data["y_pred"] for data in test_result_curepoch],dim=0)
        y_true = torch.cat([data["y_true"] for data in test_result_curepoch],dim=0)
        y_ob = torch.cat([data["y_ob"] for data in test_result_curepoch],dim=0)

        R_input_y_pred = y_pred[torch.where(y_ob==1)].detach().numpy()
        R_input_y_true = y_true[torch.where(y_ob==1)].detach().numpy()
        R = np.corrcoef(R_input_y_pred,R_input_y_true)[0,1]
        return R

    def buid_series(self,R,train_param_dict,realnet_param_dict):
        info = {"R":R}
        info.update(train_param_dict)
        info.update(realnet_param_dict)
        return pd.Series(info)

    def analyze_model_to_realnet(self,model_exp,real_exp,train_param_dict,realnet_param_dict):
        test_result_curepoch = model_exp.model.get_test_result(0, real_exp.test_loader)
        R = self.compute_R(test_result_curepoch)
        series = self.buid_series(R,train_param_dict,realnet_param_dict)
        self.R_df = self.R_df.append(series,ignore_index=True)

    def apply_trained_model_to_realnet(self):
        train_params = self.experiment_manager_train.set_train_params()
        for train_param in train_params:
            train_param_keys = ["ModelNet","ModelDynamics","ModelGnn","ModelIsWeight"]
            train_param_dict = {k:v for k,v in zip(train_param_keys,train_param)}
            model_exp = self.experiment_manager_train.get_loaded_model_exp(train_param)

            dynamics_params = self.experiment_manager_realnet.get_available_realnet_dynamics(train_param[1])
            realnet_params = self.experiment_manager_realnet.set_realnet_params(dynamics_params=dynamics_params)
            for realnet_param in realnet_params:
                realnet_param_keys = ["RealNet", "RealDynamics"]
                realnet_param_dict = {k: v for k, v in zip(realnet_param_keys, realnet_param)}
                real_exp = self.experiment_manager_realnet.get_loaded_real_exp(realnet_param)
                self.analyze_model_to_realnet(model_exp,real_exp,train_param_dict,realnet_param_dict)
        self.save_DF()


