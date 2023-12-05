import numpy as np
import pickle
import pandas as pd
import torch
import os
from mydynalearn.config import AnalyzeConfig
from mydynalearn.drawer_old.utils.data_handler import EpochDataHandler
from mydynalearn.drawer_old.matplot_drawer import DrawerMatplotEpochFigYtrureYpred
from mydynalearn.analyze.utils.performance_data.getter import get as performance_data_getter
from mydynalearn.analyze.utils.data_handler import DynamicDataHandler
from mydynalearn.analyze.utils.utils import epochdata_datacur_2_dataT
class AnalyzeTrainedModelToRealnet():
    def __init__(self,experiment_manager_train, experiment_manager_realnet):
        self.experiment_manager_train = experiment_manager_train
        self.experiment_manager_realnet = experiment_manager_realnet
        self.config = AnalyzeConfig().analyze_trained_model_to_realnet()

        self.df_file_name = r"analyze_trained_model_to_realnet.csv"
        self.R_df = self.init_DF()

    def init_DF(self):
        file_name = os.path.join(self.config.rootpath, self.df_file_name)
        if os.path.exists(file_name):
            os.remove(file_name)
        df = pd.DataFrame(columns=["RealNet","RealDynamics","ModelNet","ModelDynamics","ModelGnn","ModelIsWeight","EpochIndex","R"])
        return df

    def save_DF(self):
        file_name = os.path.join(self.config.rootpath,self.df_file_name)
        self.R_df.to_csv(file_name,index=False)
    def compute_R(self,performance_data):
        R_input_y_pred = torch.cat(performance_data, dim=0).detach().numpy()[:, 0]
        R_input_y_true = torch.cat(performance_data, dim=0).detach().numpy()[:, 1]
        R = np.corrcoef(R_input_y_pred,R_input_y_true)[0,1]
        return R

    def buid_series(self,R,train_param_dict,realnet_param_dict,epoch_index):
        info = {"R":R,
                "EpochIndex":epoch_index}
        info.update(train_param_dict)
        info.update(realnet_param_dict)
        return pd.Series(info)
    def DF_append_R(self,result,train_param_dict,realnet_param_dict,epoch_index):
        R = self.compute_R(result['performance_data'])
        series = self.buid_series(R,train_param_dict,realnet_param_dict,epoch_index)
        self.R_df = self.R_df.append(series,ignore_index=True)

    def put_testdata_in_trained_model(self, model_exp,real_exp, epoch_index):
        test_result_curepoch = model_exp.model.get_test_result(0, real_exp.test_loader)
        kwargs = epochdata_datacur_2_dataT(model_exp,test_result_curepoch)
        dynamic_data_handler = DynamicDataHandler(**kwargs)
        get_performance_index, get_performance_data = performance_data_getter(model_exp.config)
        performance_index = get_performance_index(dynamic_data_handler)
        performance_data = get_performance_data(dynamic_data_handler)
        R = self.compute_R(performance_data)
        result = {
            "model_name": model_exp.NAME,
            "epoch_index": epoch_index,
            "dynamics": model_exp.dynamics,
            "performance_index": performance_index,
            "performance_data": performance_data,
            "w_T": kwargs['w_T'],
            "R": R
        }
        return result
    def save_test_result(self,real_exp,train_param,epoch_index,result):
        file_path = self.get_test_result_filepath(real_exp, train_param, epoch_index)
        with open(file_path,'wb') as file:
            pickle.dump(result,file)
        file.close()
        torch.cuda.empty_cache()

    def load_test_result(self,real_exp,train_param,epoch_index):
        file_path = self.get_test_result_filepath(real_exp, train_param, epoch_index)
        with open(file_path,'rb') as file:
            result = pickle.load(file)
        file.close()
        return result
    def get_test_result_filepath(self,real_exp,train_param,epoch_index):
        test_result_path = os.path.join(self.config.rootpath, self.config.test_result_dir, real_exp.NAME)
        merged_string = '_'.join((str(value) for value in train_param))
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)
        file_path = os.path.join(test_result_path, merged_string+"_epoch{:d}_test_result.pkl".format(epoch_index))
        return file_path

    def analyze_model_to_realnet(self):
        print("*"*10+" ANALYZE MODEL TO REALNET "+"*"*10)
        train_params = self.experiment_manager_train.set_train_params()
        for train_param in train_params:
            epoch_index = self.experiment_manager_train.epochs-1
            model_exp = self.experiment_manager_train.get_loaded_model_exp(train_param,epoch_index)
            dynamics_params = self.experiment_manager_realnet.get_available_realnet_dynamics(train_param[1])
            realnet_params = self.experiment_manager_realnet.set_realnet_params(dynamics_params=dynamics_params)
            for realnet_param in realnet_params:
                real_exp = self.experiment_manager_realnet.get_loaded_real_exp(realnet_param)
                file_path = self.get_test_result_filepath(real_exp,train_param,epoch_index)
                if not os.path.exists(file_path):
                    result = self.put_testdata_in_trained_model(model_exp,
                                                                real_exp,
                                                                epoch_index)
                    self.save_test_result(real_exp,train_param,epoch_index,result)
                torch.cuda.empty_cache()
        print("PROCESS COMPLETED!\n\n")

    def analyze_R(self):
        print("*"*10+"ANALYZE R "+"*"*10)
        train_params = self.experiment_manager_train.set_train_params()
        for train_param in train_params:
            epoch_index = self.experiment_manager_train.epochs-1
            train_param_keys = ["ModelNet", "ModelDynamics", "ModelGnn", "ModelIsWeight"]
            train_param_dict = {k: v for k, v in zip(train_param_keys, train_param)}
            dynamics_params = self.experiment_manager_realnet.get_available_realnet_dynamics(train_param[1])
            realnet_params = self.experiment_manager_realnet.set_realnet_params(dynamics_params=dynamics_params)
            for realnet_param in realnet_params:
                realnet_param_keys = ["RealNet", "RealDynamics"]
                realnet_param_dict = {k: v for k, v in zip(realnet_param_keys, realnet_param)}
                real_exp = self.experiment_manager_realnet.get_loaded_real_exp(realnet_param)
                file_path = self.get_test_result_filepath(real_exp, train_param, epoch_index)
                assert os.path.exists(file_path)
                result = self.load_test_result(real_exp,train_param,epoch_index)
                self.DF_append_R(result, train_param_dict, realnet_param_dict,epoch_index)
                torch.cuda.empty_cache()
        self.save_DF()
        print("PROCESS COMPLETED!\n\n")

    def draw_performance(self,test_result_curepoch, real_exp):
        epoch_data_handler = EpochDataHandler(real_exp.config, real_exp.dynamics)
        matplot_epochPerformance = DrawerMatplotEpochFigYtrureYpred(real_exp.config, real_exp.dynamics)

        kwargs = epoch_data_handler.epochdata_datacur_2_dataT(test_result_curepoch)
        matplot_epochPerformance.draw(**kwargs)

    def run(self):
        '''
        将训练好得模型应用于真实网络数据
        '''
        self.analyze_model_to_realnet()
        self.analyze_R()

