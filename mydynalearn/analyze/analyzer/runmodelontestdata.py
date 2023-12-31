import os
import pickle

import torch
import numpy as np
from mydynalearn.analyze.utils.utils import epochdata_datacur_2_dataT
from mydynalearn.analyze.utils.data_handler import DynamicDataHandler
from mydynalearn.analyze.utils.performance_data.getter import get as performance_data_getter
class runModelOnTestData():
    def __init__(self,
                 config,
                 network,
                 dynamics,
                 test_loader,
                 model_exp_epoch_index,
                 model_exp,
                 dataset_exp,
                 ):
        self.config = config
        self.network = network
        self.dynamics = dynamics
        self.IS_WEIGHT = model_exp.config.IS_WEIGHT
        self.model_exp_epoch_index = model_exp_epoch_index
        self.model_exp = model_exp

        self.test_loader = test_loader
        self.testdata_exp = dataset_exp

        self.analyze_result_file = self.get_analyze_result_filepath()
        self.need_to_run = not os.path.exists(self.analyze_result_file)
    def get_model_info(self):
        model_exp_info_dict = {
            'network_name': self.network.NAME,
            'dynamics_name': self.dynamics.NAME,
            'model_name': self.model_exp.model.NAME,
        }
        model_info = "_".join([value for value in model_exp_info_dict.values()])
        return model_info

    def get_testdata_info(self):
        testset_exp_info_dict = {
            'network_name': self.testdata_exp.config.network.NAME,
            'dynamics_name': self.testdata_exp.config.dynamics.NAME,
        }

        testdata_info = "_".join([value for value in testset_exp_info_dict.values()])
        return testdata_info
    def get_analyze_result_filepath(self):
        model_info = self.get_model_info()
        testdata_info = self.get_testdata_info()
        model_dir_name = "model_" + model_info
        testdata_dir_name = "testdata_" + testdata_info

        root_dir_path = self.config.root_dir_path
        analyze_result_dir_name = self.config.analyze_result_dir_name
        analyze_result_dir_path = os.path.join(root_dir_path, analyze_result_dir_name, model_dir_name, testdata_dir_name)

        if not os.path.exists(analyze_result_dir_path):
            os.makedirs(analyze_result_dir_path)

        analyze_result_file_name = "epoch{:d}_analyze_result.pkl".format(self.model_exp_epoch_index)
        analyze_result_file_path = os.path.join(analyze_result_dir_path, analyze_result_file_name)
        return analyze_result_file_path

    def  save_analyze_result(self, analyze_result):
        with open(self.analyze_result_file, "wb") as f:
            pickle.dump(analyze_result,f)
    def load_analyze_result(self):
        with open(self.analyze_result_file, "rb") as f:
            analyze_result = pickle.load(f)
        return analyze_result
    def set_attention_model_and_optimizer(self, attention_model, optimizer):
        self.attention_model = attention_model
        self.optimizer = optimizer

    def compute_R(self,test_result_curepoch):
        y_pred = torch.cat([data["y_pred"] for data in test_result_curepoch],dim=0)
        y_true = torch.cat([data["y_true"] for data in test_result_curepoch],dim=0)
        y_ob = torch.cat([data["y_ob"] for data in test_result_curepoch],dim=0)

        R_input_y_pred = y_pred[torch.where(y_ob==1)].detach().numpy()
        R_input_y_true = y_true[torch.where(y_ob==1)].detach().numpy()
        R = np.corrcoef(R_input_y_pred,R_input_y_true)[0,1]
        return R

    def get_analyze_result(self):
        test_result = self.model_exp.model.epoch_tasks.run_test_epoch(self.network, self.dynamics, self.test_loader,
                                                                      self.model_exp_epoch_index)
        kwargs = epochdata_datacur_2_dataT(self.dynamics, self.IS_WEIGHT, test_result)
        dynamic_data_handler = DynamicDataHandler(**kwargs)
        get_performance_index, get_performance_data = performance_data_getter(self.model_exp.config)
        performance_index = get_performance_index(dynamic_data_handler)
        performance_data = get_performance_data(dynamic_data_handler)
        R = self.compute_R(test_result)
        analyze_result = {
            "model_network": self.model_exp.config.network.NAME,
            "model_dynamics": self.model_exp.config.dynamics.NAME,
            "dataset_network": self.testdata_exp.config.network.NAME,
            "dataset_dynamics": self.testdata_exp.config.dynamics.NAME,
            "model": self.model_exp.config.model.NAME,
            "model_exp_epoch_index": self.model_exp_epoch_index,
            "performance_index": performance_index,
            "performance_data": performance_data,
            "w_T": kwargs['w_T'],
            "R": R
        }
        return analyze_result
    def run(self):
        model_info = self.get_model_info()
        testdata_info = self.get_testdata_info()
        print("model: {}\ntest data:{}".format(model_info,testdata_info))
        if self.need_to_run:
            print("analyze epoch: ",self.model_exp_epoch_index)
            analyze_result = self.get_analyze_result()
            self.save_analyze_result(analyze_result)
        else:
            analyze_result = self.load_analyze_result()
        print("output analyze_result_file: ",self.analyze_result_file)
        print("analyze completed!\n")
        return analyze_result
