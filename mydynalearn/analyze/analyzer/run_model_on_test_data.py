import os
import pickle
import torch
import numpy as np
from mydynalearn.analyze.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from multipledispatch import dispatch
class runModelOnTestData():
    def __init__(self,
                 config,
                 model_exp,
                 dataset_exp,
                 ):
        '''
        测试类，使用 testdata_exp 的测试数据来测试，model_exp的模型结果
        :param config:
        :param model_exp:
        :param dataset_exp:
        '''
        self.config = config
        self.IS_WEIGHT = model_exp.config.IS_WEIGHT
        self.model_exp = model_exp
        self.testdata_exp = dataset_exp
        self.EPOCHS = model_exp.model.config.model.EPOCHS

        self.global_info = self.get_global_info()
        self.all_need_to_run = self.check_run_necessity()

    @dispatch()
    def check_run_necessity(self):
        '''静态方法，查看是否需要加载数据来创建对象

        :param model_exp:
        :param testdata_exp:
        :return:
        '''
        all_need_to_run = False
        for model_exp_epoch_index in range(self.EPOCHS):
            need_to_run = self.check_run_necessity(model_exp_epoch_index)
            if need_to_run == True:
                all_need_to_run = True
            else:
                pass
        return all_need_to_run


    @dispatch(int)
    def check_run_necessity(self, model_exp_epoch_index):
        analyze_result_filepath = self.get_analyze_result_filepath(model_exp_epoch_index)
        need_to_run = not os.path.exists(analyze_result_filepath)
        return need_to_run





    def get_global_info(self):
        global_info = {
            "model_network_name": self.model_exp.config.network.NAME,
            "model_dynamics_name": self.model_exp.config.dynamics.NAME,
            "dataset_network_name": self.testdata_exp.config.network.NAME,
            "dataset_dynamics_name": self.testdata_exp.config.dynamics.NAME,
            "model_name": self.model_exp.config.model.NAME,
        }
        return global_info

    def get_analyze_result_filepath(self,model_exp_epoch_index):
        model_info = str.join('_',[self.global_info["model_network_name"],
                                   self.global_info["model_dynamics_name"],
                                   self.global_info["model_name"]
                                   ])
        testdata_info = str.join('_',[self.global_info["dataset_network_name"],
                                   self.global_info["dataset_dynamics_name"],
                                   self.global_info["model_name"]
                                   ])
        model_dir_name = "model_" + model_info
        testdata_dir_name = "testdata_" + testdata_info

        root_dir_path = self.config.root_dir_path
        analyze_result_dir_name = self.config.analyze_result_dir_name
        analyze_result_dir_path = os.path.join(root_dir_path, analyze_result_dir_name, model_dir_name, testdata_dir_name)

        if not os.path.exists(analyze_result_dir_path):
            os.makedirs(analyze_result_dir_path)

        analyze_result_file_name = "epoch{:d}_analyze_result.pkl".format(model_exp_epoch_index)
        analyze_result_file_path = os.path.join(analyze_result_dir_path, analyze_result_file_name)
        return analyze_result_file_path

    def save_analyze_result(self, analyze_result,analyze_result_filepath):
        with open(analyze_result_filepath, "wb") as f:
            pickle.dump(analyze_result,f)
    def load_analyze_result(self,analyze_result_filepath):
        with open(analyze_result_filepath, "rb") as f:
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
    def compute_loss(self,test_result_curepoch):
        loss_list = torch.stack([data["loss"] for data in test_result_curepoch])
        loss = loss_list.mean()
        loss = loss.detach().item()
        return loss

    def create_analyze_result(self,model_exp_epoch_index):
        '''

        :param model_exp_epoch_index: 已训练模型model_exp的epoch
        :return:
        '''
        try:
            # 读取测试数据集数据
            network, dynamics, _, _, test_loader = self.testdata_exp.create_dataset()
        except Exception as e:
            print(self.global_info)
            raise e
        test_result_time_list = self.model_exp.model.epoch_tasks.run_test_epoch(network,
                                                                                dynamics,
                                                                                test_loader,
                                                                                model_exp_epoch_index)

        dynamic_data_handler = DynamicDataHandler(dynamics, test_result_time_list)
        test_result_info = {
            "model_network_name": self.model_exp.config.network.NAME,
            "model_dynamics_name": self.model_exp.config.dynamics.NAME,
            "MAX_DIMENSION": dynamics.MAX_DIMENSION,
            "dataset_network_name": self.testdata_exp.config.network.NAME,
            "dataset_dynamics_name": self.testdata_exp.config.dynamics.NAME,
            "model_name": self.model_exp.config.model.NAME,
            "model_epoch_index": model_exp_epoch_index,
        }

        test_result_df = dynamic_data_handler.get_testresult_dataframe(test_result_info)
        model_performace_dict = dynamic_data_handler.get_model_performace(test_result_df)
        merge_data = {
            "test_result_info": test_result_info,
            "test_result_df": test_result_df,
            "dynamics_STATES_MAP": dynamics.STATES_MAP,
            "model_performace_dict": model_performace_dict,
        }
        return merge_data


    def run(self, model_exp_epoch_index):

        analyze_result_filepath = self.get_analyze_result_filepath(model_exp_epoch_index)
        need_to_run = not os.path.exists(analyze_result_filepath)
        if need_to_run:
            analyze_result = self.create_analyze_result(model_exp_epoch_index)
            self.save_analyze_result(analyze_result,analyze_result_filepath)
        else:
            analyze_result = self.load_analyze_result(analyze_result_filepath)
        print("testing:")
        print("analyze epoch: ", model_exp_epoch_index)
        print("output analyze_result_filepath: ",analyze_result_filepath)
        print("analyze completed!")
        return analyze_result
