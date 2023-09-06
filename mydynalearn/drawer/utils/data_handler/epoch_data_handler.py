import pickle
import torch
from mydynalearn.drawer.utils.performace_data.getter import get as performace_data_getter
from mydynalearn.drawer.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from mydynalearn.drawer.utils.performace_data.utils import _get_metrics
class EpochDataHandler():
    def __init__(self,config,dynamics):
        self.config = config
        self.epochs = config.dataset.epochs
        self.dynamics = dynamics
        self.datapath_to_epochdata = config.datapath_to_epochdata
        self.datapath_to_maxR = config.datapath_to_maxR

    def save_epoch_data(self, epoch_index, testResult):
        self.fileName = self.datapath_to_epochdata + "/epoch{:d}Data.pkl".format(epoch_index)
        with open(self.fileName, "wb") as file:
            pickle.dump(testResult,file)

    def load_epoch_data(self, epoch_index):
        fileName = self.datapath_to_epochdata + "/epoch{:d}Data.pkl".format(epoch_index)
        with open(fileName, "rb") as file:
            epochData = pickle.load(file)
        return epochData

    def save_max_R(self,max_R):
        fileName = self.datapath_to_maxR + "/maxR.pkl"
        with open(fileName, "wb") as file:
            pickle.dump(max_R,file)

    def load_max_R(self):
        fileName = self.datapath_to_maxR + "/maxR.pkl"
        with open(fileName, "rb") as file:
            max_R = pickle.load(file)
        return max_R


    def epochdata_datacur_2_dataT(self, epoch_index, data_curEpoch):
        is_weight = self.config.is_weight
        T = len(data_curEpoch)
        x_T = torch.zeros([T] + list(data_curEpoch[0]['x'].shape))
        y_pred_T = torch.zeros([T] + list(data_curEpoch[0]['y_pred'].shape))
        y_ob_T = torch.zeros([T] + list(data_curEpoch[0]['y_ob'].shape))
        y_true_T = torch.zeros([T] + list(data_curEpoch[0]['y_true'].shape))
        w_T = torch.zeros([T] + list(data_curEpoch[0]['w'].shape))
        for time, data in enumerate(data_curEpoch):
            x = data['x'].view(-1, x_T.shape[-1])
            y_pred = data['y_pred']
            y_ob = data['y_ob']
            y_true = data['y_true']
            w = data['w']
            x_T[time] = x
            y_pred_T[time] = y_pred
            y_ob_T[time] = y_ob
            y_true_T[time] = y_true
            w_T[time] = w
        if ~is_weight:
            w_T = torch.ones(w_T.shape).to(w_T.device)

        dataT = {
            "dynamics":self.dynamics,
            "epoch_index":epoch_index,
            "x_T":x_T.view(-1, x_T.shape[-1]),
            "y_pred_T":y_pred_T.view(-1, y_pred_T.shape[-1]),
            "y_ob_T":y_ob_T.view(-1, y_ob_T.shape[-1]),
            "y_true_T":y_true_T.view(-1, y_true_T.shape[-1]),
            "w_T":w_T.view(-1)
        }
        return dataT

    def get_max_R(self,):
        get_performance_index, get_performance_data = performace_data_getter(self.config)
        R_list = torch.zeros(self.epochs)
        for epoch_index in range(self.epochs):
            # 用matplotlib画出每个epoch的结果
            epochData = self.load_epoch_data(epoch_index)
            kwargs = self.epochdata_datacur_2_dataT(epoch_index, epochData)
            dynamic_data_handler = DynamicDataHandler(**kwargs)
            performance_index = get_performance_index(dynamic_data_handler)
            performance_data = get_performance_data(dynamic_data_handler)
            corrcoef, r2 = _get_metrics(performance_data)
            R_list[epoch_index] = corrcoef
        max_R_value = R_list.max()
        max_R_index = R_list.argmax()
        max_R = (max_R_index, max_R_value)
        self.save_max_R(max_R)

