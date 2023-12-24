import pickle
import torch
from mydynalearn.drawer_old.utils.performance_data.getter import get as performance_data_getter
from mydynalearn.drawer_old.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from mydynalearn.drawer_old.utils.performance_data.utils import _get_metrics
class EpochDataHandler():
    def __init__(self,config,dynamics):
        self.config = config
        self.EPOCHS = config.dataset.EPOCHS
        self.dynamics = dynamics

    def epochdata_datacur_2_dataT(self, IS_WEIGHT,dynamics,data_curEpoch):
        T = len(data_curEpoch)
        epoch_index = data_curEpoch[0]['epoch_index']
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
        if ~IS_WEIGHT:
            w_T = torch.ones(w_T.shape).to(w_T.device)

        dataT = {
            "dynamics":dynamics,
            "epoch_index":epoch_index,
            "x_T":x_T.view(-1, x_T.shape[-1]),
            "y_pred_T":y_pred_T.view(-1, y_pred_T.shape[-1]),
            "y_ob_T":y_ob_T.view(-1, y_ob_T.shape[-1]),
            "y_true_T":y_true_T.view(-1, y_true_T.shape[-1]),
            "w_T":w_T.view(-1)
        }
        return dataT