import torch
class DynamicDataHandler():
    def __init__(self,dynamics, x_T, y_pred_T, y_ob_T, y_true_T,**kwargs):
        self.STATES_MAP = dynamics.STATES_MAP
        self.x_T = x_T
        self.y_ob_T = y_ob_T
        self.y_true_T = y_true_T
        self.y_pred_T = y_pred_T

    def get_transfer_index(self,origin_state,trans_state):
        index = torch.where((self.x_T[:, self.STATES_MAP[origin_state]] == 1) & (self.y_ob_T[:, self.STATES_MAP[trans_state]] == 1))[0]
        return index

    def get_ytrue_ypred_pair(self,origin_state,trans_state):
        index = self.get_transfer_index(origin_state,trans_state)
        ytrue_ypred_pair = torch.cat((self.y_true_T[index, self.STATES_MAP[trans_state]].view(-1, 1),
                                      self.y_pred_T[index, self.STATES_MAP[trans_state]].view(-1, 1))
                                     , dim=1)
        return ytrue_ypred_pair