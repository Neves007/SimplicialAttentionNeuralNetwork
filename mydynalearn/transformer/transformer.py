import torch

def TP_to_class(TP):
    return TP.max(-1)[1]

def dict_to_tensor(dict):
    for key in dict.keys():
        dict

def data_curEpoch_2_data_T(data_curEpoch,is_weight):
    T = len(data_curEpoch)
    x_T = torch.zeros([T]+list(data_curEpoch[0]['x'].shape))
    y_pred_T = torch.zeros([T]+list(data_curEpoch[0]['y_pred'].shape))
    y_ob_T = torch.zeros([T]+list(data_curEpoch[0]['y_ob'].shape))
    y_true_T = torch.zeros([T]+list(data_curEpoch[0]['y_true'].shape))
    w_T = torch.zeros([T]+list(data_curEpoch[0]['w'].shape))
    for time,data in enumerate(data_curEpoch):
        x = data['x']
        y_pred = data['y_pred']
        y_ob = data['y_ob']
        y_true = data['y_true']
        w = data['w']
        x_T[time] =x
        y_pred_T[time] =y_pred
        y_ob_T[time] =y_ob
        y_true_T[time] =y_true
        w_T[time] =w
    if ~is_weight:
        w_T = torch.ones(w_T.shape).to(w_T.device)
    return x_T.view(-1,x_T.shape[-1]), y_pred_T.view(-1,y_pred_T.shape[-1]), y_ob_T.view(-1,y_ob_T.shape[-1]), y_true_T.view(-1,y_true_T.shape[-1]),w_T.view(-1)