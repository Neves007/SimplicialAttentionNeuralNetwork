import torch
def _unpacktest_result_curepochResult(test_result_curepoch):
    T = len(test_result_curepoch)
    loss_all = 1.*torch.zeros(T)
    acc_all = 1.*torch.zeros(T)
    for t,data in enumerate(test_result_curepoch):
        time_idx = data['time_idx']
        loss = data['loss']
        acc = data['acc']
        x = data['x']
        y_pred = data['y_pred']
        y_true = data['y_true']
        y_ob = data['y_ob']
        loss_all[t] = loss
        acc_all[t] = acc
    return loss_all,acc_all

def unpackBatchData(data):
    # 加weight
    epoch_index = data['epoch_index']
    time_idx = data['time_idx']
    loss = data['loss']
    acc = data['acc']
    x = data['x']
    y_pred = data['y_pred']
    y_true = data['y_true']
    y_ob = data['y_ob']
    w = data['w']
    return epoch_index, time_idx, loss, acc, x, y_pred, y_true, y_ob, w

def compute_test_result_curepoch_loss_acc(test_result_curepoch):
    loss_all,acc_all = _unpacktest_result_curepochResult(test_result_curepoch)
    return loss_all.mean(),acc_all.mean()

def epochdata_datacur_2_dataT(exp, data_curEpoch):
    is_weight = exp.config.is_weight
    dynamics = exp.dynamics
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
    if ~is_weight:
        w_T = torch.ones(w_T.shape).to(w_T.device)

    dataT = {
        "dynamics": dynamics,
        "epoch_index": epoch_index,
        "x_T": x_T.view(-1, x_T.shape[-1]),
        "y_pred_T": y_pred_T.view(-1, y_pred_T.shape[-1]),
        "y_ob_T": y_ob_T.view(-1, y_ob_T.shape[-1]),
        "y_true_T": y_true_T.view(-1, y_true_T.shape[-1]),
        "w_T": w_T.view(-1)
    }
    return dataT