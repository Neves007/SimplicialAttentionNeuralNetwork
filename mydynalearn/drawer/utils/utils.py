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
    epoch_idx = data['epoch_idx']
    time_idx = data['time_idx']
    loss = data['loss']
    acc = data['acc']
    x = data['x']
    y_pred = data['y_pred']
    y_true = data['y_true']
    y_ob = data['y_ob']
    w = data['w']
    return epoch_idx, time_idx, loss, acc, x, y_pred, y_true, y_ob, w

def compute_test_result_curepoch_loss_acc(test_result_curepoch):
    loss_all,acc_all = _unpacktest_result_curepochResult(test_result_curepoch)
    return loss_all.mean(),acc_all.mean()