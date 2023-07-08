import torch
def _unpackTestResult_curEpochResult(testResult_curEpoch):
    T = len(testResult_curEpoch)
    loss_all = 1.*torch.zeros(T)
    acc_all = 1.*torch.zeros(T)
    for t,data in enumerate(testResult_curEpoch):
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

def compute_testResult_curEpoch_loss_acc(testResult_curEpoch):
    loss_all,acc_all = _unpackTestResult_curEpochResult(testResult_curEpoch)
    return loss_all.mean(),acc_all.mean()