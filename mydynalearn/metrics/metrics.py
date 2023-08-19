from torchmetrics.functional import accuracy
def acc(y_true,y_pred):
    acc_value = accuracy(y_pred,y_true,num_classes=3,task="multiclass")
    return acc_value