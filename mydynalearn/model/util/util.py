from mydynalearn.transformer import *
from mydynalearn.metrics import *
def get_acc(y_y_ob,y_pred_TP):
    y_pred_class = TP_to_class(y_pred_TP)
    y_ob_class = TP_to_class(y_y_ob)
    acc_score = acc(y_ob_class,y_pred_class)
    return acc_score