from mydynalearn.drawer.visdomDrawer.VisdomDrawer import VisdomDrawer
from mydynalearn.drawer.utils import *

class Visdom_trainingProcess():
    def __init__(self,config):
        self.config = config
        self.visdomDrawer = VisdomDrawer()

    def visdom_do_epoch(self, epoch_idx, testResult_curEpoch):
        self.visdomDrawer.init_window()
        test_loss, test_acc = compute_testResult_curEpoch_loss_acc(testResult_curEpoch)
        self.visdomDrawer.draw_epoch(test_loss, test_acc, epoch_idx)
        pass

    def visdom_do_batch(self, train_data, val_data):
        time_idx = train_data['time_idx']
        if time_idx % 100 == 0:
            epoch_idx, time_idx, train_loss, train_acc, train_x, train_y_pred, train_y_true, train_y_ob, train_w = unpackBatchData(
                train_data)
            epoch_idx, time_idx, val_loss, val_acc, val_x, val_y_pred, val_y_true, val_y_ob, val_w = unpackBatchData(
                val_data)
            self.visdomDrawer.draw_performance(val_x, val_y_pred, val_y_ob, val_y_true)
            self.visdomDrawer.draw_acc_loss(train_loss, train_acc, val_loss, val_acc, time_idx)
