
from mydynalearn.drawer_old.visdom_drawer.visdom_batch_drawer.getter import get as visdom_batch_drawer_getter
from mydynalearn.drawer_old.visdom_drawer.visdom_epoch_drawer.getter import get as visdom_eopoch_drawer_getter
from mydynalearn.drawer_old.utils import *
class VisdomController():
    def __init__(self,config,dynamics):

        self.visdom_batch_drawer = visdom_batch_drawer_getter(config,dynamics)
        self.visdom_epoch_drawer = visdom_eopoch_drawer_getter(config)

    def visdomDrawEpoch(self,epoch_index, test_result_curepoch):
        self.visdom_batch_drawer.init_window()
        test_loss, test_acc = compute_test_result_curepoch_loss_acc(test_result_curepoch)
        self.visdom_epoch_drawer.draw_epoch(test_loss, test_acc, epoch_index)
    def visdomDrawBatch(self, val_data):
        time_idx = val_data['time_idx']
        if time_idx % 100 == 0:
            self.visdom_batch_drawer.draw_performance(val_data)
            # self.visdom_batch_drawer.draw_acc_loss(train_data,val_data)