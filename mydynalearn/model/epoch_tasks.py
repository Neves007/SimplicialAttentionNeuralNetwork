from mydynalearn.model.nn.nnlayers import *
from time import sleep
import os
import pickle
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mydynalearn.model.optimizer import get as get_optimizer
from mydynalearn.drawer import VisdomController
from mydynalearn.model.util import *
import copy
from mydynalearn.logger.logger import *
from tqdm import tqdm
from mydynalearn.model.getter import get as get_attmodel
from mydynalearn.model.batch_task import BatchTask

class EpochTasks():
    def __init__(self, config):
        self.config = config
        self.EPOCHS = config.model.EPOCHS
        self.need_to_train = self.is_need_to_train()
        self.batch_task = BatchTask(config)
        #
        self.attention_model = get_attmodel(self.config)
        self.get_optimizer = get_optimizer(config.optimizer)
        self.optimizer = self.get_optimizer(self.attention_model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=4, eps = 1e-8, threshold =0.1, verbose=True)

    def get_fileName_model_state_dict(self,epoch_index):
        fileName_model_state_dict = self.config.modelparams_dir_path + "/epoch{:d}_model_state_dict.pth".format(
            epoch_index)
        return fileName_model_state_dict

    def save(self, attention_model, optimizer, epoch_index):
        model_state_dict_file_path = self.get_fileName_model_state_dict(epoch_index)
        print("\ntraining.epochtask. output dataset_file: ", model_state_dict_file_path)
        torch.save({
            # 存储 batch的state_dict
            'model_state_dict': attention_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_state_dict_file_path)

    def load(self, epoch_index):
        '''
        输入：当前epoch_index
        规则：加载模型参数
        输出：
        '''
        checkpoint = torch.load(self.get_fileName_model_state_dict(epoch_index))
        self.attention_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def is_need_to_trian(self,epoch_index):
        '''
        输入：当前epoch_index
        规则：如果存在model_state_dict文件就不需要训练，否则需要
        输出：是否需要训练
        '''
        fileName_model_state_dict = self.get_fileName_model_state_dict(epoch_index)
        is_need_to_trian = not os.path.exists(fileName_model_state_dict)
        return is_need_to_trian

    def low_the_lr(self ,epoch_index):
        if (epoch_index>0) and (epoch_index % 5 == 0):
            self.optimizer.param_groups[0]['lr'] *= 0.5


    def pack_batch_data(self, epoch_index, time_idx, loss, x, y_pred, y_true, y_ob, w):
        data = {'epoch_index': epoch_index,
                'time_idx': time_idx,
                'loss': loss.cpu(),
                'acc': get_acc(y_ob, y_pred).cpu(),
                'x': x.cpu(),
                'y_pred': y_pred.cpu(),
                'y_true': y_true.cpu(),
                'y_ob': y_ob.cpu(),
                'w': w.cpu(),
                }
        return data

    def is_need_to_train(self):
        '''
        判断该epoch_tasks是否该重新训练
        - 若存在epoch未被训练则需重新训练

        :return: bool
        '''
        tag = False
        for epoch_id in range(self.EPOCHS):
            epoch_model_file = self.get_fileName_model_state_dict(epoch_id)
            if not os.path.exists(epoch_model_file):
                tag =  True
        return tag


    def train_epoch(self,train_set, val_set, network, dynamics, epoch_index,visdom_drawer):
        process_bar = tqdm(
            enumerate(zip(train_set, val_set)),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=len(train_set)
        )
        R_stack = []
        for time_idx, (train_dataset_per_time, val_dataset_per_time) in process_bar:
            self.attention_model.train()
            self.optimizer.zero_grad()
            train_loss, train_x, train_y_pred, train_y_true, train_y_ob, train_w = self.batch_task._do_batch_(self.attention_model,
                                                                                                              network,
                                                                                                              dynamics,
                                                                                                              train_dataset_per_time)
            train_loss.backward()
            self.optimizer.step()
            self.attention_model.eval()
            val_loss, val_x, val_y_pred, val_y_true, val_y_ob, val_w = self.batch_task._do_batch_(self.attention_model,
                                                                                                  network,
                                                                                                  dynamics,
                                                                                                  val_dataset_per_time)
            val_data = self.pack_batch_data(epoch_index,
                                            time_idx,
                                            val_loss,
                                            val_x,
                                            val_y_pred,
                                            val_y_true,
                                            val_y_ob,
                                            val_w)
            y_true_flat = val_y_true.flatten()
            y_pred_flat = val_y_pred.flatten()
            R = torch.corrcoef(torch.stack([y_true_flat, y_pred_flat]))[0, 1]
            item_info = 'Epoch:{:d} LR:{:f} R:{:f}'.format(epoch_index,
                                                           self.optimizer.param_groups[0]['lr'],
                                                           R)
            process_bar.set_postfix(custom_info=item_info)
            visdom_drawer.visdomDrawBatch(val_data)
        process_bar.close()
    def run_all(self,network, dynamics, train_set, val_set, test_set):
        visdom_drawer = VisdomController(self.config, dynamics)
        for epoch_index in range(self.EPOCHS):
            self.train_epoch(train_set,
                             val_set,
                             network,
                             dynamics,
                             epoch_index,
                             visdom_drawer)
            self.save(self.attention_model, self.optimizer, epoch_index)
            self.low_the_lr(epoch_index)

    def run_test_epoch(self, network, dynamics, test_loader, epoch_index):
        self.load(epoch_index)
        process_bar = tqdm(
            enumerate(test_loader),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=len(test_loader),
        )
        test_result_curepoch = []
        self.attention_model.eval()
        for time_idx,test_dataset_per_time in process_bar:
            test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w = self.batch_task._do_batch_(self.attention_model,
                                                                                                        network,
                                                                                                        dynamics,
                                                                                                        test_dataset_per_time)
            test_data = self.pack_batch_data(epoch_index, time_idx, test_loss, test_x, test_y_pred, test_y_true, test_y_ob,
                                             test_w)
            test_result_curepoch.append(test_data)
        process_bar.close()
        return test_result_curepoch

