from mydynalearn.model.nn.nnlayers import *
import os
import pickle
import torch.nn as nn
import torch
from mydynalearn.model.optimizer import get as get_optimizer
from mydynalearn.dataset import graphDataSetLoader
from mydynalearn.drawer import VisdomController
from mydynalearn.model.util import *
import copy
from mydynalearn.logger.logger import *
from tqdm import tqdm
from mydynalearn.dataset.getter import get as dataset_getter
from torch.utils.data import DataLoader
from mydynalearn.model.batch_task import BatchTask

class EpochTask():
    def __init__(self, config,epoch_index):
        self.config = config
        self.epoch_index = epoch_index
        self.model_state_dict_file = self.get_fileName_model_state_dict(epoch_index)
        self.batch_task = BatchTask(config)

    def get_fileName_model_state_dict(self,epoch_index):
        fileName_model_state_dict = self.config.datapath_to_model_state_dict + "/epoch{:d}_model_state_dict.pth".format(
            epoch_index)
        return fileName_model_state_dict

    def save_model_state_dict(self, attention_model, optimizer):
        # todo: 修改 save_model_state_dict 需要在哪里加载
        torch.save({
            # 存储 batch的state_dict
            'model_state_dict': attention_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, self.model_state_dict_file)

    def load_model(self,attention_model, optimizer):
        '''
        输入：当前epoch_index
        规则：加载模型参数
        输出：
        '''
        epoch_index = self.epoch_index
        # todo: 修改 load_model 需要在哪里加载
        checkpoint = torch.load(self.model_state_dict_file)
        attention_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def is_need_to_trian(self,epoch_index):
        '''
        输入：当前epoch_index
        规则：如果存在model_state_dict文件就不需要训练，否则需要
        输出：是否需要训练
        '''
        fileName_model_state_dict = self.get_fileName_model_state_dict(epoch_index)
        is_need_to_trian = not os.path.exists(fileName_model_state_dict)
        return is_need_to_trian

    # todo：这个应该是在batch里面
    def low_the_lr(self, optimizer):
        if (self.epoch_index>0) and (self.epoch_index % 5 == 0):
            lr = optimizer.param_groups[0]['lr'] * 0.5
            optimizer.param_groups[0]['lr'] = lr


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


    # save
    def get_test_result(self, epoch_index,test_loader):
        self.eval()
        process_bar = tqdm(
            enumerate(test_loader),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=len(test_loader),
        )
        test_result_curepoch = []
        for time_idx,test_dataset_per_time in process_bar:
            test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w = self._do_batch_(test_dataset_per_time)
            test_data = self.pack_batch_data(epoch_index, time_idx, test_loss, test_x, test_y_pred, test_y_true, test_y_ob,
                                             test_w)
            test_result_curepoch.append(test_data)
        return test_result_curepoch

    # 放进数据集类里面
    # todo: 换位置
    def partition_dataSet(self,dataset):
        test_size = self.config.dataset.NUM_TEST
        val_size = int((len(dataset)-test_size)/2)
        train_size = len(dataset)-test_size-val_size
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

        self.train_loader = DataLoader(train_set,shuffle=True)
        self.val_loader = DataLoader(val_set,shuffle=True)
        self.test_loader = DataLoader(test_set,shuffle=True)

    def run(self, dataset, attention_model, optimizer):
        self.partition_dataSet(dataset)
        network = dataset.network
        dynamics = dataset.dynamics
        self.VisdomDrawer = VisdomController(self.config, dynamics)
        process_bar = tqdm(
            enumerate(zip(self.train_loader, self.val_loader)),
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=len(self.train_loader)
        )
        for time_idx, (train_dataset_per_time,val_dataset_per_time) in process_bar:
            item_info = 'Epoch:{:d} LR:{:f} '.format(self.epoch_index,optimizer.param_groups[0]['lr'])
            process_bar.set_postfix(custom_info=item_info)
            optimizer.zero_grad()
            train_loss,train_x,train_y_pred,train_y_true,train_y_ob, train_w = self.batch_task._do_batch_(attention_model,
                                                                                                          network,
                                                                                                          dynamics,
                                                                                                          train_dataset_per_time)
            train_loss.backward()
            optimizer.step()
            val_loss, val_x, val_y_pred, val_y_true, val_y_ob, val_w = self.batch_task._do_batch_(attention_model,
                                                                                                  network,
                                                                                                  dynamics,
                                                                                                  val_dataset_per_time)
            val_data = self.pack_batch_data(self.epoch_index,
                                            time_idx,
                                            val_loss,
                                            val_x,
                                            val_y_pred,
                                            val_y_true,
                                            val_y_ob,
                                            val_w)
            self.VisdomDrawer.visdomDrawBatch(val_data)

        self.low_the_lr(optimizer)
        self.save_model_state_dict(attention_model, optimizer)