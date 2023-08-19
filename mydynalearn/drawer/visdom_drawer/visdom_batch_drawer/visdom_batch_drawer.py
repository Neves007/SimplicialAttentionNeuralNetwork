from visdom import Visdom
import torch
import numpy as np
from mydynalearn.drawer.utils import *
class VisdomBatchDrawer:
    def __init__(self) -> None:
        self.wind = Visdom()
        red = np.array([ 242,0,0]).reshape(1,3).astype(np.int)
        orange = np.array([ 242,157,0]).reshape(1,3).astype(np.int)
        yellow = np.array([ 255,255,51]).reshape(1,3).astype(np.int)
        green = np.array([ 146,195,47]).reshape(1,3).astype(np.int)
        cyan = np.array([ 51,255,153]).reshape(1,3).astype(np.int)
        blue = np.array([ 51,153,255]).reshape(1,3).astype(np.int)
        purple = np.array([ 153,51,255]).reshape(1,3).astype(np.int)
        self.COLORS = {
            "red":red,
            "orange":orange,
            "yellow":yellow,
            "green":green,
            "cyan":cyan,
            "blue":blue,
            "purple":purple,
        }

        # 初始化窗口参数
        self.wind.line([0.], [0.],win = 'train_loss',opts = dict(title = 'train_loss',legend = ['train_loss'],                    xtickmin=0, # 坐标设置
                            ytickmin=0,
                            ytickmax=1))
        self.wind.line([0.], [0.],win = 'train_acc',opts = dict(title = 'train_acc',legend = ['train_acc'],                    xtickmin=0, # 坐标设置
                    ytickmin=0,
                    ytickmax=1))
        self.wind.line([0.], [0.],win = 'val_loss',opts = dict(title = 'val_loss',legend = ['val_loss'],                    xtickmin=0, # 坐标设置
                    ytickmin=0,
                    ytickmax=1))
        self.wind.line([0.], [0.],win = 'val_acc',opts = dict(title = 'val_acc',legend = ['val_acc'],                    xtickmin=0, # 坐标设置
                    ytickmin=0,
                    ytickmax=1))
    def init_window(self):
        # 初始化窗口参数
        self.wind.line([0.], [0.],win = 'train_loss',opts = dict(title = 'train_loss',legend = ['train_loss']))
        self.wind.line([0.], [0.],win = 'train_acc',opts = dict(title = 'train_acc',legend = ['train_acc']))
        self.wind.line([0.], [0.],win = 'val_loss',opts = dict(title = 'val_loss',legend = ['val_loss']))
        self.wind.line([0.], [0.],win = 'val_acc',opts = dict(title = 'val_acc',legend = ['val_acc']))

    def draw_acc_loss(self,train_data,val_data):
        opts = dict(
                    ytickmin=0,
                    ytickmax=1,
                    ytickstep=0.1,
                    )
        epoch_idx, time_idx, train_loss, train_acc, train_x, train_y_pred, train_y_true, train_y_ob, train_w = unpackBatchData(
            train_data)
        epoch_idx, time_idx, val_loss, val_acc, val_x, val_y_pred, val_y_true, val_y_ob, val_w = unpackBatchData(
            val_data)
        self.wind.line([train_loss.data.item()], [time_idx], win='train_loss', opts=opts,update='append')
        self.wind.line([train_acc.data.item()], [time_idx], win='train_acc',opts=opts, update='append')
        self.wind.line([val_loss.data.item()], [time_idx], win='val_loss', opts=opts,update='append')
        self.wind.line([val_acc.data.item()], [time_idx], win='val_acc', opts=opts,update='append')