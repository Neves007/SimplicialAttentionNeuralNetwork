import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from matplotlib.lines import Line2D
class FigYtrureYpred():
    def __init__(self,ax):
        self.colors = ["green", "red", "b", "orange"]
        self.markers = ["o", "o","o", "o"]
        self.label = ["S_S", "S_I", "I_S", "I_I"]
        self.ax = ax

    def _get_performance_data(self, x, predict_TP, y_ob, true_TP):
        self.STATES_MAP = {"S": 0, "I": 1}  # [S,I]
        with torch.no_grad():
            pre_labels = predict_TP.max(1)[1].type_as(y_ob)
            ob_labels = y_ob.max(1)[1].type_as(y_ob)
            right_prediction = pre_labels == ob_labels

            S_S = torch.where((x[:, self.STATES_MAP["S"]] == 1) & (y_ob[:,self.STATES_MAP["S"]]==1))[0]
            S_I = torch.where((x[:, self.STATES_MAP["S"]] == 1) & (y_ob[:,self.STATES_MAP["I"]]==1))[0]
            I_S = torch.where((x[:, self.STATES_MAP["I"]] == 1) & (y_ob[:,self.STATES_MAP["S"]]==1))[0]
            I_I = torch.where((x[:, self.STATES_MAP["I"]] == 1) & (y_ob[:,self.STATES_MAP["I"]]==1))[0]
            # 混淆矩阵
            S_S_true,S_S_pre = (true_TP[S_S,0].view(-1,1), predict_TP[S_S,0].view(-1,1))
            S_I_true,S_I_pre = (true_TP[S_I,1].view(-1,1), predict_TP[S_I,1].view(-1,1))
            I_S_true,I_S_pre = (true_TP[I_S,0].view(-1,1), predict_TP[I_S,0].view(-1,1))
            I_I_true,I_I_pre = (true_TP[I_I,1].view(-1,1), predict_TP[I_I,1].view(-1,1))
            S_S_true = S_S_true.cpu().numpy()
            S_I_true = S_I_true.cpu().numpy()
            I_S_true = I_S_true.cpu().numpy()
            I_I_true = I_I_true.cpu().numpy()
            S_S_pre = S_S_pre.cpu().numpy()
            S_I_pre = S_I_pre.cpu().numpy()
            I_S_pre = I_S_pre.cpu().numpy()
            I_I_pre = I_I_pre.cpu().numpy()
        return S_S_true,S_I_true,I_S_true,I_I_true,S_S_pre,S_I_pre,I_S_pre,I_I_pre,S_S, S_I, I_S, I_I
    def get_marker_size(self,w,max=10,min=2):

        w_min = w.min()
        w_max = w.max()
        new_min = min
        new_max = max
        if w_min == w_max:
            size = min*torch.ones(w.shape)
        else:
            size = (w - w_min) / (w_max - w_min) * (new_max - new_min) + new_min
        return size.to(torch.int)

    def scatterT(self, epoch_index, x_T, y_pred_T, y_ob_T, y_true_T,w_T):
        S_S_true, S_I_true, I_S_true, I_I_true, S_S_pre, S_I_pre, I_S_pre, I_I_pre, S_S, S_I, I_S, I_I = self._get_performance_data(
            x_T, y_pred_T, y_ob_T, y_true_T)
        # S_S_marker_size = self.get_marker_size(w_T[S_S], max=500, min=50)
        # S_I_marker_size = self.get_marker_size(w_T[S_I], max=500, min=50)
        # I_S_marker_size = self.get_marker_size(w_T[I_S], max=500, min=50)
        # I_I_marker_size = self.get_marker_size(w_T[I_I], max=500, min=50)
        # self.scat_S_S = self.ax.scatter(x=S_S_true, y=S_S_pre, c=self.colors[0], marker=self.markers[0], s=S_S_marker_size, alpha=0.3)
        # self.scat_S_I = self.ax.scatter(x=S_I_true, y=S_I_pre, c=self.colors[1], marker=self.markers[1], s=S_I_marker_size, alpha=0.3)
        # self.scat_I_S = self.ax.scatter(x=I_S_true, y=I_S_pre, c=self.colors[2], marker=self.markers[2], s=I_S_marker_size, alpha=0.3)
        # self.scat_I_I = self.ax.scatter(x=I_I_true, y=I_I_pre, c=self.colors[3], marker=self.markers[3], s=I_I_marker_size, alpha=0.3)

        marker_size = self.get_marker_size(w_T, max=700, min=50)
        self.scat_S_S = self.ax.scatter(x=S_S_true, y=S_S_pre, c=self.colors[0], marker=self.markers[0], s=marker_size[S_S], alpha=0.3)
        self.scat_S_I = self.ax.scatter(x=S_I_true, y=S_I_pre, c=self.colors[1], marker=self.markers[1], s=marker_size[S_I], alpha=0.3)
        self.scat_I_S = self.ax.scatter(x=I_S_true, y=I_S_pre, c=self.colors[2], marker=self.markers[2], s=marker_size[I_S], alpha=0.3)
        self.scat_I_I = self.ax.scatter(x=I_I_true, y=I_I_pre, c=self.colors[3], marker=self.markers[3], s=marker_size[I_I], alpha=0.3)


    def _get_metrics(self, x_T, y_pred_T, y_ob_T, y_true_T):
        S_S_true,S_I_true,I_S_true,I_I_true,S_S_pred,S_I_pred,I_S_pred,I_I_pred,_,_,_,_ = self._get_performance_data(x_T, y_pred_T, y_ob_T, y_true_T)
        true = np.concatenate((S_S_true,S_I_true,I_S_true,I_I_true),axis=0).flatten()
        pred = np.concatenate((S_S_pred,S_I_pred,I_S_pred,I_I_pred),axis=0).flatten()

        corrcoef = np.corrcoef(true,pred)[0,1]
        r2 = r2_score(true,pred)
        return corrcoef,r2

    def editAix(self,epoch_index,x_T, y_pred_T, y_ob_T, y_true_T):
        corrcoef,r2 = self._get_metrics(x_T, y_pred_T, y_ob_T, y_true_T)

        self.ax.set_title(r'epoch = {:d}, $R$ = {:0.5f}, $R^2$ = {:0.5f}'.format(epoch_index, corrcoef, r2))
        self.ax.set_xticks(np.linspace(0,1,5))
        self.ax.set_yticks(np.linspace(0,1,5))
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.set_xlabel("Target")  # 设置x轴标注
        self.ax.set_ylabel("prediction")  # 设置y轴标注

        self.legend_elements = [plt.scatter([0], [0], c=self.colors[0], marker=self.markers[0], s=53, alpha=0.8),
                           plt.scatter([0], [0], c=self.colors[1], marker=self.markers[1], s=53, alpha=0.8),
                           plt.scatter([0], [0], c=self.colors[2], marker=self.markers[2], s=53, alpha=0.8),
                           plt.scatter([0], [0], c=self.colors[3], marker=self.markers[3], s=53, alpha=0.8)]
        self.ax.legend(handles=self.legend_elements, labels=self.label)
        self.ax.grid(True)

