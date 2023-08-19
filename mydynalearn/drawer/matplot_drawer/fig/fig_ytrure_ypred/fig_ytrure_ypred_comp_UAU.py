import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from matplotlib.lines import Line2D
class FigYtrureYpredCompUAU():
    def __init__(self,ax,dynamics):
        self.STATES_MAP = dynamics.STATES_MAP
        self.colors = ["red","orange","yellow","green","cyan","blue","purple"]
        self.markers = ["o", "o","o", "o", "o","o", "o"]
        self.label = ["U to U",
            "U to A1",
            "U to A2",
            "A1 to A1",
            "A1 to U",
            "A2 to A2",
            "A2 to U"]
        self.ax = ax

    def _get_performance_data(self, x, y_pred, y_ob, y_true):
        with torch.no_grad():
            pre_labels = y_pred.max(1)[1].type_as(y_ob)

            U_U = torch.where((x[:, self.STATES_MAP["U"]] == 1) & (y_ob[:, self.STATES_MAP["U"]] == 1))[0]
            U_A1 = torch.where((x[:, self.STATES_MAP["U"]] == 1) & (y_ob[:, self.STATES_MAP["A1"]] == 1))[0]
            U_A2 = torch.where((x[:, self.STATES_MAP["U"]] == 1) & (y_ob[:, self.STATES_MAP["A2"]] == 1))[0]
            A1_A1 = torch.where((x[:, self.STATES_MAP["A1"]] == 1) & (y_ob[:, self.STATES_MAP["A1"]] == 1))[0]
            A1_U = torch.where((x[:, self.STATES_MAP["A1"]] == 1) & (y_ob[:, self.STATES_MAP["U"]] == 1))[0]
            A2_A2 = torch.where((x[:, self.STATES_MAP["A2"]] == 1) & (y_ob[:, self.STATES_MAP["A2"]] == 1))[0]
            A2_U = torch.where((x[:, self.STATES_MAP["A2"]] == 1) & (y_ob[:, self.STATES_MAP["U"]] == 1))[0]

            U_U_true_pred = torch.cat(
                (y_true[U_U, self.STATES_MAP["U"]].view(-1, 1), y_pred[U_U, self.STATES_MAP["U"]].view(-1, 1)),
                dim=1).cpu().numpy()
            U_A1_true_pred = torch.cat(
                (y_true[U_A1, self.STATES_MAP["A1"]].view(-1, 1), y_pred[U_A1, self.STATES_MAP["A1"]].view(-1, 1)),
                dim=1).cpu().numpy()
            U_A2_true_pred = torch.cat(
                (y_true[U_A2, self.STATES_MAP["A2"]].view(-1, 1), y_pred[U_A2, self.STATES_MAP["A2"]].view(-1, 1)),
                dim=1).cpu().numpy()
            A1_A1_true_pred = torch.cat(
                (y_true[A1_A1, self.STATES_MAP["A1"]].view(-1, 1), y_pred[A1_A1, self.STATES_MAP["A1"]].view(-1, 1)),
                dim=1).cpu().numpy()
            A1_U_true_pred = torch.cat(
                (y_true[A1_U, self.STATES_MAP["U"]].view(-1, 1), y_pred[A1_U, self.STATES_MAP["U"]].view(-1, 1)),
                dim=1).cpu().numpy()
            A2_A2_true_pred = torch.cat(
                (y_true[A2_A2, self.STATES_MAP["A2"]].view(-1, 1), y_pred[A2_A2, self.STATES_MAP["A2"]].view(-1, 1)),
                dim=1).cpu().numpy()
            A2_U_true_pred = torch.cat(
                (y_true[A2_U, self.STATES_MAP["U"]].view(-1, 1), y_pred[A2_U, self.STATES_MAP["U"]].view(-1, 1)),
                dim=1).cpu().numpy()

        performance_data = (U_U_true_pred,
                            U_A1_true_pred,
                            U_A2_true_pred,
                            A1_A1_true_pred,
                            A1_U_true_pred,
                            A2_A2_true_pred,
                            A2_U_true_pred)
        performance_index = (U_U,
                             U_A1,
                             U_A2,
                             A1_A1,
                             A1_U,
                             A2_A2,
                             A2_U)

        return performance_data,performance_index
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
        performance_data,performance_index = self._get_performance_data(x_T, y_pred_T, y_ob_T, y_true_T)

        marker_size = self.get_marker_size(w_T, max=700, min=50)
        self.ax.scatter(x=performance_data[0][:,0], y=performance_data[0][:,1], c=self.colors[0], marker=self.markers[0], s=marker_size[performance_index[0]], alpha=0.3)
        self.ax.scatter(x=performance_data[1][:,0], y=performance_data[1][:,1], c=self.colors[1], marker=self.markers[1], s=marker_size[performance_index[1]], alpha=0.3)
        self.ax.scatter(x=performance_data[2][:,0], y=performance_data[2][:,1], c=self.colors[2], marker=self.markers[2], s=marker_size[performance_index[2]], alpha=0.3)
        self.ax.scatter(x=performance_data[3][:,0], y=performance_data[3][:,1], c=self.colors[3], marker=self.markers[3], s=marker_size[performance_index[3]], alpha=0.3)
        self.ax.scatter(x=performance_data[4][:,0], y=performance_data[4][:,1], c=self.colors[4], marker=self.markers[4], s=marker_size[performance_index[4]], alpha=0.3)
        self.ax.scatter(x=performance_data[5][:,0], y=performance_data[5][:,1], c=self.colors[5], marker=self.markers[5], s=marker_size[performance_index[5]], alpha=0.3)
        self.ax.scatter(x=performance_data[6][:,0], y=performance_data[6][:,1], c=self.colors[6], marker=self.markers[6], s=marker_size[performance_index[6]], alpha=0.3)
    def _get_metrics(self, x_T, y_pred_T, y_ob_T, y_true_T):
        performance_data,performance_index = self._get_performance_data(x_T, y_pred_T, y_ob_T, y_true_T)
        true_pred = np.concatenate(performance_data)
        corrcoef = np.corrcoef(true_pred.T)[0,1]
        r2 = r2_score(true_pred[:,0],true_pred[:,1])
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
                           plt.scatter([0], [0], c=self.colors[3], marker=self.markers[3], s=53, alpha=0.8),
                           plt.scatter([0], [0], c=self.colors[4], marker=self.markers[4], s=53, alpha=0.8),
                           plt.scatter([0], [0], c=self.colors[5], marker=self.markers[5], s=53, alpha=0.8),
                           plt.scatter([0], [0], c=self.colors[6], marker=self.markers[6], s=53, alpha=0.8),
                                ]
        self.ax.legend(handles=self.legend_elements, labels=self.label)
        self.ax.grid(True)

