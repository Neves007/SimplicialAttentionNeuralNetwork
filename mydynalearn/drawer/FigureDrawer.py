import torch


from mydynalearn.drawer.matplotDrawer import *
from mydynalearn.drawer.visdomDrawer.fig.Visdom_trainingProcess import Visdom_trainingProcess
class FigureDrawer():
    def __init__(self,config):

        self.matplotDrawer = MatplotDrawer(config)
        self.visdom_trainingProcess =  Visdom_trainingProcess(config)
        self.matplot_epochPerformance =  epoch_Fig_ytrure_ypred_Drawer(config)
        self.matplot_Dynamic_Stady_rho =  matplot_Dynamic_Stady_rho(config)

    def matplotDrawEpoch(self,epoch_index):
        epochData = self.matplot_epochPerformance.loadEpochData(epoch_index)
        self.matplot_epochPerformance.draw(epoch_index,epochData)

    def matplotDynamic(self, eff_infectionList,stady_rho_list):
        self.matplot_Dynamic_Stady_rho.draw(eff_infectionList,stady_rho_list)

    def visdomDrawEpoch(self,epoch_idx, testResult_curEpoch):
        self.visdom_trainingProcess.visdom_do_epoch(epoch_idx, testResult_curEpoch)

    def visdomDrawBatch(self,train_data, val_data):
        self.visdom_trainingProcess.visdom_do_batch(train_data, val_data)
