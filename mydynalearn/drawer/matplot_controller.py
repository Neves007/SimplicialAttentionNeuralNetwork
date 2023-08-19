import torch


from mydynalearn.drawer.matplot_drawer import *
class MatplotController():
    def __init__(self,config,dynamics):
        self.matplot_epochPerformance =  DrawerMatplotEpochFigYtrureYpred(config,dynamics)
        self.matplot_Dynamic_Stady_rho =  matplot_Dynamic_Stady_rho(config)

    def matplotDrawEpoch(self,epoch_index):
        epochData = self.matplot_epochPerformance.load_epoch_data(epoch_index)
        self.matplot_epochPerformance.draw(epoch_index,epochData)

    def matplotDynamic(self, eff_infectionList,stady_rho_list):
        self.matplot_Dynamic_Stady_rho.draw(eff_infectionList,stady_rho_list)
