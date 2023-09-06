import torch

from mydynalearn.drawer.matplot_drawer import DrawerMatplotEpochFigYtrureYpred
from mydynalearn.drawer.matplot_drawer import matplot_Dynamic_Stady_rho
from mydynalearn.drawer.utils.data_handler import EpochDataHandler

class MatplotController():
    def __init__(self,config,dynamics):
        self.config = config
        self.dynamics = dynamics
        self.epochs = config.dataset.epochs
        self.matplot_epochPerformance = DrawerMatplotEpochFigYtrureYpred(config,dynamics)

        self.matplot_Dynamic_Stady_rho = matplot_Dynamic_Stady_rho(config)
        self.epoch_data_handler = EpochDataHandler(config,dynamics)

    def matplot_draw_epoch(self):
        self.matplot_epochPerformance.split_subplots(self.epochs)
        # 遍历所有epoch
        for epoch_index in range(self.epochs):
            # 用matplotlib画出每个epoch的结果
            epochData = self.epoch_data_handler.load_epoch_data(epoch_index)
            kwags = self.epoch_data_handler.epochdata_datacur_2_dataT(epoch_index, epochData)
            self.matplot_epochPerformance.draw(**kwags)
        # 存储matplot图像。
        self.matplot_epochPerformance.save_fig()

    def analyze_epoch_data(self):
        self.epoch_data_handler.get_max_R()

    def matplot_dynamic(self, eff_infectionList,stady_rho_list):
        self.matplot_Dynamic_Stady_rho.draw(eff_infectionList,stady_rho_list)
