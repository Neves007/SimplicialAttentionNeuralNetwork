import torch
from .fig import *
from mydynalearn.transformer import data_curEpoch_2_data_T
from mydynalearn.drawer.matplotDrawer.MatplotDrawer import MatplotDrawer
from mydynalearn.drawer.utils import *
import pickle
class matplot_Dynamic_Stady_rho(MatplotDrawer):
    def __init__(self,config):
        super().__init__(config)
        self.figName = "/matplot_Dynamic_Stady_rho.png"

    def draw(self, eff_infectionList, stady_rho_list):
        fig_beta_rho = Fig_beta_rho()
        fig_beta_rho.plot(eff_infectionList,stady_rho_list)
        plt.tight_layout()
        fig_name = self.path_to_fig + self.figName
        plt.savefig(fig_name)

