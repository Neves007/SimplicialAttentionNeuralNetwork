from mydynalearn.drawer.matplot_drawer.fig.fig_beta_rho.Fig_beta_rho import *
from mydynalearn.drawer.matplot_drawer.drawer_matplot import MatplotDrawer

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

