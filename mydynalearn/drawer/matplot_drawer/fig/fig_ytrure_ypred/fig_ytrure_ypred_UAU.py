from mydynalearn.drawer.matplot_drawer.fig.fig_ytrure_ypred import FigYtrureYpred
class FigYtrureYpredUAU(FigYtrureYpred):
    def __init__(self,ax,dynamics):
        super().__init__(ax,dynamics)
        self.colors = ["green", "red", "b", "orange"]
        self.markers = ["o", "o","o", "o"]
        self.label = ["U to U", "U to A", "A to U", "A to A"]
