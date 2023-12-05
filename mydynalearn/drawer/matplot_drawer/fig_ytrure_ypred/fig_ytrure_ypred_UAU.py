from mydynalearn.drawer.matplot_drawer.fig_ytrure_ypred import FigYtrureYpred
class FigYtrureYpredUAU(FigYtrureYpred):
    def __init__(self,dynamics):
        super().__init__(dynamics)
        self.colors = ["green", "red", "b", "orange"]
        self.markers = ["o", "o","o", "o"]
        self.label = ["U to U", "U to A", "A to U", "A to A"]
