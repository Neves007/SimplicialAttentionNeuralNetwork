from mydynalearn.drawer.matplot_drawer.fig_ytrure_ypred import FigYtrureYpred
class FigYtrureYpredCompUAU(FigYtrureYpred):
    def __init__(self,dynamics):
        super().__init__(dynamics)
        self.colors = ["red","orange","yellow","green","cyan","blue","purple"]
        self.markers = ["o", "o","o", "o", "o","o", "o"]
        self.label = ["U to U",
            "U to A1",
            "U to A2",
            "A1 to A1",
            "A1 to U",
            "A2 to A2",
            "A2 to U"]