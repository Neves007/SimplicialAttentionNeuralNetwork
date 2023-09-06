from mydynalearn.evaluator import Evaluator
from mydynalearn.drawer import MatplotController
import pickle
import torch

class evaluatorEpoch(Evaluator):
    def __init__(self,config,dynamics):
        super().__init__(config)
        self.matplot_controller = MatplotController(config,dynamics)
        pass
    # 评估epoch的结果
    
    # 画
    def draw_epoch_performance(self):
        self.matplot_controller.matplot_draw_epoch()

    def analyze_epoch_performance(self):
        self.matplot_controller.analyze_epoch_data()
        
    def evaluate(self):
        self.analyze_epoch_performance()
        self.draw_epoch_performance()

    def draw_maxR(self):
        self.matplot_controller.matplot_draw_maxR()
