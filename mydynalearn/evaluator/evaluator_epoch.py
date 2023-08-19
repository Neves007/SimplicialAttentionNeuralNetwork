from mydynalearn.evaluator import Evaluator
from mydynalearn.drawer import MatplotController
import pickle

class evaluatorEpoch(Evaluator):
    def __init__(self,config,dynamics):
        super().__init__(config)
        self.epochs = config.dataset.epochs
        self.matplot_controller = MatplotController(config,dynamics)
        pass
    # 评估epoch的结果
    def evaluate(self):
        self.matplot_controller.matplot_epochPerformance.split_subplots(self.epochs)
        # 遍历所有epoch
        for epoch_index in range(self.epochs):
            # 用matplotlib画出每个epoch的结果
            self.matplot_controller.matplotDrawEpoch(epoch_index)
        # 存储matplot图像。
        self.matplot_controller.matplot_epochPerformance.save_fig()
