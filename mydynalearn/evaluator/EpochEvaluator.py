from mydynalearn.evaluator import Evaluator
import pickle
from mydynalearn.drawer.FigureDrawer import FigureDrawer

class EpochEvaluator(Evaluator):
    def __init__(self,config):
        super().__init__(config)
        self.epochs = config.train_details.epochs
        self.figureDrawer = FigureDrawer(config)
        pass
    # 评估epoch的结果
    def _evaluate_epochData(self):
        self.figureDrawer.matplot_epochPerformance.split_subplots(self.epochs)
        # 遍历所有epoch
        for epoch_index in range(self.epochs):
            # 用matplotlib画出每个epoch的结果
            self.figureDrawer.matplotDrawEpoch(epoch_index)
        # 存储matplot图像。
        self.figureDrawer.matplot_epochPerformance.saveFig()

    def evaluate(self):
        self._evaluate_epochData()
