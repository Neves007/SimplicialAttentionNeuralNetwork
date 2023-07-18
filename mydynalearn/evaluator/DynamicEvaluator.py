from mydynalearn.evaluator import Evaluator
import pickle
from mydynalearn.drawer.FigureDrawer import FigureDrawer

class DynamicEvaluator(Evaluator):
    def __init__(self,config):
        super().__init__(config)
        self.epochs = config.dataset.epochs
        self.figureDrawer = FigureDrawer(config)
        pass
    # 评估epoch的结果
    def _evaluate_dynamic(self,eff_infectionList,stady_rho_list):
        self.figureDrawer.matplotDynamic(eff_infectionList,stady_rho_list)

    def evaluate(self,eff_infectionList,stady_rho_list):
        self._evaluate_dynamic(eff_infectionList,stady_rho_list)
