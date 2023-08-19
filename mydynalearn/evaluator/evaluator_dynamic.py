from mydynalearn.evaluator import Evaluator
from mydynalearn.drawer import MatplotController
import pickle

class DynamicEvaluator(Evaluator):
    def __init__(self,config,dynamics):
        super().__init__(config)
        self.epochs = config.dataset.epochs
        self.matplot_controller = MatplotController(config,dynamics)
        pass
    # 评估epoch的结果
    def _evaluate_dynamic(self,eff_infectionList,stady_rho_list):
        self.matplot_controller.matplotDynamic(eff_infectionList,stady_rho_list)

    def evaluate(self,eff_infectionList,stady_rho_list):
        self._evaluate_dynamic(eff_infectionList,stady_rho_list)
