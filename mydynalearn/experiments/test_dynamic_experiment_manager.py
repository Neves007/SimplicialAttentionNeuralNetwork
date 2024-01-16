# 获取配置
from mydynalearn.config import ConfigDynamicTestingExp
from mydynalearn.experiments import ExperimentTestDynamic

from mydynalearn.util.params_dealer import PasramsDealer

class TestDynamicExperimentManager():
    def __init__(self,NUM_SAMPLES, params):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.params = params
        self.root_dir = r"./output/"

    def fix_config(self,config):
        '''调整配置
        '''
        # T总时间步
        config.dataset.NUM_SAMPLES = self.NUM_SAMPLES


    def get_test_dynamic_exp(self):
        '''通过参数获得实验对象

        :param network:
        :param dynamics:
        :return: exp
        '''
        network = self.params['network']
        dynamics = self.params['dynamics']
        exp_name = "testDynamic-" + network + "-" + dynamics
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "seed": 0,
            "root_dir": self.root_dir
        }
        config = ConfigDynamicTestingExp(**kwargs)
        self.fix_config(config)
        exp = ExperimentTestDynamic(config)
        return exp

    def run(self):
        '''
        训练模型
        输出：模型的参数 model_state_dict
        '''
        print("*"*10+" TRAINING PROCESS "+"*"*10)
        exp = self.get_test_dynamic_exp()
        exp.run()
        print("TRAINING PROCESS COMPLETED!\n\n")


