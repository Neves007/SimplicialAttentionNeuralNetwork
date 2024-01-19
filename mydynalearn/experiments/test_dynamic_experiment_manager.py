# 获取配置
from mydynalearn.config import ConfigDynamicTestingExp
from mydynalearn.experiments import ExperimentTestDynamic

from mydynalearn.util.params_dealer import PasramsDealer

class TestDynamicExperimentManager():
    def __init__(self, fix_config_dict, params_dict):
        self.fix_config_dict = fix_config_dict
        self.params_dict = params_dict
        self.root_dir = r"./output/"

    def fix_config(self,config):
        '''调整配置
        '''
        # T总时间步
        config.dataset.NUM_SAMPLES = self.fix_config_dict['NUM_SAMPLES']
        config.DEVICE = self.fix_config_dict['DEVICE']
        config.EFF_BETA_LIST = self.fix_config_dict['EFF_BETA_LIST']



    def get_test_dynamic_exp(self):
        '''通过参数获得实验对象

        :param network:
        :param dynamics:
        :return: exp
        '''
        network = self.params_dict['network']
        dynamics = self.params_dict['dynamics']
        exp_name = "testDynamic-" + network + "-" + dynamics
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "seed": 0,
            "root_dir": self.root_dir
        }
        config = ConfigDynamicTestingExp(**kwargs)
        self.fix_config_dict(config)
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


