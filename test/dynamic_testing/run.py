'''测试真实网络的动力学推断效果
    简介
        - 训练集
            - 网络：SCER
            - 动力学：
                - SCUAU
                - SCCompUAU
'''
import torch

from mydynalearn.experiments import *
from mydynalearn.analyze import *
from mydynalearn.drawer import MatplotController



''' 所有参数
    "grpah_network": ["ER"],
    "grpah_dynamics": ["UAU", "CompUAU"],

    "simplicial_network": ["SCER","CONFERENCE", "HIGHSCHOOL", "HOSPITAL", "WORKPLACE"],
    "simplicial_dynamics": ["SCUAU", "SCCompUAU"],
'''
params_dict = {
    "grpah_network": ["ER"],
    "grpah_dynamics": ["CompUAU"],
}

fix_config_dict = {
    'AVG_K': 10,
    'NUM_SAMPLES' : 1000,
    'RECOVERY':0.2,
    'EFF_BETA_LIST': torch.linspace(0, 2, 21),
    'DEVICE': torch.device('cuda'),
    'ROUND': 10,
}



if __name__ == '__main__':
    test_dynamic_experiment_manager = TestDynamicExperimentManager(fix_config_dict, params_dict)

    # # 训练
    test_dynamic_experiment_manager.run()

