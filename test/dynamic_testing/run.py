'''测试真实网络的动力学推断效果
    简介
        - 训练集
            - 网络：SCER
            - 动力学：
                - SCUAU
                - SCCompUAU
'''
from mydynalearn.experiments import *
from mydynalearn.analyze import *
from mydynalearn.drawer import MatplotController

NUM_SAMPLES = 10

''' 所有参数
    "grpah_network": ["ER"],
    "grpah_dynamics": ["UAU", "CompUAU"],

    "simplicial_network": ["SCER","CONFERENCE", "HIGHSCHOOL", "HOSPITAL", "WORKPLACE"],
    "simplicial_dynamics": ["SCUAU", "SCCompUAU"],
'''
params = {
    "network": "ER",
    "dynamics": "UAU",
}



if __name__ == '__main__':
    test_dynamic_experiment_manager = TestDynamicExperimentManager(NUM_SAMPLES, params)

    # # 训练
    test_dynamic_experiment_manager.run()

