'''测试真实网络的动力学推断效果
    简介
        - 训练集
            - 网络：SCER
            - 动力学：
                - SCUAU
                - SCCompUAU
        - 测试集
            - 真实网络：
                - CONFERENCE
            - 真实网络上的模拟动力学
                - SCUAU
            - SCCompUAU
        - 输出：
            - 【训练数据】 output\data\train_model
                - 【数据集：包含训练集和测试集】  \output\data\train_model\dynamicLearning-SCER-SCCompUAU-DiffSAT\datasets
                - 【训练参数：分为加权和不加权】 output\data\train_model\dynamicLearning-SCER-SCCompUAU-DiffSAT\not_weight\modelResult\model_state_dict
            - 【真实网络文件】 output\data\realnet
            - 【分析数据】 \output\data\analyze
                - 【训练模型的分析数据】：output\data\analyze\trained_model
                    -【存储最大的R值：显示训练实验在哪个epoch得到最大的R值，依次判定模型的优劣】 ：output\data\analyze\trained_model\maxR\maxR.csv
                    -【训练数据的测试结果】output\data\analyze\trained_model\test_result
                - 【将训练的模型应用到真实网络的分析数据】：output\data\analyze\trained_model_to_realnet
                    -【真实数据的测试结果】output\data\analyze\trained_model_to_realnet\test_result
            - 【图片】：output\fig
                - 【训练模型测试集效果图片】：output\fig\model_performance
                - 【真实模型测试集效果图片】：output\fig\realnet_performance
'''
from mydynalearn.experiments import *
from mydynalearn.analyze import *
from mydynalearn.drawer import MatplotController

''' 所有参数
    "grpah_network": ["ER", "SF"],
    "grpah_dynamics": ["UAU", "CompUAU", "CoopUAU", "AsymUAU"],

    "simplicial_network": ["SCER", "SCSF", "CONFERENCE", "HIGHSCHOOL", "HOSPITAL", "WORKPLACE"],
    "simplicial_dynamics": ["SCUAU", "SCCompUAU", "SCCoopUAU", "SCAsymUAU"],

    "model": ["GAT", "SAT", "DiffSAT"],  # 至少选一个
    "IS_WEIGHT": [False]
'''

params = {
    "grpah_network": ["ER", "SF"],
    "grpah_dynamics": ["UAU",],

    "simplicial_network": ["SCER", "SCSF"],
    "simplicial_dynamics": ["SCUAU"],

    "model": ["GAT", "SAT"],  # 至少选一个
    "IS_WEIGHT": [False]
}
fix_config = {
    "NUM_SAMPLES": 100,
    "TESTSET_TIMESTEP": 10,
    "EPOCHS": 3,
    "DEVICE": torch.device('cuda'),
}


if __name__ == '__main__':
    # 训练
    train_params = PasramsDealer.assemble_train_params(params)  # 实验的设置
    train_experiment_manager = TrainExperimentManager(fix_config, train_params)  # 返回实验对象
    train_experiment_manager.run()

    # 分析：测试集分析训练模型
    exp_generator = list(train_experiment_manager.get_exp_generator())
    analyze_manager = AnalyzeManager(exp_generator)
    analyze_manager.run()

    # 画图：
    matplot_drawer = MatplotController(analyze_manager)
    matplot_drawer.run()


