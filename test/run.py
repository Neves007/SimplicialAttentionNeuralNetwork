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

NUM_SAMPLES = 10000
TESTSET_TIMESTEP = 10
EPOCHS = 30  # 10

''' 所有参数
    "grpah_network" : ["ER"],
    "grpah_dynamics" : ["UAU","CompUAU"],

    "simplicial_network" : ["SCER"],
    "real_network" : ["CONFERENCE","HIGHSCHOOL","HOSPITAL","WORKPLACE"],
    "simplicial_dynamics" : ["SCUAU", "SCCompUAU"],

    
    "model" : ["GAT","SAT","DiffSAT"],  # 至少选一个
    "IS_WEIGHT" : [False]
'''
params = {
    # "grpah_network": ["ER"],
    # "grpah_dynamics": ["CompUAU"],

    "simplicial_network": ["SCER"],
    "real_network": ["CONFERENCE"],
    "simplicial_dynamics": ["SCCompUAU"],

    "model": ["SAT"],  # 至少选一个
    "IS_WEIGHT": [False]
}



if __name__ == '__main__':
    # todo: 合并ExperimentManagerTrain和ExperimentManagerRealnet
    experiment_manager_train = ExperimentManagerTrain(NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params)
    experiment_manager_realnet = ExperimentManagerRealnet(NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params)
    analyze_trained_model = AnalyzeTrainedModel(experiment_manager_train)
    analyze_trained_model_to_realnet = AnalyzeTrainedModelToRealnet(experiment_manager_train,
                                                               experiment_manager_realnet)


    # 训练
    experiment_manager_train.run()
    # 跑真实网络的测试数据
    experiment_manager_realnet.run()
    # 分析：测试集分析训练模型
    analyze_trained_model.run()
    # 分析：训练模型应用在真实网络.
    analyze_trained_model_to_realnet.run()



    # 画图：
    matplot_drawer = MatplotController(analyze_trained_model,analyze_trained_model_to_realnet)
    matplot_drawer.run()