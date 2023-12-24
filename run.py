from mydynalearn.experiments import ExperimentManagerRealnet,ExperimentManagerTrain
from mydynalearn.analyze import *
from mydynalearn.drawer import MatplotController

NUM_SAMPLES = 10000
TESTSET_TIMESTEP = 10
EPOCHS = 30  # 10


'''
network = ["ER","SCER","CONFERENCE","HIGHSCHOOL","HOSPITAL","WORKPLACE"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU"]
dataset = ["GAT","SAT","DiffSAT"]
'''
params = {
    "grpah_network" : ["ER"],
    "grpah_dynamics" : ["UAU","CompUAU"],

    "simplicial_network" : ["SCER"],
    "real_network" : ["CONFERENCE","HIGHSCHOOL","HOSPITAL","WORKPLACE"],
    "simplicial_dynamics" : ["SCUAU","SCCompUAU"],
    "model" : ["GAT","SAT","DiffSAT"],
    "IS_WEIGHT" : [True,False]
}


if __name__ == '__main__':
    experiment_manager_train = ExperimentManagerTrain(NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params)
    experiment_manager_realnet = ExperimentManagerRealnet(NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params)
    analyze_trained_model = AnalyzeTrainedModel(experiment_manager_train)
    analyze_trained_model_to_realnet = AnalyzeTrainedModelToRealnet(experiment_manager_train,
                                                               experiment_manager_realnet)


    # 训练模型
    experiment_manager_train.run()
    # 跑真实网络的测试数据
    experiment_manager_realnet.run()
    #
    # 分析：测试集分析训练模型
    analyze_trained_model.run()
    # 分析：训练模型应用在真实网络.
    analyze_trained_model_to_realnet.run()



    # 画图：
    matplot_drawer = MatplotController(analyze_trained_model,analyze_trained_model_to_realnet)
    matplot_drawer.run()



