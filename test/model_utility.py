'''
    接口：测试模型的性能
        使用训练的模型参数预测真实网络的数据

'''
import itertools
from mydynalearn.experiments import ExperimentManagerRealnet,ExperimentManagerTrain
from mydynalearn.analyze import AnalyzeTrainedModelToRealnet

NUM_SAMPLES = 10000
TESTSET_TIMESTEP = 50
EPOCHS = 30  # 10


'''
network = ["ER","SCER","CONFERENCE","HIGHSCHOOL","HOSPITAL","WORKPLACE"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU"]
dataset = ["GAT","SAT","DiffSAT"]
'''
params = {
    "grpah_network" : ["ER"],
    "grpah_dynamics" : ["UAU"],

    "simplicial_network" : ["SCER"],
    "real_network" : ["CONFERENCE"],
    "simplicial_dynamics" : ["SCUAU"],
    "model" : ["GAT","SAT","DiffSAT"],
    "IS_WEIGHT" : [False]
}

if __name__ == '__main__':
    # 训练模型
    train_experiment_manager = ExperimentManagerTrain(NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params)
    train_experiment_manager_realnet = ExperimentManagerRealnet(NUM_SAMPLES, TESTSET_TIMESTEP, EPOCHS, params)
    analyze_trained_model_to_realnet = AnalyzeTrainedModelToRealnet(train_experiment_manager, train_experiment_manager_realnet)

    train_experiment_manager.run()
    train_experiment_manager_realnet.run()

    analyze_trained_model_to_realnet.run()


