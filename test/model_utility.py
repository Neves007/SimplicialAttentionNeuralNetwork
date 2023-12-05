'''
    接口：测试模型的性能
        使用训练的模型参数预测真实网络的数据

'''
import itertools
from mydynalearn.experiments import ExperimentManagerRealnet,ExperimentManagerTrain
from mydynalearn.analyze import AnalyzeTrainedModelToRealnet

num_samples = 10000
testset_timestep = 50
epochs = 30  # 10


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
    "is_weight" : [False]
}

if __name__ == '__main__':
    # 训练模型
    experiment_manager_train = ExperimentManagerTrain(num_samples, testset_timestep, epochs, params)
    experiment_manager_realnet = ExperimentManagerRealnet(num_samples, testset_timestep, epochs, params)
    analyze_trained_model_to_realnet = AnalyzeTrainedModelToRealnet(experiment_manager_train, experiment_manager_realnet)

    experiment_manager_train.run()
    experiment_manager_realnet.run()

    analyze_trained_model_to_realnet.run()


