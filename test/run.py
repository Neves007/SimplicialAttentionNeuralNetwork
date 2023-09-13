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
    # "grpah_network" : ["ER"],
    # "grpah_dynamics" : ["UAU","CompUAU"],

    "simplicial_network" : ["SCER"],
    "real_network" : ["CONFERENCE","HIGHSCHOOL","HOSPITAL","WORKPLACE"],
    "simplicial_dynamics" : ["SCUAU","SCCompUAU"],
    "model" : ["GAT","SAT","DiffSAT"],
    "is_weight" : [True,False]
}

if __name__ == '__main__':
    experiment_manager_train = ExperimentManagerTrain(num_samples, testset_timestep, epochs, params)
    experiment_manager_realnet = ExperimentManagerRealnet(num_samples, testset_timestep, epochs, params)
    analyze_trained_model_to_realnet = AnalyzeTrainedModelToRealnet(experiment_manager_train, experiment_manager_realnet)

    experiment_manager_train.train_model()
    experiment_manager_realnet.create_realnet_dynamics()

    analyze_trained_model_to_realnet.apply_trained_model_to_realnet()


