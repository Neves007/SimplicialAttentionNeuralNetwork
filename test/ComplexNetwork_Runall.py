import os


# 获取配置
from mydynalearn.config import ExperimentConfig
from mydynalearn.experiments import Experiment


def fix_config(config):
    # T总时间步
    config.dataset.num_samples = num_samples
    config.dataset.num_test = testSet_timestep
    config.dataset.epochs = epochs  # 10
    # 检查点
    config.dataset.checkFirstEpoch = checkFirstEpoch  # 10
    config.dataset.checkFirstEpoch_max_time = checkFirstEpoch_max_time
    config.dataset.checkFirstEpoch_timestep = checkFirstEpoch_timestep
    config.set_path()
    
    
def getExperiment(name, network, dynamics,dataset, nn_type , weight):
    config = ExperimentConfig.default(
        name=name,
        network=network,
        dynamics=dynamics,
        dataset=dataset,
        nn_type=nn_type,
        is_weight=weight,
        seed=0
    )
    fix_config(config)
    exp = Experiment(config)
    return exp




num_samples = 10000
testSet_timestep = 10
epochs = 30 # 10
checkFirstEpoch = False # 10
checkFirstEpoch_max_time = 1000
checkFirstEpoch_timestep = 100

exp = getExperiment(name="dynamicLearning-sis-er-GAT",
                    network="er",
                    dynamics="sis",
                    dataset="graph",
                    nn_type="graph",
                    weight=False)
exp.run()

exp = getExperiment(name="dynamicLearning-sc_sis-sc_er-GAT",
                    network="sc_er",
                    dynamics="sc_sis",
                    dataset="simplicial",
                    nn_type="graph",
                    weight=False)
exp.run()

exp = getExperiment(name="dynamicLearning-sc_sis-sc_er-SAT",
                    network="sc_er",
                    dynamics="sc_sis",
                    dataset="simplicial",
                    nn_type="simplicial",
                    weight=False)
exp.run()




