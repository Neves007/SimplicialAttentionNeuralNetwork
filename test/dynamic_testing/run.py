import os
from mydynalearn.config import ExperimentConfig
from DynamicExperiment import DynamicExperiment

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


def getExperiment(name, network, dynamics, topology, weight):
    config = ExperimentConfig.default(
        name=name,
        network=network,
        dynamics=dynamics,
        nn_type=topology,
        is_weight=weight,
        seed=0
    )
    fix_config(config)
    exp = DynamicExperiment(config)
    return exp
# 获取配置

# T总时间步
num_samples = 2000
testSet_timestep = 10
epochs = 1 # 10
checkFirstEpoch = False # 10
checkFirstEpoch_max_time = 1000
checkFirstEpoch_timestep = 100


exp = getExperiment(name="dynamicTesting-sc_sis-sc_er-simplicial",
                    network="sc_er",
                    dynamics="sc_sis",
                    topology="simplicial",
                    weight=False)

exp.run()