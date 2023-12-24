import os


# 获取配置
from mydynalearn.config import ExperimentTrainConfig
from mydynalearn.experiments import ExperimentTrain


def fix_config(config):
    # T总时间步
    config.dataset.NUM_SAMPLES = NUM_SAMPLES
    config.dataset.NUM_TEST = TESTSET_TIMESTEP
    config.dataset.EPOCHS = EPOCHS  # 10
    # 检查点
    config.dataset.check_first_epoch = check_first_epoch  # 10
    config.dataset.check_first_epoch_maxtime = check_first_epoch_maxtime
    config.dataset.check_first_epoch_timestep = check_first_epoch_timestep
    config.set_path()
    
    
def get_experiment(NAME, network, dynamics, dataset, MODEL_NAME, weight):
    config = ExperimentTrainConfig.default(
        NAME=NAME,
        network=network,
        dynamics=dynamics,
        dataset=dataset,
        MODEL_NAME=MODEL_NAME,
        IS_WEIGHT=weight,
        seed=0
    )
    fix_config(config)
    exp = ExperimentTrain(config)
    return exp




NUM_SAMPLES = 100
TESTSET_TIMESTEP = 10
EPOCHS = 30 # 10
check_first_epoch = False # 10
check_first_epoch_maxtime = 1000
check_first_epoch_timestep = 100

exp = get_experiment(NAME="dynamicLearning-UAU-ER-GAT",
                     network="ER",
                     dynamics="UAU",
                     dataset="graph",
                     MODEL_NAME="graph",
                     weight=False)
exp.run()
#
# exp = getExperiment(NAME="dynamicLearning-SCUAU-SCER-GAT",
#                     network="SCER",
#                     dynamics="SCUAU",
#                     dataset="simplicial",
#                     MODEL_NAME="graph",
#                     weight=False)
# exp.run()
#
# exp = getExperiment(NAME="dynamicLearning-sc_UAU_comp-SCER-SAT",
#                     network="SCER",
#                     dynamics="sc_UAU_comp",
#                     dataset="simplicial",
#                     MODEL_NAME="simplicial",
#                     weight=False)
# exp.run()




