class Evaluator():
    def __init__(self,config):
        self.config = config
        self.path_to_fig = config.path_to_fig
        self.path_to_checkpointsData = config.path_to_checkpoints_data
        self.path_to_epochData = config.path_to_epochdata
