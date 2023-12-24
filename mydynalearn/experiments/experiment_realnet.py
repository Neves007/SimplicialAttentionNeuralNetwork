from mydynalearn.experiments import ExperimentTrain

class ExperimentRealnet(ExperimentTrain):
    def __init__(self,config):
        super().__init__(config)
        self.TASKS = [
            "generate_data",
        ]