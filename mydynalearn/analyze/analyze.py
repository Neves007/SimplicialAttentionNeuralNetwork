import os
class Analyze():
    def __init__(self,config, exp, epoch_task):
        self.config = config
        self.exp = exp
        self.epoch_task = epoch_task
        self.epoch_index = epoch_task.epoch_index
        self.test_result_filepath = self.get_test_result_filepath()

    def get_test_result_filepath(self):
        rootpath = self.config.rootpath
        test_result_dir = self.config.test_result_dir
        exp_name = self.exp.NAME
        test_result_path = os.path.join(rootpath, test_result_dir, exp_name)


        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)

        test_result_filepath = os.path.join(test_result_path, "epoch{:d}_test_result.pkl".format(self.epoch_index))
        return test_result_filepath