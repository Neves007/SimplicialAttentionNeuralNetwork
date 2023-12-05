

class MatplotDrawer:
    def __init__(self,config):
        self.config = config
        self.path_to_fig = config.figpath_to_epoch_performance_fig_ytrure_ypred_1
        self.datapath_to_epochdata = config.datapath_to_epochdata
        self.figName = None

