
class OptimizerConfig():

    def default(self):
        self.NAME = "RAdam"
        self.weight_decay = 1.0e-3
        self.lr = 1.0e-3
        self.betas = (0.9, 0.999)
        self.eps = 1.0e-8
        self.amsgrad = False
        return self
