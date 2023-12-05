import mydynalearn as md
from mydynalearn.config import *

from .config import Config
import os



class DrawerConfig(Config):
    '''
    实验类基类用于初始化参数
    '''


    def default(
            self,
    ):
        self.NAME = "Drawer"
        self.rootpath = r"./output/fig/"
        self.model_performance_figdir = os.path.join(self.rootpath,"model_performance")
        self.realnet_performance_figdir = os.path.join(self.rootpath,"realnet_performance")
        if not os.path.exists(self.model_performance_figdir):
            os.makedirs(self.model_performance_figdir)
        if not os.path.exists(self.realnet_performance_figdir):
            os.makedirs(self.realnet_performance_figdir)
        return self