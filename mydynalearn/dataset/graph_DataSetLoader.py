from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class graph_DataSetLoader():
    '''
    加载训练数据集
    每次加载一个时间步的结果。
    '''
    def __init__(self,data_set) -> None:
        self.data_set = data_set
        self.__len__ = self.data_set['x0_T'].shape[0]
        self.__index__ = 0
        pass
    def __getitem__(self,index):
        network = self.data_set.network
        x0_T = self.data_set.x0_T[index]
        x1_T = self.data_set.x1_T[index]
        y_ob_T = self.data_set.y_ob_T[index]
        y_true_T = self.data_set.y_true_T[index]
        adjActEdges_T = self.data_set.adjActEdges_T[index]
        weight = self.data_set.weight[index]
        return network, x0_T, x1_T, y_ob_T, y_true_T, adjActEdges_T, weight

    def getall_dataset(self):
        nodeFeature_T = self.data_set.nodeFeature_T
        y_ob_T = self.data_set.y_ob_T
        y_true_T = self.data_set.y_true_T
        return nodeFeature_T,y_ob_T,y_true_T