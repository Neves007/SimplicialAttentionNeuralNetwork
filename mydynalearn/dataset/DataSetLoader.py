from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class DataSetLoader():
    '''
    加载训练数据集
    每次加载一个时间步的结果。
    '''
    def __init__(self,data_set) -> None:
        self.data_set = data_set
        self.__len__ = self.data_set['nodeFeature_T'].shape[0]
        self.__index__ = 0
        pass
    def __getitem__(self,index):
        nodeFeature = self.data_set.nodeFeature_T[index]
        y_ob = self.data_set.y_ob_T[index]
        y_true = self.data_set.y_true_T[index]
        w = self.data_set.weight[index]
        edge_index = self.data_set.network.edge_index
        simplices_Dict = self.data_set.network.simplices_Dict
        simplices_incidence = self.data_set.network.simplices_incidence
        return nodeFeature,y_ob,y_true,edge_index,simplices_Dict,simplices_incidence,w

    def getall_dataset(self):
        nodeFeature_T = self.data_set.nodeFeature_T
        y_ob_T = self.data_set.y_ob_T
        y_true_T = self.data_set.y_true_T
        return nodeFeature_T,y_ob_T,y_true_T