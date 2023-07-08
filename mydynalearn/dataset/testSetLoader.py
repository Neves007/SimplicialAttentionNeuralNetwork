from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class testSetLoader():
    '''
    加载训练数据集
    每次加载一个时间步的结果。
    '''
    def __init__(self,data_set) -> None:
        self.data_set = data_set
    def gather_dataset(self):
        T_data_size = self.data_set['x_T'].shape
        T_data_gatherSize = (T_data_size[0]*T_data_size[1],T_data_size[2])
        x = self.data_set.x_T.view(T_data_gatherSize)
        y_ob = self.data_set.y_ob_T.view(T_data_gatherSize)
        y_true = self.data_set.y_true_T.view(T_data_gatherSize)
        edge_index = self.data_set.network.edge_index
        return x,y_ob,y_true,edge_index