'''数据集的构建与读取
    使用Dataset构建数据集，实现__getitem__和__len__方法。
    使用random_split()将数据集切分成训练集和测试集
    使用Dataloader分批次读取数据集
'''
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class Mydataset(Dataset):
    '''数据集类
        必须实现__getitem__和__len__方法
    '''
    def __init__(self):
        self.NUM_SAMPLES = 20
        self.dataset = {
            'x': torch.arange(self.NUM_SAMPLES),
            'y': torch.arange(self.NUM_SAMPLES),
        }

    def __getitem__(self, index) -> T_co:
        x = self.dataset['x'][index]
        y = self.dataset['y'][index]
        return x,y

    def __len__(self) -> int:
        return self.NUM_SAMPLES



# 切分数据集
mydataset = Mydataset()
train_size = int(0.8 * len(mydataset))
test_size = len(mydataset) - train_size
train_dataset, test_dataset  = torch.utils.data.random_split(mydataset, [train_size, test_size])
print(train_dataset)
print(test_dataset)


# Dataloader 读取数据
batch_size = 2
dataset_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
for index,data in enumerate(dataset_loader):
    print("batch: {0:d}  data: ".format(index),data)

