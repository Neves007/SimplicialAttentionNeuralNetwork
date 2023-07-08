import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DynamicDataset(Dataset):
    def __init__(self,dataset) -> None:
        self.x = dataset['x']
        self.y=dataset['y']
        self.len = x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len
x = np.random.random((100,4))
y = np.random.random((100,3))
dataset = {'x':x,'y':y}
dataset = DynamicDataset(dataset)
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4)
if __name__ == '__main__':
    for i, data in enumerate(train_loader):
        x,y = data
        print(x)
