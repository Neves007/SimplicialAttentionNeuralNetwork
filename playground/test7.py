'''ReduceLROnPlateau
学习率调整策略
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 创建数据集
X = torch.randn(100, 1)  # 输入特征
Y = 3 * X + 2 + torch.randn(100, 1)  # 目标变量

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x
def round_to_highest_decimal_place(value):
    value = float(value)
    if value == 0:
        return 1
    # 首先找到第一个不为零的小数位
    decimal_place = 0
    temp_value = value
    while temp_value < 1:
        temp_value *= 10
        decimal_place += 1
        # 如果找到了一个不为零的小数位，就中断循环
        if int(temp_value) % 10 != 0:
            break

    # 现在保留到这个小数位，但是仅保留这一位为1
    rounded_value = round(value, decimal_place)
    # 将保留的数字置为1
    rounded_value = 1 / (10 ** decimal_place)

    return rounded_value

# 创建模型实例和优化器
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建 ReduceLROnPlateau 学习率调整策略
scheduler = ReduceLROnPlateau(optimizer,
                               mode='max',
                               factor=0.1,
                               patience=4,
                               eps=1e-100,
                               threshold=0.1,
                               threshold_mode='abs',
                               verbose=True)

# 定义损失函数
criterion = nn.MSELoss()
log_space = torch.logspace(-5, 0, steps=100)
decreasing_diff_array = 1 - log_space.flip(0)  # 翻转数组以使其从0开始递增
loss_list = decreasing_diff_array
# loss_list = torch.linspace(0,1,101)
# 训练循环
for epoch in range(len(loss_list)):
    # 前向传播
    loss = loss_list[epoch]
    # 反向传播和优化
    optimizer.zero_grad()
    optimizer.step()

    # 在每个 epoch 结束后调用学习率调整策略
    scheduler.step(loss)

    # 输出当前学习率和损失
    print("Epoch: {}  Learning Rate: {:.1e}  Loss:{:.4f}".format(epoch,
                                                                 optimizer.param_groups[0]['lr'],
                                                                 loss.item(),))
