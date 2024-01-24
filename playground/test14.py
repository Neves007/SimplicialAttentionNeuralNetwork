import torch

# 创建一个概率分布张量，假设我们有一个离散分布
weights = torch.tensor([[0.1, 0.9],[0.2,0.8]])

想要对weights的每一行进行抽样，达到[torch.multinomial(row, 1, replacement=False) for row in weights]的目的，使用torch.multinomial该如何使用
