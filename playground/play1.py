import torch

# 创建一个包含多个最大值的张量
tensor = torch.tensor([1, 2, 3, 3, 2, 1])

# 使用 torch.argmax() 获取最大值的索引
max_indices = torch.argmax(tensor)

# 打印最大值的索引
print(max_indices)