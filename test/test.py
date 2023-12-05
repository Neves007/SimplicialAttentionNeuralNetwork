import torch

# 输入数组
input_array = torch.tensor([1, 5])

# 获取唯一元素
unique_elements = torch.unique(input_array)

# 计算分布
distribution = torch.bincount(input_array)

# 打印结果
print(unique_elements)
print(distribution)