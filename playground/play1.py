import torch

# 定义张量 a 和 b
node = torch.tensor([2, 1, 3, 5, 4])
node_id = torch.arange(node.shape[0])
edges = torch.tensor([[2, 1], [3, 5], [4,1]])
triangles = torch.tensor([[2, 1, 3], [3,1, 5], [4,1,3]])

# 使用广播和逻辑比较找到索引
edges_to_index = torch.where(edges.view(-1,1)==node.view(1,-1))[1]
edges_to_index.view(edges.shape)
# 打印索引
print(edges_to_index)