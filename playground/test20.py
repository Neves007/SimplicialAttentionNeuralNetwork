import torch

# 假设 old_x0 和 new_x0 已经是正确的形状 (10, 4)，并且是 one-hot 编码
old_x0 = torch.randint(0, 2, (10, 4)).float()
new_x0 = torch.randint(0, 2, (10, 4)).float()

# 确保是正确的one-hot编码（这里仅做演示，实际情况应确保输入数据已正确）
old_x0 = torch.nn.functional.one_hot(torch.argmax(old_x0, dim=1), num_classes=4).float()
new_x0 = torch.nn.functional.one_hot(torch.argmax(new_x0, dim=1), num_classes=4).float()

# 将 one-hot 编码转换为单个整数标签，方便统计
old_labels = torch.argmax(old_x0, dim=1)
new_labels = torch.argmax(new_x0, dim=1)

# 计算迁移类别（例如，从状态3到状态1可以编码为 3*4+1 = 13）
transitions = old_labels * 4 + new_labels

# 计算每种迁移的发生频率
transition_counts = torch.zeros(16)  # 因为有16种可能的迁移（从0到15）
for t in transitions:
    transition_counts[t] += 1
transition_probs = transition_counts / transition_counts.sum()

# 将每个节点的迁移映射到其对应的频率
node_transition_probs = transition_probs[transitions]

# 输出结果
print(node_transition_probs.view(10, 1))
