import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def netsimplicial_preferential(N, ntri, device='cpu'):
    # Initialize values
    mlinks = 2 * ntri
    N0 = mlinks + 1

    # Create initial adjacency matrix on cuda
    A0 = torch.ones((N0, N0), device=device)
    A0 = A0 - torch.diag(torch.diag(A0))
    A = torch.zeros((N, N), device=device)
    A[:N0, :N0] = A0

    # 三角参与矩阵，元素表示边（i，j）上都多少个三角形
    Dij = A * (A @ A)

    # plink：Dij的稀疏表示
    # Extract upper triangular part of Dij and find non-zero elements
    TDij = torch.triu(Dij)
    non_zero_indices = torch.nonzero(TDij > 0)
    plink = torch.cat([non_zero_indices, TDij[non_zero_indices[:, 0], non_zero_indices[:, 1]].view(-1, 1)], dim=1)

    # 由plink的度分布决定的累积概率
    pcum = torch.cat([torch.tensor([0], device=device), torch.cumsum(plink[:, 2] / torch.sum(plink[:, 2]), dim=0)])

    # 添加新节点
    for n in range(N0, N):
        A[n, n] = 0
        l = len(plink)
        isw = 0

        # 附着
        # 将所有节点附着到ntri个连边上
        while isw == 0:
            # Generate ntri random numbers
            dummy = torch.rand(ntri, device=device)

            # 根据累积概率向量 (pcum) 和随机生成的值 (dummy)，确定哪些边被选中。
            diffs = pcum.view(-1, 1) - dummy.view(1, -1)  # 计算累积概率 pcum 和随机生成的值 dummy 之间的差异。
            temp = torch.diff(torch.sign(diffs), dim=0)  # 沿着维度 0（行）计算相邻元素之间的差异。
            idx = torch.nonzero(temp != 0)[:, 0] # 确定符号变化的行索引（即找到随机值在哪个累积概率区间中）。

            # 确保选择的ntri个连边里没有重复节点
            if len(torch.unique(plink[idx, :2])) == 2 * ntri:
                isw = 1

        # 更新邻接信息A and Dij
        for k in range(len(idx)):
            inode = int(plink[idx[k], 0].item())
            jnode = int(plink[idx[k], 1].item())

            A[n, inode] = 1
            A[n, jnode] = 1
            A[inode, n] = 1
            A[jnode, n] = 1

            # Update Dij
            Dij[n, inode] = 1
            Dij[n, jnode] = 1
            Dij[inode, n] = 1
            Dij[jnode, n] = 1
            Dij[inode, jnode] += 1
            Dij[jnode, inode] += 1

            # 更新 plink：添加两条新的边，并且连边k上多一个三角形
            plink = torch.cat([plink, torch.tensor([[n, inode, 1], [n, jnode, 1]], device=device)], dim=0)
            plink[idx[k], 2] = Dij[inode, jnode].item()

        # Update cumulative probability vector
        # 由plink的度分布更新的累积概率
        pcum = torch.cat([torch.tensor([0], device=device), torch.cumsum(plink[:, 2] / torch.sum(plink[:, 2]), dim=0)])
    return A, Dij
def plot_degree_distributions(A, Dij):
    # Calculate the degree distribution
    degrees = A.sum(dim=0).cpu().numpy()
    triangle_degrees = Dij.sum(dim=0).cpu().numpy()

    # Calculate the frequency of each degree
    degree_counts = np.bincount(degrees.astype(int))
    triangle_degree_counts = np.bincount(triangle_degrees.astype(int))

    # Filter out zero counts for log-log plot
    nonzero_degrees = np.nonzero(degree_counts)[0]
    nonzero_degree_counts = degree_counts[nonzero_degrees]

    nonzero_triangle_degrees = np.nonzero(triangle_degree_counts)[0]
    nonzero_triangle_degree_counts = triangle_degree_counts[nonzero_triangle_degrees]

    # Plot the degree distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.loglog(nonzero_degrees, nonzero_degree_counts, 'bo-', markersize=5)
    plt.title('Degree Distribution (Log-Log)')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')

    # Plot the triangle degree distribution
    plt.subplot(1, 2, 2)
    plt.loglog(nonzero_triangle_degrees, nonzero_triangle_degree_counts, 'ro-', markersize=5)
    plt.title('Triangle Degree Distribution (Log-Log)')
    plt.xlabel('Triangle Degree')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example of usage
N = 10000  # Number of nodes
ntri = 2  # Number of triangles
A,Dij = netsimplicial_preferential(N, ntri)
plot_degree_distributions(A, Dij)