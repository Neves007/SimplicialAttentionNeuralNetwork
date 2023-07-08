import torch
def edge_to_node_matrix( edges, nodes, one_indexed=False):
    sigma1 = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)  # 索引从0开始
    # 边索引j
    j = 0
    # oriented
    for edge in edges:
        x, y = edge
        sigma1[x - offset][j] = 1
        sigma1[y - offset][j] = 1
        j += 1
    return sigma1