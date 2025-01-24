import math
import numpy as np


def calculate_anchor_points(csr_train, a=0.3, b=0.5, c=0.2):
    num_U, num_V = csr_train.shape
    total_nodes = num_U + num_V
    print(total_nodes)
    S = np.log(total_nodes)
    print(S)
    total_edges = csr_train.nnz
    print(total_edges)
    C = 1 - (total_edges / (total_nodes * (total_nodes - 1)))
    C=total_edges/(num_U*num_V)
    print(C)
    N = np.log(num_U)
    print(N)


    degrees = csr_train.sum(axis=1)
    D = degrees.mean()
    print(D)


    total_score = S * C * a + N * b + D * c
    virtual_nodes = int(total_score)
    virtual_nodes=nearest_power_of_two(int(total_score))
    print('virtual_nodes =', virtual_nodes)
    return virtual_nodes

def nearest_power_of_two(num):

    power = round(math.log(num, 2))

    return 2 ** power


# def calculate_laplacian_matrix(G):
#     adjacency_matrix = nx.adjacency_matrix(G)
#     degree_matrix = scipy.sparse.diags(adjacency_matrix.sum(axis=1).A1, 0)
#     laplacian_matrix = degree_matrix - adjacency_matrix
#     return torch.tensor(laplacian_matrix.toarray(), dtype=torch.float32, device='cuda:0')



