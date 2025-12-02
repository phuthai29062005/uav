import math
import numpy as np
from scipy.spatial.distance import cdist


def calculated_mmd(P, i, j, width):
    """
    Tính MMD giữa hai quần thể (Eq. 9)
    Input:
        P_source: Quần thể nguồn (trong không gian thống nhất)
        P_target: Quần thể đích (trong không gian thống nhất)
    """
    def sum_kernel_matrix(A, B, width):
        # 1. Tính bình phương khoảng cách Euclid giữa mọi cặp điểm (Pairwise Squared Euclidean)
        # Kết quả là ma trận kích thước (len(A) x len(B))
        sq_dists = cdist(A, B, metric='sqeuclidean')
        
        # 2. Áp dụng công thức Gaussian lên toàn bộ ma trận cùng lúc
        K_matrix = np.exp(-sq_dists / (2 * width**2))
        
        # 3. Tính tổng toàn bộ phần tử trong ma trận
        return np.sum(K_matrix)
    
    m = len(P[i])
    n = len(P[j])

    K_ii = sum_kernel_matrix(P[i], P[i], width)
    K_ij = sum_kernel_matrix(P[i], P[j], width)
    K_jj = sum_kernel_matrix(P[j], P[j], width)
    term1 = (1.0/(m ** 2)) * K_ii
    term2 = (2.0/(m * n)) * K_ij
    term3 = (1.0/(n ** 2)) * K_jj

    mmd = term1 - term2 + term3
    return mmd

def MTS(index, num_task, P, sources, width):

    mmd = []
    L = []
    for i in range(1, num_task + 1):
        if i != index:

            mmd_val = calculated_mmd(P, index, i, width)
            mmd.append((mmd_val, i))
        
    mmd.sort(key = lambda x: x[0])
    L = [item[1] for item in mmd[:sources]]

    return L
        