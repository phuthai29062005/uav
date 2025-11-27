
import numpy as np

def get_squared_euclidean_distance(X, Y):
    """
    Tính bình phương khoảng cách Euclid giữa hai ma trận X và Y bằng Numpy thuần.
    Thay thế cho scipy.spatial.distance.cdist(..., metric='sqeuclidean')
    """
    # 1. Tính tổng bình phương từng hàng của X
    X_sq = np.sum(np.square(X), axis=1) 
    
    # 2. Tính tổng bình phương từng hàng của Y
    Y_sq = np.sum(np.square(Y), axis=1) 
    
    # 3. Tính tích vô hướng X * Y.T
    XY = np.dot(X, Y.T) 
    
    # 4. Tổng hợp lại: (a-b)^2 = a^2 + b^2 - 2ab
    # Dùng broadcasting để cộng trừ ma trận
    Dists = X_sq[:, np.newaxis] + Y_sq - 2 * XY
    
    # Ép các giá trị âm cực nhỏ (do sai số máy tính) về 0
    Dists = np.maximum(Dists, 0)
    
    return Dists

# --- CÁC HÀM LOGIC CHÍNH CỦA MTS ---

def gaussian_kernel(X, Y, sigma=1.0):
    """
    Tính Gaussian Kernel (Eq. 10)
    """
    # Sử dụng hàm tự viết thay vì scipy
    dists_sq = get_squared_euclidean_distance(X, Y)
    
    # Tính kernel
    K = np.exp(-dists_sq / (2 * sigma**2))
    return K

def calculated_mmd(P_source, P_target, sigma):
    
    """
    Tính MMD giữa hai quần thể (Eq. 9)
    Input:
        P_source: Quần thể nguồn (trong không gian thống nhất)
        P_target: Quần thể đích (trong không gian thống nhất)
    """
    m = len(P_source)
    n = len(P_target)
    
    # 1. Term 1: Kernel giữa Source và Source
    K_xx = gaussian_kernel(P_source, P_source, sigma)
    term1 = (1.0 / (m**2)) * np.sum(K_xx)
    
    # 2. Term 2: Kernel giữa Source và Target
    K_xy = gaussian_kernel(P_source, P_target, sigma)
    term2 = (2.0 / (m * n)) * np.sum(K_xy)
    
    # 3. Term 3: Kernel giữa Target và Target
    K_yy = gaussian_kernel(P_target, P_target, sigma)
    term3 = (1.0 / (n**2)) * np.sum(K_yy)
    
    mmd_sq = term1 - term2 + term3
    
    return mmd_sq
    
    
    


def MTS(target_idx, P, num_tasks, l_transfer=5, sigma=1.0):
    
    
    mmd_value = []
    
    P_target = P[target_idx]
    
    for k in range(1, num_tasks + 1):
        if k == target_idx:
            continue
        
        P_source = P[k]
        mmd_val = calculated_mmd(P_source, P_target, sigma)
        mmd_value.append((mmd_val, k))
        
    mmd_value.sort(key=lambda x: x[0])
    num_select = min(len(mmd_value), l_transfer)
    L_c = [item[1] for item in mmd_value[:num_select]]
    
    return L_c
    