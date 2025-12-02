import numpy as np
import random 

# [QUAN TRỌNG] Giữ tham số theo bài báo
population = 100

def tournament_selection(num_candidates, pool_size):
    """
    Chọn lọc cạnh tranh (Tournament Selection) dựa trên chỉ số.
    Giả định: Quần thể đầu vào ĐÃ ĐƯỢC SẮP XẾP theo fitness (Tốt nhất ở index 0).
    Hàm này sẽ ưu tiên chọn index nhỏ.
    """
    candidates = [random.randint(0, pool_size - 1) for _ in range(num_candidates)]
    
    return min(candidates)

def SBX_crossover(P, T, lower, upper, ETA_C, ETA_M):
    
    # Chuẩn bị Parent Pool
    if len(T) == 0:
        parent = np.array(P)
    else:
        P = np.array(P)
        T = np.array(T)
        parent = np.concatenate((P, T), axis=0)
    
    child = []
    pool_size = len(parent)
    
    cnt = 0
    while cnt < population:
        # [SỬA LỖI] Dùng Tournament Selection thay vì random hoàn toàn
        # Chọn 2 ứng viên ngẫu nhiên, lấy người tốt hơn (index nhỏ hơn) làm cha/mẹ
        p1_idx = tournament_selection(3, pool_size)
        p2_idx = tournament_selection(3, pool_size)
        
        # Đảm bảo 2 cha mẹ khác nhau
        while p1_idx == p2_idx:
            p2_idx = tournament_selection(3, pool_size)
        
        parent1 = parent[p1_idx]
        parent2 = parent[p2_idx]
        
        # --- Logic SBX giữ nguyên ---
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        for i in range(len(parent1)):
            if np.random.rand() <= 0.5: # 50% swap variable
                random_index = np.random.uniform(0, 1)
                B = 0.0
                if random_index <= 0.5:
                    B = (2 * random_index) ** (1 / (ETA_C + 1))
                else:
                    B = (1 / (2 * (1 - random_index))) ** (1 / (ETA_C + 1))
                
                v1 = 0.5 * ((1 + B) * parent1[i] + (1 - B) * parent2[i])
                v2 = 0.5 * ((1 - B) * parent1[i] + (1 + B) * parent2[i])
                child1[i] = v1
                child2[i] = v2
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        # Clip biên
        child1 = np.clip(child1, lower, upper)
        child2 = np.clip(child2, lower, upper)
            
        child.append(child1)
        child.append(child2)
        cnt = len(child)
    
    # Polynomial Mutation (Giữ nguyên)
    for i in range(len(child)):
        for j in range(len(child[i])):
            r = np.random.uniform(0, 1)
            if r < 1.0 / len(child[i]):
                r1 = np.random.uniform(0, 1)
                delta = 0.0
                if r1 < 0.5:
                    delta = (2 * r1) ** (1 / (ETA_M + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r1)) ** (1 / (ETA_M + 1))
                
                mutated_value = child[i][j] + delta * (upper - lower)
                child[i][j] = np.clip(mutated_value, lower, upper)
                
    return child[:population]