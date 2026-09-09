# file: sorting.py
import numpy as np

def dominates(a, b):
    """
    a có dominate b không?
    a, b: vectors mục tiêu, shape (M,)
    a dominate b khi: tất cả a[i] <= b[i] VÀ ít nhất 1 a[i] < b[i]
    """
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sorting(F):
    """
    Phân hạng Pareto cho toàn bộ population
    F: ma trận objectives, shape (N, M)
    return: list of lists, fronts[0] = rank 1 (tốt nhất), ...
    """
    N = len(F)
    domination_count = np.zeros(N, dtype=int)   # số cá thể dominate i
    dominated_set = [[] for _ in range(N)]       # tập cá thể bị i dominate
    fronts = [[]]
    
    for i in range(N):
        for j in range(i + 1, N):
            if dominates(F[i], F[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(F[j], F[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
    
    # Front đầu tiên: các cá thể không bị ai dominate
    for i in range(N):
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Các front tiếp theo
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    
    fronts.pop()  # bỏ front rỗng cuối
    return fronts


def crowding_distance(F, front):
    """
    Tính crowding distance cho 1 front
    F: ma trận objectives toàn bộ population, shape (N, M)
    front: list các index thuộc front này
    return: dict {index: distance}
    """
    n = len(front)
    if n <= 2:
        return {idx: float('inf') for idx in front}
    
    M = F.shape[1]  # số objectives
    distances = {idx: 0.0 for idx in front}
    
    for m in range(M):
        # Sort front theo objective m
        sorted_front = sorted(front, key=lambda i: F[i, m])
        
        # Biên luôn được ưu tiên
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        # Range của objective m
        f_range = F[sorted_front[-1], m] - F[sorted_front[0], m]
        if f_range < 1e-14:
            continue
        
        # Các cá thể ở giữa
        for i in range(1, n - 1):
            distances[sorted_front[i]] += (
                (F[sorted_front[i + 1], m] - F[sorted_front[i - 1], m])
                / f_range
            )
    
    return distances


# === TEST ===
if __name__ == "__main__":
    np.random.seed(30)
    
    from pymoo.problems.dynamic.df import DF1
    
    problem = DF1(time=0.0, n_var=10)
    Pop = np.random.rand(20, 10)  # 20 cá thể để dễ nhìn
    F = problem.evaluate(Pop)
    
    # Non-dominated sorting
    fronts = non_dominated_sorting(F)
    print(f"Số fronts: {len(fronts)}")
    for i, front in enumerate(fronts):
        print(f"  Front {i}: {len(front)} cá thể, indices = {front}")
    
    # Crowding distance cho front 0
    cd = crowding_distance(F, fronts[0])
    print(f"\nCrowding distance (Front 0):")
    for idx, dist in sorted(cd.items(), key=lambda x: -x[1]):
        print(f"  Individual {idx}: CD = {dist:.4f}, "
              f"f1={F[idx,0]:.4f}, f2={F[idx,1]:.4f}")