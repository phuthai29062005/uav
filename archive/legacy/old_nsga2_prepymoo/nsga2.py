# file: nsga2.py
import numpy as np
from operators import sbx_crossover, polynomial_mutation
from sorting import non_dominated_sorting, crowding_distance


def tournament_selection(pop, F, fronts, crowd_dist):
    """
    Binary tournament: chọn 1 cá thể
    Ưu tiên: rank thấp hơn → thắng
             cùng rank → crowding distance cao hơn → thắng
    """
    N = len(pop)
    
    # Tạo bảng rank cho mỗi cá thể
    rank = np.zeros(N, dtype=int)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r
    
    # Chọn 2 ngẫu nhiên, so sánh
    i, j = np.random.randint(0, N, size=2)
    
    if rank[i] < rank[j]:
        return i
    elif rank[i] > rank[j]:
        return j
    else:
        # Cùng rank → so crowding distance
        if crowd_dist.get(i, 0) > crowd_dist.get(j, 0):
            return i
        else:
            return j


def make_offspring(pop, F, fronts, crowd_dist, N):
    """
    Tạo N offspring từ population hiện tại
    """
    offspring = []
    
    while len(offspring) < N:
        # Chọn 2 cha mẹ
        p1_idx = tournament_selection(pop, F, fronts, crowd_dist)
        p2_idx = tournament_selection(pop, F, fronts, crowd_dist)
        
        # Crossover
        c1, c2 = sbx_crossover(pop[p1_idx], pop[p2_idx])
        
        # Mutation
        c1 = polynomial_mutation(c1)
        c2 = polynomial_mutation(c2)
        
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)
    
    return np.array(offspring)


def environmental_selection(pop, F, N):
    """
    Chọn N cá thể tốt nhất từ population 2N
    Theo: rank trước, crowding distance sau
    """
    fronts = non_dominated_sorting(F)
    
    selected = []
    selected_F = []
    
    for front in fronts:
        if len(selected) + len(front) <= N:
            # Cả front vừa đủ → lấy hết
            for idx in front:
                selected.append(pop[idx])
                selected_F.append(F[idx])
        else:
            # Front này quá đông → chọn theo crowding distance
            cd = crowding_distance(F, front)
            sorted_front = sorted(front, key=lambda i: -cd[i])
            
            remain = N - len(selected)
            for idx in sorted_front[:remain]:
                selected.append(pop[idx])
                selected_F.append(F[idx])
            break
    
    return np.array(selected), np.array(selected_F)


def nsga2_one_generation(pop, F_pop, problem, N):
    """
    Chạy 1 generation NSGA-II
    Input:  pop (N, D), F_pop (N, M) hiện tại
    Output: pop mới (N, D), F mới (N, M)
    """
    # 1. Sorting + crowding cho population hiện tại
    fronts = non_dominated_sorting(F_pop)
    crowd_dist = {}
    for front in fronts:
        cd = crowding_distance(F_pop, front)
        crowd_dist.update(cd)
    
    # 2. Tạo N offspring
    offspring = make_offspring(pop, F_pop, fronts, crowd_dist, N)
    
    # 3. Evaluate offspring
    F_off = problem.evaluate(offspring)
    
    # 4. Gộp parent + offspring (2N)
    combined = np.vstack([pop, offspring])
    F_combined = np.vstack([F_pop, F_off])
    
    # 5. Chọn N tốt nhất
    new_pop, new_F = environmental_selection(combined, F_combined, N)
    
    return new_pop, new_F


# === TEST: Chạy NSGA-II trên DF1 tĩnh ===
if __name__ == "__main__":
    from pymoo.problems.dynamic.df import DF1
    from pymoo.indicators.igd import IGD
    
    np.random.seed(30)
    
    N = 100
    D = 10
    t = 0.5  # cố định t → bài toán tĩnh
    n_gens = 200
    
    problem = DF1(time=t, n_var=D)
    PF = problem.pareto_front()
    igd_calc = IGD(PF)
    
    # Khởi tạo
    pop = np.random.rand(N, D)
    F = problem.evaluate(pop)
    
    print(f"Gen   0: IGD = {igd_calc(F):.6f}")
    
    for gen in range(1, n_gens + 1):
        pop, F = nsga2_one_generation(pop, F, problem, N)
        
        if gen % 50 == 0:
            igd_val = igd_calc(F)
            print(f"Gen {gen:3d}: IGD = {igd_val:.6f}")
    
    final_igd = igd_calc(F)
    print(f"\nFinal IGD = {final_igd:.6f}")
    print("IGD < 0.01 = tot!" if final_igd < 0.01 else "Chua hoi tu het")
    
    # === Vẽ kết quả ===
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trước
    pop_init = np.random.rand(N, D)
    F_init = problem.evaluate(pop_init)
    axes[0].plot(PF[:, 0], PF[:, 1], 'r-', linewidth=2, label='True PF')
    axes[0].scatter(F_init[:, 0], F_init[:, 1], c='blue', s=15, alpha=0.5)
    axes[0].set_title(f'TRUOC (random) - IGD = {igd_calc(F_init):.4f}')
    axes[0].set_xlabel('f1')
    axes[0].set_ylabel('f2')
    axes[0].legend()
    axes[0].set_xlim(-0.1, 3)
    axes[0].set_ylim(-0.1, 3)
    
    # Sau
    axes[1].plot(PF[:, 0], PF[:, 1], 'r-', linewidth=2, label='True PF')
    axes[1].scatter(F[:, 0], F[:, 1], c='green', s=15, alpha=0.5)
    axes[1].set_title(f'SAU {n_gens} gens NSGA-II - IGD = {final_igd:.4f}')
    axes[1].set_xlabel('f1')
    axes[1].set_ylabel('f2')
    axes[1].legend()
    axes[1].set_xlim(-0.1, 3)
    axes[1].set_ylim(-0.1, 3)
    
    plt.tight_layout()
    plt.savefig("nsga2_test.png", dpi=150)
    plt.show()
    print("Da luu: nsga2_test.png")