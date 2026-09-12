import numpy as np
from pymoo.problems.dynamic.df import DF1

Generation = 3000
Seed = 30
N = 100     # population size
D = 10      # số biến
n_t = 10    # severity
tau_t = 10  # frequency
n_changes = 100  # số lần đổi môi trường

if __name__ == "__main__":
    np.random.seed(Seed)
    
    # Khởi tạo population
    Parent = np.random.rand(N, D)
    
    # Vòng lặp chính
    for change_idx in range(n_changes):
        t = change_idx / n_t
        problem = DF1(time=t, n_var=D)
        
        # Evaluate
        F = problem.evaluate(Parent)
        
        # ---- Sau này thêm vào đây: ----
        # 1. Non-dominated sorting
        # 2. Selection, crossover, mutation
        # 3. Dynamic response khi change_idx > 0
        # 4. Tính MIGD so với problem.pareto_front()
        
        if change_idx % 20 == 0:
            print(f"Change {change_idx}, t={t:.2f}, "
                  f"best f1={F[:,0].min():.4f}, best f2={F[:,1].min():.4f}")