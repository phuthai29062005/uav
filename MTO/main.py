import numpy as np
import random
import os 

from ATP import *
from MTS import *
from RDA import *
from benchmark import *
from SBX import *

num_task = 10
population = 100
generation = 1000
M = int(0.05 * generation)
if M < 2: M = 2

IDEAL_SOURCES = {
    5: [1], 6: [2], 7: [3, 4], 9: [4]
}

def log_to_file(task_id, generation, fitness):
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f"results/task{task_id}.txt"
    with open(filename, "a") as f:
        f.write(f"{generation} {fitness}\n")

def clear_old_logs():
    if not os.path.exists("results"):
        os.makedirs("results")
    for i in range(1, num_task + 1):
        with open(f"results/task{i}.txt", "w") as f:
            f.write("Generation BestFitness\n")

def write_file(Final_gen, good_task):
    with open("results/final_solutions.txt", "w") as f:
        for i in range(1, num_task + 1):
            f.write(f"Task T{i} Best Solution:\n")
            f.write(" ".join(map(str, Final_gen[i])) + "\n")
            f.write(f"Good Transfer Tasks Counts: {good_task[i][1:]}\n\n")   

# [SỬA] Hàm này giờ trả về cả (New Population, New Fitness List)
def get_best(num_task, P, child, T):
    shift, dim, bounds = get_task_info(f"T{num_task}")
    fitness_data = [] # Lưu tuple (cost, gen)
    
    P_arr = np.array(P)
    if T is None or len(T) == 0:
        parent = P_arr
    else:
        T_arr = np.array(T)
        if T_arr.ndim == P_arr.ndim:
            parent = np.concatenate((P_arr, T_arr), axis=0)
        else:
            parent = P_arr

    child_arr = np.array(child)
    if len(child_arr) > 0:
        if child_arr.ndim == parent.ndim:
            parent = np.concatenate((parent, child_arr), axis=0)
    
    for k in range(len(parent)):
        unified_gen = parent[k][:dim]
        # Clip an toàn
        unified_gen = np.clip(unified_gen, 0, 1)
        
        real_gen = bounds[0] + unified_gen * (bounds[1] - bounds[0])
        cost = calculate_objective_function(f"T{num_task}", real_gen, shift)
        fitness_data.append((cost, parent[k]))
        
    # Sắp xếp tăng dần theo cost (Minimization)
    fitness_data.sort(key = lambda x: x[0])
    
    # Lấy top Population
    top_items = fitness_data[:population]
    
    # Tách ra P mới và Fitness mới
    new_P = [item[1] for item in top_items]
    new_fitness = [item[0] for item in top_items]
    
    return new_P, new_fitness

def EMaTO_AMR():
    clear_old_logs()
    
    dim_max = 0
    for i in range(1, num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        if dim > dim_max: dim_max = dim

    P = [[] for _ in range(num_task + 1)]
    # [SỬA] Khởi tạo đúng kích thước population
    for i in range(1, num_task + 1):
        P[i] = np.random.uniform(0, 1, (population, dim_max))

    W = [[] for _ in range(num_task + 1)]
    FIR = np.zeros(num_task + 1)
    good_task = np.zeros((num_task + 1, num_task + 1), dtype=int)
    
    # [QUAN TRỌNG] P_fitness lưu fitness của quần thể P hiện tại
    P_fitness = [[] for _ in range(num_task + 1)]
    
    # Biến lưu kết quả tốt nhất lịch sử (Global Best)
    history_fit = np.full(num_task + 1, 1e9)
    Final_gen = [[] for _ in range(num_task + 1)]

    # --- KHỞI TẠO FITNESS BAN ĐẦU ---
    for i in range(1, num_task + 1):
        shift, dim, bounds = get_task_info(f"T{i}")
        P[i], fits = get_best(i, P[i], [], []) # Gọi get_best để tính fitness và sort
        P_fitness[i] = fits
        history_fit[i] = fits[0] # Best là phần tử đầu tiên sau khi sort
        
        # Lưu solution tốt nhất dạng Real
        best_u = P[i][0][:dim]
        best_r = bounds[0] + best_u * (bounds[1] - bounds[0])
        Final_gen[i] = best_r

    print("Start Evolution...")
    for gen in range(generation):
        
        ETA_C = 20 
        ETA_M = 20
        if gen <= 600:
            ETA_C = 2 
            ETA_M = 5  
        # Quản lý sliding window
        for i in range(1, num_task + 1):
             while len(W[i]) > M: W[i].pop(0)
        
        # In log và Ghi file
        if gen % 10 == 0:
            print(f"Generation {gen}:")
            for i in range(1, num_task + 1):
                print(f"  Task T{i} best: {history_fit[i]:.6e}") 
        
        for i in range(1, num_task + 1):
            log_to_file(i, gen, history_fit[i])

        # --- VÒNG LẶP TIẾN HÓA ---
        for i in range(1, num_task + 1):
            atp, t = ATP(i, gen, 1.0, W[i])
            rate = np.random.uniform(0, 1)
            
            # --- TRANSFER ---
            if rate <= atp:
                L = MTS(i, num_task, P, sources=5, width=1.0)
                # Đếm task tốt
                if i in IDEAL_SOURCES:
                    count = len(set(L).intersection(set(IDEAL_SOURCES[i])))
                    # (Optional) Log count if needed
                for source_idx in L:
                    good_task[i][source_idx] += 1
                
                # Gọi RDA (P_fitness lúc này luôn đồng bộ với P)
                T = RDA(P, P_fitness, i, L, transferred_tasks=5, Ntop=10)
                child = SBX_crossover(P[i], T, 0, 1, ETA_C, ETA_M)
                # Cập nhật P và P_fitness mới ngay lập tức
                P[i], P_fitness[i] = get_best(i, P[i], child, T)
            
            # --- NO TRANSFER ---
            else:
                child = SBX_crossover(P[i], [], 0, 1, ETA_C, ETA_M)
                P[i], P_fitness[i] = get_best(i, P[i], child, [])
            
            # --- CẬP NHẬT KẾT QUẢ ---
            current_best = P_fitness[i][0] # Phần tử 0 là tốt nhất
            
            # Cập nhật Global Best
            if current_best < history_fit[i]:
                history_fit[i] = current_best
                shift, dim, bounds = get_task_info(f"T{i}")
                best_u = P[i][0][:dim]
                Final_gen[i] = bounds[0] + best_u * (bounds[1] - bounds[0])

            # Tính FIR
            denom = history_fit[i] if abs(history_fit[i]) > 1e-9 else 1e-9
            # FIR nên tính dựa trên cải thiện của thế hệ này so với Global Best
            # Hoặc so với thế hệ trước (Local improvement). Code cũ dùng Global.
            fir = (history_fit[i] - current_best) / denom 
            # Lưu ý: Nếu current_best == history_fit (không cải thiện), fir = 0.
            
            if gen == 0:
                W[i].append((t, 0.0))
            else:
                W[i].append((t, fir))
            
    write_file(Final_gen, good_task)        
            
if __name__ == "__main__":
    EMaTO_AMR()