import os
import sys
import math
import random
import numpy as np

from read_data import cost_matrix, city
from Q_EAX import EAX
from Q_LKH import LKH, get_fitness, Q_matrix, Create_Candidate_Set
from entropy import entropy, get_tour_entropy

N_pop = 300 # Kích thước quần thể [cite: 359]
N_ch = 30   # Số con lai EAX tạo ra [cite: 415]
lambda_rate = 0.1 # Tốc độ học Q-learning [cite: 415]
gamma_factor = 0.9 # Hệ số chiết khấu Q-learning [cite: 415]

M_gen = 10 * (math.log10(city) - 1) if city > 10 else 10
best_cost30 = float('inf')
gen_found = 0
gen_cost= [0] * 10000

def Select_Survive(children, pA, edge_entropy_map):

    best_survivor = pA
    max_entropy = get_tour_entropy(pA, edge_entropy_map)
    
    for child in children:
        if not child:
            continue
            
        child_entropy = get_tour_entropy(child, edge_entropy_map)
        
        if child_entropy > max_entropy:
            max_entropy = child_entropy
            best_survivor = child
            
    return best_survivor

def Generate_Initial_Pop():

    population = []
    path_template = list(range(city))
    
    for _ in range(N_pop):
        random.shuffle(path_template)
        population.append(list(path_template))
        
    return population

def RHGA(max_generations=1000):
    
    print("--- Khởi tạo RHGA ---")
    global best_cost30, gen_found, gen_cost
    candidate_set = Create_Candidate_Set()
    population = Generate_Initial_Pop()
    
    pop_fitness = [get_fitness(ind) for ind in population]
    
    l_old_1 = float('inf')
    l_old_best = float('inf') 
    num_gen_no_improve = 0
    
    global_best_tour = population[0]
    global_best_cost = pop_fitness[0]
    
    for gen in range(max_generations):
        
        print(f"Thế hệ {gen + 1}/{max_generations} - Best cost: {global_best_cost}")
        gen_cost[gen + 1] += global_best_cost
        x1 = population[0]
        cost_x1 = pop_fitness[0]

        best_idx_other = np.argmin(pop_fitness[1:]) + 1
        x_best = population[best_idx_other]
        cost_x_best = pop_fitness[best_idx_other]
        
        if cost_x1 < global_best_cost:
            global_best_cost = cost_x1
            global_best_tour = x1
        elif cost_x_best < global_best_cost:
            global_best_cost = cost_x_best
            global_best_tour = x_best
        
        if best_cost30 > global_best_cost:
            best_cost30 = global_best_cost
            gen_found = gen + 1
        
        num_gen_no_improve += 1
        
        # --- Logic Q-LKH (Dòng 8-20) ---
        if cost_x1 < l_old_1:
            x1_new, candidate_set = LKH(x1, candidate_set, Q_matrix, lambda_rate, gamma_factor)
            population[0] = x1_new
            pop_fitness[0] = get_fitness(x1_new)
            
            l_old_1 = pop_fitness[0]
            num_gen_no_improve = 0
            
        elif cost_x_best < cost_x1 and cost_x_best < l_old_best:
            l_old_best = cost_x_best
            
            x_best_new, candidate_set = LKH(x_best, candidate_set, Q_matrix, lambda_rate, gamma_factor)
            cost_x_best_new = get_fitness(x_best_new)
        
            if cost_x_best_new < cost_x_best:
                population[0] = x_best_new 
                pop_fitness[0] = cost_x_best_new 
                l_old_1 = cost_x_best_new 
                num_gen_no_improve = 0   
        
        elif num_gen_no_improve >= M_gen:
            r_idx = random.randint(1, N_pop - 1)
            x_r = population[r_idx]
            
            x_r_new, candidate_set = LKH(x_r, candidate_set, Q_matrix, lambda_rate, gamma_factor)
            cost_x_r_new = get_fitness(x_r_new)
            
            num_gen_no_improve = 0
            
            if cost_x_r_new < pop_fitness[0]:
                population[0] = x_r_new 
                pop_fitness[0] = cost_x_r_new 
                l_old_1 = cost_x_r_new 

        l_old_1 = pop_fitness[0]

        # --- Logic EAX-GA (Dòng 21-25) ---
        
        edge_entropy_map = entropy(population)
        
        rp = list(range(N_pop))
        random.shuffle(rp)
        
        for i in range(N_pop):
            idx_A = rp[i]
            idx_B = rp[(i + 1) % N_pop]
            
            pA = population[idx_A]
            pB = population[idx_B]
            
            all_children = EAX(pA, pB, candidate_set)
            
            survivor = Select_Survive(all_children, pA, edge_entropy_map)
            
            population[idx_A] = survivor
            pop_fitness[idx_A] = get_fitness(survivor)
            
    print("--- Hoàn thành ---")
    print(f"Best cost found: {global_best_cost}")
    return global_best_tour, global_best_cost

if __name__ == "__main__":
    
   
    seed = 30
    output_filename = "rhga51.txt"
    
    with open(output_filename, "w") as f:
        
        for i in range(seed):
            print(f"--- Chạy seed thứ {i + 1}/{seed} ---")
            best_tour, best_cost = RHGA(max_generations=300)
            
        f.write(f"{best_cost30} {gen_found}\n")
        for i in range(300):
            f.write(f"{i + 1}: {gen_cost[i + 1] * 1.0 / seed}\n")
    

    print(f"--- Hoàn tất {seed} seed! ---")
    print(f"Kết quả chi tiết đã được lưu tại: {output_filename}")