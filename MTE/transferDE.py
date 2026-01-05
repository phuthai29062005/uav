import numpy as np 

from benchmark import *



def get_fitness(task, i_idx):
    real_gen = task.copy()
    shift, dim, bounds = get_task_info(f"T{i_idx}")
    real_gen = bounds[0] + real_gen[:dim] * (bounds[1] - bounds[0])
    fitness = calculate_objective_function(f"T{i_idx}", real_gen, shift)
    return fitness


def extract_state_feature(task0, task1, best_fitness, total_fitness, g, G_max):
    
    D = task0.shape[1]
    state = []
    
    oc1 = g / G_max
    
    best_idx0 = np.argmin(total_fitness[0])
    best_idx1 = np.argmin(total_fitness[1])
    best0 = task0[best_idx0]
    best1 = task1[best_idx1]
    
    #tính euclid chuẩn hoá
    oc2 = np.linalg.norm(best0 - best1) / np.sqrt(D)
    oc3 = np.linalg.norm(np.mean(task0, axis = 0) - np.mean(task1, axis = 0)) / np.sqrt(D)
    oc4 = np.linalg.norm(np.std(task0, axis = 0) - np.std(task1, axis = 0)) / np.sqrt(0.5 * D)
    
    common_feats = [oc1, oc2, oc3, oc4]
    
    
def transfer_DE(Task, agent):
    
    task = [[] for _ in range(len(Task))]
    fitness_arr = [np.zeros(population) for _ in range(len(Task))] 
    dim_max = 0
    best_fitness = [1e9 for _ in range(len(Task))]
    total_fitness = [[] for _ in range(len(Task))]
    G_roll = 100
    population = 100
    
    for i in range(len(Task)):
        
        shift, dim, bounds = get_task_info(f"T{Task[i]}")
        if dim > dim_max:
            dim_max = dim
            task[i] = np.random.uniform(0, 1, (population, dim_max))
    for i in range(len(Task)):
        
        _, dim, _ = get_task_info(f"T{Task[i]}")
        task[i] = np.random.uniform(0, 1, (population, dim_max))
        
        for j in range(population):
            fitness = get_fitness(task[i][j], Task[i])
            fitness_arr[i][j] = fitness 
            total_fitness[i].append(fitness)
            if fitness < best_fitness[i]:
                best_fitness[i] = fitness
    
    for g in range(G_roll):

        s = extract_state_feature(task[0], task[1], total_fitness, g, G_roll)
