
import numpy as np
from benchmark import *

population = 100
def EMaTO_AMR(num_tasks):


    dim_max = 0
    for i in range(1, num_tasks + 1):
        shift, dim, bounds = get_task_info(f"T{i}")
        if dim > dim_max:
            dim_max = dim
    
    P = [[] for _ in range(num_tasks + 1)]
    for i in range(1, num_tasks + 1):
        P[i] = np.random.uniform(0, 1, (population + 1, dim_max))
        
    W = [[] for _ in range(num_tasks + 1)] # sliding window
    FIR = np.zeros(num_tasks + 1)
    atp = np.random.uniform(0, 1, num_tasks + 1)
    
    best_fitness_history = np.zeros(num_tasks + 1)
    fitness = np.zeros((num_tasks + 1, population + 1))
    for i in range(1, num_tasks + 1):
        
        shift, dim, bounds = get_task_info(f"T{i}")
        for k in range(1, population + 1):

            unified_gen = P[i][k][:dim]
            real_gen = bounds[0] + unified_gen * (bounds[1] - bounds[0])
            
            cost = calculate_objective_function(f"T{i}", real_gen, shift)
            fitness[i][k] = cost
            
        best_fitness_history[i] = np.min(fitness[i][1:])
            
                                     
                        
    