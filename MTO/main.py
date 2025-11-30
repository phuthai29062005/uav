import numpy as np
import random

from ATP import *

from benchmark import *

num_task = 10
population = 100
generation = 1000
M = int(0.05 * generation)

def EMaTO_AMR():

    # initialize population
    dim_max = 0
    for i in range(1, num_task + 1):
        shift, dim, bounds = get_task_info(f"T{i}")
        if dim > dim_max:
            dim_max = dim

    P = [[] for _ in range(num_task + 1)]
    for i in range(1, num_task + 1):
        P[i] = np.random.uniform(0, 1, (population + 1, dim_max))

    # initialize sliding window W, FIR, atp:
    W = [[] for _ in range(num_task + 1)]
    FIR = np.zeros(num_task + 1)
    atp = np.random.uniform(0, 1, num_task + 1)

    # evaluate fitness values of populations on correspoding rasks
    
    history_fit = np.full(num_task + 1, 1e9)
    for i in range(1, num_task):

        shift, dim, bounds = get_task_info(f"T{i}")
        for k in range(population):
            unified_gen = P[i][k][:dim]
            real_gen = bounds[0] + unified_gen * (bounds[1] - bounds[0])
            cost = calculate_objective_function(f"T{i}", real_gen, shift)
            if cost < history_fit[i]:
                history_fit[i] = cost

    for gen in range(generation):

        while len(W) > M:
            W.pop(0)

        current_fit = np.full(num_task + 1, 1e9)

        for i in range(1, num_task + 1):
            atp, t =  ATP(i, gen, 1.0, W[i])
            
            if gen == 0:
                W[i].append({t, 0})
            else:
                fir = 1 - current_fit[i] / history_fit[i]
                W[i].append({t, fir})            
            rate = np.random.uniform(0, 1)
            if rate <= atp:

                L = MTS(i, P, sources = 5, width = 1.0)


if __name__ == "__main__":
    EMaTO_AMR()