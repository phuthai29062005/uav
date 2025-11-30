import numpy as np
import random
import math

ITVS = [
    [0.10, 0.25],
    [0.25, 0.50],
    [0.50, 0.75],
    [0.75, 1.00]
]



def ATP(task, gen, w, W): # current task, gen, scaling factor, sliding window

    nITVS = {0, 1, 2, 3}
    tried_indices = set([item[0] for item in W])

    if len(tried_indices) < len(nITVS):

        all_indices = set(range(len(nITVS)))
        untried = list(all_indices - tried_indices)
        k_star = random.choice(untried)
    
    else:

        Reward = np.zeros(len(ITVS))
        credit = 0
        total_n = 0
        k_star = 0
        max_credit = -float('inf')
        n = np.zeros(len(ITVS))

        for item in W:
            t, fir = item
            Reward[t] += fir
            credit += fir
            total_n += 1
            n[t] += 1
        
        for i in range(len(ITVS)):

            if n[i] == 0:
                continue
            CredR = 0
            CredF = 1e9
            Credit = 0
            if credit != 0:
                CredR = Reward[i] / credit
            if total_n != 0:
                CredF = math.sqrt(2 * math.log(total_n) / n[i])

            Credit =  CredR + w * CredF
            if Credit > max_credit:
                max_credit = Credit
                k_star = i
        
    lower, upper = ITVS[k_star]
    atp = np.random.uniform(lower, upper)
    return atp, k_star    


