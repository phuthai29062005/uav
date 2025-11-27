

import math
import numpy as np

from read_data import cost_matrix, city
from LHK import * 


Npop = 300 
Nch = 30
ALPHA = 0.1
BETA = 0.9
Gen = 10 * (math.log10(city) - 1)
Seed = 10 


def RHGA():

    Q_value = Create_Candidate_Set_RGHA()
    parent = np.zeros((Npop, city), dtype = int)
    path = np.arange(city)
    
    rng = np.random.default_rng()
    for i in range(Npop):
        rng.shuffle(path)
        parent[i] = path
    
    
    l_1_old = float('inf')
    l_best_old = float('inf')
    
    