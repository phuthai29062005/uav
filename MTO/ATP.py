import numpy
import queue



def atp(c, gen, itv, w, W):
    
    '''
    index of current task (c), generation counter (gen), transfer
    interval set (ITV), scaling factor (w), sliding window (W )
    '''
    
    Reward_c = [] 
    n_c = []
    for i in range(4):
        Reward_c[i] = 0
        n_c[i] = 0
    
    nITVs = queue.Queue()
    for i in range(4):
        nITVs.push_back(i)
    
    