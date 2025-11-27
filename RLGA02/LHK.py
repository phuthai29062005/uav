#%%
from ast import While
import random

from collections import deque
from read_data import city, cost_matrix
from K_opt import K_opt
Kmax = 25

def get_fitness(tour):
    total_cost = 0
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        total_cost += cost_matrix[u][v]
    return total_cost
#%%  
def minimum_one_tree(i, j):

    par = [-1] * city
    
    def findpar(u):
        if par[u] == -1:
            return u
        par[u] = findpar(u)
        return par[u]
    
    mst_cost = 0
    
    if i == 0:
        
        cost = []
        for u in range(1, city):
            for v in range (u + 1, city):
                cost.append((cost_matrix[u][v], u, v))
        
        cost.sort()
        
        for k in range(city): par[k] = -1
        
        for c, u, v in cost:
            pu, pv = findpar(u), findpar(v)
            if pu > pv:
                pu, pv = pv, pu
            if pu != pv:
                par[pv] = pu
                mst_cost += c
            
        mst_cost += cost_matrix[0][j]
        second_cost = float('inf')
        
        for k in range(1, city):
            if k != j:
                second_cost = min(second_cost, cost_matrix[0][k])
        
        mst_cost += second_cost
    else:
        
        cost = []
        for u in range(1, city):
            for v in range(u + 1, city):
                if u == i and v == j:
                    continue
                cost.append((cost_matrix[u][v], u, v))
        
        cost.sort()

        for k in range(city): par[k] = -1
        
        if i != -1 and j != -1:
            mst_cost += cost_matrix[i][j]
            par[j] = i
            
        for c, u, v in cost:
            pu, pv = findpar(u), findpar(v)
            if pu > pv:
                pu, pv = pv, pu
            if pu != pv:
                par[pv] = pu
                mst_cost += c
        
        cost0 = []
        for k in range(1, city):
            cost0.append(cost_matrix[0][k])
        cost0.sort()
        mst_cost += cost0[0] + cost0[1]
    
    return mst_cost
        

def Create_Candidate_Set ():

    candidate_Set = [[] for _ in range(city)]
    alpha_x = [[] for _ in range(city)]
    min_cost = minimum_one_tree(-1, -1)
    
    for i in range(city):
        
        for j in range(i + 1, city):
            
            if i == j:
                continue
            cost_ij = minimum_one_tree(i, j) - min_cost
            alpha_x[i].append((cost_ij, j))
            alpha_x[j].append((cost_ij, i))
        
        alpha_x[i].sort()
        for j in range (Kmax):
            candidate_Set[i].append(alpha_x[i][j][1])
            
    return candidate_Set                      
    
# %%
def LKH(M):
    
    candidate_set = Create_Candidate_Set()
    
    
    X_best = []
    cost_X_best = float('inf')
    
    for i in range(M):

        X_init = []
        for j in range(city):
            X_init.append(j)
            
        cities = []
        for j in range(city):
            cities.append(j)
            
        random.shuffle(X_init)
        
        while True:
            
            if len(cities) == 0:
                break
            
            random.shuffle(cities)
            first_city = cities[0]
            cities.remove(first_city)
            X_opt, p_list = K_opt(X_init, first_city, {first_city}, 1, candidate_set)

            if get_fitness(X_opt) < get_fitness(X_init):
                X_init = X_opt
                
                cities_set = set(cities) 
                cities_set.update(p_list) 
                # Chuyển về list để shuffle (nếu bạn muốn giữ logic cũ)
                cities = list(cities_set)
                
        if(get_fitness(X_init) < cost_X_best):
            X_best = X_init
            cost_X_best = get_fitness(X_init)
            
    return X_best

        
        

        

