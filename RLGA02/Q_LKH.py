#%%
from ast import While
import numpy as np
import random

from collections import deque
from read_data import city, cost_matrix
from K_opt import K_opt

Kmax = 25
Q_matrix = np.zeros((city, city))

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
        par[u] = findpar(par[u])
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

def Create_Candidate_Set():

    global Q_matrix
    
    candidate_Set = [[] for _ in range(city)]
    alpha_x = [[] for _ in range(city)]
    min_cost = minimum_one_tree(-1, -1)
    
    for i in range(city):
        for j in range(i + 1, city):
            
            cost_ij = min_cost * 1.0/(minimum_one_tree(i, j) - min_cost + cost_matrix[i][j])
            
            Q_matrix[i][j] = cost_ij
            Q_matrix[j][i] = cost_ij
            
            alpha_x[i].append((cost_ij, j))
            alpha_x[j].append((cost_ij, i))
    
    for i in range(city):
        alpha_x[i].sort(reverse = True)
        for j in range (Kmax):
            candidate_Set[i].append(alpha_x[i][j][1])
            
    return candidate_Set        
# %%

def update_Q_matrix(p_list, candidate_set, Q_matrix, lambda_rate, gamma_factor):
    
    if p_list is None or len(p_list) < 2:
        return
    
    k_depth = len(p_list) // 2
    for i in range(1, k_depth):
        
        x = p_list[2 * i - 2]
        y = p_list[2 * i - 1]
        z = p_list[2 * i]
        
        reward_k = cost_matrix[x][y] - cost_matrix[y][z]
        max_q_next_state = 0.0
        
        if i + 1 <= k_depth:
            aa = p_list[2 * (i + 1) - 1]
            
            candidates_a = candidate_set[aa]
            if candidates_a:
                max_q_next_state = max(Q_matrix[aa][a] for a in candidates_a)
                
        current_q = Q_matrix[y][z]
        new_q = (1.0 - lambda_rate) * current_q + \
                lambda_rate * (reward_k + gamma_factor * max_q_next_state)
        
        Q_matrix[y][z] = new_q
        Q_matrix[z][y] = new_q
    
    if k_depth > 0:
        p_2k_minus_1 = p_list[2*k_depth - 2]
        p_2k = p_list[2*k_depth - 1]
        p_1 = p_list[0]
        
        reward_k = cost_matrix[p_2k_minus_1][p_2k] - cost_matrix[p_2k][p_1]
        
        current_q = Q_matrix[p_2k][p_1]
        new_q = (1.0 - lambda_rate) * current_q + lambda_rate * (reward_k + gamma_factor * 0.0)
        
        Q_matrix[p_2k][p_1] = new_q
        Q_matrix[p_1][p_2k] = new_q
         
def LKH(X_in, candidate_set, Q_matrix, lambda_rate = 0.1, gamma_factor = 0.9):

    x_out = list(X_in)
    cost_x_out = get_fitness(x_out)
    cities = list(range(city))
    
    while True:
        
        if len(cities) == 0:
            break
        
        random.shuffle(cities)
        first_city = cities[0]
        cities.remove(first_city)
        x_opt, p_list = K_opt(x_out, first_city, [first_city], 1, candidate_set)
        update_Q_matrix(p_list, candidate_set, Q_matrix, lambda_rate, gamma_factor)
        
        cost_x_opt = get_fitness(x_opt)
        if cost_x_opt < cost_x_out:
            x_out = x_opt
            cost_x_out = cost_x_opt
            
            cities_set = set(cities) 
            cities_set.update(p_list) 
            cities = list(cities_set)
            
    for i in range(city):
        q_values_i = []
        for j in range(city):
            if i != j:
                q_values_i.append((Q_matrix[i][j], j))
        q_values_i.sort(reverse=True)
        candidate_set[i] = [neighbor for q, neighbor in q_values_i[:Kmax]]
    
    return x_out, candidate_set

        
        

        

