import sys
import random
import math
import numpy as np



# B2: tạo abcycle
def get_abcycle(adj):
    ab_cycle = [] 
    used_edge = set()
    
    #lấy các city
    cities = list(adj.key())
    random.suffle(cities) # đảm báo tính ngẫu nhiên
    for start_node in cities:
        #random các node để tăng tính ngẫu nhiên
        random.suffle(adj[start_node])
        
        for (neighbor, label) in adj[start_node]:
            
            if(start_node, neighbor, label) in used_edge or (neighbor, start_node, label) in used_edge:
                continue
            
            # tạo 1 chu trình mới
            curr_cycle = []
            curr_node = start_node
            curr_neigbor = neighbor
            curr_label = label
            
            while True:
                # đánh dấu cạnh đã dùng
                used_edge.add((curr_node, curr_neigbor, curr_label))
                used_edge.add((curr_neigbor, curr_node, curr_label))
                
                #thêm vào chu trình
                curr_cycle.append((curr_node, curr_neigbor, curr_label))
                curr_node = curr_neigbor
                next_label = 3 - curr_label
                
                found_next_edge = False
                random.suffle(adj[curr_node])
                for (next_neighbor, nex_label) in adj[curr_node]:
                    if nex_label == next_label:
                        if (curr_node, next_neighbor, nex_label) not in used_edge and (next_neighbor, curr_node, nex_label) not in used_edge:
                            curr_neigbor = next_neighbor
                            curr_label = nex_label
                            found_next_edge = True
                            break
                
                if not found_next_edge:
                    break
            ab_cycle.append(curr_cycle)
    return ab_cycle

# B4 : tạo con tạm thời
def create_intermediate_solution(parent1, parent2, e_set_nodes):
    intermediate_adj = {}
    
    def add_edge(adj, u, v):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        
    # thêm tất cả các cạnh từ parent1
    for i in range(city):
        u = parent1[i]
        v = parent1[(i + 1) % city]
        add_edge(intermediate_adj, u, v)
    
    # loại bỏ các cạnh của parent1 trong e_set
    for i in range(city):
        u = parent1[i]
        v = parent1[(i + 1) % city]
        
        if u in e_set_nodes and v in e_set_nodes:
            intermediate_adj[u].remove(v)
            intermediate_adj[v].remove(u)
            
    # thêm các cạnh từ parent2 trong e_set
    for i in range(city):
        u = parent2[i]
        v = parent2[(i + 1) % city]
        
        if u in e_set_nodes and v in e_set_nodes:
            add_edge(intermediate_adj, u, v)
            
    return intermediate_adj
            
                      
def EAX_GA(parent1, parent2):
    
    ## B1: lay cac canh
   
    adj = {}  
    #cha 1
    for i in range(city):
        u1 = parent1[i]
        v1 = parent1[(i + 1) % city]
        
        #khoi tao adj
        if u1 not in adj: adj[u1] = []
        if v1 not in adj: adj[v1] = [] 
        
        adj[u1].append((v1, 1))
        adj[v1].append((u1, 1))
        
    
    #cha2
    for i in range(city):
        u2 = parent2[i]
        v2 = parent2[(i + 1) % city]
        
        #khoi tao adj
        if u2 not in adj: adj[u2] = []
        if v2 not in adj: adj[v2] = [] 
        
        adj[u2].append((v2, 2))
        adj[v2].append((u2, 2))
        
    ## B2: tạo ab cycle
    ab_cycle = get_abcycle(adj)
    
    
    ## B3: lấy 1 ab_cycle ngẫu nhiên
    if len(ab_cycle) > 1:
        random_cyle = random.choice(ab_cycle)
        e_set_nodes = set(random_cycle)
    else:
        e_set_nodes = set()

    # B4: tạo con
    intermediate_adj = create_intermediate_solution(parent1, parent2, e_set_nodes)
    # B5 : repair con
    child = repair_intermediate_solution(intermediate_adj)
 
    
def generate_child():
    
    for i in range(1, city + 1):
        
        ## parent
        par1_idx = i
        par2_idx = random.randint(1, city + 1)
        while par2_idx == par1_idx:
            par2_idx = random.randint(1, city + 1)
          
        fitness = 0.0
        for j in range(Nch):
            
            child = EAX_GA(par[par1_idx], par[par2_idx])
            if(get_fitness(child) > fitness):
                fitness = get_fitness(child)
                best_child = child
        
        new_par[i] = best_child
            
        
        
    
    
            
