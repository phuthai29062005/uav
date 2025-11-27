#%%
import numpy
import random
from read_data import cost_matrix, city

Nch = 30
Nnear = 10
#%%
#2
def get_abcycle(adj):
    ab_cycle = []
    visited = set()
    
    # get cities
    cities = list(adj.keys())
    random.shuffle(cities)
    
    for start_node in cities:
        # random nodes
        random.shuffle(adj[start_node])
        
        for (neighbor, label) in adj[start_node]:
            
            if (start_node, neighbor, label) in visited or (neighbor, start_node, label) in visited:
                continue
            
            curr_cycle = []
            curr_node = start_node
            curr_neighbor = neighbor
            curr_label = label
            
            while True:
                
                visited.add((curr_node, curr_neighbor, curr_label))
                visited.add((curr_neighbor, curr_node, curr_label))
                
                curr_cycle.append((curr_node, curr_neighbor, curr_label))
                curr_node = curr_neighbor
                next_label = 3 - curr_label
                found_next_edge = False
                random.shuffle(adj[curr_node])
                
                for (next_node, nex_label) in adj[curr_node]:
                    if nex_label == next_label:
                        if (curr_node, next_node, nex_label) not in visited:
                            if (next_node, curr_node, nex_label) not in visited:
                                curr_neighbor = next_node
                                curr_label = nex_label
                                found_next_edge = True
                                break
                
                if not found_next_edge:
                    break
            
            ab_cycle.append(curr_cycle)
            
    return ab_cycle
#%%               
#4
def create_solution(parent1, parent2, e_set_nodes):
    new_adj = {}
    adj1 = {}
    adj2 = {}

    
    def add_edge(adj, u, v):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        
    for i in range(len(parent1)):
        u = parent1[i]
        v = parent1[(i + 1) % len(parent1)] 
        add_edge(new_adj, u, v)
        add_edge(adj1, u, v)
    
    for i in range(len(parent2)):
        u = parent2[i]
        v = parent2[(i + 1) % len(parent2)] 
        add_edge(adj2, u, v)

    if e_set_nodes and len(e_set_nodes[0]) > 0:
        cycle_nodes = e_set_nodes[0]
        for i in range(len(cycle_nodes)):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % len(cycle_nodes)]
            if u in adj1 and v in adj1[u]: 
                if v in new_adj[u]: 
                    new_adj[u].remove(v)
                    new_adj[v].remove(u)
            if u in adj2 and v in adj2[u]:
                add_edge(new_adj, u, v)
                     
    return new_adj

#%%
#5
def _find_all_subtours(new_adj):
    
    all_subtour = []
    visited = set()
    nodes = list(new_adj.keys())
    if not nodes:
        return []
    for start_node in nodes:
        if start_node in visited:
            continue
        
        curr_subtour = []
        curr = start_node
        prev = -1
        
        while curr not in visited:
            visited.add(curr)
            curr_subtour.append(curr)
            
            if curr not in new_adj or len(new_adj[curr]) == 0:
                break
            
            neibors = new_adj[curr]
            
            if len(neibors) == 1:
                if neibors[0] != prev:
                    prev = curr
                    curr = neibors[0]
                else:
                    break 
            elif neibors[0] != prev:
                prev = curr
                curr = neibors[0]
            else:
                prev = curr
                curr = neibors[1]
            
            if curr == start_node:
                break
        
        if curr_subtour: 
            all_subtour.append(curr_subtour)
    
    return all_subtour
    
    
def repair_solution(new_adj, candidate_set): 
    
    all_subtours = _find_all_subtours(new_adj)
    
    if not all_subtours:
        return []
    if len(all_subtours) == 1:
        return all_subtours[0]
    
    while len(all_subtours) > 1:
        
        all_subtours.sort(key = len)
        
        smallest_subtour = all_subtours.pop(0)
        
        min_delta = float('inf')
        best_merge_info = {}
        
        for u_idx in range(len(smallest_subtour)):
            u = smallest_subtour[u_idx]
            v = smallest_subtour[(u_idx + 1) % len(smallest_subtour)]
            
            cost_uv = cost_matrix[u][v]
            relevant_neighbors = set()
            
            if u < len(candidate_set):
                relevant_neighbors.update(candidate_set[u][:Nnear])
            
            if v < len(candidate_set):
                relevant_neighbors.update(candidate_set[v][:Nnear])
            
            for T_idx, T in enumerate(all_subtours):
     
                for w_idx in range(len(T)):
                    x = T[w_idx]
                    z = T[(w_idx + 1) % len(T)]
                    if x not in relevant_neighbors and z not in relevant_neighbors:
                        continue
                        
                    cost_xz = cost_matrix[x][z]
                    
                    delta1 = cost_matrix[u][x] + cost_matrix[v][z] - cost_uv - cost_xz
                    
                    if delta1 < min_delta:
                        min_delta = delta1
                        best_merge_info = {
                            'u_idx': u_idx,
                            'w_idx': w_idx,
                            'T_idx': T_idx,
                            'type': 'ux_vz' 
                        }
                    
                    delta2 = cost_matrix[u][z] + cost_matrix[v][x] - cost_uv - cost_xz
                    if delta2 < min_delta:
                        min_delta = delta2
                        best_merge_info = {
                            'u_idx': u_idx,
                            'w_idx': w_idx,
                            'T_idx': T_idx,
                            'type': 'uz_vx' 
                        }
         
        # Lấy thông tin gộp tốt nhất
        if not best_merge_info:
            T_to_merge = all_subtours.pop(0)
            all_subtours.append(smallest_subtour + T_to_merge)
            continue

        T_idx = best_merge_info['T_idx']
        u_idx = best_merge_info['u_idx']
        w_idx = best_merge_info['w_idx']
 
        T_to_merge = all_subtours.pop(T_idx)
        
        seg_s = smallest_subtour[u_idx+1:] + smallest_subtour[:u_idx+1]
        
        seg_T = T_to_merge[w_idx+1:] + T_to_merge[:w_idx+1]
        
        if best_merge_info['type'] == 'uz_vx':
            new_tour = seg_s + seg_T
        else: # 'ux_vz'
            new_tour = seg_s + list(reversed(seg_T))
            
        all_subtours.append(new_tour)
            
    return all_subtours[0]
        
        
#%%    

def EAX(parent1, parent2, candidate_set):
    # 1
    adj = {}
    size = len(parent1)
    
    for i in range(size):
        u = parent1[i]
        v = parent1[(i + 1) % size]
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append((v, 1))
        adj[v].append((u, 1))
    
    for i in range(size):
        u = parent2[i]
        v = parent2[(i + 1) % size]
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append((v, 2))
        adj[v].append((u, 2))
       
    #2 
    all_child = []
    ab_cycle = get_abcycle(adj)
    for i in range(Nch):
        
        e_set_nodes = []
        
        if len(ab_cycle) > 0:
            random_cycle_edges = random.choice(ab_cycle) 
            nodes_in_cycle = [u for (u, v, label) in random_cycle_edges]

            e_set_nodes.append(nodes_in_cycle)
        else:
            e_set_nodes.append([])
            
        new_adj = create_solution(parent1, parent2, e_set_nodes)
        
        final_child = repair_solution(new_adj, candidate_set)
        if final_child:
            all_child.append(final_child)
        
    return all_child
        
    
# %%