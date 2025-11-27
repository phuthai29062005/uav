#%%
import numpy as np

def entropy(parent):
    
    visited_edge = {}
    num_individuals = len(parent)
    
    for path in parent:
        for i in range(len(path)):
            
            u = path[i]
            v = path[(i + 1) % len(path)]
            
            edge = tuple(sorted((u, v)))
            visited_edge[edge] = visited_edge.get(edge, 0) + 1
            
    
    edge_entropy_map = {}
    for edge, freq in visited_edge.items():
        # p là xác suất xuất hiện của cạnh
        p = freq / num_individuals
        
        # Công thức entropy cơ bản: -p * log(p)
        # Cạnh rất hiếm (p gần 0) sẽ có entropy thấp.
        # Cạnh rất phổ biến (p gần 1) cũng có entropy thấp.
        # Cạnh 50/50 (p = 0.5) có entropy cao nhất.
        if p > 0:
            entropy = -p * np.log2(p)
            edge_entropy_map[edge] = entropy
        else:
            edge_entropy_map[edge] = 0
            
    return edge_entropy_map

def get_tour_entropy(tour, edge_entropy_map):
    
    total_entropy = 0
    
    for i in range(len(tour)):
        
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        
        edge = tuple(sorted((u, v)))
        
        total_entropy += edge_entropy_map.get(edge, 0)

    return total_entropy
    
# %%
