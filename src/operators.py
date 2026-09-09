# file: operators.py
import numpy as np

def sbx_crossover(p1, p2, eta=20, prob=0.9):
    """
    Simulated Binary Crossover
    p1, p2: 2 parent vectors, shape (D,)
    eta: distribution index (cao = con giống cha mẹ)
    prob: xác suất crossover
    return: 2 offspring
    """
    D = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    
    if np.random.rand() > prob:
        return c1, c2
    
    for i in range(D):
        if np.random.rand() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue
            
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    
    # Clip vào [0, 1]
    c1 = np.clip(c1, 0, 1)
    c2 = np.clip(c2, 0, 1)
    
    return c1, c2


def polynomial_mutation(x, eta=20, prob=None):
    """
    Polynomial Mutation
    x: vector shape (D,)
    eta: distribution index
    prob: xác suất mutation mỗi biến, mặc định 1/D
    """
    D = len(x)
    if prob is None:
        prob = 1.0 / D
    
    y = x.copy()
    for i in range(D):
        if np.random.rand() > prob:
            continue
        
        u = np.random.rand()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        
        y[i] = x[i] + delta  # vì bounds là [0,1], range = 1
        y[i] = np.clip(y[i], 0, 1)
    
    return y


# === TEST ===
if __name__ == "__main__":
    np.random.seed(30)
    
    p1 = np.random.rand(10)
    p2 = np.random.rand(10)
    
    print("Parent 1:", np.round(p1, 3))
    print("Parent 2:", np.round(p2, 3))
    
    c1, c2 = sbx_crossover(p1, p2)
    print("\nChild 1: ", np.round(c1, 3))
    print("Child 2: ", np.round(c2, 3))
    
    m1 = polynomial_mutation(c1)
    print("\nMutated: ", np.round(m1, 3))
    
    # Kiểm tra bounds
    assert np.all(c1 >= 0) and np.all(c1 <= 1), "Out of bounds!"
    assert np.all(m1 >= 0) and np.all(m1 <= 1), "Out of bounds!"
    print("\nAll bounds OK!")