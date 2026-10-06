import numpy as np


def local(X_region, n_offspring, xl, xu, sigma, rng):
    """
    LOCAL (eq. 44): x' = Pi_Omega[x + sigma*(u-l)*xi], xi ~ N(0,I).

    FIX (BUG-1): ban goc bo qua tham so n_offspring, luon chi sinh 1
    offspring va tra ve mang 1D shape (D,) -- vi pham co che ngan sach
    FE theo action (eq.48: B_r(LOCAL)=ceil(eta_L*n_r), khong phai luon
    =1) va khong nhat quan voi predict() (da lam dung: loop n_offspring
    lan, tra ve mang 2D). Sua: loop dung n_offspring lan, tra ve
    (n_offspring, D).
    """
    if len(X_region) == 0:
        raise ValueError("local(): X_region rong, khong co gi de perturb.")

    offspring = []
    for _ in range(n_offspring):
        idx = rng.integers(len(X_region))
        parent = X_region[idx]

        z = (parent - xl) / (xu - xl)
        xi = rng.normal(loc=0.0, scale=1.0, size=len(parent))
        z_child = z + sigma * xi
        z_child = np.clip(z_child, 0.0, 1.0)
        child = xl + z_child * (xu - xl)
        offspring.append(child)

    return np.array(offspring)
