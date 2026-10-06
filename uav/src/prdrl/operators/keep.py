import numpy as np


def keep(X_region, n_offspring, xl, xu, sigma, rng):
    """
    KEEP (eq. 43): giu nguyen decision vectors, khong sinh offspring
    (B_extra_r = 0). n_offspring/xl/xu/sigma/rng KHONG duoc dung -- chi
    nhan de dong bo chu ky voi local()/predict()/diversify(), cho phep
    dispatch da hinh (goi 4 operator qua CUNG mot interface) o
    loop/one_change.py sau nay ma khong can if/else rieng cho KEEP.
    """
    D = X_region.shape[1]
    return np.empty((0, D))
