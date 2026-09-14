"""
Huong tham chieu Das-Dennis (dac ta muc 5.3, eq. 26-27).

    V = {v_r}_{r=1..R},  v_r >= 0,  sum_m v_rm = 1
    R = C(H + Mo - 1, Mo - 1)

Dung truc tiep pymoo.util.reference_direction (khong tu viet lai
simplex-lattice generator).
"""
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


def das_dennis_count(n_obj, h):
    """eq. 27: R = C(H+Mo-1, Mo-1). Dung de kiem tra so region truoc khi
    sinh that su (Bang 11.1 cua dac ta)."""
    from math import comb
    return comb(h + n_obj - 1, n_obj - 1)


def make_reference_directions(n_obj, h):
    """
    return: V (R, Mo) -- moi hang la mot v_r, sum(v_r) == 1, v_r >= 0.
    R phai dung eq.27 (assert de bat loi cau hinh pymoo som).
    """
    V = get_reference_directions("das-dennis", n_obj, n_partitions=h)
    V = np.asarray(V, dtype=float)
    expected_R = das_dennis_count(n_obj, h)
    assert V.shape == (expected_R, n_obj), (V.shape, expected_R, n_obj)
    assert np.all(V >= -1e-9)
    assert np.allclose(V.sum(axis=1), 1.0, atol=1e-6)
    return V
