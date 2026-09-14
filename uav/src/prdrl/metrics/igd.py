"""IGD / MIGD (dac ta muc 9.1, eq. 64-65)."""
import numpy as np


def igd(P_star, A_t):
    """
    eq. 64:
        IGD_t = (1/|P*_t|) * sum_{y in P*_t} min_{x in A_t} ||y - x||_2

    P_star: (K, M) true/reference PF tai thoi diem t.
    A_t: (N, M) approximation set (objective vectors).
    """
    P_star = np.asarray(P_star, dtype=float)
    A_t = np.asarray(A_t, dtype=float)
    assert len(P_star) > 0
    # (K, N) pairwise distances
    diff = P_star[:, None, :] - A_t[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(np.mean(dist.min(axis=1)))


def migd(igd_values):
    """eq. 65: MIGD = mean over T environments. Lower is better."""
    igd_values = [v for v in igd_values if v is not None]
    return float(np.mean(igd_values)) if igd_values else float("inf")
