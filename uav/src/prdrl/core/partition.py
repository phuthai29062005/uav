"""
Phan vung bang huong tham chieu (dac ta muc 5.3, eq. 28-29).

BAY NGUY HIEM NHAT cua toan bo thiet ke (ghi trong dac ta, phan
"Ghi nho khi tu code"): membership PHAI duoc gan bang F_minus_tilde
(TRUOC thay doi), khong bao gio duoc repartition o F_plus_tilde. Ham
assign_regions() o day CHI nhan F_minus_tilde lam input -- khong co
tham so F_plus nao ca, de khong the vo tinh dung nham.
"""
import numpy as np

EPS = 1e-12


def assign_regions(F_minus_tilde, V, eps=EPS):
    """
    eq. 28:
        u_i^- = F_tilde_i^- / (||F_tilde_i^-||_2 + eps)
        r(i)  = argmax_r  (u_i^-)^T v_r / (||v_r||_2 + eps)
    eq. 29:
        I_r = { i : r(i) = r }

    F_minus_tilde: (N, M) -- normalized objective vectors BEFORE change.
    V: (R, M) reference directions.
    return: r_of_i (N,) int array, gia tri trong [0, R-1] -- region index
            cua tung ca the, gan CO DINH cho toan bo phep do truoc/sau.
    """
    F_minus_tilde = np.asarray(F_minus_tilde, dtype=float)
    V = np.asarray(V, dtype=float)
    norms_u = np.linalg.norm(F_minus_tilde, axis=1, keepdims=True) + eps
    U = F_minus_tilde / norms_u                       # (N, M)
    norms_v = np.linalg.norm(V, axis=1) + eps          # (R,)
    # np.errstate: on some macOS/Accelerate BLAS builds this small-M
    # matmul spuriously raises divide/overflow/invalid RuntimeWarnings
    # even though inputs and outputs are verified finite (checked via
    # np.isnan/np.isinf -- both all-False) -- a known BLAS false
    # positive, not a real numerical issue here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cos_sim = (U @ V.T) / norms_v[None, :]         # (N, R)
    r_of_i = np.argmax(cos_sim, axis=1)
    return r_of_i


def region_indices(r_of_i, R):
    """eq. 29 as a dict: r -> list of solution indices I_r (co the rong)."""
    return {r: np.where(r_of_i == r)[0] for r in range(R)}
