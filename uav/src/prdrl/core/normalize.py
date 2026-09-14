"""
Chuan hoa objective space (dac ta PR-DRL-DMOEA muc 5.2, eq. 24-25).

ideal/nadir cuc bo duoc xay tren HOP {F-} union {F+} (TRUOC va SAU thay
doi cung mot lan change), khong phai tren rieng F- hay F+. Day la diem
khac voi ideal/nadir "toan cuc qua nhieu the he" thuong gap trong NSGA-III
thong thuong -- o day chi dung 2 tap diem cua chinh change event nay.
"""
import numpy as np

EPS = 1e-12


def compute_ideal_nadir(F_minus, F_plus):
    """
    eq. 24:
        z*_m   = min_i { f-_im, f+_im }
        z_nad_m = max_i { f-_im, f+_im }

    F_minus, F_plus: (N, M) cung so hang N, cung problem M objectives.
    return: (z_star (M,), z_nad (M,))
    """
    F_minus = np.asarray(F_minus, dtype=float)
    F_plus = np.asarray(F_plus, dtype=float)
    assert F_minus.shape == F_plus.shape, (F_minus.shape, F_plus.shape)
    stacked = np.vstack([F_minus, F_plus])          # (2N, M)
    z_star = stacked.min(axis=0)
    z_nad = stacked.max(axis=0)
    return z_star, z_nad


def normalize(F, z_star, z_nad, eps=EPS):
    """
    eq. 25:
        f_tilde_im = (f_im - z*_m) / (z_nad_m - z*_m + eps)
    """
    F = np.asarray(F, dtype=float)
    return (F - z_star) / (z_nad - z_star + eps)


def normalize_pair(F_minus, F_plus, eps=EPS):
    """Convenience: compute ideal/nadir from the pair then normalize both.
    return: (F_minus_tilde, F_plus_tilde, z_star, z_nad)"""
    z_star, z_nad = compute_ideal_nadir(F_minus, F_plus)
    Fm_t = normalize(F_minus, z_star, z_nad, eps)
    Fp_t = normalize(F_plus, z_star, z_nad, eps)
    return Fm_t, Fp_t, z_star, z_nad
