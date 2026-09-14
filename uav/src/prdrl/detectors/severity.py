"""
Muc thay doi theo vung (dac ta muc 5.4, eq. 30-33).

Causal EMA giong het quy tac dung trong SA-DRL-DMOEA cu (audit lai o
day, khong copy code cu -- viet lai tu cong thuc): q duoc CAP NHAT SAU
khi da dung q_{t-1} de tinh c_r(t), khong bao gio nhin truoc.
"""
import numpy as np

EPS = 1e-12


def displacement(F_minus_tilde, F_plus_tilde):
    """eq. 30: d_i = ||F+_tilde_i - F-_tilde_i||_2, per-solution."""
    Fm = np.asarray(F_minus_tilde, dtype=float)
    Fp = np.asarray(F_plus_tilde, dtype=float)
    return np.linalg.norm(Fp - Fm, axis=1)


def region_severity_raw(d_i, region_idx):
    """eq. 31: s_r = median_{i in I_r} d_i. Vung rong -> None (inactive,
    Bang 5.4 cuoi trang: 'vung rong duoc danh dau inactive')."""
    s = {}
    for r, idx in region_idx.items():
        s[r] = float(np.median(d_i[idx])) if len(idx) > 0 else None
    return s


class CausalSeverityScale:
    """
    Trang thai q_t xuyen suot mot run (mot benchmark/seed), cap nhat
    nhan qua qua tung change event.

        c_r(t) = 1 - exp(-s_r(t) / (kappa*q_{t-1} + eps))          (32)
        q_t    = (1-rho)*q_{t-1} + rho*median_r(s_r(t))            (33)

    Cold-start (lan change dau tien, q_prev chua ton tai): dac ta khong
    noi ro gia tri c_r luc nay -- ap dung quy uoc giong ChangeDetector
    cu (da kiem chung trong SA-DRL-DMOEA): c_r=0 cho moi vung o lan dau,
    va khoi tao q_0 = median_r(s_r(t=1)) truoc khi buoc tiep theo dung.
    """

    def __init__(self, kappa=1.0, rho=0.1, eps=EPS):
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.eps = float(eps)
        self.q = None

    def update(self, s_r_dict):
        """s_r_dict: {region -> float or None (inactive)}.
        return: c_r_dict {region -> float in [0,1)} (None cho vung inactive)."""
        active = {r: s for r, s in s_r_dict.items() if s is not None}
        if not active:
            return {r: None for r in s_r_dict}

        if self.q is None:
            c_r = {r: (0.0 if s is not None else None) for r, s in s_r_dict.items()}
            self.q = float(np.median(list(active.values())))
            return c_r

        denom = self.kappa * self.q + self.eps
        c_r = {r: (1.0 - np.exp(-s / denom) if s is not None else None)
              for r, s in s_r_dict.items()}
        self.q = (1.0 - self.rho) * self.q + self.rho * float(np.median(list(active.values())))
        return c_r
