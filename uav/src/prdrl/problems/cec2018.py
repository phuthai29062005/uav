"""
Adapter cho benchmark CEC2018 DF1-DF14 (Jiang, Yang, Yao, Tan, Kaiser,
Krasnogor -- "Benchmark Problems for CEC'2018 Competition on Dynamic
Multiobjective Optimisation").

KHONG tu viet lai 14 cong thuc DF1-DF14 tu technical report: pymoo (da
cai san trong moi truong nay, dung xuyen suot du an tu dau) co san
implementation DF1-DF14 dung chuan, duoc cong dong dung rong rai lam
tham chieu. Tu chep lai cong thuc tay co rui ro sai sot rat cao va
khong can thiet khi nguon dang tin cay da co san -- file nay CHI la mot
lop adapter mong quanh pymoo.problems.dynamic.df.

Ngoai le duy nhat: DF5/DF12/DF13 dung ban da "snap" gia tri sin(...) ve
dung 0.0 truoc khi ap floor(), tranh sai so dau phay dong lam floor()
nhay so gia tai cac moc chu ky (vd sin(2*pi) ~= -2.4e-16 thay vi dung
0.0 -> floor flip tu 0 thanh -1). Day la van de so hoc CO THAT, tung
duoc phat hien va kiem chung truoc do trong chinh du an nay (khong phai
suy doan) -- dat ca ba class hardened trong CUNG file nay (khong them
module moi, dung yeu cau).
"""
import numpy as np
from pymoo.problems.dynamic.df import (DF1, DF2, DF3, DF4, DF6, DF7, DF8,
                                       DF9, DF10, DF11, DF14)
from pymoo.problems.dynamic.df import DF5 as _DF5
from pymoo.problems.dynamic.df import DF12 as _DF12
from pymoo.problems.dynamic.df import DF13 as _DF13


def _snap_near_zero(v, eps=1e-9):
    """Ep gia tri gan 0 (do sai so dau phay dong tai cac moc sin(k*pi))
    ve dung 0.0 truoc khi dua vao floor()."""
    v = np.asarray(v, dtype=float)
    return np.where(np.abs(v) < eps, 0.0, v)


class DF5Stable(_DF5):
    def _evaluate(self, x, out, *args, **kwargs):
        G = _snap_near_zero(np.sin(0.5 * np.pi * self.time))
        w = np.floor(10 * G)
        g = 1 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f1 = g * (x[:, 0] + 0.02 * np.sin(w * np.pi * x[:, 0]))
        f2 = g * (1 - x[:, 0] + 0.02 * np.sin(w * np.pi * x[:, 0]))
        out["F"] = np.column_stack([f1, f2])


class DF12Stable(_DF12):
    def _evaluate(self, x, out, *args, **kwargs):
        k = 10 * _snap_near_zero(np.sin(np.pi * self.time))
        r = 1
        x0 = x[:, 0].reshape(len(x), 1)
        tmp1 = x[:, 2:] - np.sin(self.time * x0)
        tmp2 = np.abs(np.sin(np.floor(k * (2 * x[:, 0:2] - r)) * np.pi / 2))
        g = 1 + np.sum(tmp1 ** 2, axis=1) + np.prod(tmp2)
        f1 = g * np.cos(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f2 = g * np.sin(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f3 = g * np.sin(0.5 * np.pi * x[:, 0])
        out["F"] = np.column_stack([f1, f2, f3])


class DF13Stable(_DF13):
    def _evaluate(self, x, out, *args, **kwargs):
        G = _snap_near_zero(np.sin(0.5 * np.pi * self.time))
        p = np.floor(6 * G)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)
        g = 1 + np.sum((x[:, 2:] - G) ** 2, axis=1)
        g = g.reshape(len(g), 1)
        f1 = g * np.cos(0.5 * np.pi * x0) ** 2
        f2 = g * np.cos(0.5 * np.pi * x1) ** 2
        f3 = g * (
            np.sin(0.5 * np.pi * x0) ** 2
            + np.sin(0.5 * np.pi * x0) * np.cos(p * np.pi * x0) ** 2
            + np.sin(0.5 * np.pi * x1) ** 2
            + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2
        )
        out["F"] = np.column_stack([f1, f2, f3])


_DF_CLASSES = {
    1: DF1, 2: DF2, 3: DF3, 4: DF4, 5: DF5Stable, 6: DF6, 7: DF7,
    8: DF8, 9: DF9, 10: DF10, 11: DF11, 12: DF12Stable, 13: DF13Stable,
    14: DF14,
}
TRI_OBJECTIVE_DFS = {10, 11, 12, 13, 14}


def get_n_obj(df_id):
    """n_obj = 2 cho DF1-DF9, 3 cho DF10-DF14."""
    if df_id not in _DF_CLASSES:
        raise ValueError(f"df_id phai trong 1..14, nhan duoc {df_id}")
    return 3 if df_id in TRI_OBJECTIVE_DFS else 2


def get_df_bounds(df_id, D=10):
    """
    Tra ve (xl, xu) THAT cua DF{df_id}, doc truc tiep tu pymoo (khong
    doan/hardcode [0,1]^D cho moi bai -- CANH BAO xem cuoi file: nhieu
    DF (DF3,4,5,6,7,8,9,10) co bien KHONG nam trong [0,1]).
    """
    if df_id not in _DF_CLASSES:
        raise ValueError(f"df_id phai trong 1..14, nhan duoc {df_id}")
    p = _DF_CLASSES[df_id](time=0.0, n_var=D)
    return np.asarray(p.xl, dtype=float).copy(), np.asarray(p.xu, dtype=float).copy()


def evaluate_df(df_id, X, t):
    """
    Danh gia DF{df_id} tai thoi diem t.

    Parameters
    ----------
    df_id : int, 1-14
    X     : (N, D) decision vectors
    t     : float, thoi gian lien tuc (tu compute_time)

    Returns
    -------
    F : (N, n_obj) objective vectors -- n_obj=2 (DF1-9), n_obj=3 (DF10-14)
    """
    if df_id not in _DF_CLASSES:
        raise ValueError(f"df_id phai trong 1..14, nhan duoc {df_id}")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X phai shape (N,D), nhan duoc {X.shape}")
    D = X.shape[1]
    problem = _DF_CLASSES[df_id](time=float(t), n_var=D)
    F = problem.evaluate(X)
    return np.asarray(F, dtype=float)


def get_true_pf(df_id, t, n_points=500):
    """
    True PF tai thoi diem t, shape (n_points_that, n_obj). Tra None neu
    PF khong co dang giai tich (pymoo._calc_pareto_front khong ho tro
    hoac loi).
    """
    if df_id not in _DF_CLASSES:
        raise ValueError(f"df_id phai trong 1..14, nhan duoc {df_id}")
    problem = _DF_CLASSES[df_id](time=float(t), n_var=10)
    try:
        pf = problem.pareto_front(n_pareto_points=n_points)
    except Exception:
        return None
    if pf is None:
        return None
    return np.asarray(pf, dtype=float)


# =====================================================================
# CANH BAO (khong tu sua o day -- ngoai pham vi file nay theo yeu cau):
# E1_heterogeneity_audit.py::get_df_config() dang HARDCODE
# lower=zeros(D), upper=ones(D) cho MOI DF. Dieu nay SAI cho nhieu DF --
# vi du DF3 (x1 in [-1,2]), DF7 (x0 in [1,4]), DF10 (x2.. in [-1,1]),
# v.v. Dung get_df_bounds(df_id) o file nay de lay bounds THAT khi ban
# muon sua get_df_config (ngoai pham vi yeu cau hien tai nen KHONG tu
# dong sua o day).
# =====================================================================
