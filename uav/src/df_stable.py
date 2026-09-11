"""
Numerically-stabilized CEC2018 DF5/DF12/DF13 (4.10C.1).

Loi goc (pymoo): floor() tren sin(...) khong-abs tai bien chu ky exact
sinh discontinuity gia:
    sin(2*pi) = -2.449e-16 (float)  ->  floor(6 * (-eps)) = -1
trong khi toan hoc sin(2*pi)=0 -> floor(0)=0.

Sua: snap gia tri sin ve 0 khi |.| < atol TRUOC floor. KHONG doi cong
thuc benchmark ngoai on dinh floating-zero. Chi DF5/12/13 (floor tren
sin khong-abs) bi anh huong; DF2/DF9 dung abs() nen an toan.
"""
import numpy as np
from pymoo.problems.dynamic.df import DF5, DF12, DF13, get_PF

_ATOL = 1e-12


def snap_near_zero(x, atol=_ATOL):
    """Snap gia tri gan-zero (float residue) ve 0.0. Scalar."""
    return 0.0 if abs(x) < atol else float(x)


class DF5Stable(DF5):
    def _evaluate(self, x, out, *args, **kwargs):
        G = snap_near_zero(np.sin(0.5 * np.pi * self.time))
        w = np.floor(10 * G)
        g = 1 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f1 = g * (x[:, 0] + 0.02 * np.sin(w * np.pi * x[:, 0]))
        f2 = g * (1 - x[:, 0] + 0.02 * np.sin(w * np.pi * x[:, 0]))
        out["F"] = np.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        G = snap_near_zero(np.sin(0.5 * np.pi * self.time))
        w = np.floor(10 * G)
        f1 = x + 0.02 * np.sin(w * np.pi * x)
        f2 = 1 - x + 0.02 * np.sin(w * np.pi * x)
        return np.array([f1, f2]).T


class DF12Stable(DF12):
    def _evaluate(self, x, out, *args, **kwargs):
        k = 10 * snap_near_zero(np.sin(np.pi * self.time))
        r = 1
        x0 = x[:, 0].reshape(len(x), 1)
        tmp1 = x[:, 2:] - np.sin(self.time * x0)   # phi chu ky (giu nguyen)
        tmp2 = np.abs(np.sin(np.floor(k * (2 * x[:, 0:2] - r)) * np.pi / 2))
        g = 1 + np.sum(tmp1 ** 2, axis=1) + np.prod(tmp2)
        f1 = g * np.cos(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f2 = g * np.sin(0.5 * np.pi * x[:, 1]) * np.cos(0.5 * np.pi * x[:, 0])
        f3 = g * np.sin(0.5 * np.pi * x[:, 0])
        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H),
                             indexing="xy")
        k = 10 * snap_near_zero(np.sin(np.pi * self.time))
        tmp2 = np.abs(
            (np.sin((np.floor(k * (2 * x1 - 1)) * np.pi) / 2)
             * np.sin((np.floor(k * (2 * x2 - 1)) * np.pi) / 2)))
        g = 1 + tmp2
        f1 = np.multiply(np.multiply(g, np.cos(0.5 * np.pi * x2)),
                         np.cos(0.5 * np.pi * x1))
        f2 = np.multiply(np.multiply(g, np.sin(0.5 * np.pi * x2)),
                         np.cos(0.5 * np.pi * x1))
        f3 = np.multiply(g, np.sin(0.5 * np.pi * x1))
        return get_PF(np.array([f1, f2, f3]), True)


class DF13Stable(DF13):
    def _evaluate(self, x, out, *args, **kwargs):
        G = snap_near_zero(np.sin(0.5 * np.pi * self.time))
        p = np.floor(6 * G)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)
        g = 1 + np.sum((x[:, 2:] - G) ** 2, axis=1)
        g = g.reshape(len(g), 1)
        f1 = g * np.cos(0.5 * np.pi * x0) ** 2
        f2 = g * np.cos(0.5 * np.pi * x1) ** 2
        f3 = g * (np.sin(0.5 * np.pi * x0) ** 2
                  + np.sin(0.5 * np.pi * x0) * np.cos(p * np.pi * x0) ** 2
                  + np.sin(0.5 * np.pi * x1) ** 2
                  + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2)
        out["F"] = np.column_stack([f1, f2, f3])

    def _calc_pareto_front(self, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H),
                             indexing="xy")
        G = snap_near_zero(np.sin(0.5 * np.pi * self.time))
        p = np.floor(6 * G)
        f1 = np.cos(0.5 * np.pi * x1) ** 2
        f2 = np.cos(0.5 * np.pi * x2) ** 2
        f3 = (np.sin(0.5 * np.pi * x1) ** 2
              + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2
              + np.sin(0.5 * np.pi * x2) ** 2
              + np.sin(0.5 * np.pi * x2) * np.cos(p * np.pi * x2) ** 2)
        return get_PF(np.array([f1, f2, f3]), True)
