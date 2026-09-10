import warnings
from dataclasses import dataclass

import numpy as np

_TOL = 1e-12


@dataclass
class Stencil:
    """Cong thuc sai phan cho MOT cap (probe, segment). Bat bien theo t."""
    mode: str               # "central" | "onesided2" | "onesided1"
    direction: np.ndarray   # (D,), ||direction||_2 == 1.0
    h: float                # == eps


class ChangeDetector:
    """
    Temporal segment-change estimator.

    c_s(t) do MUC DO THAY DOI cua segment s giua t-1 va t,
    KHONG phai do do nhay cua objective trong moi truong t.
    """

    def __init__(self, segments, obj_scale, eps=0.01,
                 kappa=2.0, lam=0.05, lb=0.0, ub=1.0, verbose=True):
        """
        segments : list[list[int]]  chi so bien moi segment
        obj_scale: np.ndarray (M,)  thang co dinh cho tung objective
                                    (dung ref_point)
        eps      : float  L2 norm cua perturbation, GIONG NHAU
                          cho moi segment bat ke so chieu
        kappa    : float  he so tranh saturation
        lam      : float  toc do EMA
        lb, ub   : bien duoi/tren cua bien quyet dinh
        """
        self.segments = [list(s) for s in segments]
        self.obj_scale = np.asarray(obj_scale, dtype=float)
        self.eps = float(eps)
        self.kappa = float(kappa)
        self.lam = float(lam)
        self.verbose = verbose

        self.D = 1 + max(i for s in self.segments for i in s)
        self.lb = np.broadcast_to(np.asarray(lb, dtype=float), (self.D,)).copy()
        self.ub = np.broadcast_to(np.asarray(ub, dtype=float), (self.D,)).copy()

        self.last_prime_stats = None
        self.reset()

    def reset(self):
        """Goi o dau MOI episode/dynamic sequence."""
        self._g_prev = None     # cached gradients tai t-1, (S, n, M)
        self._X_prev = None     # probe set da dung cho _g_prev
        self._stencils = None   # [probe_idx][segment_idx] -> Stencil
        self._q = None          # EMA scale, None = chua khoi tao

    # ------------------------------------------------------------ stencil
    def _in_bounds(self, y):
        return bool(np.all(y >= self.lb - _TOL) and np.all(y <= self.ub + _TOL))

    def _build_stencils(self, X):
        """
        QUY TAC S1: chi doi DAU tung toa do de huong vao trong mien.
        direction[j] = sign[j] / sqrt(n_s)  =>  ||direction||_2 == 1.0
        luon dung, khong bao gio can rescale.

        QUY TAC S2: chon mode theo tinh kha thi cua diem probe.
        """
        stencils, counts = [], {"central": 0, "onesided2": 0, "onesided1": 0}
        h = self.eps
        for i in range(len(X)):
            x, row = X[i], []
            for idx in self.segments:
                n_s = len(idx)
                a = self.eps / np.sqrt(n_s)
                d = np.zeros(self.D)
                for j in idx:
                    if x[j] - a < self.lb[j]:
                        sign = 1.0          # sat bien duoi
                    elif x[j] + a > self.ub[j]:
                        sign = -1.0         # sat bien tren
                    else:
                        sign = 1.0          # canonical
                    d[j] = sign / np.sqrt(n_s)

                if self._in_bounds(x + h * d) and self._in_bounds(x - h * d):
                    mode = "central"
                elif self._in_bounds(x + 2 * h * d):
                    mode = "onesided2"
                else:
                    mode = "onesided1"
                counts[mode] += 1
                row.append(Stencil(mode, d, h))
            stencils.append(row)
        return stencils, counts

    # ---------------------------------------------------------- gradients
    def _gradients(self, X, problem):
        """
        QUY TAC S3: cong thuc theo mode.
            central   : [F(x+hd) - F(x-hd)] / (2h)            O(h^2)
            onesided2 : [-3F(x) + 4F(x+hd) - F(x+2hd)] / (2h) O(h^2)
            onesided1 : [F(x+hd) - F(x)] / h                  O(h)

        Dung bac hai o bien vi central la O(h^2): neu bien dung forward
        bac mot, sai so cat cut khong triet tieu qua phep tru temporal
        (H(t) cua DF doi theo t) va se nhan chim tin hieu that.

        Diem probe luon nam trong mien nho stencil — KHONG clip.
        return: (g_hat (S, n, M), fe_used THUC TE)
        """
        n, S = len(X), len(self.segments)
        fe = 0

        # QUY TAC S5: F(x, t) dung chung cho moi segment cua cung probe.
        need_base = [i for i in range(n)
                     if any(st.mode != "central" for st in self._stencils[i])]
        F_base = {}
        if need_base:
            Fb = problem.evaluate(X[need_base])
            fe += len(need_base)
            F_base = {i: Fb[k] for k, i in enumerate(need_base)}

        pts, meta = [], []
        for i in range(n):
            for s in range(S):
                st = self._stencils[i][s]
                step = st.h * st.direction
                if st.mode == "central":
                    pts += [X[i] + step, X[i] - step]
                    meta += [(i, s, "p1"), (i, s, "m1")]
                elif st.mode == "onesided2":
                    pts += [X[i] + step, X[i] + 2 * step]
                    meta += [(i, s, "p1"), (i, s, "p2")]
                else:
                    pts.append(X[i] + step)
                    meta.append((i, s, "p1"))

        F_pts = problem.evaluate(np.asarray(pts))
        fe += len(pts)
        buf = {}
        for k, key in enumerate(meta):
            buf[key] = F_pts[k]

        M = F_pts.shape[1]
        g = np.zeros((S, n, M))
        for i in range(n):
            for s in range(S):
                st = self._stencils[i][s]
                h = st.h
                if st.mode == "central":
                    g[s, i] = (buf[(i, s, "p1")] - buf[(i, s, "m1")]) / (2 * h)
                elif st.mode == "onesided2":
                    g[s, i] = (-3 * F_base[i] + 4 * buf[(i, s, "p1")]
                               - buf[(i, s, "p2")]) / (2 * h)
                else:
                    g[s, i] = (buf[(i, s, "p1")] - F_base[i]) / h
        return g / self.obj_scale, fe

    # --------------------------------------------------------------- API
    def prime(self, X_probe, problem_t):
        """
        Quyet dinh stencil MOT LAN, tinh va cache g_{s,t} MA KHONG tra ve c.
        Goi o CUOI warm-up de lan doi dau tien da co t-1.
        return: fe_used (int)
        """
        X = np.array(X_probe, dtype=float, copy=True)
        self._stencils, counts = self._build_stencils(X)
        total = sum(counts.values())

        if counts["onesided1"]:
            frac = counts["onesided1"] / total
            msg = (f"onesided1 fallback used for {counts['onesided1']} "
                   f"probe-segment pairs ({frac:.1%})")
            if frac > 0.05:
                msg += " — ty le cao, eps={self.eps} co the qua lon".format(
                    self=self)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        self.last_prime_stats = counts
        if self.verbose:
            print(f"  [stencil] central={counts['central']} "
                  f"onesided2={counts['onesided2']} "
                  f"onesided1={counts['onesided1']}")

        self._g_prev, fe = self._gradients(X, problem_t)
        self._X_prev = X
        return fe

    def compute(self, X_probe, problem_t, problem_prev=None):
        """
        X_probe    : (n, D) probe set — CUNG mot set cho t-1 va t
        problem_t  : problem tai t
        problem_prev: chi dung khi cold-start

        return: dict {
            "c": np.ndarray (S,)          normalized, in [0,1)
            "c_tilde": np.ndarray (S,)    raw temporal change
            "fe_used": int                so lan evaluate ca the
        }
        """
        if self._stencils is None or self._g_prev is None:
            if problem_prev is None:
                raise RuntimeError(
                    "ChangeDetector.compute() goi khi chua co stencil/"
                    "g_{t-1}. Goi prime() o cuoi warm-up truoc.")
            self.prime(X_probe, problem_prev)

        X = np.asarray(X_probe, dtype=float)
        # QUY TAC 3: PHAI cung probe set cho t-1 va t, neu khong hieu
        # se tron environmental change LAN population movement.
        if X.shape != self._X_prev.shape or not np.allclose(X, self._X_prev):
            raise RuntimeError(
                "probe set khac voi probe set da dung cho g_{t-1}. "
                "c_tilde se tron environmental change lan population "
                "movement. Dung CUNG mot X cho ca t-1 va t.")

        # QUY TAC S4: CHI doc stencil da cache, KHONG quyet dinh lai.
        g_t, fe = self._gradients(X, problem_t)
        c_tilde = np.linalg.norm(g_t - self._g_prev, axis=2).mean(axis=1)
        if not np.all(np.isfinite(c_tilde)):
            raise RuntimeError(f"non-finite c_tilde: {c_tilde}")

        # QUY TAC 4: normalize bang scale QUA KHU truoc, roi moi update EMA.
        m_t = float(np.max(c_tilde))
        if self._q is None:
            self._q = m_t
            c = np.zeros(len(self.segments))
        else:
            denom = self.kappa * self._q + 1e-12
            c = 1.0 - np.exp(-c_tilde / denom)
        self._q = (1 - self.lam) * self._q + self.lam * m_t

        self._g_prev = g_t
        return {"c": c, "c_tilde": c_tilde, "fe_used": fe}
