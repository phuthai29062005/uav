"""CEC2018 DF1-DF14 dynamic benchmark problems.

Implemented from official MATLAB cec2018_DF.m (S2, Level 1 authoritative).
Cross-validated against pymoo (S6, Level 2).
Variable bounds from pymoo (S6), consistent with official PF code (S7).

Source: S. Jiang et al., "Benchmark Problems for CEC'2018 Competition
on Dynamic Multiobjective Optimisation", Newcastle University, Dec 2017.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import DynamicBenchmarkProblem


def _nds_filter(F: np.ndarray) -> np.ndarray:
    """Keep only non-dominated rows from F (N, n_obj)."""
    is_efficient = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not is_efficient[i]:
            continue
        dominated = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        dominated[i] = False
        is_efficient[dominated] = False
    return F[is_efficient]


class DF1(DynamicBenchmarkProblem):
    # MATLAB: G=abs(sin(0.5*pi*t)); H=0.75*sin(0.5*pi*t)+1.25
    # g=1+sum((x(2:end)-G).^2); f1=x(1); f2=g*(1-(x(1)/g)^H)

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        super().__init__(n_var, 2, nt, taut)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f1 = X[:, 0]
        f2 = g * (1.0 - (X[:, 0] / g) ** H)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** H
        return np.column_stack([f1, f2])


class DF2(DynamicBenchmarkProblem):
    # MATLAB: G=abs(sin(0.5*pi*t)); r=1+floor((n-1)*G)
    # tmp=setdiff(1:n,r); g=1+sum((x(tmp)-G).^2)
    # f1=x(r); f2=g*(1-(x(r)/g)^0.5)

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        super().__init__(n_var, 2, nt, taut)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        n = self.n_var
        G = np.abs(np.sin(0.5 * np.pi * t))
        r = int(np.floor((n - 1) * G))  # 0-based (MATLAB r=1+floor(...) is 1-based)
        not_r = [k for k in range(n) if k != r]
        g = 1.0 + np.sum((X[:, not_r] - G) ** 2, axis=1)
        f1 = X[:, r]
        f2 = g * (1.0 - (X[:, r] / g) ** 0.5)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class DF3(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); H=G+1.5
    # g=1+sum((x(2:end)-G-x(1)^H).^2); f1=x(1); f2=g*(1-(x(1)/g)^H)
    # Bounds: x1 in [0,1], x2..xn in [-1,2]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[1:] = -1.0
        xu[1:] = 2.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        H = G + 1.5
        g = 1.0 + np.sum((X[:, 1:] - G - X[:, 0:1] ** H) ** 2, axis=1)
        f1 = X[:, 0]
        f2 = g * (1.0 - (X[:, 0] / g) ** H)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        G = np.sin(0.5 * np.pi * t)
        H = G + 1.5
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** H
        return np.column_stack([f1, f2])


class DF4(DynamicBenchmarkProblem):
    # MATLAB: a=sin(0.5*pi*t); b=1+abs(cos(0.5*pi*t));
    # c=max(abs(a),a+b); H=1.5+a
    # g=1+sum((x(2:end)-a*(x(1)/c).^2./[2:n]).^2)
    # f1=g*abs(x(1)-a).^H; f2=g*abs(x(1)-a-b).^H
    # Bounds: all in [-2, 2]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.full(n_var, -2.0)
        xu = np.full(n_var, 2.0)
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        n = self.n_var
        a = np.sin(0.5 * np.pi * t)
        b = 1.0 + np.abs(np.cos(0.5 * np.pi * t))
        c = max(np.abs(a), a + b)
        H = 1.5 + a
        # [2:n] in MATLAB 1-based = indices 2,3,...,n = values 2,3,...,n
        # In 0-based: x[:,1:] corresponds to MATLAB x(2:end)
        # Divisor for x[:,i] (0-based i=1..n-1) is (i+1) = 2,3,...,n
        indices = np.arange(2, n + 1, dtype=float)  # [2, 3, ..., n]
        g = 1.0 + np.sum((X[:, 1:] - a * (X[:, 0:1] / c) ** 2 / indices) ** 2, axis=1)
        f1 = g * np.abs(X[:, 0] - a) ** H
        f2 = g * np.abs(X[:, 0] - a - b) ** H
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        a = np.sin(0.5 * np.pi * t)
        b = 1.0 + np.abs(np.cos(0.5 * np.pi * t))
        H = 1.5 + a
        x = np.linspace(a, a + b, n_points)
        f1 = np.abs(x - a) ** H
        f2 = np.abs(x - a - b) ** H
        return np.column_stack([f1, f2])


class DF5(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); w=floor(10*G)
    # g=1+sum((x(2:end)-G).^2)
    # f1=g*(x(1)+0.02*sin(w*pi*x(1)));
    # f2=g*(1-x(1)+0.02*sin(w*pi*x(1)))
    # Bounds: x1 in [0,1], x2..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[1:] = -1.0
        xu[1:] = 1.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        w = np.floor(10.0 * G)
        g = 1.0 + np.sum((X[:, 1:] - G) ** 2, axis=1)
        f1 = g * (X[:, 0] + 0.02 * np.sin(w * np.pi * X[:, 0]))
        f2 = g * (1.0 - X[:, 0] + 0.02 * np.sin(w * np.pi * X[:, 0]))
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        G = np.sin(0.5 * np.pi * t)
        w = np.floor(10.0 * G)
        x = np.linspace(0, 1, n_points)
        f1 = x + 0.02 * np.sin(w * np.pi * x)
        f2 = 1.0 - x + 0.02 * np.sin(w * np.pi * x)
        return np.column_stack([f1, f2])


class DF6(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); a=0.2+2.8*abs(G)
    # y=x(2:end)-G; g=1+sum((abs(G)*y.^2-10*cos(2*pi*y)+10))
    # f1=g*(x(1)+0.1*sin(3*pi*x(1))).^a
    # f2=g*(1-x(1)+0.1*sin(3*pi*x(1))).^a
    # Note: MATLAB has "x1" which is a typo for x(1); confirmed by pymoo
    # Bounds: x1 in [0,1], x2..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[1:] = -1.0
        xu[1:] = 1.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        a = 0.2 + 2.8 * np.abs(G)
        y = X[:, 1:] - G
        g = 1.0 + np.sum(np.abs(G) * y ** 2 - 10.0 * np.cos(2.0 * np.pi * y) + 10.0, axis=1)
        f1 = g * (X[:, 0] + 0.1 * np.sin(3.0 * np.pi * X[:, 0])) ** a
        f2 = g * (1.0 - X[:, 0] + 0.1 * np.sin(3.0 * np.pi * X[:, 0])) ** a
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        G = np.sin(0.5 * np.pi * t)
        a = 0.2 + 2.8 * np.abs(G)
        x = np.linspace(0, 1, n_points)
        f1 = (x + 0.1 * np.sin(3.0 * np.pi * x)) ** a
        f2 = (1.0 - x + 0.1 * np.sin(3.0 * np.pi * x)) ** a
        return np.column_stack([f1, f2])


class DF7(DynamicBenchmarkProblem):
    # MATLAB: a=5*cos(0.5*pi*t); tmp=1/(1+exp(a*(x(1)-2.5)))
    # g=1+sum((x(2:end)-tmp).^2)
    # f1=g*(1+t)/x(1); f2=g*x(1)/(1+t)
    # Bounds: x1 in [1,4], x2..xn in [0,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[0] = 1.0
        xu[0] = 4.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        a = 5.0 * np.cos(0.5 * np.pi * t)
        tmp = 1.0 / (1.0 + np.exp(a * (X[:, 0:1] - 2.5)))
        g = 1.0 + np.sum((X[:, 1:] - tmp) ** 2, axis=1)
        f1 = g * (1.0 + t) / X[:, 0]
        f2 = g * X[:, 0] / (1.0 + t)
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        x = np.linspace(1, 4, n_points)
        f1 = (1.0 + t) / x
        f2 = x / (1.0 + t)
        return np.column_stack([f1, f2])


class DF8(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); a=2.25+2*cos(2*pi*t)
    # tmp=G*sin(4*pi*x(1))/(1+abs(G))  [NO exponent b]
    # g=1+sum((x(2:end)-tmp).^2)
    # f1=g*(x(1)+0.1*sin(3*pi*x(1)))
    # f2=g*(1-x(1)+0.1*sin(3*pi*x(1))).^a  [uses x(1), NOT x(2)]
    # Bounds: x1 in [0,1], x2..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[1:] = -1.0
        xu[1:] = 1.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        a = 2.25 + 2.0 * np.cos(2.0 * np.pi * t)
        tmp = G * np.sin(4.0 * np.pi * X[:, 0:1]) / (1.0 + np.abs(G))
        g = 1.0 + np.sum((X[:, 1:] - tmp) ** 2, axis=1)
        f1 = g * (X[:, 0] + 0.1 * np.sin(3.0 * np.pi * X[:, 0]))
        f2 = g * (1.0 - X[:, 0] + 0.1 * np.sin(3.0 * np.pi * X[:, 0])) ** a
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        a = 2.25 + 2.0 * np.cos(2.0 * np.pi * t)
        x = np.linspace(0, 1, n_points)
        f1 = x + 0.1 * np.sin(3.0 * np.pi * x)
        f2 = (1.0 - x + 0.1 * np.sin(3.0 * np.pi * x)) ** a
        return np.column_stack([f1, f2])


class DF9(DynamicBenchmarkProblem):
    # MATLAB: N=1+floor(10*abs(sin(0.5*pi*t)))
    # g=1; for i=2:n, tmp=x(i)-cos(4*t+x(1)+x(i-1)); g=g+tmp.^2; end
    # f1=g*(x(1)+max(0,(0.1+0.5/N)*sin(2*N*pi*x(1))))
    # f2=g*(1-x(1)+max(0,(0.1+0.5/N)*sin(2*N*pi*x(1))))
    # Bounds: x1 in [0,1], x2..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[1:] = -1.0
        xu[1:] = 1.0
        super().__init__(n_var, 2, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        n = self.n_var
        N = 1.0 + np.floor(10.0 * np.abs(np.sin(0.5 * np.pi * t)))
        g = np.ones(len(X))
        for i in range(1, n):
            tmp = X[:, i] - np.cos(4.0 * t + X[:, 0] + X[:, i - 1])
            g = g + tmp ** 2
        f1 = g * (X[:, 0] + np.maximum(0.0, (0.1 + 0.5 / N) * np.sin(2.0 * N * np.pi * X[:, 0])))
        f2 = g * (1.0 - X[:, 0] + np.maximum(0.0, (0.1 + 0.5 / N) * np.sin(2.0 * N * np.pi * X[:, 0])))
        return np.column_stack([f1, f2])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1500) -> np.ndarray:
        if t is None:
            t = self.time()
        N = 1.0 + np.floor(10.0 * np.abs(np.sin(0.5 * np.pi * t)))
        x = np.linspace(0, 1, n_points)
        f1 = x + np.maximum(0.0, (0.1 + 0.5 / N) * np.sin(2.0 * N * np.pi * x))
        f2 = 1.0 - x + np.maximum(0.0, (0.1 + 0.5 / N) * np.sin(2.0 * N * np.pi * x))
        pf = np.column_stack([f1, f2])
        return _nds_filter(pf)


class DF10(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); H=2.25+2*cos(0.5*pi*t)
    # tmp=sin(4*pi*(x(1)+x(2)))/(1+abs(G))  [4*pi, NOT 2*pi]
    # g=1+sum((x(3:end)-tmp).^2)
    # f1=g*sin(0.5*pi*x(1)).^H
    # f2=g*sin(0.5*pi*x(2)).^H.*cos(0.5*pi*x(1)).^H
    # f3=g*cos(0.5*pi*x(2)).^H.*cos(0.5*pi*x(1)).^H
    # Bounds: x1,x2 in [0,1], x3..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[2:] = -1.0
        xu[2:] = 1.0
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        H = 2.25 + 2.0 * np.cos(0.5 * np.pi * t)
        tmp = np.sin(4.0 * np.pi * (X[:, 0:1] + X[:, 1:2])) / (1.0 + np.abs(G))
        g = 1.0 + np.sum((X[:, 2:] - tmp) ** 2, axis=1)
        f1 = g * np.sin(0.5 * np.pi * X[:, 0]) ** H
        f2 = g * (np.sin(0.5 * np.pi * X[:, 1]) * np.cos(0.5 * np.pi * X[:, 0])) ** H
        f3 = g * (np.cos(0.5 * np.pi * X[:, 1]) * np.cos(0.5 * np.pi * X[:, 0])) ** H
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        H = 2.25 + 2.0 * np.cos(0.5 * np.pi * t)
        n_per = int(np.ceil(np.sqrt(n_points)))
        u, v = np.meshgrid(np.linspace(0, 1, n_per), np.linspace(0, 1, n_per))
        u, v = u.ravel(), v.ravel()
        f1 = np.sin(0.5 * np.pi * u) ** H
        f2 = (np.sin(0.5 * np.pi * v) * np.cos(0.5 * np.pi * u)) ** H
        f3 = (np.cos(0.5 * np.pi * v) * np.cos(0.5 * np.pi * u)) ** H
        pf = np.column_stack([f1, f2, f3])
        if len(pf) > n_points:
            idx = np.linspace(0, len(pf) - 1, n_points, dtype=int)
            pf = pf[idx]
        return pf


class DF11(DynamicBenchmarkProblem):
    # MATLAB: G=abs(sin(0.5*pi*t))
    # g=1+G+sum((x(3:end)-0.5*G*x(1)).^2)
    # y=pi*G/6+(pi/2-pi*G/3)*x(1:2)
    # f1=g*sin(y(1)); f2=g*sin(y(2))*cos(y(1)); f3=g*cos(y(2))*cos(y(1))
    # Bounds: all in [0,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        super().__init__(n_var, 3, nt, taut)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + G + np.sum((X[:, 2:] - 0.5 * G * X[:, 0:1]) ** 2, axis=1)
        y0 = np.pi * G / 6.0 + (np.pi / 2.0 - np.pi * G / 3.0) * X[:, 0]
        y1 = np.pi * G / 6.0 + (np.pi / 2.0 - np.pi * G / 3.0) * X[:, 1]
        f1 = g * np.sin(y0)
        f2 = g * np.sin(y1) * np.cos(y0)
        f3 = g * np.cos(y1) * np.cos(y0)
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        G = np.abs(np.sin(0.5 * np.pi * t))
        g_opt = 1.0 + G
        n_per = int(np.ceil(np.sqrt(n_points)))
        u, v = np.meshgrid(np.linspace(0, 1, n_per), np.linspace(0, 1, n_per))
        u, v = u.ravel(), v.ravel()
        y0 = np.pi * G / 6.0 + (np.pi / 2.0 - np.pi * G / 3.0) * u
        y1 = np.pi * G / 6.0 + (np.pi / 2.0 - np.pi * G / 3.0) * v
        f1 = g_opt * np.sin(y0)
        f2 = g_opt * np.sin(y1) * np.cos(y0)
        f3 = g_opt * np.cos(y1) * np.cos(y0)
        pf = np.column_stack([f1, f2, f3])
        if len(pf) > n_points:
            idx = np.linspace(0, len(pf) - 1, n_points, dtype=int)
            pf = pf[idx]
        return pf


class DF12(DynamicBenchmarkProblem):
    # MATLAB: k=10*sin(pi*t)
    # tmp1=x(3:end)-sin(t*x(1))
    # tmp2=abs(sin(floor(k*(2*x(1:2)-1))*pi/2))  [uses BOTH x1,x2; with abs]
    # g=1+sum(tmp1.^2)+prod(tmp2)
    # f1=g*cos(0.5*pi*x(2))*cos(0.5*pi*x(1))
    # f2=g*sin(0.5*pi*x(2))*cos(0.5*pi*x(1))
    # f3=g*sin(0.5*pi*x(1))  [uses x(1), NOT x(2)]
    # Bounds: x1,x2 in [0,1], x3..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[2:] = -1.0
        xu[2:] = 1.0
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        k = 10.0 * np.sin(np.pi * t)
        tmp1 = X[:, 2:] - np.sin(t * X[:, 0:1])
        tmp2 = np.abs(np.sin(np.floor(k * (2.0 * X[:, 0:2] - 1.0)) * np.pi / 2.0))
        g = 1.0 + np.sum(tmp1 ** 2, axis=1) + np.prod(tmp2, axis=1)
        f1 = g * np.cos(0.5 * np.pi * X[:, 1]) * np.cos(0.5 * np.pi * X[:, 0])
        f2 = g * np.sin(0.5 * np.pi * X[:, 1]) * np.cos(0.5 * np.pi * X[:, 0])
        f3 = g * np.sin(0.5 * np.pi * X[:, 0])
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        k = 10.0 * np.sin(np.pi * t)
        n_per = int(np.ceil(np.sqrt(n_points)))
        u, v = np.meshgrid(np.linspace(0, 1, n_per), np.linspace(0, 1, n_per))
        u, v = u.ravel(), v.ravel()
        tmp2 = np.abs(
            np.sin(np.floor(k * (2.0 * u - 1.0)) * np.pi / 2.0)
            * np.sin(np.floor(k * (2.0 * v - 1.0)) * np.pi / 2.0)
        )
        g = 1.0 + tmp2
        f1 = g * np.cos(0.5 * np.pi * v) * np.cos(0.5 * np.pi * u)
        f2 = g * np.sin(0.5 * np.pi * v) * np.cos(0.5 * np.pi * u)
        f3 = g * np.sin(0.5 * np.pi * u)
        pf = np.column_stack([f1, f2, f3])
        return _nds_filter(pf)


class DF13(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); p=floor(6*G)
    # g=1+sum((x(3:end)-G).^2)
    # f1=g*cos(0.5*pi*x(1)).^2
    # f2=g*cos(0.5*pi*x(2)).^2
    # f3=g*sin(0.5*pi*x(1)).^2+sin(0.5*pi*x(1)).*cos(p*pi*x(1)).^2+...
    #    sin(0.5*pi*x(2)).^2+sin(0.5*pi*x(2)).*cos(p*pi*x(2)).^2
    # Bounds: x1,x2 in [0,1], x3..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[2:] = -1.0
        xu[2:] = 1.0
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        p = np.floor(6.0 * G)
        g = 1.0 + np.sum((X[:, 2:] - G) ** 2, axis=1)
        x0 = X[:, 0]
        x1 = X[:, 1]
        f1 = g * np.cos(0.5 * np.pi * x0) ** 2
        f2 = g * np.cos(0.5 * np.pi * x1) ** 2
        f3 = (g * np.sin(0.5 * np.pi * x0) ** 2
              + np.sin(0.5 * np.pi * x0) * np.cos(p * np.pi * x0) ** 2
              + np.sin(0.5 * np.pi * x1) ** 2
              + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2)
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        G = np.sin(0.5 * np.pi * t)
        p = np.floor(6.0 * G)
        n_per = int(np.ceil(np.sqrt(n_points)))
        u, v = np.meshgrid(np.linspace(0, 1, n_per), np.linspace(0, 1, n_per))
        u, v = u.ravel(), v.ravel()
        f1 = np.cos(0.5 * np.pi * u) ** 2
        f2 = np.cos(0.5 * np.pi * v) ** 2
        f3 = (np.sin(0.5 * np.pi * u) ** 2
              + np.sin(0.5 * np.pi * u) * np.cos(p * np.pi * u) ** 2
              + np.sin(0.5 * np.pi * v) ** 2
              + np.sin(0.5 * np.pi * v) * np.cos(p * np.pi * v) ** 2)
        pf = np.column_stack([f1, f2, f3])
        return _nds_filter(pf)


class DF14(DynamicBenchmarkProblem):
    # MATLAB: G=sin(0.5*pi*t); g=1+sum((x(3:end)-G).^2)
    # y=0.5+G*(x(1)-0.5)
    # f1=g*(1-y+0.05*sin(6*pi*y))
    # f2=g*(1-x(2)+0.05*sin(6*pi*x(2)))*(y+0.05*sin(6*pi*y))
    # f3=g*(x(2)+0.05*sin(6*pi*x(2)))*(y+0.05*sin(6*pi*y))
    # Bounds: x1,x2 in [0,1], x3..xn in [-1,1]

    def __init__(self, n_var: int = 10, nt: int = 10, taut: int = 10):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        xl[2:] = -1.0
        xu[2:] = 1.0
        super().__init__(n_var, 3, nt, taut, xl, xu)

    def evaluate(self, X: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.time()
        X = np.atleast_2d(X)
        G = np.sin(0.5 * np.pi * t)
        g = 1.0 + np.sum((X[:, 2:] - G) ** 2, axis=1)
        y = 0.5 + G * (X[:, 0] - 0.5)
        f1 = g * (1.0 - y + 0.05 * np.sin(6.0 * np.pi * y))
        f2 = g * (1.0 - X[:, 1] + 0.05 * np.sin(6.0 * np.pi * X[:, 1])) * (y + 0.05 * np.sin(6.0 * np.pi * y))
        f3 = g * (X[:, 1] + 0.05 * np.sin(6.0 * np.pi * X[:, 1])) * (y + 0.05 * np.sin(6.0 * np.pi * y))
        return np.column_stack([f1, f2, f3])

    def true_pf(self, t: Optional[float] = None, n_points: int = 1000) -> np.ndarray:
        if t is None:
            t = self.time()
        G = np.sin(0.5 * np.pi * t)
        n_per = int(np.ceil(np.sqrt(n_points)))
        u, v = np.meshgrid(np.linspace(0, 1, n_per), np.linspace(0, 1, n_per))
        u, v = u.ravel(), v.ravel()
        y = 0.5 + G * (u - 0.5)
        f1 = 1.0 - y + 0.05 * np.sin(6.0 * np.pi * y)
        f2 = (1.0 - v + 0.05 * np.sin(6.0 * np.pi * v)) * (y + 0.05 * np.sin(6.0 * np.pi * y))
        f3 = (v + 0.05 * np.sin(6.0 * np.pi * v)) * (y + 0.05 * np.sin(6.0 * np.pi * y))
        pf = np.column_stack([f1, f2, f3])
        if len(pf) > n_points:
            idx = np.linspace(0, len(pf) - 1, n_points, dtype=int)
            pf = pf[idx]
        return pf
