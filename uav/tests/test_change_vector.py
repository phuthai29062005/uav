"""
Test A5: compute_change_vector phai clip nhieu ve [0,1] truoc khi
evaluate, de khong danh gia ham muc tieu ngoai mien dinh nghia.

Chay:  python -m pytest tests/test_change_vector.py -v
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from pymoo.problems.dynamic.df import DF1  # noqa: E402
from sa_drl_dmoea import compute_change_vector  # noqa: E402

SEGMENTS = [[0], list(range(1, 10))]
D = 10


def test_boundary_variables_no_crash():
    """Elite sat bien 1.0 o segment 2 -> c van huu han va trong [0,1]."""
    elite = np.full((5, 10), 0.5)
    elite[:, 1:] = 0.999                     # segment 2 sat bien
    problem = DF1(time=0.5, n_var=D)

    c = compute_change_vector(elite, problem, SEGMENTS, D)

    assert c.shape == (2,), f"shape sai: {c.shape}"
    assert np.all(np.isfinite(c)), f"c co NaN/inf: {c}"
    assert 0.0 <= c[0] <= 1.0, f"c[0] ngoai [0,1]: {c[0]}"
    assert 0.0 <= c[1] <= 1.0, f"c[1] ngoai [0,1]: {c[1]}"


class _RecordingProblem:
    """Boc mot problem, ghi lai moi input truyen vao evaluate()."""

    def __init__(self, inner):
        self.inner = inner
        self.seen = []

    def evaluate(self, X):
        self.seen.append(np.array(X, copy=True))
        return self.inner.evaluate(X)


def test_clip_produces_valid_input():
    """Moi input truyen vao evaluate deu nam trong [0,1] (khong co 1.009)."""
    elite = np.full((3, 10), 0.999)          # bien sat bien -> +0.01 se vuot 1.0
    problem = _RecordingProblem(DF1(time=0.5, n_var=D))

    c = compute_change_vector(elite, problem, SEGMENTS, D)

    assert len(problem.seen) > 0, "evaluate chua duoc goi lan nao"
    for X in problem.seen:
        assert X.max() <= 1.0 + 1e-12, f"input vuot bien tren: max={X.max()}"
        assert X.min() >= 0.0 - 1e-12, f"input vuot bien duoi: min={X.min()}"

    # va ket qua van huu han
    assert np.all(np.isfinite(c)), f"c co NaN/inf: {c}"
