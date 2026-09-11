"""L1-L10: dynamic environment timeline (no fake first change)."""
import os
import sys

import numpy as np
import pytest
from pymoo.problems.dynamic.df import DF1

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import dynamic_runner  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402

SEGS = [[0], list(range(1, 10))]


def run(n_t=10, tau_t=5, changes=3, warm_up=20, seed=0):
    return dynamic_runner.run_sa_drl(
        DF1, n_t=n_t, tau_t=tau_t, N=20, D=10, n_changes=changes,
        warm_up=warm_up, agent=NSGA2Baseline(n_segments=2), segments=SEGS,
        seed=seed, training=False, ref_point=(2.0, 2.0), n_elite=5)


def transitions(res):
    return [(e["t_old"], e["t_new"]) for e in res["timeline"]]


# ------------------------------------------ L1: FIRST CHANGE IS REAL
def test_l1_first_change_is_real():
    tl = run(n_t=10, tau_t=5, changes=3)["timeline"]
    assert np.isclose(tl[0]["t_old"], 0.0)
    assert np.isclose(tl[0]["t_new"], 0.1)
    assert not np.isclose(tl[0]["t_old"], tl[0]["t_new"])


# ------------------------------------------ L2: EXACT CHANGE SEQUENCE
def test_l2_exact_sequence():
    tr = transitions(run(n_t=10, tau_t=5, changes=3))
    assert len(tr) == 3
    for got, exp in zip(tr, [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]):
        assert np.isclose(got[0], exp[0]) and np.isclose(got[1], exp[1])


# ------------------------------ L3: TAU_T GENERATIONS PER ENVIRONMENT
def test_l3_tau_t_generations_per_env():
    for e in run(n_t=10, tau_t=5, changes=3)["timeline"]:
        assert e["gens_in_old_env"] == 5


# ----------------------------- L4: CHANGES MEANS ACTUAL TRANSITIONS
def test_l4_changes_are_transitions():
    assert transitions(run(changes=1)) == [(0.0, 0.1)]
    tr5 = transitions(run(changes=5))
    assert len(tr5) == 5
    assert np.isclose(tr5[-1][1], 0.5)


# ---------------------------------------------- L5: NO SAME-TIME CHANGE
def test_l5_no_same_time_change():
    for t_old, t_new in transitions(run(changes=20, tau_t=3)):
        assert not np.isclose(t_old, t_new)


# --------------------------- L6: DETECTOR CALLED ONCE PER ACTUAL CHANGE
def test_l6_detector_calls(monkeypatch):
    calls = {"n": 0}
    orig = dynamic_runner.ChangeDetector.compute

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(dynamic_runner.ChangeDetector, "compute", spy)
    run(changes=4)
    assert calls["n"] == 4


# ------------------------- L7: MEMORY QUERY ONCE PER ACTUAL CHANGE
def test_l7_query_calls(monkeypatch):
    calls = {"n": 0}
    orig = dynamic_runner.MemoryArchive.query

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(dynamic_runner.MemoryArchive, "query", spy)
    run(changes=4)
    assert calls["n"] == 4


# ----------------------------------------- L8: PERIODIC TIME STILL WORKS
def test_l8_periodic():
    tr = transitions(run(n_t=10, tau_t=3, changes=40))
    assert len(tr) == 40
    assert np.isclose(tr[-1][0], 3.9) and np.isclose(tr[-1][1], 4.0)


# ------------------------------------------------- L9: REPRODUCIBILITY
def test_l9_reproducible():
    a = run(changes=5, seed=3)["timeline"]
    b = run(changes=5, seed=3)["timeline"]
    assert [(e["change_index"], e["t_old"], e["t_new"]) for e in a] == \
           [(e["change_index"], e["t_old"], e["t_new"]) for e in b]


# --------------------------- L10: WARMUP DOES NOT SHIFT ENV TIME
def test_l10_warmup_independent():
    # Spec goi y warm_up=0, nhung prime/calibrate gan voi gen==warm_up-1
    # (coupling tu 4.3, ngoai scope 4.6A) nen warm_up=0 khong prime duoc
    # probe -> dung warm_up=5 vs 50; ca hai chung minh cung invariant:
    # warm-up khong dich physical environment index.
    t5 = transitions(run(warm_up=5, changes=5))
    t50 = transitions(run(warm_up=50, changes=5))
    for a, b in zip(t5, t50):
        assert np.isclose(a[0], b[0]) and np.isclose(a[1], b[1])
    assert np.isclose(t5[0][1], 0.1)
