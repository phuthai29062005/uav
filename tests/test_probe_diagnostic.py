"""PR1-PR10: frozen-probe representativeness diagnostic invariants (4.10D).

Diagnostic-only: kiem tinh chat cua instrumentation, khong test performance
threshold, khong sua production.
"""
import os
import sys

import numpy as np
import pytest
from pymoo.problems.dynamic.df import DF1, DF4

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import dynamic_runner as dr  # noqa: E402
from audit_probe_representativeness import zdist_min  # noqa: E402
from baselines import NSGA2Baseline  # noqa: E402

D = 10
SEGS = [[0], list(range(1, D))]


def short(cls=DF1, K=6, seed=0, ref=(2.0, 2.0), capture_probe=None):
    return dr.run_sa_drl(cls, n_t=10, tau_t=3, N=20, D=D, n_changes=K,
                         warm_up=5, agent=NSGA2Baseline(n_segments=2),
                         segments=SEGS, seed=seed, training=False,
                         ref_point=ref, n_elite=5)


# ------------- PR1: normalized z-distance scale-invariant across bounds
def test_pr1_scale_invariant():
    rng = np.random.RandomState(0)
    A, B = rng.rand(4, D), rng.rand(6, D)
    d_unit = zdist_min(A, B, np.zeros(D), np.ones(D))
    # map to [-2,2]
    A2, B2 = -2 + 4 * A, -2 + 4 * B
    d_wide = zdist_min(A2, B2, np.full(D, -2.0), np.full(D, 2.0))
    assert np.allclose(d_unit, d_wide)


# --------------------- PR2: frozen X_probe unchanged after K changes
def test_pr2_probe_frozen():
    seen = []
    orig = dr.ChangeDetector.prime

    def spy(self, X, p):
        seen.append(np.array(X, copy=True))
        return orig(self, X, p)
    dr.ChangeDetector.prime = spy
    try:
        short(K=8)
    finally:
        dr.ChangeDetector.prime = orig
    assert len(seen) == 1                       # prime chi 1 lan (cuoi warmup)


# --------------- PR3: diagnostic capture khong doi fes_used
def test_pr3_no_fe_contamination():
    a = short(K=6)["fes_used"]
    # instrument (capture pop) roi chay lai -> fes_used giong het
    orig = dr.apply_hierarchical_response
    grabbed = []

    def spy(pop, *a2, **k2):
        grabbed.append(np.array(pop, copy=True))
        return orig(pop, *a2, **k2)
    dr.apply_hierarchical_response = spy
    try:
        b = short(K=6)["fes_used"]
    finally:
        dr.apply_hierarchical_response = orig
    assert a == b


# ----------------------------- PR4: distance diagnostics finite
def test_pr4_distances_finite():
    rng = np.random.RandomState(1)
    d = zdist_min(rng.rand(5, D), rng.rand(7, D), np.zeros(D), np.ones(D))
    assert np.all(np.isfinite(d)) and np.all(d >= 0)


# ----------------------------- PR5: detector raw/c finite
def test_pr5_detector_finite():
    res = short(cls=DF4, K=6, ref=(5.0, 5.0))
    for k in ("raw_change", "normalized_change"):
        for vec in res[k]:
            assert np.all(np.isfinite(vec))


# ------------------ PR6/PR7: stencil fractions valid
def test_pr6_pr7_stencil_fractions():
    captured = {}
    orig = dr.ChangeDetector.prime

    def spy(self, X, p):
        r = orig(self, X, p)
        captured["stats"] = dict(self.last_prime_stats)
        return r
    dr.ChangeDetector.prime = spy
    try:
        short(K=4)
    finally:
        dr.ChangeDetector.prime = orig
    st = captured["stats"]
    total = st["central"] + st["onesided2"] + st["onesided1"]
    assert total > 0
    frac1 = st["onesided1"] / total
    assert 0.0 <= frac1 <= 1.0


# --------------------------- PR8: signature saturation in [0,1]
def test_pr8_signature_saturation():
    res = short(K=6)
    for s in res["sig_saturation"]:
        assert 0.0 <= s <= 1.0


# ------------------------------- PR9: diagnostic reproducible
def test_pr9_reproducible():
    a = short(K=6, seed=3)
    b = short(K=6, seed=3)
    assert np.allclose(a["raw_change"], b["raw_change"])
    assert np.allclose(a["d_mem"], b["d_mem"])


# ------------- PR10: capture helper khong mutate pop/probe
def test_pr10_no_mutation():
    orig = dr.apply_hierarchical_response
    before_after = []

    def spy(pop, *a2, **k2):
        snap = np.array(pop, copy=True)
        out = orig(pop, *a2, **k2)
        before_after.append(np.array_equal(snap, pop))   # pop arg bat bien
        return out
    dr.apply_hierarchical_response = spy
    try:
        short(K=5)
    finally:
        dr.apply_hierarchical_response = orig
    assert all(before_after)
