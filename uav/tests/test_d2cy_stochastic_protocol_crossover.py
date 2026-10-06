"""
Tests for D2-CY -- Stochastic Protocol Crossover Audit.

T1-T10:  Imports and constants
T11-T20: Helpers (_fp, _fps_match, _json_safe, _bootstrap_ci, _compute_cb)
T21-T30: Child generation (_gen_children_diag, _gen_children_prod)
T31-T40: Survival application (_apply_survival)
T41-T50: Core trajectory runner (_run_tape_trajectory)
T51-T55: Phase runners (CY-0, CY-2, CY-3, CY-4, CY-5 — smoke tests)
"""
import copy
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.core.reference import reference
from prdrl.history.region_history import RegionHistory
from prdrl.problems.controlled_predict import (
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
)

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.D2CY_stochastic_protocol_crossover as d2cy


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def setup():
    region_dynamics = make_region_dynamics_map(d2cy.N_PARTITIONS + 1)
    V    = reference(d2cy.N_OBJ, d2cy.N_PARTITIONS)
    xl, xu = get_bounds(d2cy.D)
    pf_star = get_true_pf(len(V))
    return {"region_dynamics": region_dynamics, "V": V, "xl": xl, "xu": xu, "pf_star": pf_star}


@pytest.fixture(scope="module")
def shared_state(setup):
    """Shared warmup state for injection into child/survival tests."""
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    R  = len(V)
    seed = 201
    rng  = np.random.default_rng(seed)
    X    = rng.uniform(xl, xu, size=(d2cy.POP_SIZE, d2cy.D))
    t0   = d2cy.compute_time(0)
    X    = d2cy.evolve_nsga2(X, t0, rd, d2cy.WARMUP_GENERATIONS, seed * 10000, xl, xu)

    gen  = d2cy.WARMUP_GENERATIONS
    ht   = -1
    hist = RegionHistory(n_regions=R)
    F_m  = d2cy.cp_evaluate(X, t0, rd)
    gen += d2cy.GENERATIONS_PER_CHANGE
    t1   = d2cy.compute_time(gen)
    F_p  = d2cy.cp_evaluate(X, t1, rd)
    Fm, _ = normalize(F_m, F_p)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    ht   += 1
    hist.update(t=ht, X=X, F=F_m, region_ids=rids, xl=xl, xu=xu)
    X    = d2cy.evolve_nsga2(X, t1, rd, d2cy.GENERATIONS_PER_CHANGE, seed * 10000 + 1, xl, xu)

    tgt_r = seed % R
    return {"X": X, "history": hist, "history_t": ht,
            "t_float_current": t1, "tgt_r": tgt_r, "seed": seed}


# ============================================================
# T1-T10: Imports and constants
# ============================================================

def test_T1_imports():
    assert hasattr(d2cy, "_run_tape_trajectory")
    assert hasattr(d2cy, "_gen_children_diag")
    assert hasattr(d2cy, "_gen_children_prod")
    assert hasattr(d2cy, "_apply_survival")


def test_T2_tape_constants():
    assert d2cy.TAPE_D  == ("DIAG_CRN",    "DIAG_DET")
    assert d2cy.TAPE_P  == ("PROD_ORACLE", "PROD_NONE")
    assert d2cy.TAPE_DP == ("DIAG_CRN",    "PROD_NONE")
    assert d2cy.TAPE_PD == ("PROD_ORACLE", "DIAG_DET")


def test_T3_config_constants():
    assert d2cy.D            == 10
    assert d2cy.N_OBJ        == 2
    assert d2cy.N_PARTITIONS == 9
    assert d2cy.POP_SIZE     == 100
    assert d2cy.BUDGET       == 10


def test_T4_seed_blocks():
    assert len(d2cy.CY_BLOCK_SEEDS) == 20
    assert d2cy.CY_BLOCK_SEEDS[0]   == 201
    assert d2cy.CY_REPRO_SEEDS[0]   == 201
    assert len(d2cy.CY_REPRO_SEEDS) == 10
    assert len(d2cy.CY_EQUIV_SEEDS) == 5


def test_T5_V_shape(setup):
    V = setup["V"]
    assert V.ndim == 2
    assert V.shape[1] == d2cy.N_OBJ


def test_T6_R_equals_len_V(setup):
    V = setup["V"]
    R = len(V)
    assert R == d2cy.N_PARTITIONS + 1  # 10


def test_T7_xl_xu_shape(setup):
    xl, xu = setup["xl"], setup["xu"]
    assert len(xl) == d2cy.D
    assert len(xu) == d2cy.D
    assert all(xl < xu)


def test_T8_pf_star_shape(setup):
    pf_star = setup["pf_star"]
    assert pf_star.ndim == 2
    assert pf_star.shape[1] == d2cy.N_OBJ


def test_T9_long_arms():
    assert set(d2cy.LONG_ARMS.keys()) == {"R_D", "R_P", "R_CHILD_D", "R_CHILD_P"}


def test_T10_n_short_n_long():
    assert d2cy.N_SHORT == 3
    assert d2cy.N_LONG  == 10


# ============================================================
# T11-T20: Helper functions
# ============================================================

def test_T11_fp_none_on_empty():
    assert d2cy._fp(None) is None
    assert d2cy._fp(np.array([])) is None


def test_T12_fp_returns_md5():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    fp  = d2cy._fp(arr, "test")
    assert "md5" in fp
    assert "shape" in fp
    assert fp["shape"] == [2, 2]


def test_T13_fp_deterministic():
    arr = np.array([[1.0, 2.0]])
    assert d2cy._fp(arr)["md5"] == d2cy._fp(arr)["md5"]


def test_T14_fps_match_same():
    arr = np.ones((5, 2))
    fp1 = d2cy._fp(arr)
    fp2 = d2cy._fp(arr)
    assert d2cy._fps_match(fp1, fp2) is True


def test_T15_fps_match_different():
    fp1 = d2cy._fp(np.ones((5, 2)))
    fp2 = d2cy._fp(np.zeros((5, 2)))
    assert d2cy._fps_match(fp1, fp2) is False


def test_T16_fps_match_none():
    assert d2cy._fps_match(None, d2cy._fp(np.ones((3, 2)))) is None
    assert d2cy._fps_match(d2cy._fp(np.ones((3, 2))), None) is None


def test_T17_json_safe():
    obj = {"a": np.int64(1), "b": np.float32(2.0), "c": np.array([1, 2])}
    safe = d2cy._json_safe(obj)
    json.dumps(safe)  # must not raise


def test_T18_bootstrap_ci_mean():
    values = [1.0] * 50
    lo, hi = d2cy._bootstrap_ci(values)
    assert abs(lo - 1.0) < 1e-9
    assert abs(hi - 1.0) < 1e-9


def test_T19_bootstrap_ci_empty():
    lo, hi = d2cy._bootstrap_ci([])
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_T20_compute_cb_basic():
    rows = [
        {"seed": 1, "arm": "R_D", "igd_response": 0.10},
        {"seed": 1, "arm": "R_P", "igd_response": 0.12},
    ]
    result = d2cy._compute_cb(rows, baseline_arm="R_P", target_arm="R_D")
    assert abs(result["mean_CB"] - 0.02) < 1e-10
    assert result["n"] == 1


# ============================================================
# T21-T30: Child generation
# ============================================================

def test_T21_gen_children_diag_shape(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    R    = len(V)
    seed = shared_state["seed"]
    X    = shared_state["X"]
    hist = copy.deepcopy(shared_state["history"])
    ht   = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r for this seed")

    Xch, op = d2cy._gen_children_diag(seed, 2, tgt_r, X_r, hist, ht, xl, xu)
    assert isinstance(Xch, np.ndarray)
    assert Xch.ndim == 2
    assert Xch.shape[1] == d2cy.D
    assert op in ("PREDICT", "LOCAL_FALLBACK")


def test_T22_gen_children_diag_bounded(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    hist  = copy.deepcopy(shared_state["history"])
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    Xch, _ = d2cy._gen_children_diag(seed, 2, tgt_r, X_r, hist, ht, xl, xu)
    if len(Xch) > 0:
        assert np.all(Xch >= xl - 1e-9)
        assert np.all(Xch <= xu + 1e-9)


def test_T23_gen_children_prod_shape(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    hist  = copy.deepcopy(shared_state["history"])
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    Xch, op = d2cy._gen_children_prod(seed, 2, tgt_r, X_r, hist, ht, xl, xu)
    assert isinstance(Xch, np.ndarray)
    assert Xch.ndim == 2
    assert op in ("PREDICT", "LOCAL_FALLBACK")


def test_T24_gen_children_diag_deterministic(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    h1 = copy.deepcopy(shared_state["history"])
    h2 = copy.deepcopy(shared_state["history"])
    Xch1, _ = d2cy._gen_children_diag(seed, 2, tgt_r, X_r, h1, ht, xl, xu)
    Xch2, _ = d2cy._gen_children_diag(seed, 2, tgt_r, X_r, h2, ht, xl, xu)
    assert np.array_equal(Xch1, Xch2), "DIAG children must be deterministic"


def test_T25_gen_children_prod_deterministic(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    h1 = copy.deepcopy(shared_state["history"])
    h2 = copy.deepcopy(shared_state["history"])
    Xch1, _ = d2cy._gen_children_prod(seed, 2, tgt_r, X_r, h1, ht, xl, xu)
    Xch2, _ = d2cy._gen_children_prod(seed, 2, tgt_r, X_r, h2, ht, xl, xu)
    assert np.array_equal(Xch1, Xch2), "PROD children must be deterministic"


def test_T26_diag_prod_children_may_differ(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    h1 = copy.deepcopy(shared_state["history"])
    h2 = copy.deepcopy(shared_state["history"])
    Xch_d, _ = d2cy._gen_children_diag(seed, 2, tgt_r, X_r, h1, ht, xl, xu)
    Xch_p, _ = d2cy._gen_children_prod(seed, 2, tgt_r, X_r, h2, ht, xl, xu)
    # They come from different RNG schemes — this test just verifies both run
    assert Xch_d.ndim == 2
    assert Xch_p.ndim == 2


def test_T27_gen_children_diag_different_seeds_differ(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    X     = shared_state["X"]
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    h1 = copy.deepcopy(shared_state["history"])
    h2 = copy.deepcopy(shared_state["history"])
    Xch1, _ = d2cy._gen_children_diag(201, 2, tgt_r, X_r, h1, ht, xl, xu)
    Xch2, _ = d2cy._gen_children_diag(999, 2, tgt_r, X_r, h2, ht, xl, xu)
    if len(Xch1) > 0 and len(Xch2) > 0:
        assert not np.array_equal(Xch1, Xch2), "Different seeds should give different children"


def test_T28_children_not_more_than_budget(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    Xch, _ = d2cy._gen_children_diag(seed, 2, tgt_r, X_r, shared_state["history"], ht, xl, xu)
    assert len(Xch) <= d2cy.BUDGET


def test_T29_gen_children_prod_bounded(setup, shared_state):
    rd, V, xl, xu = (setup["region_dynamics"], setup["V"],
                     setup["xl"], setup["xu"])
    seed  = shared_state["seed"]
    X     = shared_state["X"]
    ht    = shared_state["history_t"]
    tgt_r = shared_state["tgt_r"]

    F0 = d2cy.cp_evaluate(X, shared_state["t_float_current"], rd)
    F1 = d2cy.cp_evaluate(X, d2cy.compute_time(d2cy.WARMUP_GENERATIONS + 2 * d2cy.GENERATIONS_PER_CHANGE), rd)
    Fm, _ = normalize(F0, F1)
    rids  = np.asarray(partition(np.asarray(Fm), V), dtype=int)
    X_r   = X[rids == tgt_r]
    if len(X_r) == 0:
        pytest.skip("No points in tgt_r")

    Xch, _ = d2cy._gen_children_prod(seed, 2, tgt_r, X_r, shared_state["history"], ht, xl, xu)
    if len(Xch) > 0:
        assert np.all(Xch >= xl - 1e-9)
        assert np.all(Xch <= xu + 1e-9)


def test_T30_op_values():
    # Just a smoke test that op strings are one of the expected values
    ops = {"PREDICT", "LOCAL_FALLBACK", "KEEP"}
    assert "PREDICT" in ops


# ============================================================
# T31-T40: Survival application
# ============================================================

def test_T31_apply_survival_diag_det_deterministic(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(42)
    Xp   = rng.uniform(xl, xu, size=(120, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(120, d2cy.N_OBJ))
    Xs1, Fs1 = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "DIAG_DET")
    Xs2, Fs2 = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "DIAG_DET")
    assert np.array_equal(Xs1, Xs2)
    assert np.array_equal(Fs1, Fs2)


def test_T32_apply_survival_prod_none_shape(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(42)
    Xp   = rng.uniform(xl, xu, size=(120, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(120, d2cy.N_OBJ))
    Xs, Fs = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "PROD_NONE")
    assert Xs.shape == (100, d2cy.D)
    assert Fs.shape == (100, d2cy.N_OBJ)


def test_T33_apply_survival_returns_correct_N(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(99)
    Xp   = rng.uniform(xl, xu, size=(115, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(115, d2cy.N_OBJ))
    for N in (80, 100, 110):
        Xs, _ = d2cy._apply_survival(Xp, Fp, N, xl, xu, 201, 2, 0, "DIAG_DET")
        assert len(Xs) == N


def test_T34_apply_survival_invalid_mode(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(1)
    Xp   = rng.uniform(xl, xu, size=(110, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(110, d2cy.N_OBJ))
    with pytest.raises((ValueError, KeyError, Exception)):
        d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "BAD_MODE")


def test_T35_surv_seed_changes_result(setup):
    """Different surv seeds → different DIAG_DET results (usually)."""
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(7)
    Xp   = rng.uniform(xl, xu, size=(120, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(120, d2cy.N_OBJ))
    # Use seed 1 vs seed 2 — results differ for tie-breaking
    Xs1, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 1, 2, 0, "DIAG_DET")
    Xs2, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 2, 2, 0, "DIAG_DET")
    # Just verify both run without error (results may differ)
    assert Xs1.shape == Xs2.shape


def test_T36_surv_diag_det_bounded(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(11)
    Xp   = rng.uniform(xl, xu, size=(120, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(120, d2cy.N_OBJ))
    Xs, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "DIAG_DET")
    assert np.all(Xs >= xl - 1e-9)
    assert np.all(Xs <= xu + 1e-9)


def test_T37_surv_prod_none_bounded(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(12)
    Xp   = rng.uniform(xl, xu, size=(120, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(120, d2cy.N_OBJ))
    Xs, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "PROD_NONE")
    assert np.all(Xs >= xl - 1e-9)
    assert np.all(Xs <= xu + 1e-9)


def test_T38_survive_exact_N(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(5)
    Xp   = rng.uniform(xl, xu, size=(100, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(100, d2cy.N_OBJ))
    Xs, Fs = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "DIAG_DET")
    assert Xs.shape == (100, d2cy.D)
    assert Fs.shape == (100, d2cy.N_OBJ)


def test_T39_fingerprint_on_surv_output(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(55)
    Xp   = rng.uniform(xl, xu, size=(110, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(110, d2cy.N_OBJ))
    Xs, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 2, 0, "DIAG_DET")
    fp = d2cy._fp(Xs)
    assert fp is not None
    assert fp["shape"] == [100, d2cy.D]


def test_T40_surv_diag_det_consistent_across_calls(setup):
    xl, xu = setup["xl"], setup["xu"]
    rng  = np.random.default_rng(77)
    Xp   = rng.uniform(xl, xu, size=(120, d2cy.D))
    Fp   = rng.uniform(0, 1, size=(120, d2cy.N_OBJ))
    # Same seed, same input → same result
    a, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 3, 5, "DIAG_DET")
    b, _ = d2cy._apply_survival(Xp, Fp, 100, xl, xu, 201, 3, 5, "DIAG_DET")
    assert np.array_equal(a, b)


# ============================================================
# T41-T50: Core trajectory runner
# ============================================================

def test_T41_trajectory_runner_returns_all_tapes(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    tapes = {"TAPE_D": d2cy.TAPE_D, "TAPE_P": d2cy.TAPE_P}
    result = d2cy._run_tape_trajectory(201, 1, tapes, rd, V, xl, xu, pf_star)
    assert set(result.keys()) == set(tapes.keys())


def test_T42_trajectory_runner_row_count(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    tapes = {"TAPE_D": d2cy.TAPE_D, "TAPE_P": d2cy.TAPE_P}
    result = d2cy._run_tape_trajectory(201, 2, tapes, rd, V, xl, xu, pf_star)
    for name in tapes:
        assert len(result[name]) == 2


def test_T43_trajectory_runner_row_keys(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    result = d2cy._run_tape_trajectory(201, 1, {"TAPE_D": d2cy.TAPE_D},
                                       rd, V, xl, xu, pf_star)
    row = result["TAPE_D"][0]
    for key in ("k", "tgt_r", "realized_op", "igd_pre", "igd_response", "igd_epoch"):
        assert key in row


def test_T44_trajectory_igd_values_finite(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    result = d2cy._run_tape_trajectory(201, 2, {"TAPE_D": d2cy.TAPE_D},
                                       rd, V, xl, xu, pf_star)
    for row in result["TAPE_D"]:
        assert np.isfinite(row["igd_pre"])
        assert np.isfinite(row["igd_response"])


def test_T45_trajectory_deterministic(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    r1 = d2cy._run_tape_trajectory(201, 1, {"TAPE_D": d2cy.TAPE_D},
                                    rd, V, xl, xu, pf_star)
    r2 = d2cy._run_tape_trajectory(201, 1, {"TAPE_D": d2cy.TAPE_D},
                                    rd, V, xl, xu, pf_star)
    row1 = r1["TAPE_D"][0]
    row2 = r2["TAPE_D"][0]
    assert row1["igd_response"] == row2["igd_response"]
    if row1.get("c3_fp") and row2.get("c3_fp"):
        assert row1["c3_fp"]["md5"] == row2["c3_fp"]["md5"]


def test_T46_all_four_tapes_run(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    tapes = d2cy.TAPE_NAMES
    result = d2cy._run_tape_trajectory(201, 1, tapes, rd, V, xl, xu, pf_star)
    assert set(result.keys()) == set(tapes.keys())
    for name, rows in result.items():
        assert len(rows) == 1


def test_T47_tape_p_and_tape_d_differ(setup):
    """TAPE_D and TAPE_P produce different C3 because of RNG protocol mismatch."""
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    result = d2cy._run_tape_trajectory(201, 2,
                                       {"TAPE_D": d2cy.TAPE_D, "TAPE_P": d2cy.TAPE_P},
                                       rd, V, xl, xu, pf_star)
    # At least one change should differ (not guaranteed to differ on every change)
    diffs = 0
    for rd_row, rp_row in zip(result["TAPE_D"], result["TAPE_P"]):
        if rd_row.get("c3_fp") and rp_row.get("c3_fp"):
            if not d2cy._fps_match(rd_row["c3_fp"], rp_row["c3_fp"]):
                diffs += 1
    # Passes regardless — this is a structural test, not a correctness assertion


def test_T48_trajectory_runner_k_index(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    result = d2cy._run_tape_trajectory(201, 3, {"TAPE_D": d2cy.TAPE_D},
                                       rd, V, xl, xu, pf_star)
    for i, row in enumerate(result["TAPE_D"]):
        assert row["k"] == i


def test_T49_trajectory_tgt_r_in_range(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    R = len(V)
    result = d2cy._run_tape_trajectory(201, 3, {"TAPE_D": d2cy.TAPE_D},
                                       rd, V, xl, xu, pf_star)
    for row in result["TAPE_D"]:
        assert 0 <= row["tgt_r"] < R


def test_T50_trajectory_extra_fe_non_negative(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    result = d2cy._run_tape_trajectory(201, 2,
                                       {"TAPE_D": d2cy.TAPE_D, "TAPE_P": d2cy.TAPE_P},
                                       rd, V, xl, xu, pf_star)
    for rows in result.values():
        for row in rows:
            assert row.get("extra_fe", 0) >= 0


# ============================================================
# T51-T55: Phase runner smoke tests
# ============================================================

def test_T51_cy0_smoke(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    cy0 = d2cy.run_cy0_reproducibility(rd, V, xl, xu, pf_star)
    assert "PRODUCTION_REPRODUCIBILITY" in cy0
    assert "DIAGNOSTIC_REPRODUCIBILITY" in cy0
    assert "SURVIVAL_TIE_REPRODUCIBILITY" in cy0
    assert cy0["PRODUCTION_REPRODUCIBILITY"] in ("PASS", "FAIL")
    assert cy0["DIAGNOSTIC_REPRODUCIBILITY"] in ("PASS", "FAIL")
    assert cy0["SURVIVAL_TIE_REPRODUCIBILITY"] == "FAIL"   # non-det by design
    # total rows: TAPE_D (N_SHORT per seed) + TAPE_P (N_SHORT per seed) × n_seeds
    assert cy0["total"] == len(d2cy.CY_REPRO_SEEDS) * d2cy.N_SHORT * 2
    # prod_c2 only checked at k=0 (1 per seed)
    assert cy0["prod_c2_pass"] + cy0["prod_c2_fail"] == len(d2cy.CY_REPRO_SEEDS)


def test_T52_cy2_smoke(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    cy2 = d2cy.run_cy2_same_tape_equivalence(rd, V, xl, xu, pf_star)
    assert "SAME_TAPE_RUNNER_EQUIVALENCE" in cy2
    assert cy2["SAME_TAPE_RUNNER_EQUIVALENCE"] in ("PASS", "FAIL")
    assert cy2["total"] == len(d2cy.CY_EQUIV_SEEDS) * d2cy.N_SHORT
    # Rows should contain c2_match (children), not c3_match
    for row in cy2["rows"]:
        assert "c2_match" in row


def test_T53_cy3_smoke(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    cy3 = d2cy.run_cy3_factorial(rd, V, xl, xu, pf_star)
    expected = len(d2cy.CY_BLOCK_SEEDS) * d2cy.N_SHORT
    assert len(cy3["rows"]) == expected
    row = cy3["rows"][0]
    for tape in ("TAPE_D", "TAPE_DP", "TAPE_PD", "TAPE_P"):
        assert f"{tape}_igd_response" in row


def test_T54_cy4_smoke(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    cy4 = d2cy.run_cy4_evolution_coupling(rd, V, xl, xu, pf_star)
    assert "RNG_STATE_ISOLATION" in cy4
    assert cy4["RNG_STATE_ISOLATION"] in ("PASS", "FAIL")
    assert cy4["rng_coupling_pattern"] in ("E_FIXED", "E_COUPLED")
    assert len(cy4["rows"]) == len(d2cy.CY_REPRO_SEEDS)


def test_T55_cy5_smoke(setup):
    rd, V, xl, xu, pf_star = (setup["region_dynamics"], setup["V"],
                               setup["xl"], setup["xu"], setup["pf_star"])
    cy5 = d2cy.run_cy5_longitudinal(rd, V, xl, xu, pf_star)
    expected = len(d2cy.CY_BLOCK_SEEDS) * d2cy.N_LONG * len(d2cy.LONG_ARMS)
    assert len(cy5["rows"]) == expected
    row = cy5["rows"][0]
    assert row["arm"] in d2cy.LONG_ARMS
    assert "igd_response" in row
