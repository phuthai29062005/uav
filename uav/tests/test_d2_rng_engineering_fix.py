"""
D2-RF — Response RNG Reproducibility Fix.

T1-T10:  Imports and source-level checks
T11-T20: make_rng RESPONSE_SURVIVAL namespace
T21-T30: nsga2_survival with response_rng parameter
T31-T40: region_protected_survival — global hack removed
T41-T50: respond_population end-to-end reproducibility
T51-T55: Stress / cross-process reproducibility
"""
import copy
import hashlib
import inspect
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.core.normalize import normalize
from prdrl.core.survival import region_protected_survival
from prdrl.history.region_history import RegionHistory
from prdrl.oracle.oracle import make_rng, nsga2_survival, respond_population
from prdrl.problems.controlled_predict import (
    get_bounds,
    get_true_pf,
    make_region_dynamics_map,
    evaluate as cp_evaluate_raw,
)

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
D          = 10
N_OBJ      = 2
N_PARTS    = 10
POP_SIZE   = 20
N_SURVIVE  = POP_SIZE
RF_SEEDS   = list(range(301, 321))  # 20 fresh seeds
N_REPEAT   = 100


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr)).hexdigest()[:12]


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    rd    = make_region_dynamics_map(N_PARTS + 1)
    V     = reference(N_OBJ, N_PARTS)
    xl, xu = get_bounds(D)
    return {"rd": rd, "V": V, "xl": xl, "xu": xu}


@pytest.fixture(scope="module")
def small_pool(env):
    """Deterministic (X, F) pool for nsga2_survival tests."""
    xl, xu = env["xl"], env["xu"]
    rng = np.random.default_rng(9999)
    X = rng.uniform(xl, xu, size=(POP_SIZE * 2, D))
    F = rng.uniform(0, 1, size=(POP_SIZE * 2, N_OBJ))
    return {"X": X, "F": F, "xl": xl, "xu": xu}


@pytest.fixture(scope="module")
def region_pool(env):
    """Minimal HARD-path survival pool with region info."""
    xl, xu = env["xl"], env["xu"]
    V      = env["V"]
    R      = len(V)
    rng    = np.random.default_rng(8888)
    N      = POP_SIZE
    B      = N // 2
    X_pop  = rng.uniform(xl, xu, size=(N, D))
    F_pop  = rng.uniform(0, 1, size=(N, N_OBJ))
    X_ch   = rng.uniform(xl, xu, size=(B, D))
    F_ch   = rng.uniform(0, 1, size=(B, N_OBJ))
    rids   = np.array([i % R for i in range(N)])
    tgt_r  = 0
    n_r_before = int(np.sum(rids == tgt_r))
    return {
        "X_pop": X_pop, "F_pop": F_pop,
        "X_ch": X_ch, "F_ch": F_ch,
        "rids": rids, "tgt_r": tgt_r,
        "n_r_before": n_r_before,
        "xl": xl, "xu": xu,
    }


@pytest.fixture(scope="module")
def respond_env(env):
    """Minimal valid environment for respond_population tests (O0 path)."""
    rd, V, xl, xu = env["rd"], env["V"], env["xl"], env["xu"]
    R = len(V)
    seed = RF_SEEDS[0]
    rng = np.random.default_rng(seed)
    X   = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t0  = 0.0
    F_m = cp_evaluate_raw(X, t0, rd)
    F_p = cp_evaluate_raw(X, 0.1, rd)
    Fm_norm, _ = normalize(F_m, F_p)
    rids = np.asarray(partition(np.asarray(Fm_norm), V), dtype=int)
    hist = RegionHistory(n_regions=R)
    hist.update(t=0, X=X, F=F_m, region_ids=rids, xl=xl, xu=xu)
    tgt_r = int(seed % R)

    def eval_fn(Xc):
        return cp_evaluate_raw(Xc, 0.2, rd)

    return {
        "X": X, "F_plus": F_p, "region_ids": rids, "V": V,
        "actions": {tgt_r: "LOCAL"},
        "history": hist, "t": 1,
        "eval_fn": eval_fn,
        "xl": xl, "xu": xu,
        "base_seed": seed * 100000 + 1 * 100 + tgt_r,
        "tgt_r": tgt_r,
    }


# ============================================================
# T1-T10: Imports and source-level checks
# ============================================================

def test_t01_make_rng_importable():
    assert callable(make_rng)


def test_t02_nsga2_survival_importable():
    assert callable(nsga2_survival)


def test_t03_respond_population_importable():
    assert callable(respond_population)


def test_t04_region_protected_survival_importable():
    assert callable(region_protected_survival)


def test_t05_nsga2_survival_has_response_rng_param():
    sig = inspect.signature(nsga2_survival)
    assert "response_rng" in sig.parameters


def test_t06_response_rng_default_is_none():
    sig = inspect.signature(nsga2_survival)
    assert sig.parameters["response_rng"].default is None


def test_t07_make_rng_action_code_has_response_survival():
    # Call make_rng with RESPONSE_SURVIVAL — should not raise
    gen = make_rng(1, 0, 0, "RESPONSE_SURVIVAL")
    assert gen is not None


def test_t08_survival_py_has_no_np_random_seed():
    import prdrl.core.survival as surv_mod
    src = inspect.getsource(region_protected_survival)
    assert "np.random.seed" not in src, (
        "region_protected_survival must not call np.random.seed (global hack removed by D2-RF)"
    )


def test_t09_make_rng_unknown_action_raises():
    with pytest.raises(ValueError, match="Unknown action"):
        make_rng(1, 0, 0, "INVALID_ACTION")


def test_t10_region_protected_survival_accepts_surv_seed_kwarg():
    """surv_seed parameter retained for API compat even though it has no effect."""
    sig = inspect.signature(region_protected_survival)
    assert "surv_seed" in sig.parameters


# ============================================================
# T11-T20: make_rng RESPONSE_SURVIVAL namespace
# ============================================================

def test_t11_make_rng_response_survival_returns_generator():
    gen = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    assert isinstance(gen, np.random.Generator)


def test_t12_make_rng_response_survival_reproducible():
    g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    g2 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    a1 = g1.random(10)
    a2 = g2.random(10)
    np.testing.assert_array_equal(a1, a2)


def test_t13_make_rng_response_survival_different_base_seed():
    g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    g2 = make_rng(99, 3, 7, "RESPONSE_SURVIVAL")
    assert not np.array_equal(g1.random(10), g2.random(10))


def test_t14_make_rng_response_survival_different_t():
    g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    g2 = make_rng(42, 9, 7, "RESPONSE_SURVIVAL")
    assert not np.array_equal(g1.random(10), g2.random(10))


def test_t15_make_rng_response_survival_different_region():
    g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    g2 = make_rng(42, 3, 0, "RESPONSE_SURVIVAL")
    assert not np.array_equal(g1.random(10), g2.random(10))


def test_t16_response_survival_differs_from_predict():
    g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    g2 = make_rng(42, 3, 7, "PREDICT")
    assert not np.array_equal(g1.random(10), g2.random(10))


def test_t17_response_survival_differs_from_l2b_response():
    g1 = make_rng(42, 3, 7, "RESPONSE_SURVIVAL")
    g2 = make_rng(42, 3, 7, "LOCAL")
    assert not np.array_equal(g1.random(10), g2.random(10))


def test_t18_response_survival_action_code_is_4():
    # Verify by comparing to manually computed SeedSequence
    import numpy.random as npr
    gen = make_rng(1, 2, 3, "RESPONSE_SURVIVAL")
    ref = np.random.default_rng(npr.SeedSequence([1, 2, 3, 4]))
    np.testing.assert_array_equal(gen.random(10), ref.random(10))


def test_t19_response_survival_rng_float_sequence_stable():
    gen = make_rng(777, 5, 1, "RESPONSE_SURVIVAL")
    seq = gen.random(20)
    gen2 = make_rng(777, 5, 1, "RESPONSE_SURVIVAL")
    seq2 = gen2.random(20)
    np.testing.assert_array_equal(seq, seq2)


def test_t20_response_survival_namespace_separation():
    """All four scalar args contribute to unique output."""
    params = [(100, 0, 0), (101, 0, 0), (100, 1, 0), (100, 0, 1)]
    seqs = [make_rng(bs, t, r, "RESPONSE_SURVIVAL").random(5) for bs, t, r in params]
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            assert not np.array_equal(seqs[i], seqs[j]), (
                f"Collision between param set {i} and {j}"
            )


# ============================================================
# T21-T30: nsga2_survival with response_rng parameter
# ============================================================

def test_t21_nsga2_survival_no_rng_backward_compat(small_pool):
    sp = small_pool
    X_new, F_new = nsga2_survival(
        X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
        xl=sp["xl"], xu=sp["xu"],
    )
    assert X_new.shape == (N_SURVIVE, D)
    assert F_new.shape == (N_SURVIVE, N_OBJ)


def test_t22_nsga2_survival_with_rng_returns_correct_shape(small_pool):
    sp = small_pool
    rng = make_rng(42, 0, 0, "RESPONSE_SURVIVAL")
    X_new, F_new = nsga2_survival(
        X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
        xl=sp["xl"], xu=sp["xu"], response_rng=rng,
    )
    assert X_new.shape == (N_SURVIVE, D)
    assert F_new.shape == (N_SURVIVE, N_OBJ)


def test_t23_nsga2_survival_deterministic_with_rng(small_pool):
    sp = small_pool
    rng1 = make_rng(42, 0, 0, "RESPONSE_SURVIVAL")
    rng2 = make_rng(42, 0, 0, "RESPONSE_SURVIVAL")
    X1, F1 = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                             xl=sp["xl"], xu=sp["xu"], response_rng=rng1)
    X2, F2 = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                             xl=sp["xl"], xu=sp["xu"], response_rng=rng2)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(F1, F2)


def test_t24_nsga2_survival_different_rng_seeds_may_differ(small_pool):
    """Different survival RNG seeds produce valid (possibly different) outcomes."""
    sp = small_pool
    rng1 = make_rng(1, 0, 0, "RESPONSE_SURVIVAL")
    rng2 = make_rng(999, 0, 0, "RESPONSE_SURVIVAL")
    X1, _ = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                            xl=sp["xl"], xu=sp["xu"], response_rng=rng1)
    X2, _ = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                            xl=sp["xl"], xu=sp["xu"], response_rng=rng2)
    # Both outputs must have correct shape regardless of whether they differ
    assert X1.shape == (N_SURVIVE, D)
    assert X2.shape == (N_SURVIVE, D)


def test_t25_nsga2_survival_repeated_same_rng_deterministic(small_pool):
    """50 repeated calls with fresh-but-equal Generators → all identical."""
    sp = small_pool
    results = []
    for _ in range(50):
        rng = make_rng(42, 1, 3, "RESPONSE_SURVIVAL")
        X_new, _ = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                                   xl=sp["xl"], xu=sp["xu"], response_rng=rng)
        results.append(_md5(X_new))
    assert len(set(results)) == 1, f"Got {len(set(results))} distinct results, expected 1"


def test_t26_nsga2_survival_does_not_mutate_input(small_pool):
    sp = small_pool
    X_orig = sp["X"].copy()
    F_orig = sp["F"].copy()
    rng = make_rng(42, 0, 0, "RESPONSE_SURVIVAL")
    nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                   xl=sp["xl"], xu=sp["xu"], response_rng=rng)
    np.testing.assert_array_equal(sp["X"], X_orig)
    np.testing.assert_array_equal(sp["F"], F_orig)


def test_t27_nsga2_survival_global_numpy_state_not_affected(small_pool):
    """response_rng path must not pollute numpy legacy global state."""
    sp = small_pool
    np.random.seed(12345)
    ref_sequence = np.random.random(100)
    np.random.seed(12345)
    rng = make_rng(42, 0, 0, "RESPONSE_SURVIVAL")
    nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                   xl=sp["xl"], xu=sp["xu"], response_rng=rng)
    after_sequence = np.random.random(100)
    np.testing.assert_array_equal(ref_sequence, after_sequence)


def test_t28_nsga2_survival_all_survived_in_bounds(small_pool):
    sp = small_pool
    rng = make_rng(42, 0, 0, "RESPONSE_SURVIVAL")
    X_new, _ = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                               xl=sp["xl"], xu=sp["xu"], response_rng=rng)
    assert np.all(X_new >= sp["xl"] - 1e-10)
    assert np.all(X_new <= sp["xu"] + 1e-10)


def test_t29_nsga2_survival_rng_none_still_functional(small_pool):
    """None response_rng path (old callers) still returns valid arrays."""
    sp = small_pool
    for _ in range(5):
        X_new, F_new = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
                                       xl=sp["xl"], xu=sp["xu"])
        assert X_new.shape == (N_SURVIVE, D)
        assert F_new.shape == (N_SURVIVE, N_OBJ)


def test_t30_nsga2_survival_n_survive_respected(small_pool):
    sp = small_pool
    n = 7
    rng = make_rng(1, 0, 0, "RESPONSE_SURVIVAL")
    X_new, F_new = nsga2_survival(X=sp["X"], F=sp["F"], n_survive=n,
                                   xl=sp["xl"], xu=sp["xu"], response_rng=rng)
    assert len(X_new) == n
    assert len(F_new) == n


# ============================================================
# T31-T40: region_protected_survival — global hack removed
# ============================================================

def test_t31_region_protected_survival_callable():
    assert callable(region_protected_survival)


def test_t32_region_protected_survival_no_error(region_pool):
    rp = region_pool
    X_surv, F_surv, idx = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
    )
    assert X_surv.shape[0] == POP_SIZE


def test_t33_region_protected_survival_surv_seed_accepted(region_pool):
    """surv_seed parameter still accepted for API compat but has no effect."""
    rp = region_pool
    X_surv, F_surv, idx = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
        surv_seed=42,
    )
    assert X_surv.shape[0] == POP_SIZE


def test_t34_region_protected_survival_source_no_np_random_seed():
    src = inspect.getsource(region_protected_survival)
    assert "np.random.seed" not in src


def test_t35_region_protected_survival_does_not_mutate_global_state(region_pool):
    """Calling region_protected_survival must not change numpy legacy state."""
    rp = region_pool
    np.random.seed(55555)
    ref = np.random.random(50)
    np.random.seed(55555)
    region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
        surv_seed=42,
    )
    after = np.random.random(50)
    np.testing.assert_array_equal(ref, after)


def test_t36_region_protected_survival_deterministic(region_pool):
    rp = region_pool
    results = []
    for _ in range(20):
        X_surv, _, _ = region_protected_survival(
            X_pop=rp["X_pop"], F_pop=rp["F_pop"],
            X_children=rp["X_ch"], F_children=rp["F_ch"],
            response_region_r=rp["tgt_r"],
            old_region_ids=rp["rids"],
            n_r_before=rp["n_r_before"],
            xl=rp["xl"], xu=rp["xu"],
        )
        results.append(_md5(X_surv))
    assert len(set(results)) == 1, "region_protected_survival non-deterministic"


def test_t37_region_protected_survival_correct_output_size(region_pool):
    rp = region_pool
    X_surv, F_surv, idx = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
    )
    assert X_surv.shape == (POP_SIZE, D)
    assert F_surv.shape == (POP_SIZE, N_OBJ)
    assert len(idx) == POP_SIZE


def test_t38_region_protected_survival_hard_invariant(region_pool):
    """n_old_dropped_other == 0: outside-r OLD individuals always survive."""
    rp = region_pool
    X_surv, F_surv, idx = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
    )
    N = len(rp["X_pop"])
    outside_old_idx = [i for i in range(N) if rp["rids"][i] != rp["tgt_r"]]
    for i in outside_old_idx:
        assert i in idx, f"Protected individual {i} was dropped — HARD invariant violated"


def test_t39_surv_seed_passthrough_no_effect_on_output(region_pool):
    """surv_seed no longer seeds numpy, so output should be identical with or without it."""
    rp = region_pool
    X_noseed, _, _ = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
    )
    X_wseed, _, _ = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
        surv_seed=12345,
    )
    np.testing.assert_array_equal(X_noseed, X_wseed)


def test_t40_region_protected_survival_returns_tuple_of_three(region_pool):
    rp = region_pool
    result = region_protected_survival(
        X_pop=rp["X_pop"], F_pop=rp["F_pop"],
        X_children=rp["X_ch"], F_children=rp["F_ch"],
        response_region_r=rp["tgt_r"],
        old_region_ids=rp["rids"],
        n_r_before=rp["n_r_before"],
        xl=rp["xl"], xu=rp["xu"],
    )
    assert len(result) == 3


# ============================================================
# T41-T50: respond_population reproducibility
# ============================================================

def test_t41_respond_population_returns_dict(respond_env):
    e = respond_env
    result = respond_population(
        X=e["X"].copy(), F_plus=e["F_plus"].copy(),
        region_ids=e["region_ids"].copy(), V=e["V"],
        actions=e["actions"], history=copy.deepcopy(e["history"]),
        t=e["t"], evaluate_fn=e["eval_fn"],
        xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
        sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
        lambda_pred=0.5, base_seed=e["base_seed"],
    )
    assert isinstance(result, dict)
    assert "available" in result
    assert "X" in result
    assert "F" in result
    assert "extra_fe" in result


def test_t42_respond_population_deterministic_o0(respond_env):
    """Two calls with same args → bitwise-identical X and F (O0 path, post D2-RF)."""
    e = respond_env

    def _run():
        return respond_population(
            X=e["X"].copy(), F_plus=e["F_plus"].copy(),
            region_ids=e["region_ids"].copy(), V=e["V"],
            actions=e["actions"], history=copy.deepcopy(e["history"]),
            t=e["t"], evaluate_fn=e["eval_fn"],
            xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
            sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
            lambda_pred=0.5, base_seed=e["base_seed"],
        )

    r1 = _run()
    r2 = _run()
    np.testing.assert_array_equal(r1["X"], r2["X"])
    np.testing.assert_array_equal(r1["F"], r2["F"])


def test_t43_respond_population_extra_fe_deterministic(respond_env):
    e = respond_env

    def _run():
        return respond_population(
            X=e["X"].copy(), F_plus=e["F_plus"].copy(),
            region_ids=e["region_ids"].copy(), V=e["V"],
            actions=e["actions"], history=copy.deepcopy(e["history"]),
            t=e["t"], evaluate_fn=e["eval_fn"],
            xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
            sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
            lambda_pred=0.5, base_seed=e["base_seed"],
        )

    r1 = _run()
    r2 = _run()
    assert r1["extra_fe"] == r2["extra_fe"]


def test_t44_respond_population_different_base_seed_different_result(respond_env):
    e = respond_env

    def _run(bs):
        return respond_population(
            X=e["X"].copy(), F_plus=e["F_plus"].copy(),
            region_ids=e["region_ids"].copy(), V=e["V"],
            actions=e["actions"], history=copy.deepcopy(e["history"]),
            t=e["t"], evaluate_fn=e["eval_fn"],
            xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
            sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
            lambda_pred=0.5, base_seed=bs,
        )

    r1 = _run(e["base_seed"])
    r2 = _run(e["base_seed"] + 1)
    # Different base_seed → at least X should differ somewhere (child generation RNG differs)
    assert not np.array_equal(r1["X"], r2["X"]) or r1["extra_fe"] != r2["extra_fe"]


def test_t45_respond_population_available_true(respond_env):
    e = respond_env
    result = respond_population(
        X=e["X"].copy(), F_plus=e["F_plus"].copy(),
        region_ids=e["region_ids"].copy(), V=e["V"],
        actions=e["actions"], history=copy.deepcopy(e["history"]),
        t=e["t"], evaluate_fn=e["eval_fn"],
        xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
        sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
        lambda_pred=0.5, base_seed=e["base_seed"],
    )
    assert result["available"] is True


def test_t46_respond_population_output_shape(respond_env):
    e = respond_env
    result = respond_population(
        X=e["X"].copy(), F_plus=e["F_plus"].copy(),
        region_ids=e["region_ids"].copy(), V=e["V"],
        actions=e["actions"], history=copy.deepcopy(e["history"]),
        t=e["t"], evaluate_fn=e["eval_fn"],
        xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
        sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
        lambda_pred=0.5, base_seed=e["base_seed"],
    )
    assert result["X"].shape == (POP_SIZE, D)
    assert result["F"].shape == (POP_SIZE, N_OBJ)


def test_t47_surv_rng_derivation_in_source():
    """Verify source of respond_population contains 'RESPONSE_SURVIVAL' namespace."""
    src = inspect.getsource(respond_population)
    assert "RESPONSE_SURVIVAL" in src


def test_t48_respond_population_surv_rng_uses_make_rng(respond_env):
    """Source shows surv_rng = make_rng(base_seed, t, ..., 'RESPONSE_SURVIVAL')."""
    src = inspect.getsource(respond_population)
    assert "surv_rng" in src
    assert "make_rng" in src


def test_t49_respond_population_keep_only_acts_as_identity(respond_env):
    """If all actions are KEEP, respond_population may return early with no children."""
    e = respond_env
    result = respond_population(
        X=e["X"].copy(), F_plus=e["F_plus"].copy(),
        region_ids=e["region_ids"].copy(), V=e["V"],
        actions={},  # empty actions = all KEEP
        history=copy.deepcopy(e["history"]),
        t=e["t"], evaluate_fn=e["eval_fn"],
        xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
        sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
        lambda_pred=0.5, base_seed=e["base_seed"],
    )
    # Should return available=True (or False if no children) — just mustn't crash
    assert "available" in result


def test_t50_respond_population_multiple_seeds_all_reproducible(respond_env):
    """5 seeds × 2 runs each → all pairs identical."""
    e = respond_env
    for seed_offset in range(5):
        bs = e["base_seed"] + seed_offset * 1000

        def _run():
            return respond_population(
                X=e["X"].copy(), F_plus=e["F_plus"].copy(),
                region_ids=e["region_ids"].copy(), V=e["V"],
                actions=e["actions"], history=copy.deepcopy(e["history"]),
                t=e["t"] + seed_offset, evaluate_fn=e["eval_fn"],
                xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
                sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
                lambda_pred=0.5, base_seed=bs,
            )

        r1 = _run()
        r2 = _run()
        np.testing.assert_array_equal(
            r1["X"], r2["X"],
            err_msg=f"Non-deterministic at seed_offset={seed_offset}",
        )


# ============================================================
# T51-T55: Stress / cross-process reproducibility
# ============================================================

def test_t51_nsga2_survival_stress_n100(small_pool):
    """N_REPEAT=100 calls with fresh equal Generators → 100/100 identical X."""
    sp = small_pool
    results = []
    for _ in range(N_REPEAT):
        rng = make_rng(301, 5, 2, "RESPONSE_SURVIVAL")
        X_new, _ = nsga2_survival(
            X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
            xl=sp["xl"], xu=sp["xu"], response_rng=rng,
        )
        results.append(_md5(X_new))
    unique = set(results)
    assert len(unique) == 1, f"Stress: {len(unique)} distinct results across {N_REPEAT} runs"


def test_t52_respond_population_stress_n100(respond_env):
    """N_REPEAT=100 respond_population calls → 100/100 identical (X, F)."""
    e = respond_env
    hashes = []
    for _ in range(N_REPEAT):
        r = respond_population(
            X=e["X"].copy(), F_plus=e["F_plus"].copy(),
            region_ids=e["region_ids"].copy(), V=e["V"],
            actions=e["actions"], history=copy.deepcopy(e["history"]),
            t=e["t"], evaluate_fn=e["eval_fn"],
            xl=e["xl"], xu=e["xu"], budget=POP_SIZE * 2,
            sigma_local=0.03, sigma_predict=0.03, sigma_diversify=0.03,
            lambda_pred=0.5, base_seed=e["base_seed"],
        )
        h = _md5(r["X"]) + _md5(r["F"])
        hashes.append(h)
    unique = set(hashes)
    assert len(unique) == 1, (
        f"respond_population non-deterministic: {len(unique)} distinct results over {N_REPEAT} runs"
    )


def test_t53_cross_process_nsga2_survival():
    """10 subprocesses each compute nsga2_survival with same seed → all identical."""
    script = (
        "import sys; sys.path.insert(0,'src'); "
        "import numpy as np; import hashlib; "
        "from prdrl.oracle.oracle import make_rng, nsga2_survival; "
        "rng_sp = np.random.default_rng(9999); "
        "X = rng_sp.uniform(0,1,size=(40,10)); "
        "F = rng_sp.uniform(0,1,size=(40,2)); "
        "rng = make_rng(301,5,2,'RESPONSE_SURVIVAL'); "
        "Xn,_ = nsga2_survival(X=X,F=F,n_survive=20,xl=np.zeros(10),xu=np.ones(10),response_rng=rng); "
        "print(hashlib.md5(np.ascontiguousarray(Xn)).hexdigest())"
    )
    hashes = []
    for _ in range(10):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd="/Users/phuthai/Desktop/lab/uav",
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        hashes.append(result.stdout.strip())
    assert len(set(hashes)) == 1, (
        f"Cross-process non-determinism: got {set(hashes)}"
    )


def test_t54_cross_process_respond_population():
    """5 subprocesses each run respond_population with same seed → all identical X hash."""
    script = (
        "import sys; sys.path.insert(0,'src'); "
        "import copy, hashlib; import numpy as np; "
        "from prdrl.core.reference import reference; "
        "from prdrl.core.partition import partition; "
        "from prdrl.core.normalize import normalize; "
        "from prdrl.history.region_history import RegionHistory; "
        "from prdrl.oracle.oracle import respond_population; "
        "from prdrl.problems.controlled_predict import get_bounds,make_region_dynamics_map,evaluate; "
        "D,N_OBJ,N_PARTS,N=10,2,10,20; "
        "rd=make_region_dynamics_map(N_PARTS+1); "
        "V=reference(N_OBJ,N_PARTS); "
        "xl,xu=get_bounds(D); "
        "seed=301; rng=np.random.default_rng(seed); "
        "X=rng.uniform(xl,xu,size=(N,D)); "
        "F_m=evaluate(X,0.0,rd); "
        "F_p=evaluate(X,0.1,rd); "
        "Fm,_=normalize(F_m,F_p); "
        "rids=np.asarray(partition(np.asarray(Fm),V),dtype=int); "
        "hist=RegionHistory(n_regions=len(V)); "
        "hist.update(t=0,X=X,F=F_m,region_ids=rids,xl=xl,xu=xu); "
        "tgt_r=int(seed%len(V)); "
        "bs=seed*100000+1*100+tgt_r; "
        "eval_fn=lambda Xc: evaluate(Xc,0.2,rd); "
        "r=respond_population(X=X.copy(),F_plus=F_p.copy(),"
        "region_ids=rids.copy(),V=V,actions={tgt_r:'LOCAL'},"
        "history=copy.deepcopy(hist),t=1,evaluate_fn=eval_fn,"
        "xl=xl,xu=xu,budget=40,sigma_local=0.03,sigma_predict=0.03,"
        "sigma_diversify=0.03,lambda_pred=0.5,base_seed=bs); "
        "print(hashlib.md5(np.ascontiguousarray(r['X'])).hexdigest())"
    )
    hashes = []
    for _ in range(5):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd="/Users/phuthai/Desktop/lab/uav",
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        hashes.append(result.stdout.strip())
    assert len(set(hashes)) == 1, (
        f"Cross-process respond_population non-determinism: {set(hashes)}"
    )


def test_t55_global_state_pollution_nsga2_survival_stress(small_pool):
    """nsga2_survival with response_rng must not be affected by external global numpy state."""
    sp = small_pool
    hashes = []
    for i in range(30):
        # Corrupt global numpy state with different seeds before each call
        np.random.seed(i * 7 + 1)
        _ = np.random.random(100)
        rng = make_rng(42, 1, 3, "RESPONSE_SURVIVAL")
        X_new, _ = nsga2_survival(
            X=sp["X"], F=sp["F"], n_survive=N_SURVIVE,
            xl=sp["xl"], xu=sp["xu"], response_rng=rng,
        )
        hashes.append(_md5(X_new))
    assert len(set(hashes)) == 1, (
        f"nsga2_survival output varies with global state pollution: {len(set(hashes))} distinct"
    )
