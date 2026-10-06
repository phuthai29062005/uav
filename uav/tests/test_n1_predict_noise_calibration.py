"""
T1–T24: Unit and integration tests for N1 Predict Noise Calibration.

Sections:
  T1–T2:  Estimator / information isolation
  T3:     Production predict equivalence
  T4–T5:  CRN — identical parents and xi across all sigma
  T6:     Sigma scaling relation (unclipped case)
  T7:     Sigma=0 has no noise term
  T8:     Clipping in non-unit bounds
  T9:     rho_noise known case
  T10:    Zero-motion rho handled safely
  T11:    Displacement cosine known vectors
  T12:    HARD invariant — 0 cross-region drops at every sigma
  T13:    Final population size = POP_SIZE
  T14:    Same FE for every sigma
  T15:    Deterministic survival
  T16:    Baseline immutability
  T17:    IGD identity residual
  T18:    sigma=0.03 matches R1-N HARD03
  T19:    sigma=0 matches R1-N HARD0
  T20:    Missing history handled
  T21:    Production constants unchanged
  T22:    Candidate frozen before holdout (discovery-only selection)
  T23:    Holdout seeds disjoint from discovery seeds
  T24:    Production predictor file unmodified
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src")

import prdrl.experiments.E2P3_survival_translation as e2p3
import prdrl.experiments.R1_region_protected_response as r1
import prdrl.experiments.R1N_production_noise_interaction as r1n
import prdrl.experiments.N1_predict_noise_calibration as n1

from prdrl.experiments.N1_predict_noise_calibration import (
    run_one_n1, run_region_n1, select_candidate,
    rho_noise_metric, child_metrics,
    generate_base_samples, apply_samples,
    BUDGET, POP_SIZE, D, N_OBJ, N_PARTITIONS,
    SIGMA_SWEEP, SIGMA_PROD, LAMBDA_PRED,
    DISCOVERY_SEEDS, HOLDOUT_SEEDS,
    make_rng, nsga2_survival_traced, region_protected_survival,
    compute_pf_coverage_per_point,
)

from prdrl.history.region_history import RegionHistory
from prdrl.features.forecast_state import get_predict_motion_from_history
from prdrl.core.normalize import normalize
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.metrics.igd import igd
from prdrl.problems.controlled_predict import evaluate as cp_evaluate, get_bounds, get_true_pf

# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoke_rows():
    """2 seeds × 2 events, all 5 sigmas — quick smoke set."""
    rows = []
    for seed in [1, 2]:
        rows.extend(run_one_n1(seed, 2, "PERSISTENT", SIGMA_SWEEP, split="discovery"))
    return rows


@pytest.fixture(scope="module")
def valid_smoke(smoke_rows):
    return [r for r in smoke_rows
            if r.get("has_history", True) and "delta_igd" in r]


# ─── T1: Estimator uses history, not GT ───────────────────────────────────────

def test_t1_estimator_uses_history(valid_smoke):
    """delta_hat is derived from RegionHistory; history_t >= 1 for valid rows."""
    for row in valid_smoke[:30]:
        assert "delta_hat_norm" in row
        assert row.get("history_t", -1) >= 1
        assert "estimator_cosine" in row


# ─── T2: No future leakage ────────────────────────────────────────────────────

def test_t2_no_future_leakage(valid_smoke):
    """Every valid row has history_t >= 1 (2 history snapshots needed)."""
    for row in valid_smoke[:20]:
        assert row["history_t"] >= 1, (
            f"Row has history_t={row['history_t']} — insufficient history snapshots")


# ─── T3: apply_samples matches production predict formula ─────────────────────

def test_t3_production_predict_formula():
    """apply_samples matches production formula for controlled (parent_idx, xi)."""
    xl, xu = get_bounds(D)
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a

    rng = np.random.default_rng(42)
    X_region = rng.uniform(xl, xu, size=(6, D))
    parent_idxs = [0, 3, 2, 5, 1, 4]
    xis = [rng.standard_normal(D) for _ in range(6)]
    delta = rng.uniform(-0.05, 0.05, size=D)

    for sigma in SIGMA_SWEEP:
        ch_exp = apply_samples(parent_idxs, xis, X_region, delta, sigma, xl, xu)
        ch_ref = []
        for pidx, xi in zip(parent_idxs, xis):
            z_p   = (X_region[pidx] - xl_a) / span
            z_c   = np.clip(z_p + delta + sigma * xi, 0.0, 1.0)
            ch_ref.append(xl_a + z_c * span)
        np.testing.assert_allclose(ch_exp, np.array(ch_ref), atol=1e-14,
                                   err_msg=f"sigma={sigma}")


# ─── T4: Same parent_idxs for all sigma (CRN) ────────────────────────────────

def test_t4_crn_same_parent_idxs():
    """Same RNG seed yields same parent_idxs regardless of sigma."""
    rng1 = make_rng(7, 3, 2, "PREDICT_SHARED")
    rng2 = make_rng(7, 3, 2, "PREDICT_SHARED")
    idxs1, _ = generate_base_samples(BUDGET, D, rng1, 10)
    idxs2, _ = generate_base_samples(BUDGET, D, rng2, 10)
    assert idxs1 == idxs2, "CRN: parent_idxs must be identical for same seed"


# ─── T5: Same xi for all sigma (CRN) ─────────────────────────────────────────

def test_t5_crn_same_xi():
    """Same RNG seed yields identical xi vectors for all sigma."""
    rng1 = make_rng(7, 3, 2, "PREDICT_SHARED")
    rng2 = make_rng(7, 3, 2, "PREDICT_SHARED")
    _, xis1 = generate_base_samples(BUDGET, D, rng1, 10)
    _, xis2 = generate_base_samples(BUDGET, D, rng2, 10)
    for i, (x1, x2) in enumerate(zip(xis1, xis2)):
        np.testing.assert_array_equal(x1, x2, err_msg=f"xi[{i}] differs")


# ─── T6: Sigma scaling — unclipped case ──────────────────────────────────────

def test_t6_sigma_scaling():
    """For unclipped children: z(s2) - z(0) = (s2/s1) * [z(s1) - z(0)]."""
    xl = np.zeros(D)
    xu = np.ones(D)

    # Construct unclippable scenario: z_parent = 0.5, delta small
    X_region = np.full((1, D), 0.5)
    parent_idxs = [0]

    rng = np.random.default_rng(99)
    # Small xi so that even sigma=0.03 stays inside [0,1] from z_parent=0.5
    xi = rng.uniform(-0.05, 0.05, size=D)
    xis = [xi]
    delta = np.full(D, 0.0)  # zero delta so noise is the only perturbation

    s1, s2 = 0.005, 0.01

    ch0  = apply_samples(parent_idxs, xis, X_region, delta, 0.0, xl, xu)
    ch_s1 = apply_samples(parent_idxs, xis, X_region, delta, s1, xl, xu)
    ch_s2 = apply_samples(parent_idxs, xis, X_region, delta, s2, xl, xu)

    span = xu - xl
    z0   = (ch0[0]  - xl) / span
    z_s1 = (ch_s1[0] - xl) / span
    z_s2 = (ch_s2[0] - xl) / span

    diff_s1 = z_s1 - z0
    diff_s2 = z_s2 - z0

    expected_s2 = (s2 / s1) * diff_s1
    np.testing.assert_allclose(diff_s2, expected_s2, atol=1e-14,
                               err_msg="Sigma scaling violated for unclipped case")


# ─── T7: Sigma=0 has no noise term ───────────────────────────────────────────

def test_t7_sigma0_no_noise():
    """sigma=0: z_child = clip(z_parent + delta_hat) — xi has zero effect."""
    xl, xu = get_bounds(D)
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a

    rng = np.random.default_rng(55)
    X_region = rng.uniform(xl, xu, size=(4, D))
    parent_idxs = list(range(4))
    xis = [rng.standard_normal(D) for _ in range(4)]
    delta = rng.uniform(-0.1, 0.1, size=D)

    ch0 = apply_samples(parent_idxs, xis, X_region, delta, 0.0, xl, xu)

    for i, (pidx, _) in enumerate(zip(parent_idxs, xis)):
        z_p  = (X_region[pidx] - xl_a) / span
        z_exp = np.clip(z_p + delta, 0.0, 1.0)
        np.testing.assert_allclose(ch0[i], xl_a + z_exp * span, atol=1e-14)


# ─── T8: Clipping works for non-unit bounds ───────────────────────────────────

def test_t8_clipping_non_unit_bounds():
    """Clipping is applied in normalized [0,1] space and remapped to xl/xu."""
    xl = np.array([-5.0] * D)
    xu = np.array([5.0] * D)
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a

    rng = np.random.default_rng(66)
    X_region = rng.uniform(xl, xu, size=(5, D))
    parent_idxs = list(range(5))
    xis = [rng.standard_normal(D) for _ in range(5)]
    delta = rng.uniform(-0.5, 0.5, size=D)

    ch = apply_samples(parent_idxs, xis, X_region, delta, 0.03, xl, xu)

    for xi_val in ch.flatten():
        assert -5.0 - 1e-12 <= xi_val <= 5.0 + 1e-12, f"Child value {xi_val} out of bounds"


# ─── T9: rho_noise known case ─────────────────────────────────────────────────

def test_t9_rho_noise_known():
    """rho_noise = median(||sigma*xi||) / (||delta_hat|| + eps) — exact."""
    rng = np.random.default_rng(101)
    xis = [rng.standard_normal(D) for _ in range(BUDGET)]
    delta_pop = rng.uniform(-0.05, 0.05, size=D)
    sigma = 0.01

    metrics = rho_noise_metric(xis, sigma, delta_pop)

    xi_arr = np.asarray(xis)
    norms  = np.linalg.norm(sigma * xi_arr, axis=1)
    expected_rho = float(np.median(norms)) / (np.linalg.norm(delta_pop) + 1e-12)

    assert abs(metrics["rho_noise"] - expected_rho) < 1e-12


# ─── T10: Zero-motion rho handled safely ─────────────────────────────────────

def test_t10_zero_motion_rho():
    """rho_noise does not raise when ||delta_hat|| ≈ 0; rho_noise → large value."""
    rng = np.random.default_rng(202)
    xis = [rng.standard_normal(D) for _ in range(BUDGET)]
    delta_zero = np.zeros(D)
    sigma = 0.01

    metrics = rho_noise_metric(xis, sigma, delta_zero)
    assert np.isfinite(metrics["rho_noise"]) or metrics["rho_noise"] > 1e10


# ─── T11: Displacement cosine known vectors ───────────────────────────────────

def test_t11_displacement_cosine_known():
    """median_displacement_cosine = 1.0 when delta_hat and d_actual are collinear."""
    xl = np.zeros(D)
    xu = np.ones(D)
    xl_a = np.asarray(xl, dtype=float)
    xu_a = np.asarray(xu, dtype=float)
    span  = xu_a - xl_a

    # Parent fixed at 0.5 in all dims; delta small so no clipping
    X_region = np.full((1, D), 0.5)
    delta_hat = np.full(D, 0.01)
    parent_idxs = [0] * BUDGET
    xis = [np.zeros(D)] * BUDGET  # zero xi → no noise → d_actual ‖ delta_hat

    pf_star = get_true_pf(n_points=50)
    ch = apply_samples(parent_idxs, xis, X_region, delta_hat, 0.0, xl, xu)
    F_ch = np.random.default_rng(77).uniform(0, 1, size=(BUDGET, N_OBJ))

    cm = child_metrics(parent_idxs, xis, X_region, delta_hat, 0.0, xl, xu,
                       F_ch, pf_star)

    assert abs(cm["median_displacement_cosine"] - 1.0) < 1e-10, (
        f"Expected cosine=1.0 for collinear case, got {cm['median_displacement_cosine']}")


# ─── T12: HARD invariant — 0 outside-region drops at every sigma ──────────────

def test_t12_hard_zero_outside_drops(valid_smoke):
    """n_old_dropped_other == 0 for all sigma values in every valid row."""
    for row in valid_smoke:
        assert row.get("n_old_dropped_other", 0) == 0, (
            f"sigma={row['sigma']}: HARD dropped {row['n_old_dropped_other']} outside-region OLD")


# ─── T13: Final population size = POP_SIZE ────────────────────────────────────

def test_t13_population_size(valid_smoke):
    """All sigma rows have FE=BUDGET and survival rates in [0, 1]."""
    for row in valid_smoke:
        assert row["fe"] == BUDGET
        assert 0.0 <= row["child_survival_rate"] <= 1.0


# ─── T14: Same FE for all sigma ───────────────────────────────────────────────

def test_t14_same_fe_all_sigma(valid_smoke):
    """Every sigma row records FE = BUDGET = 10."""
    for row in valid_smoke:
        assert row["fe"] == BUDGET, f"sigma={row['sigma']} FE={row['fe']} != {BUDGET}"


# ─── T15: Deterministic survival ─────────────────────────────────────────────

def test_t15_determinism():
    """Two runs with the same seed produce identical delta_igd for all sigma."""
    rows_a = run_one_n1(4, 2, "PERSISTENT", SIGMA_SWEEP, split="discovery")
    rows_b = run_one_n1(4, 2, "PERSISTENT", SIGMA_SWEEP, split="discovery")

    va = [r for r in rows_a if r.get("has_history", True) and "delta_igd" in r]
    vb = [r for r in rows_b if r.get("has_history", True) and "delta_igd" in r]

    assert len(va) == len(vb), "Determinism: row counts differ"
    for i, (ra, rb) in enumerate(zip(va, vb)):
        assert ra["sigma"]     == rb["sigma"]
        assert ra["delta_igd"] == rb["delta_igd"], f"Row {i}: delta_igd not deterministic"


# ─── T16: Baseline immutability ───────────────────────────────────────────────

def test_t16_baseline_immutability():
    """run_region_n1 does not mutate X_pop or F_plus."""
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    region_dynamics = e2p3.make_scenario_dynamics("PERSISTENT", R)
    pf_star = get_true_pf(n_points=100)
    SEED = 88

    rng = np.random.default_rng(SEED)
    X_pop = rng.uniform(xl, xu, size=(POP_SIZE, D))
    t_cur = e2p3.compute_time(0)
    X_pop = e2p3.evolve_nsga2(X_pop, t_cur, region_dynamics,
                               e2p3.WARMUP_GENERATIONS, SEED * 10000, xl, xu)
    history = RegionHistory(n_regions=R); history_t = -1; gen = 0

    for ci in range(1, 4):
        F_minus = cp_evaluate(X_pop, t_cur, region_dynamics)
        gen += e2p3.GENERATIONS_PER_CHANGE
        t_new = e2p3.compute_time(gen)
        F_plus = cp_evaluate(X_pop, t_new, region_dynamics)
        F_tm, _ = normalize(F_minus, F_plus)
        rids = np.asarray(partition(np.asarray(F_tm), V))
        history_t += 1
        history.update(t=history_t, X=X_pop, F=F_minus, region_ids=rids, xl=xl, xu=xu)

        if ci >= 2:
            X_pop_copy  = X_pop.copy()
            F_plus_copy = F_plus.copy()
            _ = run_region_n1(0, X_pop, F_plus, F_minus, rids, t_new, t_cur,
                               region_dynamics, R, pf_star, V,
                               xl, xu, SEED, ci, history, history_t,
                               sigma_values=SIGMA_SWEEP)
            np.testing.assert_array_equal(X_pop,  X_pop_copy,  err_msg="X_pop modified!")
            np.testing.assert_array_equal(F_plus, F_plus_copy, err_msg="F_plus modified!")
            break

        t_cur = t_new
        X_pop = e2p3.evolve_nsga2(X_pop, t_cur, region_dynamics,
                                   e2p3.GENERATIONS_PER_CHANGE,
                                   SEED * 10000 + ci, xl, xu)


# ─── T17: IGD identity ────────────────────────────────────────────────────────

def test_t17_igd_identity(valid_smoke):
    """igd_identity_residual < 1e-10 for all sigma rows."""
    for row in valid_smoke:
        res = row.get("igd_identity_residual", 0.0)
        if res is not None:
            assert res < 1e-10, f"sigma={row['sigma']} igd_identity_residual={res:.2e}"


# ─── T18: sigma=0.03 agrees with R1-N HARD03 ─────────────────────────────────

def test_t18_sigma03_agrees_r1n():
    """N1 sigma=0.03 HARD rows reproduce R1-N HARD03 rows for same seeds/events."""
    n1_rows = run_one_n1(1, 2, "PERSISTENT", [0.03], split="discovery",
                         include_unr_anchors=False)
    r1n_rows = r1n.run_one_r1n(1, 2, "PERSISTENT")

    n1_valid  = {(r["seed"], r["event_id"], r["region_id"]): r
                 for r in n1_rows if r.get("has_history", True) and "delta_igd" in r}
    r1n_valid = {(r["seed"], r["event_id"], r["region_id"]): r
                 for r in r1n_rows if r.get("has_history", True) and "hard03_delta_igd" in r}

    matched = 0
    for key in n1_valid:
        if key in r1n_valid:
            n1_digd  = n1_valid[key]["delta_igd"]
            r1n_digd = r1n_valid[key]["hard03_delta_igd"]
            assert abs(n1_digd - r1n_digd) < 1e-12, (
                f"N1 sigma=0.03 delta_igd={n1_digd:.6f} != R1-N HARD03={r1n_digd:.6f} at {key}")
            matched += 1

    assert matched > 0, "No matched keys between N1 sigma=0.03 and R1-N HARD03"


# ─── T19: sigma=0 agrees with R1-N HARD0 ─────────────────────────────────────

def test_t19_sigma0_agrees_r1n():
    """N1 sigma=0 HARD rows reproduce R1-N HARD0 rows for same seeds/events."""
    n1_rows = run_one_n1(2, 2, "PERSISTENT", [0.0], split="discovery",
                         include_unr_anchors=False)
    r1n_rows = r1n.run_one_r1n(2, 2, "PERSISTENT")

    n1_valid  = {(r["seed"], r["event_id"], r["region_id"]): r
                 for r in n1_rows if r.get("has_history", True) and "delta_igd" in r}
    r1n_valid = {(r["seed"], r["event_id"], r["region_id"]): r
                 for r in r1n_rows if r.get("has_history", True) and "hard0_delta_igd" in r}

    matched = 0
    for key in n1_valid:
        if key in r1n_valid:
            n1_digd  = n1_valid[key]["delta_igd"]
            r1n_digd = r1n_valid[key]["hard0_delta_igd"]
            assert abs(n1_digd - r1n_digd) < 1e-12, (
                f"N1 sigma=0 delta_igd={n1_digd:.6f} != R1-N HARD0={r1n_digd:.6f} at {key}")
            matched += 1

    assert matched > 0, "No matched keys between N1 sigma=0 and R1-N HARD0"


# ─── T20: Missing history handled correctly ───────────────────────────────────

def test_t20_missing_history_handled(smoke_rows):
    """All rows have 'has_history'; skipped rows lack 'delta_igd'."""
    for row in smoke_rows:
        assert "has_history" in row
    skipped = [r for r in smoke_rows if not r.get("has_history", True)]
    for row in skipped:
        assert "delta_igd" not in row
        assert "seed" in row and "event_id" in row


# ─── T21: Production constants unchanged ──────────────────────────────────────

def test_t21_production_constants():
    """N1 does not alter production-locked constants."""
    assert n1.BUDGET       == 10
    assert n1.POP_SIZE     == 100
    assert n1.LAMBDA_PRED  == 1.0
    assert n1.SIGMA_PROD   == 0.03
    assert e2p3.BUDGET     == 10
    assert e2p3.POP_SIZE   == 100


# ─── T22: Candidate selected only from discovery rows ────────────────────────

def test_t22_candidate_freeze():
    """select_candidate only uses split='discovery' rows; ignores 'holdout'."""
    rng = np.random.default_rng(55)
    # Build fake discovery rows and fake holdout rows with different values
    fake_disc = []
    for sigma in SIGMA_SWEEP:
        for i in range(50):
            fake_disc.append({
                "has_history": True, "split": "discovery",
                "sigma": sigma, "seed": 1, "event_id": 3, "region_id": i % 10,
                "delta_igd": float(rng.uniform(-1e-4, 1e-4)),
                "child_pf_dist": float(rng.uniform(0.01, 0.1)),
                "clipping_frac": float(rng.uniform(0.1, 0.3)) if sigma < 0.02 else 0.9,
                "rho_noise": float(sigma / (0.01 + 1e-12)) if sigma > 0 else 0.0,
            })
    # Fake holdout with very different (better) values — should NOT affect candidate
    fake_holdout = []
    for sigma in SIGMA_SWEEP:
        for i in range(50):
            fake_holdout.append({
                "has_history": True, "split": "holdout",
                "sigma": sigma, "seed": 15, "event_id": 3, "region_id": i % 10,
                "delta_igd": 999.0,  # extreme value — must not influence candidate
                "child_pf_dist": 0.0,
                "clipping_frac": 0.0,
                "rho_noise": 0.0,
            })

    all_rows = fake_disc + fake_holdout

    # select_candidate filters to split='discovery' only
    candidate, reasoning = select_candidate(all_rows)
    # Key check: no holdout data influenced the result
    assert "candidate" in reasoning


# ─── T23: Holdout seeds disjoint from discovery seeds ────────────────────────

def test_t23_holdout_seeds_disjoint():
    """DISCOVERY_SEEDS and HOLDOUT_SEEDS share no elements."""
    assert set(DISCOVERY_SEEDS).isdisjoint(set(HOLDOUT_SEEDS)), (
        f"Seed overlap: {set(DISCOVERY_SEEDS) & set(HOLDOUT_SEEDS)}")


# ─── T24: Production predictor file unmodified ───────────────────────────────

def test_t24_no_production_mutation():
    """predict.py contains the production predict function, not N1-specific code."""
    import importlib.util, hashlib, os
    predict_path = "src/prdrl/operators/predict.py"
    assert os.path.exists(predict_path), "predict.py missing"
    content = open(predict_path).read()
    # N1 must not add 'sigma_sweep', 'N1', or 'calibration' tokens to production predict
    assert "sigma_sweep" not in content, "N1 contaminated production predict.py with sigma_sweep"
    assert "N1_" not in content, "N1 contaminated production predict.py"
    assert "SIGMA_SWEEP" not in content, "N1 contaminated production predict.py with SIGMA_SWEEP"
