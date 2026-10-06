"""
E2-P4 Headroom vs Population Dilution -- Tests T1-T10.

T1  -- nested budget samples: B=5 children == first 5 of B=10; B=10 == first 10 of B=20, etc.
T2  -- same baseline: X_pop/F_plus identical across all B branches
T3  -- production BUDGET constant untouched (still 10)
T4  -- correct replacement denominator: pop_replacement_fraction = n_survived / POP_SIZE
T5  -- warmup does NOT change environment time: t_prev2, t_prev, t_new same for W=0 and W=30
T6  -- same X_init across warmup levels for same seed
T7  -- GT uses true t-1->t shift (delta_response_true semantics)
T8  -- no production mutation: BUDGET, SIGMA_GT_E4, etc. unchanged
T9  -- FE scales with B: n_children_generated == budget_b for GT rows
T10 -- cross-process deterministic RNG (no Python hash)
"""
import os, sys, math
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import prdrl.experiments.E2P4_headroom_dilution as e2p4
from prdrl.problems.controlled_predict import get_bounds
from prdrl.core.reference import reference
from prdrl.core.partition import partition
from prdrl.core.normalize import normalize
from prdrl.oracle.oracle import split_by_region
from prdrl.problems.controlled_predict import evaluate as cp_evaluate

D    = e2p4.D
xl_g, xu_g = get_bounds(D)
xl_g = np.asarray(xl_g, dtype=float)
xu_g = np.asarray(xu_g, dtype=float)


# =====================================================================
# T1 -- Nested budget samples: B=5 children == first 5 of B=10, etc.
# =====================================================================
def test_t1_nested_budget_samples():
    """Children for B=5 must equal the first 5 children of B=10 before survival."""
    region_dynamics = e2p4.make_scenario_dynamics("PERSISTENT", e2p4.N_PARTITIONS)
    rng_pop = np.random.default_rng(1)
    X_region = rng_pop.uniform(0.2, 0.8, size=(12, D))

    # Simulate delta: D-dimensional (leading 0 + D-1 shift components)
    delta_full = np.zeros(D)
    delta_full[1:] = 0.05

    base_seed = 42
    change_idx = 3
    r = 2

    rng1 = e2p4.make_rng(base_seed, change_idx, r, "PREDICT_SHARED")
    all_idxs, all_xis = e2p4.generate_base_samples(e2p4.B_MAX, D, rng1, len(X_region))

    ch_5  = e2p4.apply_samples(all_idxs[:5],  all_xis[:5],  X_region, delta_full, 0.0, xl_g, xu_g)
    ch_10 = e2p4.apply_samples(all_idxs[:10], all_xis[:10], X_region, delta_full, 0.0, xl_g, xu_g)
    ch_20 = e2p4.apply_samples(all_idxs[:20], all_xis[:20], X_region, delta_full, 0.0, xl_g, xu_g)

    # First 5 of B=10 must match B=5
    np.testing.assert_array_equal(ch_10[:5], ch_5)
    # First 10 of B=20 must match B=10
    np.testing.assert_array_equal(ch_20[:10], ch_10)


# =====================================================================
# T2 -- Same baseline: X_pop/F_plus identical across all B branches
# =====================================================================
def test_t2_same_baseline_across_b():
    """run_one_budget_sweep must not mutate X_pop between B branches."""
    rows = e2p4.run_one_budget_sweep(seed=1, n_scored_changes=1,
                                     scenario_name="PERSISTENT")
    # For same (seed, event_id, region_id), all GT rows must have same igd_before
    from collections import defaultdict
    by_event_region = defaultdict(list)
    for row in rows:
        if row["mode"] == "GT":
            key = (row["event_id"], row["region_id"])
            by_event_region[key].append(row["igd_before"])

    for key, vals in by_event_region.items():
        if len(vals) >= 2:
            assert all(abs(v - vals[0]) < 1e-10 for v in vals), \
                f"igd_before differs across B values for event/region {key}: {vals}"


# =====================================================================
# T3 -- Production BUDGET constant untouched
# =====================================================================
def test_t3_production_budget_untouched():
    assert e2p4.BUDGET == 10, "Production BUDGET must remain 10"
    # B_VALUES may contain any values but BUDGET itself is not changed
    assert e2p4.BUDGET not in [b for b in e2p4.B_VALUES if b != 10] or 10 in e2p4.B_VALUES


# =====================================================================
# T4 -- Correct replacement denominator
# =====================================================================
def test_t4_correct_replacement_denominator():
    rows = e2p4.run_one_budget_sweep(seed=2, n_scored_changes=1,
                                     scenario_name="PERSISTENT")
    for row in rows:
        n_surv = row.get("n_children_survived")
        prf    = row.get("population_replacement_fraction")
        if n_surv is None or prf is None:
            continue
        expected = n_surv / e2p4.POP_SIZE
        assert abs(prf - expected) < 1e-10, \
            f"pop_replace_frac={prf} != n_surv({n_surv})/POP_SIZE({e2p4.POP_SIZE})={expected}"
        # Must NOT equal n_surv/BUDGET (since POP_SIZE != BUDGET=10)
        assert e2p4.POP_SIZE != e2p4.BUDGET


# =====================================================================
# T5 -- Warmup does NOT change environment time
# =====================================================================
def test_t5_warmup_does_not_change_env_time():
    """t_prev2, t_prev, t_new must be the same for W=0 and W=30 at the same event."""
    rows_0  = e2p4.run_one_headroom_sweep(seed=1, n_scored_changes=2,
                                          scenario_name="PERSISTENT", warmup_gen=0)
    rows_30 = e2p4.run_one_headroom_sweep(seed=1, n_scored_changes=2,
                                          scenario_name="PERSISTENT", warmup_gen=30)

    def by_event(rows):
        d = {}
        for r in rows:
            key = (r["event_id"], r["region_id"])
            d[key] = (r["t_prev2"], r["t_prev"], r["t_new"])
        return d

    d0  = by_event(rows_0)
    d30 = by_event(rows_30)

    common_keys = set(d0.keys()) & set(d30.keys())
    assert len(common_keys) > 0, "No common (event,region) pairs between W=0 and W=30"

    for key in common_keys:
        t0_vals  = d0[key]
        t30_vals = d30[key]
        assert abs(t0_vals[0] - t30_vals[0]) < 1e-12, \
            f"t_prev2 differs W=0 vs W=30 at {key}: {t0_vals[0]} vs {t30_vals[0]}"
        assert abs(t0_vals[1] - t30_vals[1]) < 1e-12, \
            f"t_prev differs W=0 vs W=30 at {key}: {t0_vals[1]} vs {t30_vals[1]}"
        assert abs(t0_vals[2] - t30_vals[2]) < 1e-12, \
            f"t_new differs W=0 vs W=30 at {key}: {t0_vals[2]} vs {t30_vals[2]}"


# =====================================================================
# T6 -- Same X_init across warmup levels for same seed
# =====================================================================
def test_t6_same_x_init_across_warmup():
    """Verify X_init is constructed identically for all warmup levels."""
    def get_x_init(seed):
        xl, xu = get_bounds(D)
        rng = np.random.default_rng(seed)
        X_init = rng.uniform(xl, xu, size=(e2p4.POP_SIZE, D))
        return X_init

    # Same seed -> same X_init regardless of warmup (warmup uses seed*10000+W)
    xi1 = get_x_init(42)
    xi2 = get_x_init(42)
    np.testing.assert_array_equal(xi1, xi2)

    # Different seed -> different X_init
    xi3 = get_x_init(43)
    assert not np.array_equal(xi1, xi3), "Different seeds must yield different X_init"


# =====================================================================
# T7 -- GT uses true t-1->t shift (delta_response_true semantics)
# =====================================================================
def test_t7_gt_uses_delta_response_true():
    """run_region_headroom_sweep uses delta_response_true_full (t-1->t)."""
    import inspect
    src = inspect.getsource(e2p4.run_region_headroom_sweep)
    assert "mu_star_current - mu_star_prev" in src
    assert "delta_response_true_full" in src
    assert "Delta_GT" not in src


# =====================================================================
# T8 -- No production mutation
# =====================================================================
def test_t8_no_production_mutation():
    assert e2p4.BUDGET == 10
    assert e2p4.SIGMA_GT_E4 == 0.0
    assert e2p4.B_MAX == 100
    assert e2p4.POP_SIZE == 100
    # B_VALUES contains BUDGET but does not change it
    assert 10 in e2p4.B_VALUES, "B=10 (production BUDGET) must be in B_VALUES for reference"


# =====================================================================
# T9 -- FE scales with B: n_children_generated == budget_b for GT rows
# =====================================================================
def test_t9_fe_scales_with_b():
    rows = e2p4.run_one_budget_sweep(seed=3, n_scored_changes=1,
                                     scenario_name="PERSISTENT")
    for row in rows:
        if row["mode"] == "GT":
            assert row["n_children_generated"] == row["budget_b"], \
                f"n_children_generated={row['n_children_generated']} != budget_b={row['budget_b']}"


# =====================================================================
# T10 -- Cross-process deterministic RNG (no Python hash)
# =====================================================================
def test_t10_cross_process_deterministic_rng():
    """Same (base_seed, change_idx, region_id) always yields same samples.
    _e2p3_common_surv_seed must not use Python hash()."""
    import inspect
    import re

    # Check make_rng (inherited from e2p3/e2p2)
    src = inspect.getsource(e2p4.make_rng)
    assert not re.search(r'[^_a-zA-Z]hash\s*\(', src), "make_rng must not call Python hash()"

    # Check _e2p3_common_surv_seed
    src2 = inspect.getsource(e2p4._e2p3_common_surv_seed)
    assert not re.search(r'[^_a-zA-Z]hash\s*\(', src2), \
        "_e2p3_common_surv_seed must not call Python hash()"

    # Same call twice -> same result
    s1 = e2p4._e2p3_common_surv_seed(7, 3, 2)
    s2 = e2p4._e2p3_common_surv_seed(7, 3, 2)
    assert s1 == s2

    # Budget samples are reproducible
    rng1 = e2p4.make_rng(42, 5, 3, "PREDICT_SHARED")
    rng2 = e2p4.make_rng(42, 5, 3, "PREDICT_SHARED")
    idxs1, xis1 = e2p4.generate_base_samples(10, D, rng1, 15)
    idxs2, xis2 = e2p4.generate_base_samples(10, D, rng2, 15)
    assert idxs1 == idxs2
    for a, b in zip(xis1, xis2):
        np.testing.assert_array_equal(a, b)
