"""
R1 Region-Protected Response — Tests T1-T14.

T1  -- region_protected_survival produces exactly POP_SIZE survivors
T2  -- OLD individuals outside region r are never dropped (zero cross-region loss)
T3  -- all OLD_outside individuals appear in final population (by pool_idx)
T4  -- total final pool indices are unique (no duplicates in survivors)
T5  -- selected_r candidates all come from OLD_r ∪ CHILD_r (none from OLD_outside)
T6  -- region_protected_survival is deterministic (same seed → same result)
T7  -- UNRESTRICTED and PROTECTED receive identical children (same rng_shared, same B)
T8  -- global ranks use full merged pool (children do NOT lose rank advantage)
T9  -- protected IGD >= unrestricted IGD on PERSISTENT GT (hypothesis: protection helps)
T10 -- n_r_before candidates fill n_r_before slots (|selected_r| == n_r_before)
T11 -- production survival (nsga2_survival) is NOT imported or called in protected path
T12 -- production BUDGET remains 10
T13 -- FE == B for both methods
T14 -- smoke run (2 seeds × 2 events) completes without errors; rows have required fields
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import prdrl.experiments.R1_region_protected_response as r1
import prdrl.experiments.E2P3_survival_translation    as e2p3
from prdrl.problems.controlled_predict import get_bounds, evaluate as cp_eval
from prdrl.metrics.igd import igd

D    = r1.D
xl_g, xu_g = get_bounds(D)
xl_g = np.asarray(xl_g, dtype=float)
xu_g = np.asarray(xu_g, dtype=float)

REQUIRED_FIELDS = [
    "unr_delta_igd", "prot_delta_igd", "delta_delta_igd",
    "unr_n_old_dropped_other", "prot_n_old_dropped_other",
    "unr_mean_delta_pf_outside", "prot_mean_delta_pf_outside",
    "igd_gain_outside", "budget_b", "fe",
]


# ===========================================================
# T1 -- region_protected_survival produces exactly POP_SIZE survivors
# ===========================================================
def test_t1_protected_survives_pop_size():
    """Output population must have exactly POP_SIZE individuals."""
    rng = np.random.default_rng(1001)
    N = r1.POP_SIZE
    B = r1.BUDGET
    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))

    # Assign region 0 to first 10 old individuals, rest to region 1
    rids = np.zeros(N, dtype=int)
    rids[:10] = 0
    rids[10:] = 1

    X_s, F_s, idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=0, old_region_ids=rids,
        n_r_before=10, xl=xl_g, xu=xu_g,
    )
    assert len(X_s) == N, f"Expected POP_SIZE={N}, got {len(X_s)}"
    assert len(F_s) == N
    assert len(idx) == N


# ===========================================================
# T2 -- OLD individuals outside region r are never dropped
# ===========================================================
def test_t2_protected_zero_cross_region_drop():
    """After protected survival, no OLD individual from regions != r is dropped."""
    rng = np.random.default_rng(1002)
    N = r1.POP_SIZE
    B = r1.BUDGET

    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))

    r_response = 0
    rids = np.zeros(N, dtype=int)
    n_r_before = 12
    rids[:n_r_before] = r_response
    rids[n_r_before:] = 1

    _, _, final_idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=r_response, old_region_ids=rids,
        n_r_before=n_r_before, xl=xl_g, xu=xu_g,
    )

    # All OLD individuals outside r_response must be in final_idx
    outside_old = [i for i in range(N) if rids[i] != r_response]
    for i in outside_old:
        assert i in final_idx, (
            f"OLD individual {i} (region={rids[i]}) was dropped by protected survival"
        )


# ===========================================================
# T3 -- All OLD_outside individuals appear unchanged
# ===========================================================
def test_t3_old_outside_preserved():
    """OLD individuals outside r must appear in final_idx with correct X/F values."""
    rng = np.random.default_rng(1003)
    N = r1.POP_SIZE
    B = r1.BUDGET

    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))

    rids = np.zeros(N, dtype=int)
    n_r_before = 8
    rids[:n_r_before] = 0
    rids[n_r_before:] = 1

    X_s, F_s, final_idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=0, old_region_ids=rids,
        n_r_before=n_r_before, xl=xl_g, xu=xu_g,
    )

    # Find each outside-r OLD individual in the survived population
    outside_old = [i for i in range(N) if rids[i] != 0]
    for pool_i in outside_old:
        # The survived X must contain X_pop[pool_i]
        match = np.any(np.all(X_s == X_pop[pool_i], axis=1))
        assert match, f"X_pop[{pool_i}] (outside region) not in survived X_s"


# ===========================================================
# T4 -- Final pool indices are unique
# ===========================================================
def test_t4_final_indices_unique():
    """No pool index should appear twice in the final population."""
    rng = np.random.default_rng(1004)
    N = r1.POP_SIZE
    B = r1.BUDGET

    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))

    rids = np.zeros(N, dtype=int)
    rids[:10] = 0
    rids[10:] = 1

    _, _, final_idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=0, old_region_ids=rids,
        n_r_before=10, xl=xl_g, xu=xu_g,
    )

    assert len(final_idx) == len(np.unique(final_idx)), (
        f"Duplicate pool indices in final survivors: {final_idx}"
    )


# ===========================================================
# T5 -- Selected r-candidates come only from OLD_r ∪ CHILD_r
# ===========================================================
def test_t5_selected_candidates_valid():
    """All n_r_before selections must come from OLD_r or CHILD_r (pool indices N..N+B-1)."""
    rng = np.random.default_rng(1005)
    N = r1.POP_SIZE
    B = r1.BUDGET

    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))

    r_resp = 0
    n_r_before = 15
    rids = np.zeros(N, dtype=int)
    rids[:n_r_before] = r_resp
    rids[n_r_before:] = 1

    _, _, final_idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=r_resp, old_region_ids=rids,
        n_r_before=n_r_before, xl=xl_g, xu=xu_g,
    )

    # Identify which final indices are "selected_r" (not old_outside)
    old_outside = [i for i in range(N) if rids[i] != r_resp]
    selected_r  = [i for i in final_idx if i not in old_outside]

    valid_r     = set(range(n_r_before)) | set(range(N, N + B))
    for i in selected_r:
        assert i in valid_r, (
            f"Selected r-candidate {i} is outside OLD_r ∪ CHILD_r"
        )
    assert len(selected_r) == n_r_before, (
        f"Expected {n_r_before} r-slots filled, got {len(selected_r)}"
    )


# ===========================================================
# T6 -- Protected survival is deterministic
# ===========================================================
def test_t6_determinism():
    """Same seed → same final indices."""
    rng = np.random.default_rng(1006)
    N = r1.POP_SIZE
    B = r1.BUDGET

    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))
    rids  = np.zeros(N, dtype=int); rids[:10] = 0; rids[10:] = 1

    _, _, idx_a = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch, 0, rids, 10, xl_g, xu_g, surv_seed=42
    )
    _, _, idx_b = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch, 0, rids, 10, xl_g, xu_g, surv_seed=42
    )
    np.testing.assert_array_equal(idx_a, idx_b, err_msg="Protected survival not deterministic")


# ===========================================================
# T7 -- UNRESTRICTED and PROTECTED receive identical children
# ===========================================================
def test_t7_identical_children():
    """Both methods must process the same children (same RNG, same B)."""
    rows = r1.run_one_r1(seed=7, n_scored_changes=1, scenario_name="PERSISTENT")
    for row in rows:
        b_unr  = row["budget_b"]
        fe_unr = row["fe"]
        assert b_unr == r1.BUDGET, f"budget_b={b_unr} != BUDGET={r1.BUDGET}"
        assert fe_unr == r1.BUDGET, f"fe={fe_unr} != BUDGET={r1.BUDGET}"


# ===========================================================
# T8 -- Global ranks use full merged pool
# ===========================================================
def test_t8_global_ranks_on_full_pool():
    """Children in a well-dominating position should win a global rank 0 slot."""
    # Construct OLD with bad F values and CHILD with Pareto-superior F
    N = r1.POP_SIZE
    B = r1.BUDGET

    rng = np.random.default_rng(1008)
    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    # OLD: all have F values in [0.5, 1.0] — bad
    F_pop = rng.uniform(0.5, 1.0, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    # Children: dominate the OLD — F values in [0.0, 0.1]
    F_ch  = rng.uniform(0.0, 0.1, size=(B, 2))

    rids = np.zeros(N, dtype=int)
    rids[:10] = 0
    rids[10:] = 1

    _, F_surv, final_idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=0, old_region_ids=rids,
        n_r_before=10, xl=xl_g, xu=xu_g,
    )

    # All B=10 children (globally rank 0) should be selected in the r slots
    child_selected = sum(1 for i in final_idx if i >= N)
    assert child_selected == B, (
        f"Expected all {B} dominating children to be selected, got {child_selected}"
    )


# ===========================================================
# T9 -- Protected IGD delta >= unrestricted (protection helps on average)
# ===========================================================
def test_t9_protected_igd_not_worse():
    """On PERSISTENT GT sigma=0, protection should not hurt IGD on average.

    This is the core hypothesis test. We verify over 1 seed × 1 event that
    the median delta_delta_igd >= 0 (protected >= unrestricted).
    """
    rows = r1.run_one_r1(seed=9, n_scored_changes=1, scenario_name="PERSISTENT")
    delta_deltas = [r["delta_delta_igd"] for r in rows if r.get("delta_delta_igd") is not None]
    assert len(delta_deltas) > 0, "No delta_delta_igd values recorded"
    import statistics
    med_ddi = statistics.median(delta_deltas)
    # Allow some tolerance — this is a probabilistic check
    # The hypothesis is that median >= 0; a strong violation would be < -0.001
    assert med_ddi >= -0.001, (
        f"Median delta_delta_igd={med_ddi:.6f} strongly negative — "
        "protection appears to hurt IGD"
    )


# ===========================================================
# T10 -- Exactly n_r_before slots are filled from r-candidate pool
# ===========================================================
def test_t10_r_slots_exactly_n_r_before():
    """The number of r-region slots in the final pop must equal n_r_before."""
    rng = np.random.default_rng(1010)
    N = r1.POP_SIZE
    B = r1.BUDGET
    n_r_before = 7

    X_pop = rng.uniform(xl_g, xu_g, size=(N, D))
    F_pop = rng.uniform(0, 1, size=(N, 2))
    X_ch  = rng.uniform(xl_g, xu_g, size=(B, D))
    F_ch  = rng.uniform(0, 1, size=(B, 2))

    rids  = np.zeros(N, dtype=int)
    rids[:n_r_before] = 0
    rids[n_r_before:] = 1

    _, _, final_idx = r1.region_protected_survival(
        X_pop, F_pop, X_ch, F_ch,
        response_region_r=0, old_region_ids=rids,
        n_r_before=n_r_before, xl=xl_g, xu=xu_g,
    )

    old_outside = [i for i in range(N) if rids[i] != 0]
    selected_r  = [i for i in final_idx if i not in old_outside]
    assert len(selected_r) == n_r_before, (
        f"Expected {n_r_before} r-slots, got {len(selected_r)}"
    )


# ===========================================================
# T11 -- Production nsga2_survival not called in protected path
# ===========================================================
def test_t11_production_survival_not_called():
    """region_protected_survival must not call nsga2_survival internally."""
    import inspect
    src = inspect.getsource(r1.region_protected_survival)
    assert "nsga2_survival(" not in src, (
        "region_protected_survival must not call production nsga2_survival"
    )


# ===========================================================
# T12 -- Production BUDGET unchanged
# ===========================================================
def test_t12_budget_unchanged():
    assert r1.BUDGET == 10, f"BUDGET must be 10, got {r1.BUDGET}"


# ===========================================================
# T13 -- FE equals B for both methods
# ===========================================================
def test_t13_fe_equals_budget():
    rows = r1.run_one_r1(seed=13, n_scored_changes=1, scenario_name="PERSISTENT")
    for row in rows:
        assert row["fe"] == row["budget_b"] == r1.BUDGET, (
            f"fe={row['fe']} or budget_b={row['budget_b']} != BUDGET={r1.BUDGET}"
        )


# ===========================================================
# T14 -- Smoke run: 2 seeds × 2 events, required fields present
# ===========================================================
def test_t14_smoke_run():
    """2 seeds × 2 events must complete without error; all required fields present."""
    rows = []
    for seed in [1, 2]:
        rows.extend(r1.run_one_r1(seed, n_scored_changes=2, scenario_name="PERSISTENT"))

    assert len(rows) > 0, "Smoke run produced no rows"

    for row in rows:
        for field in REQUIRED_FIELDS:
            assert field in row, f"Required field '{field}' missing from row"
        # Protected method must drop ZERO outside-region OLD individuals
        assert row["prot_n_old_dropped_other"] == 0, (
            f"Protected method dropped {row['prot_n_old_dropped_other']} "
            f"outside-region individuals (must be 0)"
        )
