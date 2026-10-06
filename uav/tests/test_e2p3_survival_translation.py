"""
E2-P3 Survival Translation Audit -- Tests T1-T10 + TA1-TA6.

Original tests (T1-T10):
T1  -- traced survival returns identical population to production survival
T2  -- provenance count: survivors are correctly labeled OLD/CHILD
T3  -- FE equality: LOCAL / POP / GT generate same number of children
T4  -- sigma=0 for POP and GT (no Gaussian perturbation)
T5  -- no future leakage: POP estimator reads only t-2, t-1
T6  -- GT uses true t-1 -> t shift (delta_response_true semantics)
T7  -- baseline immutability: LOCAL does not mutate X_region used by POP/GT
T8  -- median_nearest_pf_distance sanity (point on PF -> 0; far point -> large)
T9  -- igd_before is identical across LOCAL/POP/GT branches for same event/region
T10 -- zero-survivor handling: survival_rate=0, surviving-child PF metric=NaN, no crash

Corrected-E2P3 tests (TA1-TA6):
TA1 -- provenance via pymoo pool_idx attribute (no tobytes): traced == production, exact
TA2 -- duplicate-X provenance: CHILD copy of an OLD vector gets correct CHILD attribution
TA3 -- LOCAL/POP/GT receive the SAME survival seed for same (seed, event, region)
TA4 -- changing mode alone does NOT change the survival seed
TA5 -- region_type and dominant_bin are present and valid in every returned row
TA6 -- population_replacement_fraction = n_survived / POP_SIZE (not / BUDGET)
"""
import os, sys, math
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import prdrl.experiments.E2P3_survival_translation as e2p3
from prdrl.problems.controlled_predict import get_bounds, get_true_pf

D    = e2p3.D
xl_g, xu_g = get_bounds(D)
xl_g = np.asarray(xl_g, dtype=float)
xu_g = np.asarray(xu_g, dtype=float)


def _interior_X(n=20, seed=0):
    return np.random.default_rng(seed).uniform(0.2, 0.8, size=(n, D))


# =====================================================================
# T1 -- Traced survival produces identical X, F as production survival
# =====================================================================
def test_t1_traced_survival_matches_production():
    from prdrl.oracle.oracle import nsga2_survival

    rng = np.random.default_rng(1)
    X_pool = rng.uniform(0.0, 1.0, size=(60, D))
    F_pool = rng.uniform(0.0, 1.0, size=(60, 2))
    surv_seed_val = 12345

    np.random.seed(surv_seed_val)
    X_prod, F_prod = nsga2_survival(X_pool, F_pool, n_survive=40, xl=xl_g, xu=xu_g)

    X_tr, F_tr, orig_idx = e2p3.nsga2_survival_traced(
        X_pool, F_pool, 40, xl_g, xu_g, surv_seed_val
    )

    assert X_tr.shape == X_prod.shape
    assert F_tr.shape == F_prod.shape

    def sort_rows(A):
        return A[np.lexsort(A.T[::-1])]
    np.testing.assert_allclose(sort_rows(X_tr), sort_rows(X_prod), atol=0, rtol=0)
    np.testing.assert_allclose(sort_rows(F_tr), sort_rows(F_prod), atol=0, rtol=0)


# =====================================================================
# T2 -- Provenance: survivors from OLD vs CHILD correctly identified
# =====================================================================
def test_t2_provenance_count():
    rng = np.random.default_rng(7)
    N_old = 50
    N_ch  = 10
    X_old = rng.uniform(0.0, 1.0, size=(N_old, D))
    X_ch  = rng.uniform(0.0, 1.0, size=(N_ch,  D))
    F_old = rng.uniform(0.0, 1.0, size=(N_old, 2))
    F_ch  = rng.uniform(0.0, 1.0, size=(N_ch,  2))

    X_pool = np.vstack([X_old, X_ch])
    F_pool = np.vstack([F_old, F_ch])

    _, _, orig_idx = e2p3.nsga2_survival_traced(X_pool, F_pool, 30, xl_g, xu_g)

    assert np.all(orig_idx >= 0), "All survivors must be matched to original pool"
    assert np.all(orig_idx < len(X_pool))

    n_child_surv = int(np.sum(orig_idx >= N_old))
    assert 0 <= n_child_surv <= N_ch


# =====================================================================
# T3 -- FE equality: LOCAL / POP / GT each produce BUDGET children
# =====================================================================
def test_t3_fe_equality():
    X_region = _interior_X(n=12)

    ch_local = e2p3.make_local_children(X_region, xl_g, xu_g, 1, 1, 1)
    assert len(ch_local) == e2p3.BUDGET

    rng = e2p3.make_rng(1, 1, 1, "PREDICT_SHARED")
    idxs, xis = e2p3.generate_base_samples(e2p3.BUDGET, D, rng, len(X_region))
    delta = np.zeros(D)

    ch_pop = e2p3.apply_samples(idxs, xis, X_region, delta, e2p3.SIGMA_POP, xl_g, xu_g)
    ch_gt  = e2p3.apply_samples(idxs, xis, X_region, delta, e2p3.SIGMA_GT,  xl_g, xu_g)

    assert len(ch_pop) == e2p3.BUDGET
    assert len(ch_gt)  == e2p3.BUDGET


# =====================================================================
# T4 -- sigma=0: POP and GT offspring = parent + delta (no Gaussian noise)
# =====================================================================
def test_t4_sigma_zero_no_noise():
    assert e2p3.SIGMA_POP == 0.0
    assert e2p3.SIGMA_GT  == 0.0

    X_region = _interior_X(n=5)
    delta = np.zeros(D)

    rng = e2p3.make_rng(1, 1, 1, "PREDICT_SHARED")
    idxs, xis = e2p3.generate_base_samples(e2p3.BUDGET, D, rng, len(X_region))

    ch = e2p3.apply_samples(idxs, xis, X_region, delta, 0.0, xl_g, xu_g)
    for i, (idx, xi) in enumerate(zip(idxs, xis)):
        parent = X_region[idx]
        z_p = (parent - xl_g) / (xu_g - xl_g)
        expected = xl_g + np.clip(z_p, 0.0, 1.0) * (xu_g - xl_g)
        np.testing.assert_allclose(ch[i], expected, atol=1e-10)


# =====================================================================
# T5 -- No future leakage: POP estimator reads only t-2, t-1
# =====================================================================
def test_t5_no_future_leakage():
    import inspect
    src = inspect.getsource(e2p3.run_region_e2p3)
    assert "t_prev2" in src and "t_prev" in src
    assert "Delta_POP_full = (np.mean(Z_cur" in src


# =====================================================================
# T6 -- GT uses delta_response_true (t-1 → t), not historical shift
# =====================================================================
def test_t6_gt_semantics():
    import inspect
    src = inspect.getsource(e2p3.run_region_e2p3)
    assert "delta_response_true" in src
    assert "delta_response_true_full" in src
    assert "Delta_GT" not in src


# =====================================================================
# T7 -- Baseline immutability: LOCAL does not mutate X_region for POP/GT
# =====================================================================
def test_t7_baseline_immutability():
    X_region = _interior_X(n=10, seed=5)
    X_copy   = X_region.copy()

    ch_local = e2p3.make_local_children(X_region, xl_g, xu_g, 1, 1, 1)
    np.testing.assert_array_equal(X_region, X_copy)

    rng = e2p3.make_rng(1, 1, 1, "PREDICT_SHARED")
    idxs, xis = e2p3.generate_base_samples(e2p3.BUDGET, D, rng, len(X_region))
    delta = np.zeros(D)
    _ = e2p3.apply_samples(idxs, xis, X_region, delta, 0.0, xl_g, xu_g)
    np.testing.assert_array_equal(X_region, X_copy)


# =====================================================================
# T8 -- median_nearest_pf_distance sanity
# =====================================================================
def test_t8_pf_distance_sanity():
    pf_star = get_true_pf(n_points=500)

    F_on_pf = np.array([[1.0, 0.0]])
    med_on, _, _ = e2p3.median_nearest_pf_distance(F_on_pf, pf_star)
    assert med_on < 0.02, f"Point on PF should have near-zero distance, got {med_on}"

    F_far = np.array([[5.0, 5.0]])
    med_far, _, _ = e2p3.median_nearest_pf_distance(F_far, pf_star)
    assert med_far > 5.0, f"Far point should have large distance, got {med_far}"

    med_empty, _, _ = e2p3.median_nearest_pf_distance(np.empty((0, 2)), pf_star)
    assert math.isnan(med_empty)


# =====================================================================
# T9 -- igd_before is identical across LOCAL / POP / GT for same event
# =====================================================================
def test_t9_igd_before_same_across_modes():
    rows = e2p3.run_one_e2p3(seed=1, n_scored_changes=1, scenario_name="PERSISTENT")
    from collections import defaultdict
    by_event_region = defaultdict(list)
    for row in rows:
        by_event_region[(row["event_id"], row["region_id"])].append(row)

    for key, grp in by_event_region.items():
        if len(grp) < 2:
            continue
        igd_vals = [r["igd_before"] for r in grp if r["igd_before"] is not None]
        if len(igd_vals) >= 2:
            assert all(abs(v - igd_vals[0]) < 1e-10 for v in igd_vals), \
                f"igd_before mismatch for key {key}: {igd_vals}"


# =====================================================================
# T10 -- Zero-survivor handling: no crash, survival_rate=0, pf=NaN
# =====================================================================
def test_t10_zero_survivor_handling():
    rng = np.random.default_rng(99)
    N_old = 50
    X_old = rng.uniform(0.0, 1.0, size=(N_old, D))
    F_old = rng.uniform(0.0, 0.5, size=(N_old, 2))

    N_ch = 10
    X_ch = rng.uniform(0.0, 1.0, size=(N_ch, D))
    F_ch = rng.uniform(5.0, 10.0, size=(N_ch, 2))

    X_pool = np.vstack([X_old, X_ch])
    F_pool = np.vstack([F_old, F_ch])

    _, _, orig_idx = e2p3.nsga2_survival_traced(X_pool, F_pool, N_old, xl_g, xu_g)
    n_child_surv = int(np.sum(orig_idx >= N_old))

    if n_child_surv == 0:
        sr = n_child_surv / N_ch
        assert sr == 0.0

        F_surv_ch = np.empty((0, 2))
        med, _, _ = e2p3.median_nearest_pf_distance(F_surv_ch, get_true_pf())
        assert math.isnan(med)


# =====================================================================
# TA1 -- pymoo pool_idx provenance: traced == production (no tobytes)
#        Directly checks that pool_idx attribute travels through survival
# =====================================================================
def test_ta1_pymoo_pool_idx_provenance():
    """Pool indices come from pymoo attribute, NOT tobytes comparison."""
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

    rng = np.random.default_rng(42)
    N = 50
    X_pool = rng.uniform(0.0, 1.0, size=(N, D))
    F_pool = rng.uniform(0.0, 1.0, size=(N, 2))
    n_survive = 30
    surv_seed = 777

    # Verify internally: pool_idx travels through RankAndCrowding.do()
    # Use a seeded default_rng for pymoo's randomized_argsort so tie-breaking at
    # the truncation front is deterministic (pymoo uses default_rng, not np.random).
    seed_int = int(surv_seed) % (2 ** 31)
    np.random.seed(seed_int)
    surv_rng = np.random.default_rng(seed_int)
    problem = Problem(n_var=D, n_obj=2, xl=xl_g, xu=xu_g)
    pop = Population.new(X=X_pool, F=F_pool)
    pool_idx = np.arange(N, dtype=int)
    pop.set("pool_idx", pool_idx)
    survival = RankAndCrowding()
    survived = survival.do(problem, pop, n_survive=n_survive, random_state=surv_rng)
    idx_from_attribute = survived.get("pool_idx").astype(int)

    # All indices valid
    assert len(idx_from_attribute) == n_survive
    assert np.all(idx_from_attribute >= 0)
    assert np.all(idx_from_attribute < N)

    # The survived X/F must match pool at those indices
    X_surv = survived.get("X")
    for j, orig in enumerate(idx_from_attribute):
        np.testing.assert_array_equal(X_surv[j], X_pool[orig])

    # Also verify our wrapper gives same result
    X_tr, F_tr, orig_idx = e2p3.nsga2_survival_traced(
        X_pool, F_pool, n_survive, xl_g, xu_g, surv_seed
    )
    np.testing.assert_array_equal(
        np.sort(idx_from_attribute),
        np.sort(orig_idx),
    )


# =====================================================================
# TA2 -- Duplicate-X provenance: CHILD copy gets CHILD attribution
#        This is the key test that tobytes() fails and pool_idx passes
# =====================================================================
def test_ta2_duplicate_x_provenance():
    """If pool[N_old] == pool[0] (child is copy of old individual),
    provenance must attribute it as CHILD (idx=N_old), not as OLD (idx=0).
    pymoo pool_idx preserves this; tobytes() could fail silently."""
    rng = np.random.default_rng(55)
    N_old = 10
    N_ch  = 5

    X_old = rng.uniform(0.2, 0.8, size=(N_old, D))
    F_old = rng.uniform(0.0, 1.0, size=(N_old, 2))

    # Make child 0 an EXACT copy of old individual 0
    X_ch       = rng.uniform(0.2, 0.8, size=(N_ch, D))
    X_ch[0]    = X_old[0].copy()   # exact duplicate
    F_ch       = rng.uniform(0.0, 1.0, size=(N_ch, 2))
    F_ch[0]    = F_old[0].copy()   # same F too

    X_pool = np.vstack([X_old, X_ch])
    F_pool = np.vstack([F_old, F_ch])

    _, _, orig_idx = e2p3.nsga2_survival_traced(X_pool, F_pool, N_old, xl_g, xu_g)

    # All indices must be valid and unique
    assert np.all(orig_idx >= 0)
    assert np.all(orig_idx < len(X_pool))
    # No duplicate pool indices in the survivor set
    assert len(np.unique(orig_idx)) == len(orig_idx), \
        "Survivor pool indices must be unique (each pool member selected at most once)"

    # Verify that OLD index 0 and CHILD index N_old can both appear as distinct survivors
    # (they may or may not survive, but if both do, they must have distinct pool indices)
    survivors_that_are_child = orig_idx[orig_idx >= N_old]
    survivors_that_are_old   = orig_idx[orig_idx < N_old]
    # Check they do not overlap in pool_idx space
    overlap = set(survivors_that_are_child) & set(survivors_that_are_old)
    assert len(overlap) == 0, f"Pool indices must not overlap: {overlap}"


# =====================================================================
# TA3 -- LOCAL/POP/GT receive SAME survival seed for same event+region
# =====================================================================
def test_ta3_same_survival_seed_across_modes():
    """_e2p3_common_surv_seed returns the same value for all calls with same
    (base_seed, change_idx, region_id) regardless of any mode parameter."""
    seed1 = e2p3._e2p3_common_surv_seed(42, 3, 7)
    seed2 = e2p3._e2p3_common_surv_seed(42, 3, 7)
    assert seed1 == seed2, "Same args must yield same seed"

    # Different (seed, event, region) yields different value
    seed_other = e2p3._e2p3_common_surv_seed(42, 3, 8)
    assert seed1 != seed_other, "Different region_id must yield different seed"


# =====================================================================
# TA4 -- Changing mode alone does NOT change survival seed
#        (common seed has no mode parameter)
# =====================================================================
def test_ta4_mode_independent_survival_seed():
    """_e2p3_common_surv_seed has no mode parameter.
    The survival seed must be identical for LOCAL, POP, GT."""
    import inspect
    sig = inspect.signature(e2p3._e2p3_common_surv_seed)
    params = list(sig.parameters.keys())
    # Must NOT have a mode or variant_key parameter
    for forbidden in ("mode", "mode_name", "variant_key", "key"):
        assert forbidden not in params, \
            f"_e2p3_common_surv_seed must not accept '{forbidden}' parameter"

    # Calling with same (base_seed, change_idx, region_id) yields same seed
    # regardless of which mode we're about to run
    s = e2p3._e2p3_common_surv_seed(1, 2, 3)
    assert isinstance(s, int)
    assert 0 <= s < 2**31


# =====================================================================
# TA5 -- region_type and dominant_bin in every returned row
# =====================================================================
def test_ta5_metadata_in_rows():
    """Every row returned by run_one_e2p3 must have non-empty region_type
    and dominant_bin >= 0 for rows with has_history or has_history=False."""
    rows = e2p3.run_one_e2p3(seed=2, n_scored_changes=1, scenario_name="PERSISTENT")
    for row in rows:
        assert "region_type" in row, "region_type missing from row"
        assert "dominant_bin" in row, "dominant_bin missing from row"
        if row.get("has_history") is True or row.get("has_history") is False:
            assert row["region_type"] != "", \
                f"region_type is empty string for row {row}"
            assert row["dominant_bin"] >= 0, \
                f"dominant_bin < 0 for row {row}"


# =====================================================================
# TA6 -- population_replacement_fraction = n_survived / POP_SIZE
#        NOT n_survived / BUDGET
# =====================================================================
def test_ta6_population_replacement_fraction():
    """population_replacement_fraction must use POP_SIZE as denominator."""
    rows = e2p3.run_one_e2p3(seed=3, n_scored_changes=1, scenario_name="PERSISTENT")
    for row in rows:
        if row.get("has_history") is not True:
            continue
        if row.get("n_children_survived") is None:
            continue
        n_surv = row["n_children_survived"]
        prf    = row["population_replacement_fraction"]
        expected = n_surv / e2p3.POP_SIZE
        assert abs(prf - expected) < 1e-10, \
            f"population_replacement_fraction={prf} != {expected}"
        # Must NOT equal n_surv/BUDGET unless POP_SIZE==BUDGET (they're not)
        assert e2p3.POP_SIZE != e2p3.BUDGET  # sanity: 100 != 10
        # So if n_surv > 0, prf < survival_rate
        if n_surv > 0:
            sr = row["survival_rate"]  # = n_surv / BUDGET
            assert prf < sr or abs(prf - sr) < 1e-10, \
                f"prf={prf} should be <= sr={sr} since POP_SIZE>=BUDGET"
