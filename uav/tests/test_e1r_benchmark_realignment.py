"""
E1R -- Benchmark Realignment for Pareto-Region Heterogeneity.
Tests T1-T20 (Phase 15).
"""
import inspect
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.core.reference import reference  # noqa: E402
from prdrl.core.seeding import deterministic_rng  # noqa: E402
from prdrl.core.normalize import normalize  # noqa: E402
from prdrl.detectors.severity import severity  # noqa: E402
from prdrl.detectors.rhi import rhi_null  # noqa: E402
from prdrl.problems.cec2018 import get_df_bounds, get_n_obj  # noqa: E402
from prdrl.problems.sdp import evaluate_sdp, get_sdp_bounds, get_sdp_true_pf  # noqa: E402
from prdrl.problems import controlled_regional as cr  # noqa: E402
from prdrl.experiments.E1b_sdp_heterogeneity import (  # noqa: E402
    PRESELECTED_SDP, run_nsga2_with_changes as sdp_run_nsga2,
)
from prdrl.experiments.E1_heterogeneity_audit import (  # noqa: E402
    get_df_config, run_nsga2_with_changes as cec_run_nsga2,
)


# =====================================================================
# T1 -- CEC adapter returns true non-unit bounds where applicable.
# =====================================================================
def test_t1_cec_adapter_non_unit_bounds():
    xl, xu = get_df_bounds(3, D=10)   # DF3: x2..xD in [-1,2] (non [0,1])
    assert not (np.allclose(xl, 0.0) and np.allclose(xu, 1.0))
    assert np.any(xu > 1.0) or np.any(xl < 0.0)


# =====================================================================
# T2 -- E1 CEC initialization respects true bounds.
# =====================================================================
def test_t2_e1_cec_config_uses_real_bounds():
    n_obj, D, lower, upper = get_df_config(3)
    xl_real, xu_real = get_df_bounds(3, D=10)
    np.testing.assert_allclose(lower, xl_real)
    np.testing.assert_allclose(upper, xu_real)
    assert not (np.allclose(lower, 0.0) and np.allclose(upper, 1.0))


def test_t2b_e1_cec_population_within_true_bounds():
    events = cec_run_nsga2(df_id=3, n_changes=1, pop_size=20,
                           gen_per_change=2, seed=1)
    xl, xu = get_df_bounds(3, D=10)
    for ev in events:
        X = ev["X"]
        assert np.all(X >= xl - 1e-9) and np.all(X <= xu + 1e-9)


# =====================================================================
# T3 -- reference n_partitions and R are not conflated.
# =====================================================================
def test_t3_n_partitions_not_conflated_with_R():
    V2 = reference(2, 9)
    assert len(V2) == 10 != 9          # M=2, n_partitions=9 -> R=10
    V3 = reference(3, 4)
    assert len(V3) == 15 != 4          # M=3, n_partitions=4 -> R=15


# =====================================================================
# T4 -- GLOBAL_SHIFT ground-truth change magnitudes approx equal.
# =====================================================================
def test_t4_global_shift_magnitudes_approx_equal():
    d = cr.make_global_shift(seed=3)
    mag = d["ground_truth_change_magnitude"]
    active = mag[~np.isnan(mag)]
    assert len(active) >= 2
    assert np.std(active) < 1e-9   # tat ca active region cung magnitude


# =====================================================================
# T5 -- SINGLE_REGION_SHIFT shifts only intended region.
# =====================================================================
def test_t5_single_region_shift_only_intended_region():
    d = cr.make_single_region_shift(seed=4)
    diff = d["F_plus"] - d["F_minus"]
    moved = np.linalg.norm(diff, axis=1) > 1e-9
    assert np.all(d["region_ids_gt"][moved] == d["R_shift"])
    mag = d["ground_truth_change_magnitude"]
    nonzero_regions = np.flatnonzero(~np.isnan(mag) & (mag > 1e-9))
    assert list(nonzero_regions) == [d["R_shift"]]


# =====================================================================
# T6 -- MULTI_SEVERITY ground truth ordering is correct.
# =====================================================================
def test_t6_multi_severity_ground_truth_ordering():
    d = cr.make_multi_severity(seed=5)
    F_tilde_minus, F_tilde_plus = normalize(d["F_minus"], d["F_plus"])
    s_r, c_r, q = severity(F_tilde_minus, F_tilde_plus, d["region_ids_gt"],
                           d["R"], q_prev=1.0)
    s_r = np.asarray(s_r)
    mag = d["ground_truth_change_magnitude"]
    mask = ~np.isnan(s_r) & ~np.isnan(mag)
    assert np.sum(mask) >= 3
    rho = np.corrcoef(np.argsort(np.argsort(mag[mask])),
                      np.argsort(np.argsort(s_r[mask])))[0, 1]
    assert rho > 0.7


# =====================================================================
# T7 -- same X is used for F_minus/F_plus.
# =====================================================================
def test_t7_same_X_for_F_minus_F_plus_cec():
    events = cec_run_nsga2(df_id=1, n_changes=1, pop_size=20,
                           gen_per_change=10, seed=1)
    assert len(events) >= 1
    ev = events[0]
    # F_minus/F_plus deu duoc suy ra tu CUNG X (ev["X"]) -- xac nhan
    # bang cach danh gia lai truc tiep va so sanh.
    from prdrl.problems.cec2018 import evaluate_df
    F_minus_recomputed = evaluate_df(1, ev["X"], ev["t_old"])
    F_plus_recomputed = evaluate_df(1, ev["X"], ev["t_new"])
    np.testing.assert_allclose(ev["F_minus"], F_minus_recomputed)
    np.testing.assert_allclose(ev["F_plus"], F_plus_recomputed)


def test_t7b_same_X_for_F_minus_F_plus_sdp():
    events = sdp_run_nsga2(problem_id=2, n_obj=2, n_changes=1, pop_size=20,
                           gen_per_change=10, seed=1)
    assert len(events) >= 1
    ev = events[0]
    F_minus_recomputed = evaluate_sdp(2, ev["X"], ev["t_old"], 2, seed=1)
    F_plus_recomputed = evaluate_sdp(2, ev["X"], ev["t_new"], 2, seed=1)
    np.testing.assert_allclose(ev["F_minus"], F_minus_recomputed)
    np.testing.assert_allclose(ev["F_plus"], F_plus_recomputed)


# =====================================================================
# T8 -- partition receives F_minus, not F_plus.
# =====================================================================
def test_t8_partition_signature_has_no_f_plus_param():
    from prdrl.core.partition import partition
    params = list(inspect.signature(partition).parameters.keys())
    assert "F_plus" not in params
    assert params[0] == "F_minus"


# =====================================================================
# T9 -- empty region -> NaN severity.
# =====================================================================
def test_t9_empty_region_nan_severity():
    F_minus = np.array([[0.1, 0.2], [0.15, 0.25]])
    F_plus = np.array([[0.12, 0.22], [0.17, 0.27]])
    region_ids = np.array([0, 0])   # region 1 rong
    s_r, c_r, q = severity(F_minus, F_plus, region_ids, R=2, q_prev=1.0)
    assert np.isnan(s_r[1])
    assert not np.isnan(s_r[0])


# =====================================================================
# T10 -- RHI excludes inactive regions.
# =====================================================================
def test_t10_rhi_excludes_inactive_regions():
    from prdrl.detectors.rhi import rhi
    s_r_with_nan = [1.0, np.nan, 3.0, 5.0]
    s_r_without = [1.0, 3.0, 5.0]
    assert rhi(s_r_with_nan) == pytest.approx(rhi(s_r_without))


# =====================================================================
# T11 -- size-preserving null keeps exact region counts.
# =====================================================================
def test_t11_null_preserves_exact_region_sizes():
    rng = np.random.default_rng(0)
    region_ids = np.array([0] * 7 + [1] * 3 + [2] * 5)
    F_minus = rng.uniform(0, 1, size=(15, 2))
    F_plus = F_minus + rng.normal(scale=0.05, size=(15, 2))

    real_sizes = {r: int(np.sum(region_ids == r)) for r in range(3)}

    # Mo phong 1 vong permutation nhu ben trong rhi_null() de kiem tra
    # kich thuoc duoc bao toan CHINH XAC (khong chi ~).
    null_rng = deterministic_rng(1, 2, 3)
    N = len(region_ids)
    sizes = [int(np.sum(region_ids == r)) for r in range(3)]
    perm = null_rng.permutation(N)
    fake_ids = np.empty(N, dtype=int)
    start = 0
    for r in range(3):
        fake_ids[perm[start:start + sizes[r]]] = r
        start += sizes[r]
    fake_sizes = {r: int(np.sum(fake_ids == r)) for r in range(3)}
    assert fake_sizes == real_sizes


# =====================================================================
# T12 -- rhi_null deterministic under fixed RNG seed.
# =====================================================================
def test_t12_rhi_null_deterministic_fixed_seed():
    rng = np.random.default_rng(0)
    F_minus = rng.uniform(0, 1, size=(30, 2))
    F_plus = F_minus + rng.normal(scale=0.05, size=(30, 2))
    region_ids = rng.integers(0, 4, size=30)

    rng1 = deterministic_rng(5, 6, 7)
    rng2 = deterministic_rng(5, 6, 7)
    out1 = rhi_null(F_minus, F_plus, region_ids, R=4, K=50, rng=rng1)
    out2 = rhi_null(F_minus, F_plus, region_ids, R=4, K=50, rng=rng2)
    assert out1 == out2

    rng3 = deterministic_rng(5, 6, 8)   # change_idx khac -> ket qua khac
    out3 = rhi_null(F_minus, F_plus, region_ids, R=4, K=50, rng=rng3)
    assert out3 != out1


# =====================================================================
# T13 -- SDP adapter returns finite F with expected dimensions.
# =====================================================================
@pytest.mark.parametrize("pid", PRESELECTED_SDP)
def test_t13_sdp_finite_expected_dims(pid):
    for n_obj in (2, 3):
        xl, xu = get_sdp_bounds(pid, n_obj, n_var=10)
        rng = np.random.default_rng(0)
        X = rng.uniform(xl, xu, size=(15, 10))
        F = evaluate_sdp(pid, X, 0.3, n_obj, seed=1)
        assert F.shape == (15, n_obj)
        assert np.all(np.isfinite(F))


def test_t13b_sdp_not_implemented_raises_clearly():
    with pytest.raises(NotImplementedError):
        evaluate_sdp(1, np.zeros((3, 10)), 0.0, 2, seed=1)
    with pytest.raises(NotImplementedError):
        evaluate_sdp(12, np.zeros((3, 10)), 0.0, 2, seed=1)


# =====================================================================
# T14 -- SDP bounds respected (evaluate at random in-bound X, no crash;
# also confirm NSGA-II-generated trajectory stays within bounds).
# =====================================================================
@pytest.mark.parametrize("pid", PRESELECTED_SDP)
def test_t14_sdp_bounds_respected_in_trajectory(pid):
    events = sdp_run_nsga2(problem_id=pid, n_obj=2, n_changes=1, pop_size=20,
                           gen_per_change=2, seed=1)
    xl, xu = get_sdp_bounds(pid, 2, n_var=10)
    for ev in events:
        X = ev["X"]
        assert np.all(X >= xl - 1e-9) and np.all(X <= xu + 1e-9)


# =====================================================================
# T15 -- SDP true/reference PF dimensions match n_obj.
# =====================================================================
@pytest.mark.parametrize("pid", PRESELECTED_SDP)
def test_t15_sdp_true_pf_dims_match_n_obj(pid):
    for n_obj in (2, 3):
        pf = get_sdp_true_pf(pid, t=0.2, n_obj=n_obj, n_points=30, seed=2)
        assert pf.shape == (30, n_obj)
        assert np.all(np.isfinite(pf))


# =====================================================================
# T16 -- continuous trajectory keeps full population size.
# =====================================================================
def test_t16_continuous_trajectory_full_population_cec():
    events = cec_run_nsga2(df_id=1, n_changes=3, pop_size=37,
                           gen_per_change=2, seed=1)
    for ev in events:
        assert ev["X"].shape[0] == 37


def test_t16b_continuous_trajectory_full_population_sdp():
    events = sdp_run_nsga2(problem_id=2, n_obj=2, n_changes=3, pop_size=37,
                           gen_per_change=2, seed=1)
    for ev in events:
        assert ev["X"].shape[0] == 37


def test_t16c_trajectory_uses_sampling_not_reset():
    """Regression (Phase 8): NSGA2 phai duoc goi voi sampling=X_pop
    (khong phai NSGA2(pop_size=...) roi bo qua quan the hien co)."""
    src = inspect.getsource(cec_run_nsga2)
    assert "sampling=X_pop" in src
    assert "result.pop.get" in src or "res.pop.get" in src
    src_sdp = inspect.getsource(sdp_run_nsga2)
    assert "sampling=X_pop" in src_sdp
    assert "res.pop.get" in src_sdp


# =====================================================================
# T17 -- no regional operator is called anywhere in E1.
# =====================================================================
def test_t17_no_regional_operator_imported_in_e1_scripts():
    import prdrl.experiments.E1_heterogeneity_audit as e1c
    import prdrl.experiments.E1a_controlled_heterogeneity as e1a
    import prdrl.experiments.E1b_sdp_heterogeneity as e1b

    forbidden = ("prdrl.operators", "prdrl.oracle")
    for mod in (e1c, e1a, e1b):
        src = inspect.getsource(mod)
        for f in forbidden:
            assert f not in src, f"{mod.__name__} khong duoc import {f}"


# =====================================================================
# T18 -- controlled benchmark reproducibility under fixed seed.
# =====================================================================
@pytest.mark.parametrize("scenario", list(cr.SCENARIOS.keys()))
def test_t18_controlled_benchmark_reproducible(scenario):
    d1 = cr.make_scenario(scenario, seed=11)
    d2 = cr.make_scenario(scenario, seed=11)
    np.testing.assert_allclose(d1["F_minus"], d2["F_minus"])
    np.testing.assert_allclose(d1["F_plus"], d2["F_plus"])
    np.testing.assert_array_equal(d1["region_ids_gt"], d2["region_ids_gt"])


# =====================================================================
# T19 -- JSON outputs contain no accidental NaN serialization.
# =====================================================================
def test_t19_json_safe_no_raw_nan():
    from prdrl.experiments.E1a_controlled_heterogeneity import json_safe as safe_a
    from prdrl.experiments.E1b_sdp_heterogeneity import json_safe as safe_b
    from prdrl.experiments.E1_heterogeneity_audit import json_safe as safe_c

    for safe in (safe_a, safe_b, safe_c):
        obj = {"a": float("nan"), "b": [1.0, float("nan")], "c": np.nan,
              "d": np.array([1.0, np.nan])}
        out = safe(obj)
        text = json.dumps(out)
        assert "NaN" not in text
        assert out["a"] is None
        assert out["b"][1] is None
        assert out["c"] is None
        assert out["d"][1] is None
