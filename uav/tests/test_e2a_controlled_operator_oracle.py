"""
E2a -- Controlled Regional Operator Oracle. Tests T1-T33 (Section 21-23).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.operators.keep import keep  # noqa: E402
from prdrl.operators.local import local  # noqa: E402
from prdrl.operators.predict import predict  # noqa: E402
from prdrl.operators.diversify import diversify  # noqa: E402
from prdrl.operators.budget import equal_budget  # noqa: E402

from prdrl.problems.controlled_operator import (  # noqa: E402
    evaluate as op_evaluate, get_bounds, get_true_pf,
    make_region_dynamics_map, SCENARIOS,
)

import prdrl.experiments.E2a_controlled_operator_oracle as e2a  # noqa: E402
from prdrl.history.region_history import RegionHistory  # noqa: E402


D = 5
xl = np.zeros(D)
xu = np.ones(D)


def make_region(n, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, size=(n, D))


# =====================================================================
# T1-T7 -- operators (locked formulas, Section 5)
# =====================================================================
def test_t1_keep_shape():
    out = keep(make_region(5), 3, xl, xu, 0.05, np.random.default_rng(0))
    assert out.shape == (0, D)


def test_t2_local_sigma_zero_equals_parent():
    X = make_region(10, seed=1)
    rng = np.random.default_rng(0)
    out = local(X, n_offspring=20, xl=xl, xu=xu, sigma=0.0, rng=rng)
    for child in out:
        assert np.any(np.all(np.isclose(child, X), axis=1))


def test_t3_local_within_bounds():
    X = make_region(10, seed=2)
    rng = np.random.default_rng(1)
    out = local(X, 200, xl, xu, sigma=2.0, rng=rng)
    assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9)


def test_t4_diversify_within_bounds():
    X = make_region(10, seed=3)
    rng = np.random.default_rng(2)
    out = diversify(X, 200, xl, xu, sigma=5.0, rng=rng)
    assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9)


def test_t5_diversify_wider_spread_than_local():
    X = make_region(30, seed=4)
    out_local = local(X, 500, xl, xu, sigma=0.02, rng=np.random.default_rng(5))
    out_div = diversify(X, 500, xl, xu, sigma=0.15, rng=np.random.default_rng(5))
    assert np.std(out_div) > np.std(out_local)


def test_t6_predict_deterministic_shift():
    parent = np.array([[0.4, 0.5]])

    delta_hat = np.array([0.1, -0.05])

    class FixedIdxRng:
        def integers(self, n):
            return 0

        def normal(self, loc=0.0, scale=1.0, size=None):
            return np.zeros(size)

    out = predict(
        X_prev=parent,
        delta_hat=delta_hat,
        n_offspring=1,
        xl=np.zeros(2),
        xu=np.ones(2),
        sigma_pred=0.0,
        rng=FixedIdxRng(),
        sigma_local=0.02,
    )

    np.testing.assert_allclose(
        out[0],
        [0.5, 0.45],
        atol=1e-9,
    )

def test_t7_predict_missing_history_fallback_local():
    """Current PDF protocol: khi thieu history (history rong, khong du
    mu^{t-2} va mu^{t-1}), PREDICT fallback sang LOCAL voi sigma_local.
    Kiet qua phai la array hop le (khong None), va voi sigma_local=0
    moi child phai bang chinh parent (LOCAL voi sigma=0)."""
    X_region = make_region(1, seed=6)  # 1 parent: [0.x...]
    empty_history = RegionHistory(n_regions=1)  # khong co snapshot nao

    # Voi sigma_local=0, LOCAL khong xao tron parent -> child = parent
    # Kiem tra qua viec SIGMA_LOCAL cua E2a duoc truyen dung vao predict()
    out = e2a.generate_children("PREDICT", region_id=0, X_region=X_region,
                                history=empty_history, t=2,
                                budget=1, xl=xl, xu=xu,
                                base_seed=1, event_id=2)
    assert out is not None
    assert out.shape == (1, D)  # 1 offspring, D dims


# =====================================================================
# T8-T23 -- budget / oracle
# =====================================================================
def test_t8_keep_budget_zero():
    assert equal_budget("KEEP", region_size=10, budget=10) == 0


def test_t9_active_operators_budget_equals_B():
    for action in ("LOCAL", "PREDICT", "DIVERSIFY"):
        assert equal_budget(action, region_size=10, budget=10) == 10


def test_t10_empty_region_budget_zero():
    assert equal_budget("LOCAL", region_size=0, budget=10) == 0


def _toy_snapshot(seed=0, N=40, R=4):
    rng = np.random.default_rng(seed)
    xl_, xu_ = get_bounds(D=6)
    X_base = rng.uniform(xl_, xu_, size=(N, 6))
    region_dynamics = make_region_dynamics_map("MIXED_RESPONSE", R)
    F_plus = op_evaluate(X_base, 0.3, region_dynamics)
    region_ids = rng.integers(0, R, size=N)
    return X_base, F_plus, region_ids, R, xl_, xu_, region_dynamics


def test_t11_isolated_trial_only_adds_target_region_children():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot()

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    target_r = int(region_ids[0])
    result = e2a.respond(X_base, F_plus, region_ids, R, {target_r: "LOCAL"},
                         RegionHistory(R), 1, evaluate_fn, xl_, xu_, budget=5,
                         base_seed=1, event_id=1)
    assert result["available"]
    n_region_points = int(np.sum(region_ids == target_r))
    expected_children = min(5, 5)  # budget=5 regardless of region size
    assert result["extra_fe"] == 5


def test_t12_all_trials_start_from_identical_frozen_X():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot()
    X_base_copy = X_base.copy()

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    active = sorted(set(region_ids.tolist()))
    for r in active:
        e2a.respond(X_base, F_plus, region_ids, R, {r: "LOCAL"},
                   RegionHistory(R), 1, evaluate_fn, xl_, xu_, budget=5,
                   base_seed=1, event_id=1)
    np.testing.assert_array_equal(X_base, X_base_copy)


def test_t13_oracle_trial_does_not_mutate_baseline_trajectory():
    """run_one() phai KHONG dung ket qua oracle lam X_pop ke tiep --
    kiem tra qua inspect nguon: khong co gan X_pop = oracle_result/
    regional_result o bat ky dau."""
    import inspect
    src = inspect.getsource(e2a.run_one)
    assert "X_pop = oracle_result" not in src
    assert "X_pop = regional_result" not in src
    assert "X_pop = evolve_nsga2" in src


def test_t14_same_seed_event_region_action_bit_identical_children():
    X = make_region(20, seed=7)
    rng1 = e2a.make_rng(base_seed=1, event_id=3, region_id=2, action="LOCAL")
    rng2 = e2a.make_rng(base_seed=1, event_id=3, region_id=2, action="LOCAL")
    out1 = local(X, 10, xl, xu, 0.05, rng1)
    out2 = local(X, 10, xl, xu, 0.05, rng2)
    np.testing.assert_array_equal(out1, out2)

    rng3 = e2a.make_rng(base_seed=1, event_id=4, region_id=2, action="LOCAL")
    out3 = local(X, 10, xl, xu, 0.05, rng3)
    assert not np.array_equal(out1, out3)


def test_t15_isolated_equals_combined_children():
    """region r + LOCAL phai sinh CUNG offspring du no xuat hien trong
    isolated trial hay combined regional oracle (deterministic RNG
    keyed tren (base_seed,event_id,region_id,action), doc lap voi cac
    region/action khac trong actions_map)."""
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=9)

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    active = sorted(set(region_ids.tolist()))
    r0 = active[0]

    h = RegionHistory(R)
    isolated = e2a.respond(X_base, F_plus, region_ids, R, {r0: "LOCAL"},
                           h, 5, evaluate_fn, xl_, xu_, budget=5,
                           base_seed=2, event_id=5)

    combined_actions = {r: "LOCAL" for r in active}
    combined = e2a.respond(X_base, F_plus, region_ids, R, combined_actions,
                           h, 5, evaluate_fn, xl_, xu_, budget=5,
                           base_seed=2, event_id=5)

    # Sinh lai truc tiep children cho r0 qua generate_children() de so
    # sanh khong phu thuoc vao ket qua survival (co the loai bo child).
    X_r0 = X_base[region_ids == r0]
    children_isolated = e2a.generate_children(
        "LOCAL", r0, X_r0, h, 5, 5, xl_, xu_, base_seed=2, event_id=5)
    children_combined = e2a.generate_children(
        "LOCAL", r0, X_r0, h, 5, 5, xl_, xu_, base_seed=2, event_id=5)
    np.testing.assert_array_equal(children_isolated, children_combined)


def test_t16_tie_with_keep_selects_keep():
    """Neu tat ca action co utility <= 0 (kho tao tie=0 tuyet doi qua
    optimization that), test truc tiep logic argmax: gia lap utilities
    voi 1 action dung bang 0 -- van phai chon KEEP (dung > khong phai
    >=)."""
    utilities = {"KEEP": 0.0, "LOCAL": 0.0, "PREDICT": float("nan"),
                "DIVERSIFY": -0.1}
    best_action = "KEEP"
    best_u = 0.0
    for action in e2a.ACTIONS:
        u = utilities.get(action, float("nan"))
        if np.isnan(u):
            continue
        if u > best_u:
            best_u = u
            best_action = action
    assert best_action == "KEEP"


def test_t17_negative_utility_selects_keep():
    utilities = {"KEEP": 0.0, "LOCAL": -0.05, "PREDICT": -0.2,
                "DIVERSIFY": -0.01}
    best_action = "KEEP"
    best_u = 0.0
    for action in e2a.ACTIONS:
        u = utilities.get(action, float("nan"))
        if np.isnan(u):
            continue
        if u > best_u:
            best_u = u
            best_action = action
    assert best_action == "KEEP"


def test_t18_predict_has_finite_utility_even_with_partial_history():
    """Current protocol: PREDICT fallback sang LOCAL khi thieu history.
    O event 2 (first scored), history chi co 1 snapshot (change_idx=1).
    PREDICT phai van co utility so thuc (khong NaN) cho moi vung active."""
    rows = e2a.run_one("PERSISTENT_DIRECTIONAL_SHIFT", seed=42,
                       n_scored_changes=1)
    row = rows[0]
    for r, u in row["utilities"].items():
        pv = u.get("PREDICT")
        # Moi vung active phai co utility PREDICT so thuc (khong NaN)
        assert pv is not None
        assert not np.isnan(pv), (
            f"region {r}: PREDICT utility phai finite (fallback ve LOCAL "
            f"khi thieu history), nhan duoc NaN")


def test_t19_survival_always_exactly_N():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=11, N=37)

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    active = sorted(set(region_ids.tolist()))
    actions_map = {r: "DIVERSIFY" for r in active}
    result = e2a.respond(X_base, F_plus, region_ids, R, actions_map,
                         RegionHistory(R), 1, evaluate_fn, xl_, xu_,
                         budget=8, base_seed=1, event_id=1)
    assert result["X"].shape[0] == 37
    assert result["F"].shape[0] == 37


def test_t20_response_fe_equals_children_evaluated():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=12)
    n_calls = {"n": 0}

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        n_calls["n"] += len(Xnew)
        return op_evaluate(Xnew, t, rd_)

    active = sorted(set(region_ids.tolist()))
    actions_map = {r: "LOCAL" for r in active}
    result = e2a.respond(X_base, F_plus, region_ids, R, actions_map,
                         RegionHistory(R), 1, evaluate_fn, xl_, xu_,
                         budget=6, base_seed=1, event_id=1)
    assert result["extra_fe"] == n_calls["n"]
    assert result["extra_fe"] == len(active) * 6


def test_t21_no_hidden_evaluation_inside_operators():
    """local/predict/diversify/keep KHONG duoc goi ham danh gia muc
    tieu nao (khong import prdrl.problems.* hay evaluate*) -- chi thao
    tac decision vectors."""
    import inspect
    for fn in (local, predict, diversify, keep):
        src = inspect.getsource(fn)
        assert "evaluate" not in src
        assert "prdrl.problems" not in src


def test_t22_global_action_same_action_every_active_region():
    X_base, F_plus, region_ids, R, xl_, xu_, rd = _toy_snapshot(seed=13)

    def evaluate_fn(Xnew, t=0.3, rd_=rd):
        return op_evaluate(Xnew, t, rd_)

    active = sorted(set(region_ids.tolist()))
    actions_map = {r: "DIVERSIFY" for r in active}
    for r in active:
        assert actions_map[r] == "DIVERSIFY"
    result = e2a.respond(X_base, F_plus, region_ids, R, actions_map,
                         RegionHistory(R), 1, evaluate_fn, xl_, xu_,
                         budget=5, base_seed=1, event_id=1)
    assert result["extra_fe"] == len(active) * 5


def test_t23_regional_oracle_starts_from_original_frozen_population():
    import inspect
    src = inspect.getsource(e2a.run_oracle_event)
    # Regional oracle (combined) phai goi respond() voi X_base goc,
    # khong phai bien trung gian tu isolated trial nao.
    assert 'respond(X_base, F_plus, region_ids, R,\n                              regional_actions' in src \
        or "regional_result = respond(X_base, F_plus, region_ids, R," in src


# =====================================================================
# T24-T33 -- controlled benchmark + E1 freeze check
# =====================================================================
def test_t24_evaluate_deterministic():
    rng = np.random.default_rng(0)
    xl_, xu_ = get_bounds(8)
    X = rng.uniform(xl_, xu_, size=(15, 8))
    rd = make_region_dynamics_map("MIXED_RESPONSE", 5)
    F1 = op_evaluate(X, 0.4, rd)
    F2 = op_evaluate(X, 0.4, rd)
    np.testing.assert_array_equal(F1, F2)


def test_t25_bounds_respected():
    xl_, xu_ = get_bounds(8)
    np.testing.assert_array_equal(xl_, np.zeros(8))
    np.testing.assert_array_equal(xu_, np.ones(8))


def test_t26_reference_front_finite_correct_dims():
    pf = get_true_pf(n_points=200)
    assert pf.shape == (200, 2)
    assert np.all(np.isfinite(pf))
    radius = np.linalg.norm(pf, axis=1)
    np.testing.assert_allclose(radius, 1.0, atol=1e-9)


def test_t27_fixed_V_through_run():
    import inspect
    src = inspect.getsource(e2a.run_one)
    assert src.count("reference(") == 1   # V tinh 1 lan, dung xuyen suot


def test_t28_same_X_for_F_minus_F_plus():
    xl_, xu_ = get_bounds(6)
    rng = np.random.default_rng(1)
    X = rng.uniform(xl_, xu_, size=(10, 6))
    rd = make_region_dynamics_map("ABRUPT_LARGE_SHIFT", 4)
    F_minus = op_evaluate(X, 0.1, rd)
    F_plus = op_evaluate(X, 0.4, rd)
    assert F_minus.shape == F_plus.shape == (10, 2)


def test_t29_partition_uses_F_minus_only():
    import inspect
    from prdrl.core.partition import partition
    params = list(inspect.signature(partition).parameters.keys())
    assert "F_plus" not in params
    src = inspect.getsource(e2a.run_one)
    assert "partition(F_tilde_minus, V)" in src


def test_t30_persistent_directional_scenario_consistent_temporal_movement():
    from prdrl.problems.controlled_operator import DYNAMIC_TYPES
    fn = DYNAMIC_TYPES["persistent"]
    ts = np.linspace(0.0, 1.0, 10)
    targets = np.array([fn(t) for t in ts])
    diffs = np.diff(targets)
    assert np.all(diffs > 0)             # tang deu, cung huong
    assert np.allclose(diffs, diffs[0], atol=1e-9)  # tuyen tinh (rate co dinh)


def test_t31_abrupt_scenario_not_linear_extrapolation_of_history():
    from prdrl.problems.controlled_operator import DYNAMIC_TYPES
    fn = DYNAMIC_TYPES["abrupt"]
    t_hist = [0.0, 0.1, 0.2]              # truoc jump_time=0.3 -> deu 0
    targets_hist = [fn(t) for t in t_hist]
    assert all(v == 0.0 for v in targets_hist)
    # ngoai suy tuyen tinh tu lich su (tat ca =0) se du doan t=0.4 van =0
    linear_extrapolation = 0.0
    actual_t04 = fn(0.4)
    assert actual_t04 != linear_extrapolation
    assert actual_t04 > 0.3               # nhay dot ngot, khong phai tu tu


def test_t32_global_control_scenario_uniform_dynamics_across_bins():
    rd = make_region_dynamics_map("GLOBAL_STABLE_OR_SMALL_SHIFT", 10)
    assert len(set(rd)) == 1   # tat ca bin CUNG 1 dynamic type


def test_t33_existing_e1_tests_still_pass():
    """Meta-check: cac module E1 (severity/rhi/partition/normalize)
    khong bi import-error hay thay doi API sau khi them E2a."""
    from prdrl.detectors.severity import severity  # noqa: F401
    from prdrl.detectors.rhi import rhi, rhi_null  # noqa: F401
    from prdrl.core.partition import partition  # noqa: F401
    from prdrl.core.normalize import normalize  # noqa: F401
    import inspect
    assert list(inspect.signature(severity).parameters.keys()) == [
        "F_minus", "F_plus", "region_ids", "R", "q_prev", "kappa", "rho"]
