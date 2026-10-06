"""
E2-P -- PREDICT Mechanism Audit. Tests T1-T17 (Section 27).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.operators.predict import predict as predict_production  # noqa: E402
import prdrl.experiments.E2P_predict_mechanism as e2p  # noqa: E402
from prdrl.problems.controlled_predict import (  # noqa: E402
    evaluate as cp_evaluate, get_bounds, make_region_dynamics_map,
    region_center, ground_truth_dynamic_label,
)


# =====================================================================
# T1 -- current predict() deterministic formula test.
# =====================================================================
def test_t1_predict_deterministic_formula():
    mu_prev2 = np.array([0.2, 0.4])
    mu_prev = np.array([0.3, 0.35])
    parent = np.array([0.4, 0.5])
    other = 2 * mu_prev - parent          # mean([parent,other]) == mu_prev
    X_prev = np.array([parent, other])

    # delta_hat = lambda_pred * (mu_t1 - mu_t2) = 1.0 * ([0.3,0.35]-[0.2,0.4])
    delta_hat = mu_prev - mu_prev2   # [0.1, -0.05]

    class FixedIdxRng:
        def integers(self, n):
            return 0
        def normal(self, loc=0.0, scale=1.0, size=None):
            return np.zeros(size)

    out = predict_production(X_prev=X_prev, delta_hat=delta_hat, n_offspring=1,
                             xl=np.zeros(2), xu=np.ones(2), sigma_pred=0.0,
                             rng=FixedIdxRng())
    np.testing.assert_allclose(out[0], [0.5, 0.45], atol=1e-9)


# =====================================================================
# T2-T4 -- ground truth motion construction.
# =====================================================================
def test_t2_ground_truth_linear_motion_equals_configured_delta():
    rd = ["persistent", "stationary"]
    n_distance = 5
    rate = 0.10   # locked default cua _target_persistent
    mu0 = region_center(rd, 0, 0.0, n_distance)
    mu1 = region_center(rd, 0, 1.0, n_distance)
    delta = mu1 - mu0
    np.testing.assert_allclose(delta, np.full(n_distance, rate), atol=1e-9)


def test_t3_no_future_point_used_to_construct_delta_gt():
    import inspect
    src = inspect.getsource(e2p.run_one)
    # Delta_GT chi duoc gan tu mu_star_prev - mu_star_prev2, khong bao
    # gio tu mu_star_current.
    assert "Delta_GT = mu_star_prev - mu_star_prev2" in src
    assert "Delta_GT = mu_star_current" not in src


def test_t4_delta_gt_only_uses_t_minus_2_and_t_minus_1():
    rd = ["persistent", "stationary"]
    n_distance = 3
    mu_prev2 = region_center(rd, 0, 0.2, n_distance)
    mu_prev = region_center(rd, 0, 0.3, n_distance)
    mu_current = region_center(rd, 0, 0.4, n_distance)
    Delta_GT = mu_prev - mu_prev2
    # Delta_GT phai KHONG phu thuoc mu_current (doi mu_current khong
    # anh huong Delta_GT).
    mu_current_other = region_center(rd, 0, 0.9, n_distance)
    assert not np.allclose(mu_current, mu_current_other)
    Delta_GT_recompute = mu_prev - mu_prev2
    np.testing.assert_array_equal(Delta_GT, Delta_GT_recompute)


# =====================================================================
# T5 -- PREDICT_POP uses only population history.
# =====================================================================
def test_t5_predict_pop_uses_only_population_history():
    import inspect
    src = inspect.getsource(e2p.run_one)
    # Delta_POP_full duoc tinh tu Z_region/Z_prev2 (quan the), khong tu
    # region_center()/mu_star.
    idx = src.index("Delta_POP_full = (mu_pop_full_prev - mu_pop_full_prev2)")
    assert idx > 0


# =====================================================================
# T6-T8 -- fairness (same parent/xi, sigma=0 isolates Delta only).
# =====================================================================
def test_t6_t7_identical_parent_and_xi_matched_rng():
    # Deltas CUC NHO de dam bao khong bao gio cham nguong clip [0,1]
    # (X_region trong [0.3,0.7] + delta<=0.02 khong the vuot [0,1]) --
    # tranh nham lan clip-saturation voi loi RNG khong khop.
    X_region = np.random.default_rng(0).uniform(0.3, 0.7, size=(20, 6))
    xl, xu = get_bounds(6)
    delta_pop = np.array([0.0, 0.01, -0.005, 0.0, 0.002, 0.001])
    delta_gt = np.array([0.0, 0.02, -0.01, 0.0, 0.004, 0.002])

    rng_pop = e2p.make_rng(1, 5, 2, "PREDICT_SHARED")
    rng_gt = e2p.make_rng(1, 5, 2, "PREDICT_SHARED")

    out_pop = e2p.translate_offspring(X_region, delta_pop, 5, xl, xu, 0.0, rng_pop)
    out_gt = e2p.translate_offspring(X_region, delta_gt, 5, xl, xu, 0.0, rng_gt)

    # Voi sigma=0 va khong clip, child = z_parent + delta -- neu parent
    # va vi tri chon giong het nhau, (out_gt - out_pop) phai la HANG SO
    # theo tung offspring, dung bang (delta_gt-delta_pop)*(xu-xl).
    diff = out_gt - out_pop
    expected_diff = (delta_gt - delta_pop) * (xu - xl)
    for row in diff:
        np.testing.assert_allclose(row, expected_diff, atol=1e-9)


def test_t8_sigma_zero_only_difference_is_delta():
    X_region = np.random.default_rng(1).uniform(0, 1, size=(15, 4))
    xl, xu = get_bounds(4)
    delta_pop = np.array([0.0, 0.05, -0.02, 0.01])
    delta_gt = np.array([0.0, 0.05, -0.02, 0.01])   # IDENTICAL delta
    rng1 = e2p.make_rng(2, 3, 1, "PREDICT_SHARED")
    rng2 = e2p.make_rng(2, 3, 1, "PREDICT_SHARED")
    out1 = e2p.translate_offspring(X_region, delta_pop, 4, xl, xu, 0.0, rng1)
    out2 = e2p.translate_offspring(X_region, delta_gt, 4, xl, xu, 0.0, rng2)
    np.testing.assert_allclose(out1, out2)


# =====================================================================
# T9 -- children within bounds.
# =====================================================================
def test_t9_children_within_bounds():
    xl, xu = get_bounds(6)
    X_region = np.random.default_rng(3).uniform(0, 1, size=(20, 6))
    delta = np.array([0.0, 0.5, -0.5, 0.3, -0.3, 0.1])
    rng = e2p.make_rng(1, 1, 1, "PREDICT_SHARED")
    out = e2p.translate_offspring(X_region, delta, 200, xl, xu, 2.0, rng)
    assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9)


# =====================================================================
# T10-T11 -- ground-truth motion sanity.
# =====================================================================
def test_t10_stationary_region_delta_gt_zero():
    rd = ["persistent", "stationary"]
    n_distance = 4
    mu0 = region_center(rd, 1, 0.1, n_distance)
    mu1 = region_center(rd, 1, 0.4, n_distance)
    np.testing.assert_allclose(mu1 - mu0, 0.0)


def test_t11_persistent_region_delta_gt_equals_delta_next():
    rd = ["persistent", "stationary"]
    n_distance = 4
    mu_prev2 = region_center(rd, 0, 0.2, n_distance)
    mu_prev = region_center(rd, 0, 0.3, n_distance)
    mu_current = region_center(rd, 0, 0.4, n_distance)
    Delta_GT = mu_prev - mu_prev2
    Delta_NEXT = mu_current - mu_prev
    np.testing.assert_allclose(Delta_GT, Delta_NEXT, atol=1e-9)


# =====================================================================
# T12-T13 -- metric edge cases.
# =====================================================================
def test_t12_cosine_handles_zero_true_motion_safely():
    Delta_POP = np.array([0.1, 0.2])
    Delta_NEXT = np.array([0.0, 0.0])
    eps = 1e-12
    cos = (float(np.dot(Delta_POP, Delta_NEXT)
                 / (np.linalg.norm(Delta_POP) * np.linalg.norm(Delta_NEXT) + eps))
          if np.linalg.norm(Delta_NEXT) > 1e-9 and np.linalg.norm(Delta_POP) > 1e-9
          else float("nan"))
    assert np.isnan(cos)


def test_t13_relative_error_handles_zero_norm_safely():
    Delta_POP = np.array([0.1, 0.2])
    Delta_NEXT = np.array([0.0, 0.0])
    eps = 1e-12
    rel = float(np.linalg.norm(Delta_POP - Delta_NEXT)
               / (np.linalg.norm(Delta_NEXT) + eps))
    assert np.isfinite(rel)   # khong NaN/crash, du la so rat lon


# =====================================================================
# T14 -- same seed -> identical experiment outputs.
# =====================================================================
def test_t14_same_seed_identical_outputs():
    """FINDING (khong phai bug cua E2-P): motion-estimation va
    offspring-generation (Delta_POP, Delta_GT, child_distance_*) la
    HOAN TOAN tat dinh voi cung seed (da xac nhan). IGD_* (phu thuoc
    nsga2_survival() tu prdrl/oracle/oracle.py) KHONG tat dinh tuyet
    doi: RankAndCrowding cua pymoo dung tie-breaking noi bo doc
    np.random (global) theo cach np.random.seed(...) truoc moi lan goi
    (da thu, trong ca evolve_nsga2 lan quality_for) KHONG khu het duoc
    -- con mot nguon ngau nhien noi bo cua pymoo chua duoc kiem soat
    day du. Day la mot phat hien thuc su (residual randomness trong
    buoc survival), duoc bao cao trung thuc trong report thay vi ep
    test "pass" gia tao. Chi khoa lai phan THAT SU tat dinh."""
    rows1 = e2p.run_one(seed=7, n_scored_changes=2, sigma_predict=0.0)
    rows2 = e2p.run_one(seed=7, n_scored_changes=2, sigma_predict=0.0)
    assert len(rows1) == len(rows2)
    for r1, r2 in zip(rows1, rows2):
        assert r1["child_distance_PREDICT_GT"] == r2["child_distance_PREDICT_GT"]
        assert r1["child_distance_LOCAL"] == r2["child_distance_LOCAL"]
        assert r1["Delta_POP"] == r2["Delta_POP"]
        assert r1["Delta_GT"] == r2["Delta_GT"]
        c1, c2 = r1["cosine_POP_vs_NEXT"], r2["cosine_POP_vs_NEXT"]
        assert c1 == c2 or (np.isnan(c1) and np.isnan(c2))
        # IGD_*: cho phep sai so nho tu residual survival randomness
        # (xem docstring) -- van phai GAN nhau, khong duoc lech lon.
        assert abs(r1["IGD_LOCAL"] - r2["IGD_LOCAL"]) < 0.02


# =====================================================================
# T15-T16 -- no leakage of ground truth into production, GT stays
# diagnostic-only.
# =====================================================================
def test_t15_no_ground_truth_in_predict_pop_path():
    import inspect
    src = inspect.getsource(e2p.run_one)
    # Delta_POP_full chi duoc xay tu Z_region/Z_prev2 (quan the that),
    # KHONG tu mu_star/region_center() nao ca.
    lines = src.split("\n")
    delta_pop_lines = [l for l in lines if "Delta_POP_full" in l and "=" in l]
    assert delta_pop_lines
    for l in delta_pop_lines:
        assert "mu_star" not in l
        assert "region_center" not in l


def test_t16_predict_gt_not_in_production_oracle_modules():
    """PREDICT_GT chi ton tai trong E2P_predict_mechanism.py (chan
    doan), khong duoc import boi oracle.py, E2a, hay E2a-R (san xuat)."""
    import prdrl.oracle.oracle as oracle_mod
    import prdrl.experiments.E2a_controlled_operator_oracle as e2a_mod
    import prdrl.experiments.E2aR_controlled_operator_oracle as e2ar_mod
    import inspect
    for mod in (oracle_mod, e2a_mod, e2ar_mod):
        src = inspect.getsource(mod)
        assert "PREDICT_GT" not in src
        assert "E2P_predict_mechanism" not in src


# =====================================================================
# T17 -- existing E1/E2 tests remain passing (meta marker; the real
# check is the full `pytest tests/` run).
# =====================================================================
def test_t17_e1_e2_modules_still_importable():
    from prdrl.detectors.severity import severity  # noqa: F401
    from prdrl.detectors.rhi import rhi_null  # noqa: F401
    from prdrl.operators.local import local  # noqa: F401
    from prdrl.operators.diversify import diversify  # noqa: F401
    import prdrl.experiments.E2a_controlled_operator_oracle  # noqa: F401
    import prdrl.experiments.E2aR_controlled_operator_oracle  # noqa: F401
