"""
Review 4 operator (KEEP/LOCAL/PREDICT/DIVERSIFY) -- eq.43-47.
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
from prdrl.features.forecast_state import compute_current_predict_motion  # noqa: E402


D = 4
xl = np.zeros(D)
xu = np.ones(D)


def make_region(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n, D))


# =====================================================================
# KEEP -- eq.43
# =====================================================================
def test_keep_returns_zero_offspring_correct_ncols():
    X = make_region(5)
    rng = np.random.default_rng(0)
    out = keep(X, n_offspring=3, xl=xl, xu=xu, sigma=0.05, rng=rng)
    assert out.shape == (0, D)


def test_keep_empty_region_still_correct_shape():
    X = np.empty((0, D))
    rng = np.random.default_rng(0)
    out = keep(X, n_offspring=3, xl=xl, xu=xu, sigma=0.05, rng=rng)
    assert out.shape == (0, D)


def test_keep_signature_matches_other_operators():
    """Dong bo chu ky voi local()/predict()/diversify() de dispatch da
    hinh khong can if/else rieng cho KEEP."""
    import inspect
    params = list(inspect.signature(keep).parameters.keys())
    assert params == ["X_region", "n_offspring", "xl", "xu", "sigma", "rng"]


# =====================================================================
# LOCAL -- eq.44
# =====================================================================
def test_local_respects_n_offspring_bug1_regression():
    """(BUG-1) truoc fix: n_offspring bi bo qua, luon tra ve 1 offspring
    dang 1D. Regression khoa lai fix."""
    X = make_region(10)
    rng = np.random.default_rng(1)
    out = local(X, n_offspring=7, xl=xl, xu=xu, sigma=0.05, rng=rng)
    assert out.shape == (7, D)


def test_local_n_offspring_1_still_2d():
    X = make_region(10)
    rng = np.random.default_rng(2)
    out = local(X, n_offspring=1, xl=xl, xu=xu, sigma=0.05, rng=rng)
    assert out.shape == (1, D)


def test_local_stays_within_bounds():
    X = make_region(10)
    rng = np.random.default_rng(3)
    out = local(X, n_offspring=50, xl=xl, xu=xu, sigma=2.0, rng=rng)  # sigma lon
    assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9)


def test_local_empty_region_raises():
    X = np.empty((0, D))
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        local(X, n_offspring=5, xl=xl, xu=xu, sigma=0.05, rng=rng)


def test_local_single_solution_region():
    X = make_region(1)
    rng = np.random.default_rng(5)
    out = local(X, n_offspring=3, xl=xl, xu=xu, sigma=0.05, rng=rng)
    assert out.shape == (3, D)


def test_local_deterministic_with_seeded_rng():
    X = make_region(10)
    out1 = local(X, 5, xl, xu, 0.05, np.random.default_rng(42))
    out2 = local(X, 5, xl, xu, 0.05, np.random.default_rng(42))
    np.testing.assert_allclose(out1, out2)


# =====================================================================
# PREDICT -- eq.45-46
# =====================================================================
def test_predict_normal_case_shape():
    X_prev = make_region(10, seed=10)
    X_prev2 = make_region(10, seed=11)
    rng = np.random.default_rng(6)
    mu_t2 = np.mean((X_prev2 - xl) / (xu - xl), axis=0)
    mu_t1 = np.mean((X_prev - xl) / (xu - xl), axis=0)
    delta_hat = compute_current_predict_motion(mu_t2, mu_t1, lambda_pred=0.5)["delta_hat"]
    out = predict(X_prev, delta_hat=delta_hat, n_offspring=6, xl=xl, xu=xu,
                  sigma_pred=0.05, rng=rng)
    assert out.shape == (6, D)


def test_predict_falls_back_to_local_when_no_history_bug2_regression():
    """(BUG-2) truoc fix: X_prev2 rong -> raise ValueError, se crash
    thuc te o change event dau tien/thu hai cua MOI run. Sua: fallback
    ve LOCAL, khong crash."""
    X_prev = make_region(10, seed=20)
    rng = np.random.default_rng(7)
    out = predict(X_prev, delta_hat=None, n_offspring=4, xl=xl, xu=xu,
                  sigma_pred=0.05, rng=rng, sigma_local=0.05)
    assert out.shape == (4, D)   # khong crash, tra ve dung so offspring


def test_predict_raises_when_current_population_empty():
    """X_prev rong (vung hien tai khong co nghiem) -- khong the perturb
    du la PREDICT hay LOCAL fallback, PHAI raise (khong co gi de lam)."""
    X_prev = np.empty((0, D))
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError):
        predict(X_prev, delta_hat=None, n_offspring=4, xl=xl, xu=xu,
               sigma_pred=0.05, rng=rng, sigma_local=0.05)


def test_predict_stays_within_bounds():
    X_prev = make_region(10, seed=40)
    X_prev2 = make_region(10, seed=41)
    rng = np.random.default_rng(9)
    mu_t2 = np.mean((X_prev2 - xl) / (xu - xl), axis=0)
    mu_t1 = np.mean((X_prev - xl) / (xu - xl), axis=0)
    delta_hat = compute_current_predict_motion(mu_t2, mu_t1, lambda_pred=5.0)["delta_hat"]
    out = predict(X_prev, delta_hat=delta_hat, n_offspring=30, xl=xl, xu=xu,
                  sigma_pred=2.0, rng=rng)  # tham so lon
    assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9)


def test_predict_shift_direction_matches_history_movement():
    """Neu quan the dich chuyen 1 huong ro rang qua 2 moc thoi gian,
    shift phai cung dau voi huong dich chuyen do (sigma=0 de kiem tra
    thuan shift, khong nhieu)."""
    rng_data = np.random.default_rng(50)
    X_prev2 = rng_data.uniform(0.1, 0.2, size=(20, D))   # centroid ~0.15
    X_prev = rng_data.uniform(0.5, 0.6, size=(20, D))    # centroid ~0.55 (da dich +)
    rng = np.random.default_rng(10)
    mu_t2 = np.mean((X_prev2 - xl) / (xu - xl), axis=0)
    mu_t1 = np.mean((X_prev - xl) / (xu - xl), axis=0)
    delta_hat = compute_current_predict_motion(mu_t2, mu_t1, lambda_pred=1.0)["delta_hat"]
    out = predict(X_prev, delta_hat=delta_hat, n_offspring=20, xl=xl, xu=xu,
                  sigma_pred=0.0, rng=rng)
    # child phai lech ve phia +shift so voi parent tuong ung trong X_prev
    assert np.mean(out) > np.mean(X_prev) - 1e-6


# =====================================================================
# DIVERSIFY -- eq.47 (cung cong thuc LOCAL, sigma lon hon)
# =====================================================================
def test_diversify_respects_n_offspring():
    X = make_region(10)
    rng = np.random.default_rng(11)
    out = diversify(X, n_offspring=9, xl=xl, xu=xu, sigma=0.3, rng=rng)
    assert out.shape == (9, D)


def test_diversify_wider_spread_than_local_on_average():
    """sigma_D > sigma_L (do caller truyen) -> DIVERSIFY spread ra xa
    parent hon LOCAL trung binh, cung cong thuc nen chi khac do lon."""
    X = make_region(30, seed=60)
    out_local = local(X, 200, xl, xu, sigma=0.02, rng=np.random.default_rng(12))
    out_div = diversify(X, 200, xl, xu, sigma=0.30, rng=np.random.default_rng(12))
    spread_local = np.std(out_local)
    spread_div = np.std(out_div)
    assert spread_div > spread_local


def test_diversify_stays_within_bounds():
    X = make_region(10)
    rng = np.random.default_rng(13)
    out = diversify(X, n_offspring=50, xl=xl, xu=xu, sigma=5.0, rng=rng)
    assert np.all(out >= xl - 1e-9) and np.all(out <= xu + 1e-9)
