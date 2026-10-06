import numpy as np

from prdrl.features.forecast_state import (
    compute_current_predict_motion,
    compute_extended_motion,
    get_predict_motion_from_history,
)
from prdrl.history.region_history import RegionHistory

def test_current_predict_motion():
    mu_t2 = np.array([0.2, 0.4])
    mu_t1 = np.array([0.3, 0.45])

    result = compute_current_predict_motion(
        mu_t2,
        mu_t1,
        lambda_pred=1.0,
    )

    expected_d = np.array([0.1, 0.05])

    np.testing.assert_allclose(
        result["d_prev"],
        expected_d,
    )

    np.testing.assert_allclose(
        result["delta_hat"],
        expected_d,
    )


def test_current_predict_motion_with_lambda():
    mu_t2 = np.array([0.2, 0.4])
    mu_t1 = np.array([0.3, 0.5])

    result = compute_current_predict_motion(
        mu_t2,
        mu_t1,
        lambda_pred=0.5,
    )

    np.testing.assert_allclose(
        result["d_prev"],
        np.array([0.1, 0.1]),
    )

    np.testing.assert_allclose(
        result["delta_hat"],
        np.array([0.05, 0.05]),
    )


def test_extended_motion_same_direction():
    mu_t2 = np.array([0.1, 0.2])
    mu_t1 = np.array([0.2, 0.2])
    mu_t = np.array([0.3, 0.2])

    result = compute_extended_motion(
        mu_t2,
        mu_t1,
        mu_t,
    )

    np.testing.assert_allclose(
        result["d_prev"],
        np.array([0.1, 0.0]),
    )

    np.testing.assert_allclose(
        result["d_curr"],
        np.array([0.1, 0.0]),
    )

    np.testing.assert_allclose(
        result["speed_prev"],
        0.1,
    )

    np.testing.assert_allclose(
        result["speed_curr"],
        0.1,
    )

    np.testing.assert_allclose(
        result["cosine"],
        1.0,
    )

    np.testing.assert_allclose(
        result["acceleration"],
        np.array([0.0, 0.0]),
        atol=1e-12,
    )


def test_extended_motion_reverse_direction():
    mu_t2 = np.array([0.1, 0.0])
    mu_t1 = np.array([0.2, 0.0])
    mu_t = np.array([0.1, 0.0])

    result = compute_extended_motion(
        mu_t2,
        mu_t1,
        mu_t,
    )

    np.testing.assert_allclose(
        result["cosine"],
        -1.0,
    )


def test_extended_motion_stationary():
    mu_t2 = np.array([0.2, 0.3])
    mu_t1 = np.array([0.2, 0.3])
    mu_t = np.array([0.3, 0.3])

    result = compute_extended_motion(
        mu_t2,
        mu_t1,
        mu_t,
    )

    assert result["speed_prev"] == 0.0
    assert np.isnan(result["cosine"])

#TEST
def test_get_predict_motion_from_history():
    history = RegionHistory(n_regions=1)

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    # t = 0  -> μ^(t-2)
    history.update(
        t=0,
        X=np.array([[0.2, 0.4]]),
        F=np.array([[1.0, 2.0]]),
        region_ids=np.array([0]),
        xl=xl,
        xu=xu,
    )

    # t = 1 -> μ^(t-1)
    history.update(
        t=1,
        X=np.array([[0.3, 0.5]]),
        F=np.array([[1.1, 1.9]]),
        region_ids=np.array([0]),
        xl=xl,
        xu=xu,
    )

    # đang chuẩn bị response cho t = 2
    result = get_predict_motion_from_history(
        history=history,
        region_id=0,
        t=2,
        lambda_pred=1.0,
    )

    np.testing.assert_allclose(
        result["d_prev"],
        np.array([0.1, 0.1]),
    )

    np.testing.assert_allclose(
        result["delta_hat"],
        np.array([0.1, 0.1]),
    )

def test_get_predict_motion_from_history_missing_history():
    history = RegionHistory(n_regions=1)

    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    # Chỉ có t = 1, thiếu t = 0
    history.update(
        t=1,
        X=np.array([[0.3, 0.5]]),
        F=np.array([[1.0, 2.0]]),
        region_ids=np.array([0]),
        xl=xl,
        xu=xu,
    )

    result = get_predict_motion_from_history(
        history=history,
        region_id=0,
        t=2,
        lambda_pred=1.0,
    )

    assert result is None