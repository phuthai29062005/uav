"""
Review budget.py / quality_loss.py / oracle.py / E2_operator_oracle.py -- eq.48
(budget), eq.34-35 (quality_loss), eq.59-63 (oracle).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from prdrl.operators.budget import equal_budget, proportional_budget  # noqa: E402
from prdrl.detectors.quality_loss import quality_loss  # noqa: E402
from prdrl.oracle.oracle import (  # noqa: E402
    regional_oracle,
    split_by_region,
    operator_heterogeneity_index,
)
from prdrl.history.region_history import RegionHistory  # noqa: E402


# =====================================================================
# budget.py -- eq.48
# =====================================================================
def test_equal_budget_keep_is_zero():
    assert equal_budget("KEEP", region_size=10, budget=7) == 0


def test_equal_budget_empty_region_is_zero():
    assert equal_budget("LOCAL", region_size=0, budget=7) == 0


def test_equal_budget_normal_case():
    assert equal_budget("LOCAL", region_size=10, budget=7) == 7
    assert equal_budget("PREDICT", region_size=1, budget=7) == 7


def test_equal_budget_unknown_action_raises():
    with pytest.raises(ValueError):
        equal_budget("BOGUS", region_size=10, budget=7)


def test_proportional_budget_matches_eq48():
    import math
    n_r = 17
    eta = 0.3
    expected = math.ceil(eta * n_r)
    got = proportional_budget(
        "LOCAL", region_size=n_r,
        eta_local=eta, eta_predict=0.1, eta_diversify=0.9,
    )
    assert got == expected


def test_proportional_budget_keep_is_zero():
    got = proportional_budget(
        "KEEP", region_size=17,
        eta_local=0.3, eta_predict=0.1, eta_diversify=0.9,
    )
    assert got == 0


def test_proportional_budget_uses_correct_eta_per_action():
    import math
    n_r = 10
    for action, eta in [("LOCAL", 0.2), ("PREDICT", 0.5), ("DIVERSIFY", 0.9)]:
        kwargs = dict(eta_local=0.2, eta_predict=0.5, eta_diversify=0.9)
        got = proportional_budget(action, region_size=n_r, **kwargs)
        assert got == math.ceil(eta * n_r), f"{action}: expected eta={eta}"


# =====================================================================
# quality_loss.py -- eq.34-35 (PBI-based quality loss)
# regression: ban goc dung bien khong ton tai (Fm/Fp) va tham so sai ten
# (theta vs theta_pbi) -- crash NameError/TypeError tren MOI call co it
# nhat 1 vung non-empty.
# =====================================================================
D_OBJ = 2
V2 = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_quality_loss_does_not_crash_regression():
    """(BUG) ban goc: loop dung Fm/Fp chua khai bao (NameError), va
    dung bien theta chua khai bao (tham so ham la theta_pbi).
    Regression: goi voi region non-empty phai chay duoc, khong crash."""
    F_minus = np.array([[0.1, 0.2], [0.3, 0.1]])
    F_plus = np.array([[0.15, 0.25], [0.35, 0.05]])
    region_ids = np.array([0, 1])
    out = quality_loss(F_minus, F_plus, region_ids, V2, theta=5.0)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_quality_loss_keyword_theta_matches_caller():
    """E2_operator_oracle.py goi quality_loss(..., theta=THETA_PBI) --
    tham so phai ten la `theta`, khong phai `theta_pbi`."""
    import inspect
    params = list(inspect.signature(quality_loss).parameters.keys())
    assert params == ["F_minus", "F_plus", "region_ids", "V", "theta"]


def test_quality_loss_empty_region_is_nan():
    F_minus = np.array([[0.1, 0.2]])
    F_plus = np.array([[0.15, 0.25]])
    region_ids = np.array([0])  # vung 1 khong co nghiem nao
    out = quality_loss(F_minus, F_plus, region_ids, V2, theta=5.0)
    assert out[0] == pytest.approx(0.0) or np.isfinite(out[0])
    assert np.isnan(out[1])


def test_quality_loss_never_negative():
    """loss = max(0, g_plus - g_minus) -- khong bao gio am, du F_plus
    tot hon F_minus (huong ve goc toa do)."""
    rng = np.random.default_rng(1)
    F_minus = rng.uniform(0.5, 1.0, size=(30, 2))
    F_plus = F_minus - 0.3  # dich ve phia goc -> "tot len"
    region_ids = np.array([0 if f[0] >= f[1] else 1 for f in F_minus])
    out = quality_loss(F_minus, F_plus, region_ids, V2, theta=5.0)
    active = out[~np.isnan(out)]
    assert np.all(active >= -1e-9)


def test_quality_loss_zero_when_no_change():
    F = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4]])
    region_ids = np.array([0, 1, 1])
    out = quality_loss(F, F.copy(), region_ids, V2, theta=5.0)
    active = out[~np.isnan(out)]
    np.testing.assert_allclose(active, 0.0, atol=1e-9)


def test_quality_loss_zero_norm_reference_direction_raises():
    F_minus = np.array([[0.1, 0.2]])
    F_plus = np.array([[0.15, 0.25]])
    region_ids = np.array([0])
    V_bad = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        quality_loss(F_minus, F_plus, region_ids, V_bad, theta=5.0)


# =====================================================================
# oracle.py -- eq.59-63
# =====================================================================
D = 3
xl = np.zeros(D)
xu = np.ones(D)
V_oracle = np.array([[1.0, 0.0], [0.0, 1.0]])


def make_toy_oracle_inputs(seed=0, n=20):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, D))
    F_plus = rng.uniform(0, 1, size=(n, 2))
    region_ids = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return rng, X, F_plus, region_ids


def test_oracle_import_path_regression():
    """(BUG) ban goc: `from prdrl.core.budget import equal_budget` --
    budget.py thuc te nam o prdrl/operators/budget.py, khong phai
    prdrl/core/. Import sai -> ModuleNotFoundError ngay khi import
    oracle.py. Regression: module phai import duoc."""
    import importlib
    import prdrl.oracle.oracle as oracle_mod
    importlib.reload(oracle_mod)
    assert hasattr(oracle_mod, "regional_oracle")


def test_regional_oracle_runs_without_history():
    """history rong (change event dau tien cua run, chua co lich su nao ca)."""
    rng, X, F_plus, region_ids = make_toy_oracle_inputs()

    def evaluate_fn(Xnew):
        return rng.uniform(0, 1, size=(len(Xnew), 2))

    def quality_fn(F):
        return -np.mean(F)

    result = regional_oracle(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V_oracle,
        history=RegionHistory(n_regions=2), t=1,
        evaluate_fn=evaluate_fn, quality_fn=quality_fn,
        xl=xl, xu=xu, budget=5,
    )
    assert "OHI" in result and "ROG" in result
    assert 0 in result["utilities"] and 1 in result["utilities"]


def test_regional_oracle_predict_falls_back_not_unavailable_regression():
    """PREDICT phai co utility thuc (khong NaN) ngay ca khi history rong
    -- fallback ve LOCAL."""
    rng, X, F_plus, region_ids = make_toy_oracle_inputs(seed=7)

    def evaluate_fn(Xnew):
        return rng.uniform(0, 1, size=(len(Xnew), 2))

    def quality_fn(F):
        return -np.mean(F)

    result = regional_oracle(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V_oracle,
        history=RegionHistory(n_regions=2), t=1,
        evaluate_fn=evaluate_fn, quality_fn=quality_fn,
        xl=xl, xu=xu, budget=5,
    )
    for r in (0, 1):
        assert not np.isnan(result["utilities"][r]["PREDICT"]), (
            f"vung {r}: PREDICT phai fallback ve LOCAL va co utility "
            f"thuc, khong duoc la NaN/unavailable")

    # global PREDICT action cung phai available, khong bi loai vi thieu history
    assert result["global_results"]["PREDICT"]["available"] is True


def test_regional_oracle_predict_with_partial_history():
    """Mot vung co lich su, vung con lai khong -- ca hai deu phai co
    utility PREDICT thuc (nho fallback rieng tung vung)."""
    rng, X, F_plus, region_ids = make_toy_oracle_inputs(seed=9)

    # Tao history chi co du lieu cho vung 0 (tat ca X thuoc region 0)
    history = RegionHistory(n_regions=2)
    rng2 = np.random.default_rng(200)
    X_r0 = rng2.uniform(0, 1, size=(10, D))
    rids_r0 = np.zeros(10, dtype=int)
    history.update(t=0, X=X_r0, F=rng2.uniform(0, 1, size=(10, 2)),
                   region_ids=rids_r0, xl=xl, xu=xu)
    X_r0_2 = rng2.uniform(0, 1, size=(10, D))
    history.update(t=1, X=X_r0_2, F=rng2.uniform(0, 1, size=(10, 2)),
                   region_ids=rids_r0, xl=xl, xu=xu)

    def evaluate_fn(Xnew):
        return rng.uniform(0, 1, size=(len(Xnew), 2))

    def quality_fn(F):
        return -np.mean(F)

    result = regional_oracle(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V_oracle,
        history=history, t=2,
        evaluate_fn=evaluate_fn, quality_fn=quality_fn,
        xl=xl, xu=xu, budget=5,
    )
    assert not np.isnan(result["utilities"][0]["PREDICT"])
    assert not np.isnan(result["utilities"][1]["PREDICT"])


def test_regional_oracle_keep_baseline_zero_utility():
    """KEEP so voi chinh no phai co utility = 0 (U_keep = Q_keep - Q_keep)."""
    rng, X, F_plus, region_ids = make_toy_oracle_inputs(seed=3)

    def evaluate_fn(Xnew):
        return rng.uniform(0, 1, size=(len(Xnew), 2))

    def quality_fn(F):
        return -np.mean(F)

    result = regional_oracle(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V_oracle,
        history=RegionHistory(n_regions=2), t=1,
        evaluate_fn=evaluate_fn, quality_fn=quality_fn,
        xl=xl, xu=xu, budget=5,
    )
    for r in (0, 1):
        assert result["utilities"][r]["KEEP"] == 0.0


def test_regional_oracle_ohi_in_valid_range():
    rng, X, F_plus, region_ids = make_toy_oracle_inputs(seed=5)

    def evaluate_fn(Xnew):
        return rng.uniform(0, 1, size=(len(Xnew), 2))

    def quality_fn(F):
        return -np.mean(F)

    result = regional_oracle(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V_oracle,
        history=RegionHistory(n_regions=2), t=1,
        evaluate_fn=evaluate_fn, quality_fn=quality_fn,
        xl=xl, xu=xu, budget=5,
    )
    assert -1e-9 <= result["OHI"] <= 1.0 + 1e-9


def test_operator_heterogeneity_index_extremes():
    assert operator_heterogeneity_index(["LOCAL"]) == 0.0  # <2 regions
    assert operator_heterogeneity_index(["LOCAL", "LOCAL", "LOCAL"]) == pytest.approx(0.0)
    het = operator_heterogeneity_index(["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"])
    assert het == pytest.approx(1.0, abs=1e-6)


def test_split_by_region_shapes():
    X = np.arange(20 * D).reshape(20, D).astype(float)
    region_ids = np.array([0] * 8 + [1] * 12)
    regions = split_by_region(X, region_ids, R=2)
    assert regions[0].shape == (8, D)
    assert regions[1].shape == (12, D)


def test_split_by_region_empty_region_shape():
    X = np.arange(10 * D).reshape(10, D).astype(float)
    region_ids = np.zeros(10, dtype=int)
    regions = split_by_region(X, region_ids, R=2)
    assert regions[0].shape == (10, D)
    assert regions[1].shape == (0, D)


# =====================================================================
# E2_operator_oracle.py -- smoke test tich hop toan bo pipeline
# (normalize -> partition -> severity -> quality_loss -> oracle),
# thu nho tham so de chay nhanh (khong chay full 14 DF x 3 seed x
# 15 change that).
# =====================================================================
def test_e2_run_one_smoke_integration():
    """Regression tong hop cho ca 3 fix (budget import path, quality_loss
    Fm/Fp + theta, oracle PREDICT fallback): chay that DF1 qua
    prdrl.experiments.E2_operator_oracle.run_one() voi tham so thu nho,
    phai khong crash va tra ve row hop le."""
    import prdrl.experiments.E2_operator_oracle as e2

    old = (e2.WARMUP_GENERATIONS, e2.N_CHANGES, e2.GENERATIONS_PER_CHANGE,
           e2.OPERATOR_BUDGET)
    try:
        e2.WARMUP_GENERATIONS = 5
        e2.N_CHANGES = 2
        e2.GENERATIONS_PER_CHANGE = 3
        e2.OPERATOR_BUDGET = 3

        rows = e2.run_one(df_id=1, seed=1)

        assert len(rows) == 2
        for row in rows:
            assert np.isfinite(row["OHI"])
            assert np.isfinite(row["ROG"])
            assert np.isfinite(row["IGD_keep"])
            assert np.isfinite(row["regional_IGD"])
            # change event dau tien (change_idx=1) khong co lich su t-2
            # nao ca -- PREDICT phai van co utility thuc (khong NaN het)
            # nho fallback ve LOCAL, dung cho it nhat 1 vung active.
            predict_utils = [
                u["PREDICT"] for u in row["utilities"].values()
                if "PREDICT" in u
            ]
            assert any(not np.isnan(u) for u in predict_utils), (
                "PREDICT phai co it nhat 1 vung tra utility thuc, "
                "khong duoc NaN toan bo do bi chan boi thieu lich su")
    finally:
        (e2.WARMUP_GENERATIONS, e2.N_CHANGES, e2.GENERATIONS_PER_CHANGE,
         e2.OPERATOR_BUDGET) = old
