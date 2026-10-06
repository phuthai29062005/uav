import numpy as np

from prdrl.oracle.oracle import (
    make_rng,
    generate_children,
    respond_population,
    operator_heterogeneity_index,
    regional_oracle,
)

from prdrl.history.region_history import RegionHistory


# ============================================================
# Helper
# ============================================================

def simple_evaluate(X):
    """
    Objective giả để test.

    F = X

    X shape (N, 2)
    F shape (N, 2)
    """
    X = np.asarray(X, dtype=float)
    return X.copy()


def simple_quality(F):
    """
    Quality càng lớn càng tốt.

    Ta dùng:
        Q = - mean(sum(F))

    Chỉ phục vụ unit test.
    """
    F = np.asarray(F, dtype=float)

    return -float(
        np.mean(
            np.sum(F, axis=1)
        )
    )


# ============================================================
# 1. RNG phải deterministic trong cùng event/region/action
# ============================================================

def test_make_rng_is_deterministic():

    rng1 = make_rng(
        base_seed=42,
        t=2,
        region_id=1,
        action="LOCAL",
    )

    rng2 = make_rng(
        base_seed=42,
        t=2,
        region_id=1,
        action="LOCAL",
    )

    a = rng1.normal(size=5)
    b = rng2.normal(size=5)

    np.testing.assert_allclose(
        a,
        b,
        atol=1e-12,
    )


# ============================================================
# 2. PREDICT lấy đúng motion từ history
#
# mu(t-2) = [0.2, 0.4]
# mu(t-1) = [0.3, 0.5]
#
# delta_hat = [0.1, 0.1]
#
# parent = [0.3, 0.5]
#
# sigma_predict = 0
#
# child = [0.4, 0.6]
# ============================================================

def test_generate_children_predict_from_history():

    history = RegionHistory(
        n_regions=1
    )

    xl = np.array([
        0.0,
        0.0,
    ])

    xu = np.array([
        1.0,
        1.0,
    ])

    # ------------------------------------
    # t = 0 = t-2
    # ------------------------------------

    history.update(
        t=0,
        X=np.array([
            [0.2, 0.4],
        ]),
        F=np.array([
            [1.0, 2.0],
        ]),
        region_ids=np.array([0]),
        xl=xl,
        xu=xu,
    )

    # ------------------------------------
    # t = 1 = t-1
    # ------------------------------------

    history.update(
        t=1,
        X=np.array([
            [0.3, 0.5],
        ]),
        F=np.array([
            [1.0, 2.0],
        ]),
        region_ids=np.array([0]),
        xl=xl,
        xu=xu,
    )

    # Population ngay trước change t=2
    X_region = np.array([
        [0.3, 0.5],
    ])

    children = generate_children(
        action="PREDICT",
        region_id=0,
        X_region=X_region,
        history=history,
        t=2,
        budget=1,
        xl=xl,
        xu=xu,
        sigma_local=0.0,
        sigma_predict=0.0,
        sigma_diversify=0.0,
        lambda_pred=1.0,
        base_seed=42,
    )

    expected = np.array([
        [0.4, 0.6],
    ])

    np.testing.assert_allclose(
        children,
        expected,
        atol=1e-12,
    )


# ============================================================
# 3. Thiếu history -> PREDICT fallback LOCAL
#
# sigma_local = 0
#
# LOCAL khi sigma=0:
# child = parent
# ============================================================

def test_generate_children_predict_missing_history_fallback_local():

    history = RegionHistory(
        n_regions=1
    )

    xl = np.array([
        0.0,
        0.0,
    ])

    xu = np.array([
        1.0,
        1.0,
    ])

    # Chỉ có t=1.
    # Thiếu t=0 -> không đủ t-2, t-1.
    history.update(
        t=1,
        X=np.array([
            [0.3, 0.5],
        ]),
        F=np.array([
            [1.0, 2.0],
        ]),
        region_ids=np.array([0]),
        xl=xl,
        xu=xu,
    )

    X_region = np.array([
        [0.3, 0.5],
    ])

    children = generate_children(
        action="PREDICT",
        region_id=0,
        X_region=X_region,
        history=history,
        t=2,
        budget=1,
        xl=xl,
        xu=xu,

        # LOCAL fallback không thay đổi parent
        sigma_local=0.0,

        # Cố tình để sigma PREDICT rất lớn.
        # Nếu code dùng nhầm sigma_predict thì test sẽ fail.
        sigma_predict=0.9,

        sigma_diversify=0.0,
        lambda_pred=1.0,
        base_seed=42,
    )

    expected = np.array([
        [0.3, 0.5],
    ])

    np.testing.assert_allclose(
        children,
        expected,
        atol=1e-12,
    )


# ============================================================
# 4. respond_population với toàn KEEP
#
# Không có offspring
# -> extra FE = 0
# -> population giữ nguyên
# ============================================================

def test_respond_population_all_keep():

    X = np.array([
        [0.2, 0.8],
        [0.8, 0.2],
    ])

    F_plus = simple_evaluate(X)

    region_ids = np.array([
        0,
        1,
    ])

    V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    history = RegionHistory(
        n_regions=2
    )

    result = respond_population(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,

        # {} nghĩa là tất cả default KEEP
        actions={},

        history=history,
        t=2,
        evaluate_fn=simple_evaluate,
        xl=np.array([0.0, 0.0]),
        xu=np.array([1.0, 1.0]),
        budget=1,
        sigma_local=0.02,
        sigma_predict=0.03,
        sigma_diversify=0.15,
        lambda_pred=1.0,
        base_seed=42,
    )

    assert result["available"] is True
    assert result["extra_fe"] == 0

    np.testing.assert_allclose(
        result["X"],
        X,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["F"],
        F_plus,
        atol=1e-12,
    )


# ============================================================
# 5. OHI
#
# Nếu tất cả cùng action:
# OHI = 0
#
# Nếu 50% KEEP, 50% LOCAL:
#
# H = log(2)
# OHI = log(2)/log(4) = 0.5
# ============================================================

def test_operator_heterogeneity_index():

    same = [
        "KEEP",
        "KEEP",
        "KEEP",
        "KEEP",
    ]

    ohi_same = operator_heterogeneity_index(
        same
    )

    np.testing.assert_allclose(
        ohi_same,
        0.0,
        atol=1e-10,
    )

    mixed = [
        "KEEP",
        "KEEP",
        "LOCAL",
        "LOCAL",
    ]

    ohi_mixed = operator_heterogeneity_index(
        mixed
    )

    np.testing.assert_allclose(
        ohi_mixed,
        0.5,
        atol=1e-10,
    )


# ============================================================
# 6. Full regional oracle sanity test
#
# Dùng sigma = 0 để LOCAL/DIVERSIFY không dịch parent.
#
# PREDICT có history:
#
# region 0:
# [0.20, 0.80] -> [0.25, 0.75]
#
# region 1:
# [0.80, 0.20] -> [0.75, 0.25]
#
# Tổng mỗi vector luôn = 1.
#
# simple_quality = -mean(sum(F))
#
# => mọi action có cùng quality
# => KEEP phải thắng tie.
# ============================================================

def test_regional_oracle_keep_wins_ties():

    xl = np.array([
        0.0,
        0.0,
    ])

    xu = np.array([
        1.0,
        1.0,
    ])

    V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    history = RegionHistory(
        n_regions=2
    )

    # ------------------------------------
    # t = 0
    # ------------------------------------

    X_t0 = np.array([
        [0.20, 0.80],
        [0.80, 0.20],
    ])

    region_ids = np.array([
        0,
        1,
    ])

    history.update(
        t=0,
        X=X_t0,
        F=simple_evaluate(X_t0),
        region_ids=region_ids,
        xl=xl,
        xu=xu,
    )

    # ------------------------------------
    # t = 1
    # ------------------------------------

    X_t1 = np.array([
        [0.25, 0.75],
        [0.75, 0.25],
    ])

    history.update(
        t=1,
        X=X_t1,
        F=simple_evaluate(X_t1),
        region_ids=region_ids,
        xl=xl,
        xu=xu,
    )

    # ------------------------------------
    # Change event t = 2
    #
    # Population trước response chính là
    # population đã có từ environment t-1.
    # ------------------------------------

    X = X_t1.copy()
    F_plus = simple_evaluate(X)

    result = regional_oracle(
        X=X,
        F_plus=F_plus,
        region_ids=region_ids,
        V=V,
        history=history,
        t=2,
        evaluate_fn=simple_evaluate,
        quality_fn=simple_quality,
        xl=xl,
        xu=xu,
        budget=1,

        # Không noise để test deterministic
        sigma_local=0.0,
        sigma_predict=0.0,
        sigma_diversify=0.0,

        lambda_pred=1.0,
        base_seed=42,
    )

    # ------------------------------------
    # Cấu trúc output
    # ------------------------------------

    assert "utilities" in result
    assert "preferred_action" in result
    assert "OHI" in result
    assert "best_global_action" in result
    assert "regional_Q" in result
    assert "ROG" in result

    # ------------------------------------
    # KEEP phải thắng tie
    # ------------------------------------

    assert result["preferred_action"][0] == "KEEP"
    assert result["preferred_action"][1] == "KEEP"

    assert result["best_global_action"] == "KEEP"

    # Hai region đều KEEP
    # -> OHI = 0
    np.testing.assert_allclose(
        result["OHI"],
        0.0,
        atol=1e-10,
    )

    # Regional cũng giống best-global
    # -> ROG = 0
    np.testing.assert_allclose(
        result["ROG"],
        0.0,
        atol=1e-10,
    )

    # Regional oracle toàn KEEP
    # -> không có extra response FE
    assert result["regional_extra_fe"] == 0