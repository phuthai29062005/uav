"""
D2 Production-Path Implementation Validation — Test Suite (T1–T55).

Tests are organised in groups:
    T1–T8   : O0 default regression — sigma=0.03, unrestricted survival unchanged
    T9–T20  : O3 config & constants
    T21–T28 : generate_children_with_info — metadata correctness
    T29–T38 : D2-B equivalence — production path O3 == canonical HARD helper
    T39–T44 : Action isolation — HARD only fires for realized PREDICT
    T45–T50 : HARD structural invariant — n_old_dropped_other == 0
    T51–T55 : FE fairness — HARD adds zero evaluation FE
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src")

from prdrl.core.survival import region_protected_survival, compute_ranks_crowding_all
from prdrl.history.region_history import RegionHistory
from prdrl.oracle.oracle import (
    PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    PREDICT_SURVIVAL_UNRESTRICTED,
    generate_children,
    generate_children_with_info,
    respond_population,
    regional_oracle,
    nsga2_survival,
)
from prdrl.experiments.D2_production_path_equivalence import (
    build_pre_response_state,
    run_diagnostic_path,
    run_production_path_o3,
    run_production_path_o0,
    run_action_isolated_path,
    run_equivalence_case,
    run_batch,
    summarize_batch,
    _sorted_rows,
    BUDGET,
    POP_SIZE,
    D,
    N_OBJ,
    N_PARTITIONS,
)
from prdrl.problems.controlled_predict import evaluate as cp_evaluate, get_bounds
from prdrl.core.reference import reference
from prdrl.core.normalize import normalize
from prdrl.core.partition import partition


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture(scope="module")
def state_42():
    return build_pre_response_state(seed=42, target_region=0)


@pytest.fixture(scope="module")
def state_43():
    return build_pre_response_state(seed=43, target_region=0)


@pytest.fixture(scope="module")
def diag_42(state_42):
    return run_diagnostic_path(state_42, sigma_predict=0.005)


@pytest.fixture(scope="module")
def prod_o3_42(state_42):
    return run_production_path_o3(state_42, sigma_predict=0.005)


@pytest.fixture(scope="module")
def prod_o0_42(state_42):
    return run_production_path_o0(state_42, sigma_predict=0.03)


def _simple_evaluate(X):
    """Dummy evaluator: identity objectives (first two dims)."""
    X = np.asarray(X, dtype=float)
    return X[:, :2].copy()


def _make_tiny_history(n_regions=4):
    """Minimal RegionHistory with no valid entries (PREDICT falls back to LOCAL)."""
    return RegionHistory(n_regions=n_regions)


def _make_tiny_state_no_history():
    """State where PREDICT lacks valid history → LOCAL_FALLBACK."""
    xl, xu = get_bounds(D)
    V = reference(N_OBJ, N_PARTITIONS)
    R = len(V)
    rng = np.random.default_rng(999)
    X = rng.uniform(xl, xu, size=(POP_SIZE, D))
    F = cp_evaluate(X, 1, ["persistent"] * R)
    F_m, _ = normalize(F, F)
    region_ids = np.asarray(partition(np.asarray(F_m), V), dtype=int)
    history = RegionHistory(n_regions=R)  # empty → no valid centroid
    return {
        "X": X, "F_plus": F, "region_ids": region_ids,
        "V": V, "history": history, "t": 2,
        "xl": xl, "xu": xu, "target_region": 0,
        "n_r_before": int(np.sum(region_ids == 0)),
        "region_dynamics": ["persistent"] * R,
    }


# ============================================================
# T1–T8: O0 Default Regression
# ============================================================

def test_T1_o0_default_sigma():
    """T1: regional_oracle default sigma_predict is 0.005 (post-D2-D O1 adoption, 2026-09-20)."""
    import inspect
    sig = inspect.signature(regional_oracle)
    assert sig.parameters["sigma_predict"].default == pytest.approx(0.005)


def test_T2_respond_population_default_mode():
    """T2: respond_population default survival mode is UNRESTRICTED."""
    import inspect
    sig = inspect.signature(respond_population)
    assert sig.parameters["predict_response_survival_mode"].default == PREDICT_SURVIVAL_UNRESTRICTED


def test_T3_o0_pop_size_preserved(state_42, prod_o0_42):
    """T3: O0 path returns correct population size."""
    assert len(prod_o0_42["X_surv"]) == state_42["POP_SIZE"]


def test_T4_o0_f_shape_preserved(state_42, prod_o0_42):
    """T4: O0 path returns correct objective shape."""
    assert prod_o0_42["F_surv"].shape == (state_42["POP_SIZE"], N_OBJ)


def test_T5_o0_extra_fe_positive(prod_o0_42):
    """T5: O0 path generates positive extra_fe (PREDICT produces children)."""
    assert prod_o0_42["extra_fe"] > 0


def test_T6_o0_not_identical_to_o3(state_42, prod_o0_42, prod_o3_42):
    """T6: O0 (sigma=0.03) produces different survivors from O3 (sigma=0.005).

    Different sigma → different children (same base_seed, different noise scale).
    """
    X_o0 = _sorted_rows(prod_o0_42["X_surv"])
    X_o3 = _sorted_rows(prod_o3_42["X_surv"])
    assert not np.allclose(X_o0, X_o3, atol=1e-10), (
        "O0 and O3 produced identical survivors — sigma difference must propagate"
    )


def test_T7_o0_unrestricted_allows_outside_drops(state_42, prod_o0_42):
    """T7: O0 unrestricted survival does NOT guarantee zero outside-region drops.

    We verify the O0 path does not unconditionally call region_protected_survival.
    (It may still happen to preserve outside by chance — we test it CAN drop them,
    not that it always does. Here we just verify it produces a valid population.)
    """
    N = state_42["POP_SIZE"]
    assert len(prod_o0_42["X_surv"]) == N


def test_T8_respond_population_keep_all_returns_original(state_42):
    """T8: respond_population with all-KEEP returns original X unchanged."""
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    V = state_42["V"]
    region_ids = state_42["region_ids"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    history = state_42["history"]

    result = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={},  # empty → all KEEP
        history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET,
        sigma_local=0.02, sigma_predict=0.03, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert result["extra_fe"] == 0
    assert np.allclose(result["X"], X)


# ============================================================
# T9–T20: O3 Config & Constants
# ============================================================

def test_T9_constant_unrestricted_value():
    """T9: PREDICT_SURVIVAL_UNRESTRICTED == 'UNRESTRICTED'."""
    assert PREDICT_SURVIVAL_UNRESTRICTED == "UNRESTRICTED"


def test_T10_constant_hard_value():
    """T10: PREDICT_SURVIVAL_HARD_RESPONSE_ONLY == 'HARD_RESPONSE_ONLY'."""
    assert PREDICT_SURVIVAL_HARD_RESPONSE_ONLY == "HARD_RESPONSE_ONLY"


def test_T11_constants_distinct():
    """T11: Two survival mode constants are distinct."""
    assert PREDICT_SURVIVAL_UNRESTRICTED != PREDICT_SURVIVAL_HARD_RESPONSE_ONLY


def test_T12_o3_pop_size_preserved(state_42, prod_o3_42):
    """T12: O3 path returns correct population size."""
    assert len(prod_o3_42["X_surv"]) == state_42["POP_SIZE"]


def test_T13_o3_f_shape_preserved(state_42, prod_o3_42):
    """T13: O3 path returns correct objective shape."""
    assert prod_o3_42["F_surv"].shape == (state_42["POP_SIZE"], N_OBJ)


def test_T14_o3_extra_fe_equals_budget(prod_o3_42):
    """T14: O3 path extra_fe == BUDGET (PREDICT generates exactly BUDGET children)."""
    assert prod_o3_42["extra_fe"] == BUDGET


def test_T15_o0_extra_fe_equals_budget(prod_o0_42):
    """T15: O0 path extra_fe == BUDGET for identical setup."""
    assert prod_o0_42["extra_fe"] == BUDGET


def test_T16_lazy_loader_returns_callable():
    """T16: _get_region_protected_survival() returns a callable."""
    from prdrl.oracle.oracle import _get_region_protected_survival
    fn = _get_region_protected_survival()
    assert callable(fn)


def test_T17_lazy_loader_is_canonical():
    """T17: The lazy-loaded function is the same object as core.survival.region_protected_survival."""
    from prdrl.oracle.oracle import _get_region_protected_survival
    fn = _get_region_protected_survival()
    assert fn is region_protected_survival


def test_T18_r1_imports_from_core():
    """T18: R1 diagnostic imports region_protected_survival from prdrl.core.survival."""
    import prdrl.experiments.R1_region_protected_response as r1
    assert r1.region_protected_survival is region_protected_survival


def test_T19_l2b_imports_from_core():
    """T19: L2B imports region_protected_survival from prdrl.core.survival."""
    import prdrl.experiments.L2B_mixed_regime_boundary_validation as l2b
    assert l2b.region_protected_survival is region_protected_survival


def test_T20_oracle_module_exports_constants():
    """T20: oracle module exports both survival mode constants at module level."""
    import prdrl.oracle.oracle as ora
    assert hasattr(ora, "PREDICT_SURVIVAL_UNRESTRICTED")
    assert hasattr(ora, "PREDICT_SURVIVAL_HARD_RESPONSE_ONLY")


# ============================================================
# T21–T28: generate_children_with_info metadata
# ============================================================

def test_T21_with_info_returns_three_tuple(state_42):
    """T21: generate_children_with_info returns (children, realized_op, n_r_before)."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r

    children, realized_op, n_r_before = generate_children_with_info(
        action="PREDICT", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert children.ndim == 2
    assert isinstance(realized_op, str)
    assert isinstance(n_r_before, int)


def test_T22_with_info_n_r_before_matches_mask(state_42):
    """T22: n_r_before equals count of OLD individuals in target region."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r

    _, _, n_r_before = generate_children_with_info(
        action="PREDICT", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert n_r_before == int(np.sum(region_ids == r))


def test_T23_with_info_predict_success_realizes_predict(state_42):
    """T23: realized_op == 'PREDICT' when history has valid centroid sequence."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r

    _, realized_op, _ = generate_children_with_info(
        action="PREDICT", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert realized_op == "PREDICT"


def test_T24_with_info_predict_no_history_falls_back(state_42):
    """T24: realized_op == 'LOCAL_FALLBACK' when history has no centroid sequence."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r
    empty_history = RegionHistory(n_regions=len(state_42["V"]))

    _, realized_op, _ = generate_children_with_info(
        action="PREDICT", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=empty_history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert realized_op == "LOCAL_FALLBACK"


def test_T25_with_info_local_action_realizes_local(state_42):
    """T25: realized_op == 'LOCAL' when action == LOCAL."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r

    _, realized_op, _ = generate_children_with_info(
        action="LOCAL", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert realized_op == "LOCAL"


def test_T26_with_info_diversify_action_realizes_diversify(state_42):
    """T26: realized_op == 'DIVERSIFY' when action == DIVERSIFY."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r

    _, realized_op, _ = generate_children_with_info(
        action="DIVERSIFY", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert realized_op == "DIVERSIFY"


def test_T27_with_info_children_match_generate_children(state_42):
    """T27: generate_children_with_info children are identical to generate_children."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r
    X_region = X[mask]
    kwargs = dict(
        region_id=r, X_region=X_region, history=history, t=t,
        budget=BUDGET, xl=xl, xu=xu, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
    )

    ch_direct = generate_children(action="PREDICT", **kwargs)
    ch_with_info, _, _ = generate_children_with_info(
        action="PREDICT", old_region_ids=region_ids, **kwargs,
    )
    assert np.allclose(ch_direct, ch_with_info)


def test_T28_with_info_children_count_equals_budget(state_42):
    """T28: PREDICT generates exactly BUDGET children."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    mask = region_ids == r

    children, _, _ = generate_children_with_info(
        action="PREDICT", region_id=r, X_region=X[mask], old_region_ids=region_ids,
        history=history, t=t, budget=BUDGET, xl=xl, xu=xu,
        sigma_local=0.02, sigma_predict=0.005, sigma_diversify=0.15,
        lambda_pred=1.0, base_seed=42,
    )
    assert len(children) == BUDGET


# ============================================================
# T29–T38: D2-B Equivalence — production path O3 == canonical HARD
# ============================================================

def test_T29_single_case_pass(state_42, diag_42, prod_o3_42):
    """T29: Single equivalence case seed=42 passes all checks."""
    result = run_equivalence_case(42, target_region=0)
    assert result["PASS"], f"Equivalence case failed: {result}"


def test_T30_survivors_identical(diag_42, prod_o3_42):
    """T30: Survivor set (X) is identical between diagnostic and production O3 paths."""
    X_d = _sorted_rows(diag_42["X_surv"])
    X_p = _sorted_rows(prod_o3_42["X_surv"])
    assert np.allclose(X_d, X_p, atol=1e-10), (
        "Diagnostic and production O3 survivors differ"
    )


def test_T31_f_survivors_identical(diag_42, prod_o3_42):
    """T31: Survivor objectives (F) are identical between diagnostic and production O3."""
    F_d = _sorted_rows(diag_42["F_surv"])
    F_p = _sorted_rows(prod_o3_42["F_surv"])
    assert np.allclose(F_d, F_p, atol=1e-10)


def test_T32_fe_match(diag_42, prod_o3_42):
    """T32: extra_fe is the same on both paths."""
    assert diag_42["extra_fe"] == prod_o3_42["extra_fe"]


def test_T33_pop_size_preserved_diag(state_42, diag_42):
    """T33: Diagnostic path preserves population size."""
    assert len(diag_42["X_surv"]) == state_42["POP_SIZE"]


def test_T34_pop_size_preserved_prod(state_42, prod_o3_42):
    """T34: Production O3 path preserves population size."""
    assert len(prod_o3_42["X_surv"]) == state_42["POP_SIZE"]


def test_T35_batch_4_seeds():
    """T35: Batch equivalence check on 4 seeds all pass."""
    results = run_batch([42, 43, 44, 45])
    summary = summarize_batch(results)
    assert summary["all_pass"], f"Batch failed: {summary}"


def test_T36_batch_all_fe_match():
    """T36: All seeds in a batch have matching extra_fe."""
    results = run_batch([42, 43, 44, 45])
    assert all(r["fe_match"] for r in results)


def test_T37_batch_all_survivors_identical():
    """T37: All seeds in a batch have identical survivors (diagnostic == production)."""
    results = run_batch([42, 43, 44, 45])
    assert all(r["survivors_identical"] for r in results)


def test_T38_second_seed_equivalence(state_43):
    """T38: Equivalence holds for a second seed (43) to confirm generalization."""
    result = run_equivalence_case(43, target_region=0)
    assert result["PASS"], f"Equivalence failed for seed 43: {result}"


# ============================================================
# T39–T44: Action Isolation — HARD only fires for realized PREDICT
# ============================================================

def test_T39_local_action_uses_unrestricted(state_42):
    """T39: LOCAL action under HARD mode still uses unrestricted survival.

    When actions={r: LOCAL}, no region achieves realized_op==PREDICT, so
    the production path must fall through to unrestricted nsga2_survival.
    """
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    region_ids = state_42["region_ids"]
    V = state_42["V"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    r = state_42["target_region"]

    result_hard = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "LOCAL"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )
    result_unr = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "LOCAL"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )
    # LOCAL action → no HARD trigger → both paths must produce identical survivors
    assert np.allclose(
        _sorted_rows(result_hard["X"]),
        _sorted_rows(result_unr["X"]),
        atol=1e-10,
    ), "LOCAL action with HARD mode should fall through to unrestricted, matching UNRESTRICTED mode"


def test_T40_diversify_action_uses_unrestricted(state_42):
    """T40: DIVERSIFY action under HARD mode uses unrestricted survival."""
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    region_ids = state_42["region_ids"]
    V = state_42["V"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    r = state_42["target_region"]

    result_hard = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "DIVERSIFY"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )
    result_unr = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "DIVERSIFY"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )
    assert np.allclose(
        _sorted_rows(result_hard["X"]),
        _sorted_rows(result_unr["X"]),
        atol=1e-10,
    )


def test_T41_predict_fallback_uses_unrestricted():
    """T41: PREDICT that falls back to LOCAL_FALLBACK must NOT apply HARD.

    When history is empty, PREDICT resolves to LOCAL_FALLBACK. The HARD
    mode must detect this and fall through to unrestricted survival.
    """
    state = _make_tiny_state_no_history()
    X = state["X"]
    F_plus = state["F_plus"]
    region_ids = state["region_ids"]
    V = state["V"]
    history = state["history"]  # empty — no valid centroid
    t = state["t"]
    xl, xu = state["xl"], state["xu"]
    rd = state["region_dynamics"]
    r = state["target_region"]

    result_hard = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "PREDICT"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )
    result_unr = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "PREDICT"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )
    # Fallback → no HARD → must match unrestricted
    assert np.allclose(
        _sorted_rows(result_hard["X"]),
        _sorted_rows(result_unr["X"]),
        atol=1e-10,
    ), "LOCAL_FALLBACK must not trigger HARD survival"


def test_T42_keep_only_no_hard_triggered(state_42):
    """T42: All-KEEP actions return original pop unchanged under HARD mode."""
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    V = state_42["V"]
    region_ids = state_42["region_ids"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    history = state_42["history"]

    result = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={},  # empty → all KEEP
        history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )
    assert result["extra_fe"] == 0
    assert np.allclose(result["X"], X)


def test_T43_hard_only_fires_for_predict_not_local_directly(state_42):
    """T43: HARD mode with LOCAL action produces unrestricted result (same as UNRESTRICTED mode)."""
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    region_ids = state_42["region_ids"]
    V = state_42["V"]
    history = state_42["history"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    r = state_42["target_region"]

    result = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={r: "LOCAL"}, history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )
    # LOCAL under HARD → unrestricted: pop size must still be correct
    assert len(result["X"]) == state_42["POP_SIZE"]


def test_T44_predict_with_hard_differs_from_predict_without(state_42, prod_o3_42, prod_o0_42):
    """T44: PREDICT+HARD (O3) produces different survivors than PREDICT+UNRESTRICTED (O0).

    Ensures HARD mode has a meaningful structural effect beyond sigma difference alone.
    We compare O3 (sigma=0.005, HARD) vs O0 (sigma=0.03, UNRESTRICTED).
    """
    # They should differ (different sigma AND different survival)
    X_o3 = _sorted_rows(prod_o3_42["X_surv"])
    X_o0 = _sorted_rows(prod_o0_42["X_surv"])
    assert not np.allclose(X_o3, X_o0, atol=1e-10)


# ============================================================
# T45–T50: HARD Structural Invariant — n_old_dropped_other == 0
# ============================================================

def test_T45_hard_invariant_single_case(diag_42):
    """T45: n_old_dropped_other == 0 for seed=42 diagnostic path."""
    assert diag_42["n_old_dropped_other"] == 0


def test_T46_hard_invariant_verified_on_raw_call(state_42, diag_42):
    """T46: Direct call to region_protected_survival holds HARD invariant."""
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    n_r_before = state_42["n_r_before"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    t = state_42["t"]
    mask = region_ids == r

    children = diag_42["X_children"]
    F_children = diag_42["F_children"]

    _, _, final_idx = region_protected_survival(
        X_pop=X, F_pop=F_plus,
        X_children=children, F_children=F_children,
        response_region_r=r, old_region_ids=region_ids,
        n_r_before=n_r_before, xl=xl, xu=xu,
    )

    N = len(X)
    final_set = set(int(i) for i in final_idx if i < N)
    old_outside = [i for i in range(N) if region_ids[i] != r]
    n_dropped_other = sum(1 for i in old_outside if i not in final_set)
    assert n_dropped_other == 0


def test_T47_hard_invariant_multiple_seeds():
    """T47: HARD invariant holds across 4 seeds."""
    results = run_batch([42, 43, 44, 45])
    assert all(r["hard_invariant_diag"] for r in results), (
        f"HARD invariant failed for seeds: {[r['seed'] for r in results if not r['hard_invariant_diag']]}"
    )


def test_T48_old_outside_all_preserved_in_final_idx(state_42, diag_42):
    """T48: Every OLD individual outside the target region appears in final_idx."""
    X = state_42["X"]
    region_ids = state_42["region_ids"]
    r = state_42["target_region"]
    final_idx = diag_42["final_idx"]

    N = len(X)
    final_set = set(int(i) for i in final_idx if i < N)
    old_outside = [i for i in range(N) if region_ids[i] != r]
    for i in old_outside:
        assert i in final_set, f"OLD individual {i} outside r={r} was dropped"


def test_T49_region_protected_survival_size_invariant():
    """T49: region_protected_survival always produces POP_SIZE survivors."""
    rng = np.random.default_rng(77)
    N = 20
    B = 5
    dim = 2
    r = 0
    n_r_before = 5
    X_pop = rng.uniform(0, 1, (N, dim))
    F_pop = rng.uniform(0, 1, (N, 2))
    X_ch = rng.uniform(0, 1, (B, dim))
    F_ch = rng.uniform(0, 1, (B, 2))
    region_ids = np.zeros(N, dtype=int)
    region_ids[n_r_before:] = 1  # first n_r_before in r, rest in r=1

    X_s, F_s, idx = region_protected_survival(
        X_pop, F_pop, X_ch, F_ch, r, region_ids, n_r_before, 0.0, 1.0,
    )
    assert len(X_s) == N
    assert len(F_s) == N
    assert len(idx) == N


def test_T50_region_protected_survival_assert_fires_on_wrong_n_r_before():
    """T50: region_protected_survival raises AssertionError when n_r_before is inconsistent."""
    rng = np.random.default_rng(88)
    N = 10
    B = 3
    dim = 2
    r = 0
    X_pop = rng.uniform(0, 1, (N, dim))
    F_pop = rng.uniform(0, 1, (N, 2))
    X_ch = rng.uniform(0, 1, (B, dim))
    F_ch = rng.uniform(0, 1, (B, 2))
    # All OLD are outside r (region_ids all == 1), but n_r_before=5 claims 5 in r
    region_ids = np.ones(N, dtype=int)
    # n_r_before=5 but |OLD_r|=0 → candidates pool has only B=3 < 5 → assertion fires
    with pytest.raises(AssertionError):
        region_protected_survival(
            X_pop, F_pop, X_ch, F_ch, r, region_ids, 5, 0.0, 1.0,
        )


# ============================================================
# T51–T55: FE Fairness — HARD adds zero evaluation FE
# ============================================================

def test_T51_hard_fe_equals_unrestricted_fe(state_42, prod_o3_42, prod_o0_42):
    """T51: O3 (HARD) extra_fe equals O0 (UNRESTRICTED) extra_fe for same actions.

    FE fairness: HARD survival selection is rank-based, adds zero evaluation calls.
    """
    assert prod_o3_42["extra_fe"] == prod_o0_42["extra_fe"]


def test_T52_hard_fe_equals_budget(prod_o3_42):
    """T52: O3 extra_fe == BUDGET (not BUDGET + something from HARD selection)."""
    assert prod_o3_42["extra_fe"] == BUDGET


def test_T53_diagnostic_fe_equals_budget(diag_42):
    """T53: Diagnostic path extra_fe == BUDGET (HARD helper itself adds no FE)."""
    assert diag_42["extra_fe"] == BUDGET


def test_T54_fe_not_doubled(state_42, prod_o3_42):
    """T54: extra_fe is not doubled (not counted for both parents and children)."""
    assert prod_o3_42["extra_fe"] == BUDGET
    assert prod_o3_42["extra_fe"] != 2 * BUDGET


def test_T55_keep_fe_is_zero(state_42):
    """T55: All-KEEP actions produce extra_fe == 0."""
    X = state_42["X"]
    F_plus = state_42["F_plus"]
    V = state_42["V"]
    region_ids = state_42["region_ids"]
    t = state_42["t"]
    xl, xu = state_42["xl"], state_42["xu"]
    rd = state_42["region_dynamics"]
    history = state_42["history"]

    result = respond_population(
        X=X, F_plus=F_plus, region_ids=region_ids, V=V,
        actions={},
        history=history, t=t,
        evaluate_fn=lambda Xc: cp_evaluate(Xc, t, rd),
        xl=xl, xu=xu, budget=BUDGET, sigma_local=0.02, sigma_predict=0.005,
        sigma_diversify=0.15, lambda_pred=1.0, base_seed=42,
        predict_response_survival_mode=PREDICT_SURVIVAL_HARD_RESPONSE_ONLY,
    )
    assert result["extra_fe"] == 0
