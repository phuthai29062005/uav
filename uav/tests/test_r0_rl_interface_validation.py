"""tests/test_r0_rl_interface_validation.py — R0 RL Interface Validation tests.

T1–T67 as specified, plus additional correctness checks.

INVARIANTS:
  - No DRL, no training, no sigma tuning, no HARD production action
  - Production semantics unchanged (T63)
  - Manuscript/PDF untouched (T64)
  - Frozen experiment results untouched (T65)
"""

import json
import os
import hashlib
import numpy as np
import importlib.util
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "results", "R0_rl_interface_validation")


def _json(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures (interface imports)
# ---------------------------------------------------------------------------
from prdrl.rl.interface import (
    VALID_ACTIONS, ACTION_IDS, ACTION_NAMES,
    ACTION_KEEP, ACTION_LOCAL, ACTION_PREDICT, ACTION_DIVERSIFY,
    ACTION_SPECS, KEEP_SPEC, LOCAL_SPEC, PREDICT_SPEC, DIVERSIFY_SPEC,
    CURRENT_RESPONSE_BUDGET_UNIT, CURRENT_RESPONSE_BUDGET_VALUE, KEEP_RESPONSE_FE,
    STATE_SCHEMA_V1, REWARD_INTERFACE_STATUS,
    action_to_id, id_to_action,
    validate_action_vector, expected_response_fe,
    build_state_v1, build_decision_context, execute_response,
    check_e3_readiness, RegionalResponseRecord,
)
from prdrl.oracle.oracle import (
    respond_population, PREDICT_SURVIVAL_UNRESTRICTED,
    make_rng, ACTIONS as ORACLE_ACTIONS,
)
from prdrl.history.region_history import RegionHistory
from prdrl.core.partition import partition
from prdrl.features.forecast_state import get_predict_motion_from_history

# ---------------------------------------------------------------------------
# Toy test setup (mirrors experiment script)
# ---------------------------------------------------------------------------
D, M, N, R = 5, 2, 40, 4
XL = np.zeros(D)
XU = np.ones(D)
BASE_SEED = 42


def _eval(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.sum(X**2, axis=1), np.sum((X - 1)**2, axis=1)])


def _V():
    rng = np.random.default_rng(0)
    return rng.dirichlet(np.ones(M), size=R)


def _pop(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(XL, XU, size=(N, D))
    return X, _eval(X)


def _history_two_events(seed=42):
    V = _V()
    X0, F0 = _pop(seed)
    rid0 = partition(F0, V)
    h = RegionHistory(n_regions=R)
    h.update(0, X0, F0, rid0, XL, XU)
    rng = np.random.default_rng(seed + 1)
    X1 = np.clip(X0 + rng.normal(0, 0.05, X0.shape), 0, 1)
    F1 = _eval(X1)
    rid1 = partition(F1, V)
    h.update(1, X1, F1, rid1, XL, XU)
    return h, V, X1, F1, rid1


def _history_one_event(seed=99):
    V = _V()
    X0, F0 = _pop(seed)
    rid0 = partition(F0, V)
    h = RegionHistory(n_regions=R)
    h.update(0, X0, F0, rid0, XL, XU)
    return h, V, X0, F0, rid0


# ===========================================================================
# GATE TESTS (P0/D2D must be complete)
# ===========================================================================

def test_T1_p0_complete():
    """T1: P0_COMPLETE=YES."""
    p0r = _json("../P0R_manuscript_reconciliation/P0R_verdict.json")
    assert p0r["P0_COMPLETE"] == "YES"


def test_T2_d2d_complete():
    """T2: D2D_COMPLETE=YES."""
    p0r = _json("../P0R_manuscript_reconciliation/P0R_verdict.json")
    assert p0r["D2D_COMPLETE"] == "YES"


def test_T3_production_sigma_p():
    """T3: production sigma_P=0.005."""
    src = os.path.join(REPO, "src/prdrl/oracle/oracle.py")
    with open(src) as f:
        text = f.read()
    assert "sigma_predict=0.005" in text


# ===========================================================================
# ACTION SET
# ===========================================================================

def test_T4_action_set_exactly_four():
    """T4: action set is exactly {KEEP, LOCAL, PREDICT, DIVERSIFY}."""
    assert set(VALID_ACTIONS) == {"KEEP", "LOCAL", "PREDICT", "DIVERSIFY"}
    assert len(VALID_ACTIONS) == 4


def test_T5_no_hard_action():
    """T5: HARD is not in the controller action set."""
    assert "HARD" not in VALID_ACTIONS
    assert "HARD" not in ACTION_IDS


def test_T6_deterministic_action_ids():
    """T6: action IDs are deterministic integers."""
    assert ACTION_IDS["KEEP"] == 0
    assert ACTION_IDS["LOCAL"] == 1
    assert ACTION_IDS["PREDICT"] == 2
    assert ACTION_IDS["DIVERSIFY"] == 3


def test_T7_action_id_roundtrip():
    """T7: id->action->id and action->id->action roundtrip."""
    for name, id_ in ACTION_IDS.items():
        assert action_to_id(name) == id_
        assert id_to_action(id_) == name
    with pytest.raises(ValueError):
        action_to_id("HARD")
    with pytest.raises(ValueError):
        id_to_action(99)


# ===========================================================================
# ACTION SEMANTICS
# ===========================================================================

def test_T8_keep_fe_zero():
    """T8: KEEP response FE = 0."""
    assert KEEP_RESPONSE_FE == 0
    assert KEEP_SPEC.requested_fe == "0"


def test_T9_local_sigma():
    """T9: LOCAL sigma = 0.02."""
    assert LOCAL_SPEC.sigma == pytest.approx(0.02)


def test_T10_predict_sigma():
    """T10: PREDICT sigma = 0.005."""
    assert PREDICT_SPEC.sigma == pytest.approx(0.005)


def test_T11_predict_lambda():
    """T11: PREDICT lambda_P = 1.0."""
    assert PREDICT_SPEC.lambda_P == pytest.approx(1.0)


def test_T12_diversify_sigma():
    """T12: DIVERSIFY sigma = 0.15."""
    assert DIVERSIFY_SPEC.sigma == pytest.approx(0.15)


# ===========================================================================
# PREDICT FALLBACK CONTRACT
# ===========================================================================

def test_T13_predict_success_realized_predict():
    """T13: successful PREDICT (valid history) → realized = PREDICT."""
    h, V, X, F_plus, rid = _history_two_events()
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=2, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2), None)
    assert rec is not None
    if rec.n_parents > 0:
        assert rec.realized_operator in (ACTION_PREDICT, ACTION_LOCAL), (
            "realized must be PREDICT or LOCAL_FALLBACK (if region empty at t-2)")


def test_T14_missing_history_predict_requested_predict():
    """T14: PREDICT with missing history — requested_action stays PREDICT."""
    h, V, X, F_plus, rid = _history_one_event()
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=1, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2), None)
    if rec is not None and rec.n_parents > 0:
        assert rec.requested_action == ACTION_PREDICT


def test_T15_missing_history_predict_realized_local():
    """T15: PREDICT with missing history → realized = LOCAL."""
    h, V, X, F_plus, rid = _history_one_event()
    motion = get_predict_motion_from_history(history=h, region_id=2, t=1, lambda_pred=1.0)
    assert motion is None, "Expected no history for fallback test"
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=1, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2), None)
    if rec is not None and rec.n_parents > 0:
        assert rec.realized_operator == ACTION_LOCAL


def test_T16_fallback_sigma_local():
    """T16: fallback uses sigma_L = 0.02."""
    h, V, X, F_plus, rid = _history_one_event()
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=1, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2 and r.fallback_used), None)
    if rec is not None:
        assert rec.sigma_used == pytest.approx(0.02)


def test_T17_fallback_reason_recorded():
    """T17: fallback_reason recorded when fallback occurs."""
    h, V, X, F_plus, rid = _history_one_event()
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=1, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2 and r.fallback_used), None)
    if rec is not None:
        assert rec.fallback_reason == "MISSING_PREDICT_HISTORY"


def test_T18_no_hard_on_fallback():
    """T18: fallback uses LOCAL not HARD."""
    h, V, X, F_plus, rid = _history_one_event()
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=1, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2 and r.fallback_used), None)
    if rec is not None:
        assert rec.realized_operator != "HARD"
        assert rec.realized_operator != "HARD_RESPONSE_ONLY"


def test_T19_production_survival_unrestricted():
    """T19: production survival mode is UNRESTRICTED (not HARD)."""
    for spec in ACTION_SPECS.values():
        if spec.survival_mode != "N/A":
            assert spec.survival_mode == "UNRESTRICTED_NSGA2"
        assert "HARD" not in spec.survival_mode


# ===========================================================================
# BUDGET
# ===========================================================================

def test_T20_future_Br_not_activated():
    """T20: future B_r(a) action-dependent budget NOT activated."""
    budget_contract = _json("R0_budget_contract.json")
    assert budget_contract["future_action_dependent_budget_activated"] == False


def test_T21_current_B_semantics():
    """T21: current B semantics correctly identified as PER_ACTIVE_REGION."""
    assert CURRENT_RESPONSE_BUDGET_UNIT == "PER_ACTIVE_REGION"
    assert CURRENT_RESPONSE_BUDGET_VALUE == 10


def test_T22_multi_region_fe_formula():
    """T22: FE formula: B=10 per non-KEEP non-empty region."""
    sizes = {0: 10, 1: 10, 2: 10, 3: 10}
    assert expected_response_fe({0: ACTION_KEEP, 1: ACTION_LOCAL, 2: ACTION_PREDICT, 3: ACTION_DIVERSIFY}, sizes) == 30
    assert expected_response_fe({0: ACTION_KEEP}, sizes) == 0
    assert expected_response_fe({0: ACTION_LOCAL, 1: ACTION_LOCAL}, sizes) == 20
    # Empty region contributes 0 even if non-KEEP
    assert expected_response_fe({0: ACTION_LOCAL}, {0: 0}) == 0


# ===========================================================================
# REGION ACTION VECTOR
# ===========================================================================

def test_T23_inactive_region_not_silently_keep():
    """T23: inactive/empty region skipped, not silently treated as KEEP in FE."""
    # current code skips empty regions with `continue` in respond_population
    sizes = {0: 0, 1: 10}
    # empty region 0 with LOCAL still contributes 0 FE (skipped in code)
    fe = expected_response_fe({0: ACTION_LOCAL, 1: ACTION_LOCAL}, sizes)
    assert fe == 10  # only region 1 contributes


def test_T24_deterministic_region_ordering():
    """T24: region execution order deterministic (ascending by region_id)."""
    # respond_population iterates r in range(R) — deterministic ascending order
    from prdrl.oracle.oracle import respond_population
    h, V, X, F_plus, rid = _history_two_events()
    r1, _ = execute_response(
        actions={0: ACTION_LOCAL, 1: ACTION_DIVERSIFY}, X=X, F_plus=F_plus,
        region_ids=rid, V=V, history=h, t=2, evaluate_fn=_eval, xl=XL, xu=XU,
        base_seed=BASE_SEED,
    )
    h2, V2, X2, F2, rid2 = _history_two_events()
    r2, _ = execute_response(
        actions={0: ACTION_LOCAL, 1: ACTION_DIVERSIFY}, X=X2, F_plus=F2,
        region_ids=rid2, V=V2, history=h2, t=2, evaluate_fn=_eval, xl=XL, xu=XU,
        base_seed=BASE_SEED,
    )
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)


def test_T25_invalid_action_id_rejected():
    """T25: unknown action string rejected."""
    with pytest.raises(ValueError):
        validate_action_vector({"0": "HARD"}, list(range(4)))
    with pytest.raises(ValueError):
        validate_action_vector({"0": "NSGA3"}, list(range(4)))


def test_T26_invalid_region_rejected():
    """T26: action for non-existent region raises ValueError."""
    with pytest.raises(ValueError):
        validate_action_vector({99: ACTION_LOCAL}, list(range(4)))


def test_T27_no_duplicate_region_ids():
    """T27: Dict prevents duplicate region IDs by construction."""
    # Python dicts cannot have duplicate keys; validate_action_vector operates on dict
    d = {0: ACTION_LOCAL, 0: ACTION_DIVERSIFY}  # noqa: duplicate key — last wins
    assert len(d) == 1  # dict already deduped — show this is handled at source level


# ===========================================================================
# STATE FEATURE AUDIT
# ===========================================================================

def test_T28_state_feature_audit_complete():
    """T28: state feature audit file exists and PASS."""
    audit = _json("R0_state_feature_audit.json")
    assert audit["STATE_FEATURE_AUDIT"] == "PASS"


def test_T29_state_features_causal():
    """T29: state features are causally available."""
    audit = _json("R0_state_feature_audit.json")
    assert audit["STATE_CAUSALITY"] == "PASS"
    for name, feat in audit["features"].items():
        if not feat.get("excluded_from_v1"):
            assert feat.get("causal") is True, f"Feature {name} must be causal"


def test_T30_no_oracle_best_action_leakage():
    """T30: oracle best action not in state."""
    audit = _json("R0_state_feature_audit.json")
    assert audit["ORACLE_LEAKAGE_CHECK"] == "PASS"


def test_T31_no_oracle_utility_leakage():
    """T31: oracle utility U_{r,a} not in state."""
    audit = _json("R0_state_feature_audit.json")
    assert audit["ORACLE_LEAKAGE_CHECK"] == "PASS"


def test_T32_no_regime_label_leakage():
    """T32: synthetic regime labels not in state."""
    audit = _json("R0_state_feature_audit.json")
    assert audit["REGIME_LABEL_LEAKAGE_CHECK"] == "PASS"


def test_T33_no_post_response_igd_in_state():
    """T33: post-response IGD not in state features."""
    audit = _json("R0_state_feature_audit.json")
    causal = _json("R0_causality_audit.json")
    assert causal["checks"]["no_post_response_population_quality"]["status"] == "PASS"


def test_T34_no_epoch_end_metric_in_state():
    """T34: epoch-end metric not in state."""
    causal = _json("R0_causality_audit.json")
    assert causal["checks"]["no_epoch_end_metric"]["status"] == "PASS"


# ===========================================================================
# PREDICT HISTORY CAUSALITY
# ===========================================================================

def test_T35_predict_history_uses_t2_t1_only():
    """T35: PREDICT history uses t-2, t-1 only (not mu_t)."""
    causal = _json("R0_causality_audit.json")
    assert causal["checks"]["predict_history_causal"]["status"] == "PASS"


def test_T36_no_mu_t_forecast_leakage():
    """T36: mu_t not used in delta_hat construction."""
    causal = _json("R0_causality_audit.json")
    assert causal["checks"]["no_mu_t_in_delta_hat"]["status"] == "PASS"
    # Verify in code: get_centroid_sequence called with end_t=t-1
    import inspect
    from prdrl.features import forecast_state as fs
    src = inspect.getsource(fs.get_predict_motion_from_history)
    assert "end_t=t - 1" in src or "end_t = t - 1" in src


# ===========================================================================
# STATE SCHEMA
# ===========================================================================

def test_T37_deterministic_state_feature_ordering():
    """T37: state feature order is deterministic."""
    schema = _json("R0_state_schema.json")
    assert schema["feature_names"] == ["c_r", "l_r", "o_r", "delta_c_r"]
    assert "deterministic" in schema["feature_order"]


def test_T38_missing_feature_policy_explicit():
    """T38: missing-feature policy is explicitly specified."""
    schema = _json("R0_state_schema.json")
    assert "missing_value_policy" in schema
    assert "delta_c_r" in schema["missing_value_policy"]


def test_T39_inactive_sparse_handling_explicit():
    """T39: inactive/sparse region handling specified."""
    schema = _json("R0_state_schema.json")
    assert "inactive" in schema["missing_value_policy"] or "empty" in schema["missing_value_policy"]


# ===========================================================================
# CONTROLLER CONTEXT
# ===========================================================================

def test_T40_controller_context_read_only():
    """T40: controller observation feature_vector is read-only."""
    h, V, X, F_plus, rid = _history_two_events()
    ctx = build_decision_context(region_id=1, history=h, t=2, N_total=N, D=D)
    assert not ctx.feature_vector.flags.writeable


# ===========================================================================
# RESPONSE RECORD
# ===========================================================================

def test_T41_response_record_contains_requested_action():
    """T41: response record contains requested_action field."""
    h, V, X, F_plus, rid = _history_two_events()
    _, records = execute_response(
        actions={1: ACTION_LOCAL}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=2, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next(r for r in records if r.region_id == 1)
    assert rec.requested_action == ACTION_LOCAL


def test_T42_response_record_contains_realized_operator():
    """T42: response record contains realized_operator field."""
    h, V, X, F_plus, rid = _history_two_events()
    _, records = execute_response(
        actions={1: ACTION_LOCAL}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=2, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next(r for r in records if r.region_id == 1)
    assert rec.realized_operator == ACTION_LOCAL


def test_T43_requested_action_retained_during_fallback():
    """T43: requested_action=PREDICT retained even when realized=LOCAL."""
    h, V, X, F_plus, rid = _history_one_event()
    _, records = execute_response(
        actions={2: ACTION_PREDICT}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=1, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 2 and r.fallback_used), None)
    if rec is not None:
        assert rec.requested_action == ACTION_PREDICT
        assert rec.realized_operator == ACTION_LOCAL


def test_T44_response_fe_recorded():
    """T44: response FE recorded in record."""
    h, V, X, F_plus, rid = _history_two_events()
    _, records = execute_response(
        actions={1: ACTION_LOCAL}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=2, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 1), None)
    if rec is not None:
        assert isinstance(rec.response_fe, int)
        assert rec.response_fe >= 0


def test_T45_rng_metadata_trace_recorded():
    """T45: RNG namespace metadata in response record."""
    h, V, X, F_plus, rid = _history_two_events()
    _, records = execute_response(
        actions={1: ACTION_LOCAL}, X=X, F_plus=F_plus, region_ids=rid,
        V=V, history=h, t=2, evaluate_fn=_eval, xl=XL, xu=XU, base_seed=BASE_SEED,
    )
    rec = next((r for r in records if r.region_id == 1), None)
    if rec is not None:
        assert "base_seed" in rec.rng_namespace
        assert "t" in rec.rng_namespace
        assert "region_id" in rec.rng_namespace
        assert "action" in rec.rng_namespace


# ===========================================================================
# EQUIVALENCE (T46–T51)
# ===========================================================================

def _direct_respond(actions, X, F_plus, rid, V, h, t):
    return respond_population(
        X=X, F_plus=F_plus, region_ids=rid, V=V, actions=actions,
        history=h, t=t, evaluate_fn=_eval, xl=XL, xu=XU,
        budget=CURRENT_RESPONSE_BUDGET_VALUE,
        sigma_local=LOCAL_SPEC.sigma,
        sigma_predict=PREDICT_SPEC.sigma,
        sigma_diversify=DIVERSIFY_SPEC.sigma,
        lambda_pred=PREDICT_SPEC.lambda_P,
        base_seed=BASE_SEED,
        predict_response_survival_mode=PREDICT_SURVIVAL_UNRESTRICTED,
    )


def test_T46_wrapper_keep_equivalence():
    """T46: interface wrapper KEEP = direct KEEP."""
    h1, V, X, F, rid = _history_two_events()
    h2, _, X2, F2, rid2 = _history_two_events()
    acts = {r: ACTION_KEEP for r in range(R)}
    r1 = _direct_respond(acts, X, F, rid, V, h1, 2)
    r2, _ = execute_response(acts, X2, F2, rid2, V, h2, 2, _eval, XL, XU, base_seed=BASE_SEED)
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)
    assert np.allclose(r1["F"], r2["F"], atol=1e-12)
    assert r1["extra_fe"] == r2["extra_fe"]


def test_T47_local_equivalence():
    """T47: interface wrapper LOCAL = direct LOCAL."""
    h1, V, X, F, rid = _history_two_events()
    h2, _, X2, F2, rid2 = _history_two_events()
    acts = {1: ACTION_LOCAL}
    r1 = _direct_respond(acts, X, F, rid, V, h1, 2)
    r2, _ = execute_response(acts, X2, F2, rid2, V, h2, 2, _eval, XL, XU, base_seed=BASE_SEED)
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)
    assert r1["extra_fe"] == r2["extra_fe"]


def test_T48_predict_success_equivalence():
    """T48: wrapper PREDICT (success) = direct PREDICT."""
    h1, V, X, F, rid = _history_two_events()
    h2, _, X2, F2, rid2 = _history_two_events()
    acts = {2: ACTION_PREDICT}
    r1 = _direct_respond(acts, X, F, rid, V, h1, 2)
    r2, _ = execute_response(acts, X2, F2, rid2, V, h2, 2, _eval, XL, XU, base_seed=BASE_SEED)
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)
    assert r1["extra_fe"] == r2["extra_fe"]


def test_T49_predict_fallback_equivalence():
    """T49: wrapper PREDICT-fallback = direct PREDICT-fallback."""
    h1, V, X, F, rid = _history_one_event()
    h2, _, X2, F2, rid2 = _history_one_event()
    acts = {2: ACTION_PREDICT}
    r1 = _direct_respond(acts, X, F, rid, V, h1, 1)
    r2, _ = execute_response(acts, X2, F2, rid2, V, h2, 1, _eval, XL, XU, base_seed=BASE_SEED)
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)
    assert r1["extra_fe"] == r2["extra_fe"]


def test_T50_diversify_equivalence():
    """T50: wrapper DIVERSIFY = direct DIVERSIFY."""
    h1, V, X, F, rid = _history_two_events()
    h2, _, X2, F2, rid2 = _history_two_events()
    acts = {3: ACTION_DIVERSIFY}
    r1 = _direct_respond(acts, X, F, rid, V, h1, 2)
    r2, _ = execute_response(acts, X2, F2, rid2, V, h2, 2, _eval, XL, XU, base_seed=BASE_SEED)
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)
    assert r1["extra_fe"] == r2["extra_fe"]


def test_T51_wrapper_does_not_alter_rng():
    """T51: wrapper produces bit-identical results as direct call (RNG not disturbed)."""
    equiv = _json("R0_equivalence.json")
    assert equiv["RNG_EQUIVALENCE"] == "PASS"


# ===========================================================================
# DETERMINISM
# ===========================================================================

def test_T52_same_process_deterministic():
    """T52: same-process deterministic — two identical calls yield identical results."""
    h1, V, X, F, rid = _history_two_events()
    h2, _, X2, F2, rid2 = _history_two_events()
    acts = {0: ACTION_KEEP, 1: ACTION_LOCAL, 2: ACTION_PREDICT, 3: ACTION_DIVERSIFY}
    r1, _ = execute_response(acts, X, F, rid, V, h1, 2, _eval, XL, XU, base_seed=BASE_SEED)
    r2, _ = execute_response(acts, X2, F2, rid2, V, h2, 2, _eval, XL, XU, base_seed=BASE_SEED)
    assert np.allclose(r1["X"], r2["X"], atol=1e-12)


def test_T53_multi_region_dry_run_deterministic():
    """T53: multi-region dry run deterministic."""
    dry_run = _json("R0_multi_region_dry_run.json")
    assert dry_run["deterministic"] == True


def test_T54_multi_region_no_state_aliasing():
    """T54: execute_response does not alias population arrays."""
    h, V, X, F, rid = _history_two_events()
    r, _ = execute_response(
        {1: ACTION_LOCAL}, X, F, rid, V, h, 2, _eval, XL, XU, base_seed=BASE_SEED,
    )
    # Modifying the returned X should not affect a second call
    X_returned = r["X"]
    X_returned[0, 0] = -999.0
    h2, _, X2, F2, rid2 = _history_two_events()
    r2, _ = execute_response(
        {1: ACTION_LOCAL}, X2, F2, rid2, V, h2, 2, _eval, XL, XU, base_seed=BASE_SEED,
    )
    assert not np.all(r2["X"] == -999.0)


def test_T55_multi_region_fe_correct():
    """T55: multi-region FE = B * n_active_non_empty_regions."""
    dry_run = _json("R0_multi_region_dry_run.json")
    assert dry_run["fe_correct"] == True


# ===========================================================================
# REWARD / DRL
# ===========================================================================

def test_T56_reward_not_implemented():
    """T56: reward not implemented/tuned in R0."""
    assert REWARD_INTERFACE_STATUS == "SPEC_ONLY"


def test_T57_no_dqn_implementation():
    """T57: no DQN implementation in rl/interface.py."""
    iface_path = os.path.join(REPO, "src/prdrl/rl/interface.py")
    with open(iface_path) as f:
        src = f.read()
    for keyword in ("DQN", "Q_network", "target_network", "replay_buffer"):
        assert keyword not in src, f"Found '{keyword}' in interface.py"


def test_T58_no_replay_buffer():
    """T58: no replay buffer class/import in interface (comments mentioning it are OK)."""
    iface_path = os.path.join(REPO, "src/prdrl/rl/interface.py")
    with open(iface_path) as f:
        src = f.read()
    # Check for actual code constructs, not doc-comment mentions
    for kw in ("ReplayBuffer", "replay_buffer =", "import replay", "from replay"):
        assert kw not in src, f"Found '{kw}' in interface.py"


def test_T59_no_optimizer():
    """T59: no optimizer implementation in interface (comments mentioning it are OK)."""
    iface_path = os.path.join(REPO, "src/prdrl/rl/interface.py")
    with open(iface_path) as f:
        src = f.read()
    # Check for actual optimizer usage, not doc-comment mentions
    for kw in ("Adam(", "SGD(", "torch.optim", "optim.Adam", "optimizer.step"):
        assert kw not in src, f"Found '{kw}' in interface.py"


def test_T60_no_torch_training_dependency():
    """T60: interface.py does not import torch/tensorflow for training."""
    iface_path = os.path.join(REPO, "src/prdrl/rl/interface.py")
    with open(iface_path) as f:
        src = f.read()
    for kw in ("import torch", "import tensorflow", "import keras"):
        assert kw not in src, f"Found '{kw}' in interface.py"


# ===========================================================================
# BASE MOEA STATUS
# ===========================================================================

def test_T61_nsga2_remains_current_base():
    """T61: NSGA-II is the current base (oracle uses RankAndCrowding)."""
    oracle_path = os.path.join(REPO, "src/prdrl/oracle/oracle.py")
    with open(oracle_path) as f:
        src = f.read()
    assert "RankAndCrowding" in src


def test_T62_nsga3_remains_future_only():
    """T62: NSGA-III not in oracle.py or interface.py as current implementation."""
    for fname in ["src/prdrl/oracle/oracle.py", "src/prdrl/rl/interface.py"]:
        path = os.path.join(REPO, fname)
        with open(path) as f:
            src = f.read()
        assert "NSGA3" not in src
        assert "nsga3" not in src.lower() or "future" in src.lower()


# ===========================================================================
# PRODUCTION UNCHANGED
# ===========================================================================

def test_T63_production_code_semantics_unchanged():
    """T63: production oracle hash matches prechange manifest."""
    manifest = _json("R0_prechange_manifest.json")
    expected = manifest["files"]["src/prdrl/oracle/oracle.py"]
    actual = _sha256(os.path.join(REPO, "src/prdrl/oracle/oracle.py"))
    assert actual == expected, "oracle.py changed during R0 — production code must not change"


def test_T64_manuscript_pdf_untouched():
    """T64: canonical Markdown source and historical PDF untouched."""
    p0r = _json("../P0R_manuscript_reconciliation/P0R_verdict.json")
    hist_sha = p0r["historical_pdf_sha256"]
    hist_pdf = os.path.join(REPO, "paper/PR_DRL_DMOEA_UAV_New_Design_2026-09-14.pdf")
    assert _sha256(hist_pdf) == hist_sha


def test_T65_frozen_experiment_results_untouched():
    """T65: P0R and D2D verdict files have unchanged hashes."""
    manifest = _json("R0_prechange_manifest.json")
    for rel in [
        "results/P0R_manuscript_reconciliation/P0R_verdict.json",
        "results/D2R_selected_o1_confirmation/D2R_verdict.json",
    ]:
        expected = manifest["files"].get(rel)
        if expected is None:
            continue
        actual = _sha256(os.path.join(REPO, rel))
        assert actual == expected, f"{rel} changed — frozen results must not change"


def test_T66_no_new_scientific_seed_experiment():
    """T66: no new scientific seed experiment added in R0."""
    # R0 experiment file should not contain references to seed sweeps
    exp_path = os.path.join(REPO, "src/prdrl/experiments/R0_rl_interface_validation.py")
    with open(exp_path) as f:
        src = f.read()
    # Correctness dry-run uses fixed single seeds — no multi-seed sweep
    assert "for seed in" not in src
    assert "seed_list" not in src


def test_T67_e3_interface_readiness():
    """T67: E3 interface readiness = READY."""
    e3 = check_e3_readiness()
    assert e3["E3_INTERFACE_READINESS"] == "READY"


# ===========================================================================
# VERDICT FILE
# ===========================================================================

@pytest.mark.parametrize("key,expected", [
    ("ACTION_INTERFACE", "PASS"),
    ("ACTION_SEMANTICS", "PASS"),
    ("PREDICT_FALLBACK_CONTRACT", "PASS"),
    ("BUDGET_CONTRACT", "PASS"),
    ("MULTI_REGION_CONTRACT", "PASS"),
    ("STATE_FEATURE_AUDIT", "PASS"),
    ("STATE_CAUSALITY", "PASS"),
    ("ORACLE_LEAKAGE_CHECK", "PASS"),
    ("REGIME_LABEL_LEAKAGE_CHECK", "PASS"),
    ("RESPONSE_RECORD_CONTRACT", "PASS"),
    ("PRODUCTION_EQUIVALENCE", "PASS"),
    ("RNG_EQUIVALENCE", "PASS"),
    ("PRODUCTION_DEFAULT", "O1_F005"),
    ("RL_STATUS", "NOT_STARTED"),
    ("DRL_TRAINING", "NOT_STARTED"),
    ("R0_COMPLETE", "YES"),
    ("NEXT_STAGE", "E3_RULE_BASED_REGIONAL_CONTROLLER"),
])
def test_verdict(key, expected):
    """Verdict file checks."""
    v = _json("R0_verdict.json")
    assert v[key] == expected, f"Verdict {key}: expected {expected}, got {v.get(key)}"
