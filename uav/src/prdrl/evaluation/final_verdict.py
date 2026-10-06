"""
Effect classification, safety/robustness gates, and scientific verdict logic
for PR-DRL-DMOEA final evaluation (frozen in E4E, consumed unchanged in E4F).

Practical margin DELTA_E4 = 1e-5 (same order used in earlier controller stages).
Positive effect => favorable to the learned DDQN policy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

DELTA_E4 = 1e-5

# The five frozen labels. Their meaning is fixed; the exhaustive extension below
# only assigns the (near-impossible / sub-margin CI) combinations NOT covered by
# the five canonical definitions, without changing any label's meaning.
BENEFICIAL = "BENEFICIAL"
DIRECTIONAL_POSITIVE = "DIRECTIONAL_POSITIVE"
PRACTICALLY_NEUTRAL = "PRACTICALLY_NEUTRAL"
DIRECTIONAL_NEGATIVE = "DIRECTIONAL_NEGATIVE"
HARMFUL = "HARMFUL"

VALIDATED = "VALIDATED"
PARTIAL = "PARTIAL"
FAILED = "FAILED"


def classify_effect(mean: float, lo: float, hi: float, delta: float = DELTA_E4) -> str:
    """Classify an MIGD effect and its 95% CI into one of five frozen labels.

    Canonical definitions (spec E4E-9 EFFECT CLASSIFICATION):
      BENEFICIAL           : mean > +delta AND lo > 0
      DIRECTIONAL_POSITIVE : mean > +delta AND lo <= 0 <= hi
      PRACTICALLY_NEUTRAL  : abs(mean) <= delta AND NOT (hi < 0)   [CI ~ no harm]
      DIRECTIONAL_NEGATIVE : mean < -delta AND lo <= 0 <= hi
      HARMFUL              : mean < -delta AND hi < 0

    Exhaustive deterministic extension for the 3 combinations left uncovered
    (documented BEFORE E4F; meaning of the five labels unchanged):
      * abs(mean) <= delta AND hi < 0
            -> DIRECTIONAL_NEGATIVE  (CI negative-direction, sub-margin magnitude)
      * mean > +delta AND hi < 0    (point/CI contradiction; defer to CI)
            -> DIRECTIONAL_NEGATIVE
      * mean < -delta AND lo > 0    (point/CI contradiction; defer to CI)
            -> DIRECTIONAL_POSITIVE
    """
    straddle = (lo <= 0.0 <= hi)
    ci_neg = (hi < 0.0)
    ci_pos = (lo > 0.0)

    # ---- canonical five ----
    if mean > delta and ci_pos:
        return BENEFICIAL
    if mean > delta and straddle:
        return DIRECTIONAL_POSITIVE
    if abs(mean) <= delta and not ci_neg:
        return PRACTICALLY_NEUTRAL
    if mean < -delta and straddle:
        return DIRECTIONAL_NEGATIVE
    if mean < -delta and ci_neg:
        return HARMFUL

    # ---- exhaustive extension (uncovered combinations) ----
    if abs(mean) <= delta and ci_neg:
        return DIRECTIONAL_NEGATIVE
    if mean > delta and ci_neg:
        return DIRECTIONAL_NEGATIVE
    if mean < -delta and ci_pos:
        return DIRECTIONAL_POSITIVE
    # total fallback (should be unreachable): sub-margin, treat as neutral
    return PRACTICALLY_NEUTRAL


def is_positive_direction(label: str) -> bool:
    return label in (BENEFICIAL, DIRECTIONAL_POSITIVE)


def policy_robustness(per_policy_deltas: Sequence[float], *, threshold: int = 4,
                      margin: float = 0.0) -> Dict:
    """MIXED_POLICY_ROBUSTNESS = PASS iff at least `threshold` of the per-policy
    mean effects are > margin (strictly positive => DDQN better)."""
    n_positive = sum(1 for d in per_policy_deltas if d > margin)
    return {
        "n_policy": len(per_policy_deltas),
        "n_positive": n_positive,
        "threshold": threshold,
        "PASS": n_positive >= threshold,
    }


def safety_gate(response_label: str, epoch_label: str) -> Dict:
    """Scenario safety = PASS iff neither endpoint is HARMFUL (improvement not
    required)."""
    harmful = (response_label == HARMFUL) or (epoch_label == HARMFUL)
    return {
        "response_label": response_label,
        "epoch_label": epoch_label,
        "PASS": not harmful,
    }


def mechanism_decollapse_supported(delta_predict: float, delta_multi: float,
                                   delta_entropy: float) -> Dict:
    """Delta_predict = PREDICT_frac_V1 - PREDICT_frac_V3 (>0 expected: V3 less PREDICT).
    Delta_multi   = multi_action_V3 - multi_action_V1 (>0 expected).
    Delta_entropy = entropy_V3 - entropy_V1 (>0 expected).
    SUPPORTED iff >= 2 of 3 point-estimate effects are in the expected (>0) direction."""
    n = int(delta_predict > 0) + int(delta_multi > 0) + int(delta_entropy > 0)
    return {"delta_predict": delta_predict, "delta_multi": delta_multi,
            "delta_entropy": delta_entropy, "n_expected": n,
            "MECHANISM_DECOLLAPSE_SUPPORTED": "YES" if n >= 2 else "NO"}


def normalization_ab_verdict(
    *,
    mixed_primary_label: str,
    mechanism_supported: bool,
    policy_robustness_pass: bool,
    persistent_safety_pass: bool,
    stationary_safety_pass: bool,
    macro_safety_pass: bool,
    checkpoint_integrity_pass: bool,
    protocol_integrity_pass: bool,
    version_integrity_pass: bool,
    no_final_tuning: bool,
) -> Dict:
    """Frozen E4HT/E4HE success logic for the V1-vs-V3 NORMALIZATION CAUSALITY question.

    Verdicts: NORMALIZATION_CAUSAL_VALIDATED / NORMALIZATION_CAUSAL_PARTIAL
    (subtype PERFORMANCE_PARTIAL or MECHANISTIC_PARTIAL) / NORMALIZATION_CAUSAL_FAILED.
    Mechanism change never rescues harmful performance; a mechanistic win with neutral
    performance is MECHANISTIC_PARTIAL, never VALIDATED.
    """
    VALIDATED = "NORMALIZATION_CAUSAL_VALIDATED"
    PARTIAL = "NORMALIZATION_CAUSAL_PARTIAL"
    FAILED_V = "NORMALIZATION_CAUSAL_FAILED"
    engineering = (checkpoint_integrity_pass and protocol_integrity_pass
                   and version_integrity_pass and no_final_tuning)
    safety_all = persistent_safety_pass and stationary_safety_pass and macro_safety_pass

    # ---- FAILED conditions first ----
    if not engineering:
        return {"verdict": FAILED_V, "subtype": None, "reason": "engineering/integrity failure",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if not safety_all:
        return {"verdict": FAILED_V, "subtype": None, "reason": "safety gate HARMFUL",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label in (HARMFUL, DIRECTIONAL_NEGATIVE):
        return {"verdict": FAILED_V, "subtype": None,
                "reason": f"MIXED primary {mixed_primary_label}",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}
    if mixed_primary_label == PRACTICALLY_NEUTRAL and not mechanism_supported:
        return {"verdict": FAILED_V, "subtype": None,
                "reason": "MIXED neutral and mechanism not supported",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}

    # ---- MECHANISTIC_PARTIAL (neutral performance + mechanism, safe) ----
    if mixed_primary_label == PRACTICALLY_NEUTRAL and mechanism_supported:
        return {"verdict": PARTIAL, "subtype": "MECHANISTIC_PARTIAL",
                "reason": "mechanism de-collapse supported but MIXED performance neutral",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}

    # ---- VALIDATED (beneficial + mechanism + robustness + safety + engineering) ----
    if (mixed_primary_label == BENEFICIAL and mechanism_supported and policy_robustness_pass):
        return {"verdict": VALIDATED, "subtype": None,
                "reason": "beneficial MIXED + mechanism + robustness + safety + integrity",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "SUPPORTED"}

    # ---- PERFORMANCE_PARTIAL (beneficial/directional-positive but a supporting gate fails) ----
    if mixed_primary_label in (BENEFICIAL, DIRECTIONAL_POSITIVE):
        return {"verdict": PARTIAL, "subtype": "PERFORMANCE_PARTIAL",
                "reason": "positive MIXED but a supporting gate (mechanism/robustness) unmet or CI crosses 0",
                "PERFORMANCE_IMPROVEMENT_CLAIM": "PARTIAL"}

    return {"verdict": FAILED_V, "subtype": None, "reason": "unclassified; conservative FAILED",
            "PERFORMANCE_IMPROVEMENT_CLAIM": "NOT_SUPPORTED"}


def scientific_verdict(
    *,
    mixed_primary_label: str,
    ddqn_vs_rule_mixed_label: str,
    mixed_policy_robustness_pass: bool,
    persistent_safety_pass: bool,
    stationary_safety_pass: bool,
    macro_safety_pass: bool,
    checkpoint_integrity_pass: bool,
    protocol_integrity_pass: bool,
    no_training_during_final: bool,
) -> Dict:
    """Freeze the three-way scientific verdict (spec E4E-18).

    VALIDATED requires ALL nine conditions.
    FAILED   if MIXED primary is HARMFUL / DIRECTIONAL_NEGATIVE / PRACTICALLY_NEUTRAL,
             OR any safety gate FAIL, OR integrity failure.
    PARTIAL  is the promising-but-incomplete middle band.
    """
    engineering_ok = (checkpoint_integrity_pass and protocol_integrity_pass
                      and no_training_during_final)

    # ---- FAILED conditions (checked first) ----
    failed = False
    reasons = []
    if mixed_primary_label in (HARMFUL, DIRECTIONAL_NEGATIVE, PRACTICALLY_NEUTRAL):
        failed = True
        reasons.append(f"MIXED_PRIMARY={mixed_primary_label}")
    if not persistent_safety_pass:
        failed = True; reasons.append("PERSISTENT_SAFETY=FAIL")
    if not stationary_safety_pass:
        failed = True; reasons.append("STATIONARY_SAFETY=FAIL")
    if not macro_safety_pass:
        failed = True; reasons.append("MACRO_SAFETY=FAIL")
    if not engineering_ok:
        failed = True; reasons.append("ENGINEERING_INTEGRITY=FAIL")
    if failed:
        return {"verdict": FAILED, "reasons": reasons}

    # ---- VALIDATED conditions ----
    validated = (
        mixed_primary_label == BENEFICIAL
        and ddqn_vs_rule_mixed_label == BENEFICIAL
        and mixed_policy_robustness_pass
        and persistent_safety_pass
        and stationary_safety_pass
        and macro_safety_pass
        and engineering_ok
    )
    if validated:
        return {"verdict": VALIDATED, "reasons": ["all confirmatory + safety gates pass"]}

    # ---- PARTIAL (promising, safe, engineering-ok, but a secondary gate fails) ----
    # reachable here only when no FAILED condition and not full VALIDATED,
    # i.e. MIXED primary is BENEFICIAL or DIRECTIONAL_POSITIVE and no safety FAIL.
    partial_ok = (
        mixed_primary_label in (BENEFICIAL, DIRECTIONAL_POSITIVE)
        and persistent_safety_pass and stationary_safety_pass and macro_safety_pass
        and engineering_ok
    )
    if partial_ok:
        secondary = []
        if ddqn_vs_rule_mixed_label != BENEFICIAL:
            secondary.append(f"DDQN_VS_RULE_MIXED={ddqn_vs_rule_mixed_label}")
        if not mixed_policy_robustness_pass:
            secondary.append("MIXED_POLICY_ROBUSTNESS=FAIL")
        if mixed_primary_label == DIRECTIONAL_POSITIVE:
            secondary.append("MIXED_PRIMARY=DIRECTIONAL_POSITIVE")
        return {"verdict": PARTIAL, "reasons": secondary or ["secondary gate(s) unmet"]}

    # Defensive default (unreachable given the exhaustive FAILED checks above).
    return {"verdict": FAILED, "reasons": ["unclassified; conservative FAILED"]}
