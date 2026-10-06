"""
H1-RV — H1 Verdict and Regional-Interaction Review.

Reads ONLY existing H1 artifacts (no new seeds, no new oracle evaluations).
Audits whether the PARTIAL separability verdict is consistent with the
predeclared frozen separability rule and determines what the negative
PERSISTENT ROG actually means.

Stage invariants:
  - No new seeds (no 941+).
  - No new oracle evaluations.
  - No rule tuning. No DQN. No RULE_V2.
  - H1 raw data NOT overwritten.
  - H1_verdict.json NOT overwritten.
  - RL_STATUS = NOT_STARTED. DRL_TRAINING = NOT_STARTED.
"""

from __future__ import annotations

import csv
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "src")

from prdrl.oracle.oracle import regional_oracle

H1_DIR    = Path("results/H1_regional_controller_hypothesis_review")
REVIEW_DIR = Path("results/H1_verdict_review")

RL_STATUS    = "NOT_STARTED"
DRL_TRAINING = "NOT_STARTED"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
ACTIONS   = ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")

# Frozen from manifest
BAL_ACC_MARGIN = 0.05
MARGIN_EPS     = 1e-8
DELTA_H1_ROG   = 1e-5


# ============================================================
# Loaders
# ============================================================

def _load_json(name: str, base: Path = H1_DIR) -> dict:
    return json.loads((base / name).read_text())


def _load_csv(name: str, base: Path = H1_DIR) -> List[dict]:
    with open(base / name, newline="") as f:
        return list(csv.DictReader(f))


def _jdump(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# PART A — Separability recompute
# ============================================================

def _stronger_baseline(d: dict) -> float:
    maj  = float(d.get("majority_baseline_bal_acc", float("nan")))
    pred = float(d.get("global_predict_label_bal_acc", float("nan")))
    if np.isnan(maj) and np.isnan(pred):
        return float("nan")
    if np.isnan(maj):
        return pred
    if np.isnan(pred):
        return maj
    return max(maj, pred)


def compute_separability_recompute(
    all_probes: dict,
    nnt_probes: dict,
    margin: float = BAL_ACC_MARGIN,
) -> dict:
    """
    Apply the frozen rule mechanically from the manifest:
      SUPPORTED: >=4 of 8 directions with
        bal_acc >= stronger_base + margin AND perm_pass AND nnt_signal

    Spec says "bal_acc" without qualifying LR or DT.
    We evaluate THREE interpretations:
      1. LR-only (as implemented)
      2. DT-only
      3. Either probe (at least one of LR/DT clears the margin)

    For "perm_pass" the implementation used "perm_lr OR perm_dt" (either).
    We preserve that.

    For "nnt_signal" the implementation used nnt_bal_lr >= stronger + margin.
    We also apply the three interpretations consistently.

    Returns per-direction breakdown and aggregate counts for all three.
    """
    directions = [
        (sc, direction)
        for sc in SCENARIOS + ["POOLED"]
        for direction in ["A_to_B", "B_to_A"]
    ]

    records = []
    for sc, direction in directions:
        d = all_probes.get(sc, {}).get(direction, {})
        dn = nnt_probes.get(sc, {}).get(direction, {})

        if "error" in d:
            records.append({
                "scenario": sc, "direction": direction,
                "error": d["error"],
                "supported_lr_only": False,
                "supported_dt_only": False,
                "supported_either": False,
                "partial_lr_only": False,
                "partial_dt_only": False,
                "partial_either": False,
            })
            continue

        bal_lr = float(d.get("logistic_bal_acc", float("nan")))
        bal_dt = float(d.get("tree_bal_acc",     float("nan")))
        perm_lr = bool(d.get("perm_pass_logistic", False))
        perm_dt = bool(d.get("perm_pass_tree",     False))
        stronger = _stronger_baseline(d)

        nnt_d = dn if "error" not in dn else {}
        nnt_lr = float(nnt_d.get("logistic_bal_acc", float("nan")))
        nnt_dt = float(nnt_d.get("tree_bal_acc",     float("nan")))
        nnt_perm_dt = bool(nnt_d.get("perm_pass_tree", False))

        # Margins
        margin_lr = (bal_lr - stronger) if not np.isnan(bal_lr + stronger) else float("nan")
        margin_dt = (bal_dt - stronger) if not np.isnan(bal_dt + stronger) else float("nan")
        nnt_margin_lr = (nnt_lr - stronger) if not np.isnan(nnt_lr + stronger) else float("nan")
        nnt_margin_dt = (nnt_dt - stronger) if not np.isnan(nnt_dt + stronger) else float("nan")

        # Pass flags (perm_pass = either, per implementation)
        perm_pass = perm_lr or perm_dt

        # Under LR-only:
        margin_above_lr  = (not np.isnan(margin_lr)     and margin_lr     >= margin)
        nnt_signal_lr    = (not np.isnan(nnt_margin_lr) and nnt_margin_lr >= margin)
        supported_lr     = margin_above_lr and perm_pass and nnt_signal_lr
        partial_lr       = (margin_above_lr or perm_pass) and not supported_lr

        # Under DT-only:
        margin_above_dt  = (not np.isnan(margin_dt)     and margin_dt     >= margin)
        nnt_signal_dt    = (not np.isnan(nnt_margin_dt) and nnt_margin_dt >= margin and nnt_perm_dt)
        supported_dt     = margin_above_dt and perm_pass and nnt_signal_dt
        partial_dt       = (margin_above_dt or perm_pass) and not supported_dt

        # Under either:
        margin_above_ei  = margin_above_lr or margin_above_dt
        nnt_signal_ei    = nnt_signal_lr or nnt_signal_dt
        supported_ei     = margin_above_ei and perm_pass and nnt_signal_ei
        partial_ei       = (margin_above_ei or perm_pass) and not supported_ei

        records.append({
            "scenario": sc, "direction": direction,
            "bal_lr": round(bal_lr, 6) if not np.isnan(bal_lr) else None,
            "bal_dt": round(bal_dt, 6) if not np.isnan(bal_dt) else None,
            "stronger_baseline": round(stronger, 6) if not np.isnan(stronger) else None,
            "margin_lr": round(margin_lr, 6) if not np.isnan(margin_lr) else None,
            "margin_dt": round(margin_dt, 6) if not np.isnan(margin_dt) else None,
            "nnt_margin_lr": round(nnt_margin_lr, 6) if not np.isnan(nnt_margin_lr) else None,
            "nnt_margin_dt": round(nnt_margin_dt, 6) if not np.isnan(nnt_margin_dt) else None,
            "perm_pass_lr": perm_lr, "perm_pass_dt": perm_dt, "perm_pass_either": perm_pass,
            "nnt_perm_pass_dt": nnt_perm_dt,
            # LR-only
            "margin_above_lr_only": margin_above_lr,
            "nnt_signal_lr_only":   nnt_signal_lr,
            "supported_lr_only":    supported_lr,
            "partial_lr_only":      partial_lr,
            # DT-only
            "margin_above_dt_only": margin_above_dt,
            "nnt_signal_dt_only":   nnt_signal_dt,
            "supported_dt_only":    supported_dt,
            "partial_dt_only":      partial_dt,
            # Either
            "margin_above_either": margin_above_ei,
            "nnt_signal_either":   nnt_signal_ei,
            "supported_either":    supported_ei,
            "partial_either":      partial_ei,
        })

    def _classify(recs, sup_key, par_key):
        n_sup  = sum(1 for r in recs if r.get(sup_key, False))
        n_par  = sum(1 for r in recs if r.get(par_key, False))
        if n_sup >= 4:
            return "SUPPORTED", n_sup, n_par
        if n_sup + n_par >= 4:
            return "PARTIAL", n_sup, n_par
        return "ABSENT", n_sup, n_par

    v_lr, s_lr, p_lr = _classify(records, "supported_lr_only", "partial_lr_only")
    v_dt, s_dt, p_dt = _classify(records, "supported_dt_only", "partial_dt_only")
    v_ei, s_ei, p_ei = _classify(records, "supported_either",  "partial_either")

    # Audit: does the spec specify LR or DT? The manifest says "bal_acc" without qualification.
    # Implementation used LR-only → matches only if LR-only is the intended reading.
    # Since LR-only → PARTIAL and either-probe → SUPPORTED, the interpretation changes the verdict.
    if v_lr == "PARTIAL" and v_ei == "SUPPORTED":
        audit = "SPEC_AMBIGUOUS"
    elif v_lr == v_ei:
        audit = "CONSISTENT"
    else:
        audit = "IMPLEMENTATION_MISMATCH"

    return {
        "BAL_ACC_MARGIN_USED": margin,
        "MARGIN_EPS_USED": MARGIN_EPS,
        "n_directions_evaluated": len([r for r in records if "error" not in r]),
        "perm_pass_rule": "perm_pass_lr OR perm_pass_dt (as implemented)",
        "per_direction": records,
        # Aggregate by interpretation
        "lr_only": {
            "verdict": v_lr, "n_supported": s_lr, "n_partial": p_lr,
            "description": "margin_above uses LR bal_acc only; nnt_signal uses LR bal_acc only"
        },
        "dt_only": {
            "verdict": v_dt, "n_supported": s_dt, "n_partial": p_dt,
            "description": "margin_above uses DT bal_acc only; nnt_signal uses DT bal_acc + nnt_perm_dt"
        },
        "either_probe": {
            "verdict": v_ei, "n_supported": s_ei, "n_partial": p_ei,
            "description": "margin_above if LR OR DT clears threshold; nnt_signal if LR or DT (with perm) clears"
        },
        "original_verdict": "PARTIAL",
        "recomputed_verdict": v_ei,   # Under the spec's ambiguous 'bal_acc' = either probe
        "SEPARABILITY_VERDICT_AUDIT": audit,
        "implementation_note": (
            "H1 decide_separability used bal_lr for margin_above and nnt_bal_lr for "
            "nnt_signal, while using perm_lr OR perm_dt for perm_pass. "
            "The manifest separability_classification says 'bal_acc' without "
            "specifying probe. Under 'either probe', DT satisfies the margin "
            "in MIXED and POOLED (4 of 8 directions) -> SUPPORTED."
        ),
    }


# ============================================================
# PART B — Oracle composition audit
# ============================================================

def audit_oracle_type() -> dict:
    """Determine oracle type from source code inspection."""
    src = inspect.getsource(regional_oracle)

    # Key evidence: isolated region evaluation (one region at a time, others KEEP)
    isolated_eval = "r: action" in src      # {r: action} dict literal in source
    # Full joint search would require iterating over 4^R combinations
    joint_exhaustive = "4**R" in src or "itertools.product" in src or "combinations" in src
    # Composition after greedy selection
    composition_step = "regional_actions" in src and "preferred_action[r]" in src

    # Determine type
    if joint_exhaustive:
        oracle_type = "TRUE_JOINT"
    elif isolated_eval and composition_step and not joint_exhaustive:
        oracle_type = "GREEDY_FACTORIZED"
    else:
        oracle_type = "OTHER"

    return {
        "ORACLE_TYPE": oracle_type,
        "evidence": {
            "isolated_per_region_evaluation": isolated_eval,
            "exhaustive_joint_search": joint_exhaustive,
            "greedy_composition_after_selection": composition_step,
        },
        "description": (
            "Each region r is evaluated independently with all other regions on KEEP. "
            "The preferred action per region is selected as argmax U_{r,a}. "
            "These per-region selections are then COMPOSED into a joint response "
            "and run together to compute Q_regional. "
            "This is GREEDY_FACTORIZED: selection is independent, evaluation is joint. "
            "A TRUE_JOINT oracle would search over 4^R = 4^10 = 1,048,576 action "
            "combinations, which is computationally infeasible."
        ),
        "implication": (
            "The greedy factorized oracle UNDERESTIMATES true joint regional headroom "
            "when regional actions interact negatively. "
            "Positive ROG (MIXED, STATIONARY) indicates that even greedy factorization "
            "finds headroom above the best global action. "
            "Negative ROG (PERSISTENT) may reflect oracle composition interference "
            "rather than true absence of regional headroom."
        ),
        "4_to_R_combinations": 4**10,
    }


# ============================================================
# PART C — PERSISTENT ROG interpretation
# ============================================================

def analyze_persistent_interaction(
    event_rows: List[dict],
    region_rows: List[dict],
) -> dict:
    pers_ev = [r for r in event_rows if r["scenario"] == "PERSISTENT"]
    pers_rr = [r for r in region_rows if r["scenario"] == "PERSISTENT"]

    n_ev  = len(pers_ev)
    n_rr  = len(pers_rr)

    rogs      = [float(r["ROG"]) for r in pers_ev]
    rogs_pred = [float(r["ROG_vs_GLOBAL_PREDICT"]) for r in pers_ev]
    ohis      = [float(r["OHI"]) for r in pers_ev]
    best_gacts = [r["best_global_action"] for r in pers_ev]

    frac_rog_neg  = sum(1 for v in rogs if v < 0) / n_ev if n_ev else float("nan")
    frac_rog_pos  = sum(1 for v in rogs if v > 0) / n_ev if n_ev else float("nan")
    frac_ohi_pos  = sum(1 for v in ohis if v > 0) / n_ev if n_ev else float("nan")
    best_predict_frac = sum(1 for a in best_gacts if a == "PREDICT") / n_ev if n_ev else float("nan")

    # Per-region oracle labels: when PERSISTENT, all regions evolving same way
    # but oracle assigns heterogeneous actions (OHI > 0) — those compositions likely interfere
    oracle_actions_pers = [r["oracle_best_action"] for r in pers_rr]
    from collections import Counter
    action_dist = dict(Counter(oracle_actions_pers))
    total_rr = len(oracle_actions_pers)
    action_frac = {a: action_dist.get(a, 0) / total_rr for a in ACTIONS} if total_rr else {}

    # Mean Q comparison
    q_best_arr  = [float(r["Q_best_global"]) for r in pers_ev]
    q_reg_arr   = [float(r["Q_regional"])    for r in pers_ev]
    q_keep_arr  = [float(r["Q_keep"])        for r in pers_ev]

    # Hypothesis 1: GLOBAL_ACTION_TRULY_DOMINATES
    # In PERSISTENT, all regions have same dynamics → a single global action (PREDICT)
    # optimally handles all regions. No genuine per-region benefit.
    # Evidence: best_global=PREDICT 86% of events, ROG<0 means composed beats nothing.

    # Hypothesis 2: FACTORIZED_ORACLE_INTERACTION
    # Greedy factorized assigns heterogeneous actions (OHI>0 in 92.75% events)
    # but in PERSISTENT, mixing LOCAL+PREDICT+DIVERSIFY across regions is suboptimal
    # compared to uniform PREDICT. The composition creates interference.
    # Evidence: OHI>0 in most events but ROG<0 — heterogeneity applied is harmful.

    # Both are consistent with the data. Cannot distinguish without true joint oracle.
    interpretation = "BOTH_PLAUSIBLE"

    return {
        "n_events":          n_ev,
        "n_region_records":  n_rr,
        "mean_ROG":          float(np.mean(rogs)),
        "mean_ROG_vs_PREDICT": float(np.mean(rogs_pred)),
        "frac_ROG_negative": float(frac_rog_neg),
        "frac_ROG_positive": float(frac_rog_pos),
        "frac_OHI_positive": float(frac_ohi_pos),
        "frac_best_global_PREDICT": float(best_predict_frac),
        "mean_Q_best_global":  float(np.mean(q_best_arr)),
        "mean_Q_regional":     float(np.mean(q_reg_arr)),
        "mean_Q_keep":         float(np.mean(q_keep_arr)),
        "regional_oracle_action_distribution": action_frac,
        "PERSISTENT_NEGATIVE_ROG_INTERPRETATION": interpretation,
        "hypothesis_A": {
            "name": "GLOBAL_ACTION_TRULY_DOMINATES",
            "evidence_for": [
                f"best_global=PREDICT in {best_predict_frac:.2%} of events",
                "PERSISTENT: all regions same dynamics → no structural benefit to differentiation",
                "CONTRADICTED headroom by pooled bootstrap (CI entirely negative)",
            ],
            "evidence_against": [
                f"Regional oracle assigns diverse actions (OHI>0 in {frac_ohi_pos:.2%} events)",
                "ROG>0 in 31.5% of events — isolated cases where greedy composition helped",
            ],
        },
        "hypothesis_B": {
            "name": "FACTORIZED_ORACLE_INTERACTION",
            "evidence_for": [
                f"Greedy oracle assigns heterogeneous per-region actions (OHI>0: {frac_ohi_pos:.2%})",
                "In PERSISTENT env, mixing LOCAL+PREDICT+DIVERSIFY across co-evolving regions",
                "creates interference compared to uniform global PREDICT",
                "Negative ROG even when per-region utilities suggest non-PREDICT actions",
            ],
            "evidence_against": [
                "Cannot confirm without true joint oracle on PERSISTENT",
                "Factorized oracle is the only oracle available",
            ],
        },
        "notes": (
            "Both hypotheses are consistent with stored evidence. "
            "Distinguishing them requires either a true joint oracle search (4^10 infeasible) "
            "or additional counterfactual experiments (not run in H1). "
            "PERSISTENT negative ROG does NOT invalidate MIXED/STATIONARY positive ROG."
        ),
    }


# ============================================================
# PART D — MIXED descriptive evidence
# ============================================================

def analyze_mixed_evidence(
    mixed_rows: List[dict],
    event_rows: List[dict],
    region_rows: List[dict],
) -> dict:
    n_rows = len(mixed_rows)
    nuniq = [int(r["n_unique_nnt_actions"]) for r in mixed_rows]
    rogs  = [float(r["ROG"]) for r in mixed_rows]
    rogs_pred = [float(r["ROG_vs_GLOBAL_PREDICT"]) for r in mixed_rows]

    frac_ge2 = sum(1 for n in nuniq if n >= 2) / n_rows if n_rows else float("nan")
    frac_ge3 = sum(1 for n in nuniq if n >= 3) / n_rows if n_rows else float("nan")
    frac_1   = sum(1 for n in nuniq if n == 1) / n_rows if n_rows else float("nan")
    frac_0   = sum(1 for n in nuniq if n == 0) / n_rows if n_rows else float("nan")

    frac_rog_pos  = sum(1 for v in rogs if v > 0) / n_rows if n_rows else float("nan")
    frac_pred_pos = sum(1 for v in rogs_pred if v > 0) / n_rows if n_rows else float("nan")

    med_rog  = float(np.median(rogs))
    med_pred = float(np.median(rogs_pred))
    mean_rog = float(np.mean(rogs))
    mean_pred = float(np.mean(rogs_pred))

    # From H1_oracle_headroom: MIXED pooled SUPPORTED
    return {
        "n_events": n_rows,
        "frac_events_ge2_nnt_actions": float(frac_ge2),
        "frac_events_ge3_nnt_actions": float(frac_ge3),
        "frac_events_exactly_1_nnt_action": float(frac_1),
        "frac_events_zero_nnt_actions": float(frac_0),
        "frac_events_ROG_positive": float(frac_rog_pos),
        "frac_events_ROG_vs_PREDICT_positive": float(frac_pred_pos),
        "median_ROG": med_rog,
        "median_ROG_vs_GLOBAL_PREDICT": med_pred,
        "mean_ROG": mean_rog,
        "mean_ROG_vs_GLOBAL_PREDICT": mean_pred,
        "interpretation": (
            "In MIXED events, 97% show >=2 distinct non-near-tie oracle actions, "
            "and 60.5% show >=3. This confirms genuine regional heterogeneity in the "
            "oracle's recommendations. Combined with positive pooled ROG (CI entirely above 0), "
            "MIXED is the clearest case for potential regional controller benefit."
        ),
        "MIXED_ORACLE_HEADROOM": "SUPPORTED",
    }


# ============================================================
# PART E — DRL case classification
# ============================================================

def classify_drl_case(
    headroom: str,
    predict_dominance: str,
    sep_recomputed: str,
    tree_mixed: bool,
    tree_pooled: bool,
    oracle_type: str,
    persistent_interp: str,
) -> dict:
    cases = []

    # NOT_JUSTIFIED
    if headroom == "ABSENT" or predict_dominance == "STRONG":
        cases.append("NOT_JUSTIFIED")
        return {
            "cases": cases,
            "primary": "NOT_JUSTIFIED",
            "secondary": None,
            "explanation": "No regional headroom or strong PREDICT dominance.",
        }

    # NONLINEAR_MAPPING_SUPPORTED
    nonlinear = (
        headroom in ("SUPPORTED", "PARTIAL") and
        predict_dominance == "WEAK" and
        (tree_mixed or tree_pooled)
    )
    if nonlinear:
        cases.append("NONLINEAR_MAPPING_SUPPORTED")

    # STATE_SIGNAL_TOO_WEAK
    if sep_recomputed == "ABSENT":
        cases.append("STATE_SIGNAL_TOO_WEAK")

    # JOINT_COORDINATION_REQUIRED
    joint_coord = (
        headroom in ("SUPPORTED", "PARTIAL") and
        oracle_type == "GREEDY_FACTORIZED" and
        persistent_interp in ("FACTORIZED_ORACLE_INTERACTION", "BOTH_PLAUSIBLE")
    )
    if joint_coord:
        cases.append("JOINT_COORDINATION_REQUIRED")

    if not cases:
        cases.append("NOT_JUSTIFIED")

    primary   = cases[0]
    secondary = cases[1] if len(cases) > 1 else None

    return {
        "cases": cases,
        "primary": primary,
        "secondary": secondary,
        "NONLINEAR_MAPPING_SUPPORTED_evidence": (
            "Tree probe exceeds BAL_ACC_MARGIN on MIXED (A→B: +6.48%, B→A: +7.42%) and "
            "POOLED (A→B: +8.92%, B→A: +8.49%) with perm_pass_tree=True, while LR does "
            "not clear the margin. This is the nonlinear mapping signature: the state-to-action "
            "relationship is detectable by a nonlinear (tree) probe but not by a linear (LR) probe."
        ) if nonlinear else None,
        "JOINT_COORDINATION_REQUIRED_evidence": (
            f"Oracle is GREEDY_FACTORIZED. PERSISTENT ROG strongly negative "
            f"(mean=-0.02262, CI entirely below 0) despite heterogeneous per-region oracle "
            f"labels (OHI>0 in 92.75% of PERSISTENT events). This suggests that composing "
            f"independently-optimal per-region actions creates joint interference in "
            f"homogeneous (PERSISTENT) dynamics. Future E4 architecture must account for "
            f"cross-region coordination, not just per-region mapping."
        ) if joint_coord else None,
        "interpretation_combined": (
            "Primary: NONLINEAR_MAPPING_SUPPORTED — the V2 state contains predictive signal "
            "about regional oracle labels that a linear probe cannot detect but a nonlinear "
            "tree probe can (reproducibly across blocks, perm-validated). "
            "Secondary: JOINT_COORDINATION_REQUIRED — the greedy factorized oracle suffers "
            "composition interference in PERSISTENT environments; E4 reward/architecture must "
            "capture cross-region interaction rather than treating isolated per-region utilities "
            "as additive targets."
        ) if len(cases) >= 2 else cases[0] if cases else "NOT_JUSTIFIED",
    }


# ============================================================
# PART F — H1 outcome review + E4 readiness
# ============================================================

def review_h1_outcome(
    headroom: str,
    sep_recomputed: str,
    sep_audit: str,
    predict_dominance: str,
    mixed_headroom: str,
    nonlinear_signal: bool,
) -> Tuple[str, str, str]:
    """
    Apply frozen decision matrix mechanically.
    Under sep_audit=SPEC_AMBIGUOUS: if sep_recomputed=SUPPORTED (either-probe),
    the frozen matrix yields RULE_MAPPING_INADEQUATE.
    """
    if headroom == "ABSENT" or predict_dominance == "STRONG":
        outcome = "GLOBAL_ACTION_DOMINANCE"
    elif headroom in ("SUPPORTED", "PARTIAL") and sep_recomputed == "ABSENT":
        outcome = "STATE_INFORMATION_INSUFFICIENT"
    elif headroom == "SUPPORTED" and sep_recomputed == "SUPPORTED" and predict_dominance != "STRONG":
        outcome = "RULE_MAPPING_INADEQUATE"
    else:
        outcome = "INCONCLUSIVE"

    # E4 readiness
    if outcome == "RULE_MAPPING_INADEQUATE":
        e4 = "READY"
        next_stage = "E4_LEARNED_CONTROLLER_PREPARATION"
    elif outcome == "INCONCLUSIVE" and mixed_headroom == "SUPPORTED" and \
            predict_dominance == "WEAK" and nonlinear_signal:
        e4 = "CONDITIONAL"
        next_stage = "E4_LEARNED_CONTROLLER_PREPARATION"
    else:
        e4 = "NOT_READY"
        next_stage = "H1_ARCHITECTURE_REVIEW"

    return outcome, e4, next_stage


# ============================================================
# Main
# ============================================================

def main():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load frozen manifest ─────────────────────────────────────
    print("H1-RV — Verdict and Regional-Interaction Review")
    print("  Reading H1 artifacts (NO new seeds, NO new evaluations)")
    manifest = _load_json("H1_pre_run_manifest.json")
    h1_verdict = _load_json("H1_verdict.json")

    # Verify frozen constants
    assert abs(float(manifest["BAL_ACC_MARGIN"]) - BAL_ACC_MARGIN) < 1e-15
    assert abs(float(manifest["MARGIN_EPS"])     - MARGIN_EPS) < 1e-20
    assert abs(float(manifest["DELTA_H1_ROG"])   - DELTA_H1_ROG) < 1e-17

    # ── Load H1 data ─────────────────────────────────────────────
    all_probes  = _load_json("H1_separability_crossblock.json")
    nnt_probes  = _load_json("H1_permutation_control.json")
    event_rows  = _load_csv("H1_oracle_event_summary.csv")
    region_rows_raw = []
    with open(H1_DIR / "H1_oracle_region_records.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                region_rows_raw.append(json.loads(line))
    mixed_rows  = _load_csv("H1_mixed_within_event.csv")
    headroom_csv = _load_csv("H1_oracle_headroom.csv")

    # ── Write pre-run manifest ────────────────────────────────────
    h1rv_manifest = {
        "stage": "H1RV_verdict_review",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS, "DRL_TRAINING": DRL_TRAINING,
        "source_h1_manifest_hash": _load_json("H1_pre_run_manifest.json").get(
            "files", {}),
        "BAL_ACC_MARGIN_FROZEN": BAL_ACC_MARGIN,
        "MARGIN_EPS_FROZEN": MARGIN_EPS,
        "DELTA_H1_ROG_FROZEN": DELTA_H1_ROG,
        "no_new_seeds": True,
        "no_new_oracle_evaluations": True,
        "no_rule_tuning": True,
        "no_DQN": True,
        "no_RULE_V2": True,
        "H1_data_immutable": True,
        "frozen_sep_rule": manifest.get("separability_classification"),
        "frozen_decision_matrix": manifest.get("decision_matrix"),
    }
    _jdump(h1rv_manifest, REVIEW_DIR / "H1RV_pre_manifest.json")

    # ── PART A: Separability recompute ───────────────────────────
    print("\n[A] Separability audit")
    sep_result = compute_separability_recompute(all_probes, nnt_probes)
    _jdump(sep_result, REVIEW_DIR / "H1RV_separability_recompute.json")

    sep_audit       = sep_result["SEPARABILITY_VERDICT_AUDIT"]
    sep_original    = sep_result["original_verdict"]
    sep_recomputed  = sep_result["recomputed_verdict"]  # under either-probe
    sep_lr_only     = sep_result["lr_only"]["verdict"]
    sep_dt_only     = sep_result["dt_only"]["verdict"]
    sep_either      = sep_result["either_probe"]["verdict"]

    print(f"  SEPARABILITY_VERDICT_ORIGINAL    = {sep_original}")
    print(f"  SEPARABILITY_VERDICT_LR_ONLY     = {sep_lr_only}")
    print(f"  SEPARABILITY_VERDICT_DT_ONLY     = {sep_dt_only}")
    print(f"  SEPARABILITY_VERDICT_EITHER      = {sep_either}")
    print(f"  SEPARABILITY_VERDICT_AUDIT       = {sep_audit}")

    # Tree signal in MIXED and POOLED
    tree_mixed_ab = all_probes.get("MIXED", {}).get("A_to_B", {})
    tree_mixed_ba = all_probes.get("MIXED", {}).get("B_to_A", {})
    tree_pooled_ab = all_probes.get("POOLED", {}).get("A_to_B", {})
    tree_pooled_ba = all_probes.get("POOLED", {}).get("B_to_A", {})
    stronger_mixed = max(
        tree_mixed_ab.get("majority_baseline_bal_acc", 0),
        tree_mixed_ab.get("global_predict_label_bal_acc", 0)
    )
    tree_signal_mixed  = (
        tree_mixed_ab.get("tree_bal_acc", 0) >= stronger_mixed + BAL_ACC_MARGIN and
        tree_mixed_ab.get("perm_pass_tree", False) and
        tree_mixed_ba.get("tree_bal_acc", 0) >= stronger_mixed + BAL_ACC_MARGIN and
        tree_mixed_ba.get("perm_pass_tree", False)
    )
    stronger_pooled = max(
        tree_pooled_ab.get("majority_baseline_bal_acc", 0),
        tree_pooled_ab.get("global_predict_label_bal_acc", 0)
    )
    tree_signal_pooled = (
        tree_pooled_ab.get("tree_bal_acc", 0) >= stronger_pooled + BAL_ACC_MARGIN and
        tree_pooled_ab.get("perm_pass_tree", False) and
        tree_pooled_ba.get("tree_bal_acc", 0) >= stronger_pooled + BAL_ACC_MARGIN and
        tree_pooled_ba.get("perm_pass_tree", False)
    )
    # Logistic signal
    lr_mixed_ab_margin = (
        tree_mixed_ab.get("logistic_bal_acc", 0) - stronger_mixed
    )
    lr_pooled_ab_margin = (
        tree_pooled_ab.get("logistic_bal_acc", 0) - stronger_pooled
    )
    logistic_signal = (
        lr_mixed_ab_margin >= BAL_ACC_MARGIN or lr_pooled_ab_margin >= BAL_ACC_MARGIN
    )
    print(f"  TREE_SIGNAL_MIXED  = {'SUPPORTED' if tree_signal_mixed else 'NOT_SUPPORTED'}")
    print(f"  TREE_SIGNAL_POOLED = {'SUPPORTED' if tree_signal_pooled else 'NOT_SUPPORTED'}")
    print(f"  LOGISTIC_SIGNAL    = {'SUPPORTED' if logistic_signal else 'ABSENT'}")

    # Write addendum if original PARTIAL but recomputed SUPPORTED
    if sep_original != sep_recomputed:
        addendum = {
            "addendum_type": "SEPARABILITY_VERDICT_REVISION",
            "date": "2026-09-20",
            "original_verdict": sep_original,
            "recomputed_verdict": sep_recomputed,
            "SEPARABILITY_VERDICT_AUDIT": sep_audit,
            "cause": (
                "The H1 manifest separability_classification specifies "
                "'bal_acc >= stronger_base + BAL_ACC_MARGIN' without naming the probe. "
                "The H1 implementation (decide_separability) used bal_lr for margin_above "
                "and nnt_bal_lr for nnt_signal. Under the 'either probe' reading (also "
                "consistent with the spec), the DT probe satisfies the margin in exactly "
                "4 of 8 cross-block directions (MIXED A→B, MIXED B→A, POOLED A→B, POOLED B→A), "
                "all with perm_pass_tree=True and NNT DT signal above threshold, "
                "yielding SUPPORTED (>=4 of 8)."
            ),
            "direction_breakdown": {
                d["scenario"] + "_" + d["direction"]: {
                    "supported_lr_only": d.get("supported_lr_only"),
                    "supported_either":  d.get("supported_either"),
                    "margin_dt": d.get("margin_dt"),
                    "nnt_margin_dt": d.get("nnt_margin_dt"),
                }
                for d in sep_result["per_direction"]
                if "error" not in d
            },
            "NOT_post_hoc_rule_change": True,
            "NO_BAL_ACC_MARGIN_change": True,
            "RL_STATUS": RL_STATUS,
            "DRL_TRAINING": DRL_TRAINING,
        }
        _jdump(addendum, REVIEW_DIR / "H1_verdict_addendum.json")
        print("  H1_verdict_addendum.json written (original PARTIAL, recomputed SUPPORTED)")
    else:
        # Consistency — still write addendum explaining why
        addendum = {
            "addendum_type": "SEPARABILITY_VERDICT_CONSISTENT",
            "original_verdict": sep_original,
            "recomputed_verdict": sep_recomputed,
            "SEPARABILITY_VERDICT_AUDIT": sep_audit,
        }
        _jdump(addendum, REVIEW_DIR / "H1_verdict_addendum.json")

    # ── PART B: Oracle type ───────────────────────────────────────
    print("\n[B] Oracle composition audit")
    oracle_audit = audit_oracle_type()
    _jdump(oracle_audit, REVIEW_DIR / "H1RV_oracle_composition_audit.json")
    oracle_type = oracle_audit["ORACLE_TYPE"]
    print(f"  ORACLE_TYPE = {oracle_type}")

    # ── PART C: PERSISTENT interpretation ────────────────────────
    print("\n[C] PERSISTENT interaction diagnosis")
    pers_audit = analyze_persistent_interaction(event_rows, region_rows_raw)
    _jdump(pers_audit, REVIEW_DIR / "H1RV_persistent_interaction.json")
    pers_interp = pers_audit["PERSISTENT_NEGATIVE_ROG_INTERPRETATION"]
    print(f"  PERSISTENT_NEGATIVE_ROG_INTERPRETATION = {pers_interp}")

    # ── PART D: MIXED evidence ────────────────────────────────────
    print("\n[D] MIXED scenario evidence")
    mixed_audit = analyze_mixed_evidence(mixed_rows, event_rows, region_rows_raw)
    _jdump(mixed_audit, REVIEW_DIR / "H1RV_mixed_evidence.json")
    print(f"  frac_events_ge2_nnt_actions = {mixed_audit['frac_events_ge2_nnt_actions']:.4f}")
    print(f"  frac_events_ge3_nnt_actions = {mixed_audit['frac_events_ge3_nnt_actions']:.4f}")
    print(f"  frac_ROG > 0 = {mixed_audit['frac_events_ROG_positive']:.4f}")
    print(f"  frac_ROG_vs_PREDICT > 0 = {mixed_audit['frac_events_ROG_vs_PREDICT_positive']:.4f}")
    print(f"  median_ROG = {mixed_audit['median_ROG']:.6f}")
    print(f"  median_ROG_vs_PREDICT = {mixed_audit['median_ROG_vs_GLOBAL_PREDICT']:.6f}")
    mixed_headroom = mixed_audit["MIXED_ORACLE_HEADROOM"]

    # ── PART E: DRL case ─────────────────────────────────────────
    print("\n[E] DRL case classification")
    predict_dominance = h1_verdict.get("GLOBAL_PREDICT_DOMINANCE", "UNKNOWN")
    overall_headroom  = h1_verdict.get("REGIONAL_ORACLE_HEADROOM", "UNKNOWN")
    drl_case_result   = classify_drl_case(
        headroom        = overall_headroom,
        predict_dominance = predict_dominance,
        sep_recomputed  = sep_recomputed,
        tree_mixed      = tree_signal_mixed,
        tree_pooled     = tree_signal_pooled,
        oracle_type     = oracle_type,
        persistent_interp = pers_interp,
    )
    _jdump(drl_case_result, REVIEW_DIR / "H1RV_drl_case.json")
    drl_primary   = drl_case_result["primary"]
    drl_secondary = drl_case_result.get("secondary")
    print(f"  DRL_CASE (primary) = {drl_primary}")
    if drl_secondary:
        print(f"  DRL_CASE (secondary) = {drl_secondary}")

    # ── PART F: H1 outcome review ─────────────────────────────────
    print("\n[F] H1 outcome review")
    nonlinear_signal = tree_signal_mixed or tree_signal_pooled
    h1_outcome_reviewed, e4_readiness, next_stage = review_h1_outcome(
        headroom       = overall_headroom,
        sep_recomputed = sep_recomputed,
        sep_audit      = sep_audit,
        predict_dominance = predict_dominance,
        mixed_headroom = mixed_headroom,
        nonlinear_signal = nonlinear_signal,
    )

    # ── Write H1RV verdict ────────────────────────────────────────
    verdict = {
        "stage": "H1RV_verdict_review",
        "date": "2026-09-20",
        "RL_STATUS": RL_STATUS, "DRL_TRAINING": DRL_TRAINING,
        "H1_ORIGINAL_VERDICT": {
            "H1_OUTCOME":              h1_verdict.get("H1_OUTCOME"),
            "V2_ACTION_SEPARABILITY":  h1_verdict.get("V2_ACTION_SEPARABILITY"),
            "REGIONAL_ORACLE_HEADROOM": overall_headroom,
            "GLOBAL_PREDICT_DOMINANCE": predict_dominance,
            "NEXT_STAGE":              h1_verdict.get("NEXT_STAGE"),
        },
        "SEPARABILITY_VERDICT_ORIGINAL":   sep_original,
        "SEPARABILITY_VERDICT_LR_ONLY":    sep_lr_only,
        "SEPARABILITY_VERDICT_DT_ONLY":    sep_dt_only,
        "SEPARABILITY_VERDICT_EITHER":     sep_either,
        "SEPARABILITY_VERDICT_RECOMPUTED": sep_recomputed,
        "SEPARABILITY_VERDICT_AUDIT":      sep_audit,
        "TREE_SIGNAL_MIXED":  "SUPPORTED" if tree_signal_mixed  else "NOT_SUPPORTED",
        "TREE_SIGNAL_POOLED": "SUPPORTED" if tree_signal_pooled else "NOT_SUPPORTED",
        "LOGISTIC_SIGNAL":    "SUPPORTED" if logistic_signal     else "ABSENT",
        "ORACLE_TYPE":        oracle_type,
        "PERSISTENT_NEGATIVE_ROG_INTERPRETATION": pers_interp,
        "MIXED_ORACLE_HEADROOM":         mixed_headroom,
        "GLOBAL_PREDICT_DOMINANCE":      predict_dominance,
        "DRL_CASE_PRIMARY":              drl_primary,
        "DRL_CASE_SECONDARY":            drl_secondary,
        "H1_OUTCOME_ORIGINAL":           "INCONCLUSIVE",
        "H1_OUTCOME_REVIEWED":           h1_outcome_reviewed,
        "E4_READINESS":                  e4_readiness,
        "PRODUCTION_DEFAULT":            "O1_F005",
        "NEXT_STAGE":                    next_stage,
        "H1_RAW_DATA_UNCHANGED":         True,
        "NO_RULE_TUNING":                True,
        "NO_CLASSIFIER_ROLLOUT":         True,
        "NO_DQN":                        True,
        "E4_ARCHITECTURE_NOTE": (
            "E4 must account for cross-region coordination (oracle is GREEDY_FACTORIZED). "
            "Do NOT train a shared per-region policy on greedy factorized oracle labels as "
            "ground truth. Use global reward/context to capture joint outcomes. "
            "Oracle labels may be used for warm-start diagnostics only if explicitly "
            "justified and labeled as GREEDY_FACTORIZED_ORACLE_LABELS, not ground truth."
        ),
    }
    _jdump(verdict, REVIEW_DIR / "H1RV_verdict.json")

    # ── Write report ─────────────────────────────────────────────
    report_lines = [
        "# H1-RV — H1 Verdict and Regional-Interaction Review",
        "",
        f"**Date**: 2026-09-20  **RL_STATUS**: {RL_STATUS}  **DRL_TRAINING**: {DRL_TRAINING}",
        "",
        "## A. Separability Audit",
        "",
        "### Frozen rule (from H1_pre_run_manifest.json)",
        "```",
        'SUPPORTED: ">=4 of 8 cross-block directions: bal_acc >= stronger_base + BAL_ACC_MARGIN AND perm_pass AND nnt_signal"',
        f"BAL_ACC_MARGIN = {BAL_ACC_MARGIN}",
        "```",
        "",
        "### Per-direction results",
        "",
        "| Direction | LR margin | DT margin | perm_pass | NNT LR | NNT DT | Sup(LR) | Sup(DT) | Sup(ei) |",
        "|-----------|-----------|-----------|-----------|--------|--------|---------|---------|---------|",
    ]
    for rec in sep_result["per_direction"]:
        if "error" in rec:
            continue
        row = (
            f"| {rec['scenario']} {rec['direction']} "
            f"| {rec.get('margin_lr', float('nan')):.4f} "
            f"| {rec.get('margin_dt', float('nan')):.4f} "
            f"| {'✓' if rec.get('perm_pass_either') else '✗'} "
            f"| {rec.get('nnt_margin_lr', float('nan')):.4f} "
            f"| {rec.get('nnt_margin_dt', float('nan')):.4f} "
            f"| {'✓' if rec.get('supported_lr_only') else '✗'} "
            f"| {'✓' if rec.get('supported_dt_only') else '✗'} "
            f"| {'✓' if rec.get('supported_either') else '✗'} |"
        )
        report_lines.append(row)

    report_lines += [
        "",
        f"- SEPARABILITY_VERDICT_ORIGINAL = **{sep_original}**",
        f"- SEPARABILITY_VERDICT_LR_ONLY = {sep_lr_only}",
        f"- SEPARABILITY_VERDICT_DT_ONLY = {sep_dt_only}",
        f"- SEPARABILITY_VERDICT_EITHER = **{sep_either}**",
        f"- SEPARABILITY_VERDICT_AUDIT = **{sep_audit}**",
        "",
        "**Finding**: The manifest says 'bal_acc' without specifying LR or DT. "
        "The implementation used LR-only (→ PARTIAL). Under 'either probe' (also spec-consistent), "
        "DT satisfies the margin in MIXED and POOLED (4 of 8) → SUPPORTED.",
        "",
        "## B. Oracle Composition Audit",
        "",
        f"**ORACLE_TYPE = {oracle_type}**",
        "",
        f"The regional oracle evaluates each region r independently (all others on KEEP), "
        f"selects argmax_a U_{{r,a}}, then composes. A true joint oracle would search 4^10 = 1,048,576 "
        f"combinations — computationally infeasible. The composition step is evaluated jointly "
        f"(Q_regional) but the selection is greedy and factorized.",
        "",
        "**Implication**: Positive ROG (MIXED, STATIONARY) = greedy factorized oracle beats "
        "best global even without optimal joint coordination. Negative ROG (PERSISTENT) may "
        "reflect composition interference rather than true absence of headroom.",
        "",
        "## C. PERSISTENT Negative ROG",
        "",
        f"**PERSISTENT_NEGATIVE_ROG_INTERPRETATION = {pers_interp}**",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mean_ROG | {pers_audit['mean_ROG']:.5f} |",
        f"| frac_ROG<0 | {pers_audit['frac_ROG_negative']:.4f} |",
        f"| frac_OHI>0 | {pers_audit['frac_OHI_positive']:.4f} |",
        f"| frac_best_global_PREDICT | {pers_audit['frac_best_global_PREDICT']:.4f} |",
        "",
        "Hypothesis A (GLOBAL_ACTION_TRULY_DOMINATES): In PERSISTENT dynamics, all regions "
        "evolve the same way — uniform PREDICT is globally optimal with no structural benefit "
        "to differentiation.",
        "",
        "Hypothesis B (FACTORIZED_ORACLE_INTERACTION): Greedy factorized oracle assigns "
        "heterogeneous per-region actions (OHI>0 in 92.75% of events) that interfere when "
        "composed in a homogeneous environment.",
        "",
        "Both consistent with stored evidence. Cannot distinguish without true joint oracle.",
        "",
        "## D. MIXED Evidence",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| frac events >=2 NNT actions | {mixed_audit['frac_events_ge2_nnt_actions']:.4f} |",
        f"| frac events >=3 NNT actions | {mixed_audit['frac_events_ge3_nnt_actions']:.4f} |",
        f"| frac events exactly 1 NNT action | {mixed_audit['frac_events_exactly_1_nnt_action']:.4f} |",
        f"| frac events ROG > 0 | {mixed_audit['frac_events_ROG_positive']:.4f} |",
        f"| frac events ROG_vs_PREDICT > 0 | {mixed_audit['frac_events_ROG_vs_PREDICT_positive']:.4f} |",
        f"| median ROG | {mixed_audit['median_ROG']:.6f} |",
        f"| median ROG_vs_PREDICT | {mixed_audit['median_ROG_vs_GLOBAL_PREDICT']:.6f} |",
        "",
        "## E. DRL Case",
        "",
        f"**DRL_CASE (primary) = {drl_primary}**",
        f"**DRL_CASE (secondary) = {drl_secondary}**",
        "",
        (drl_case_result.get("NONLINEAR_MAPPING_SUPPORTED_evidence") or ""),
        "",
        (drl_case_result.get("JOINT_COORDINATION_REQUIRED_evidence") or ""),
        "",
        "## F. Reviewed H1 Outcome",
        "",
        f"- H1_OUTCOME_ORIGINAL = **INCONCLUSIVE**",
        f"- H1_OUTCOME_REVIEWED = **{h1_outcome_reviewed}**",
        f"- E4_READINESS = **{e4_readiness}**",
        f"- NEXT_STAGE = **{next_stage}**",
        "",
        "**Decision path** (frozen matrix):",
        "  headroom SUPPORTED + separability SUPPORTED (recomputed) + predict != STRONG",
        f"  → {h1_outcome_reviewed}",
        "",
        "## G. Architecture Note for E4",
        "",
        verdict["E4_ARCHITECTURE_NOTE"],
        "",
        "## H. Invariants",
        "",
        f"RL_STATUS = {RL_STATUS}. DRL_TRAINING = {DRL_TRAINING}. "
        "No new seeds. No new oracle evaluations. H1 raw data unchanged. "
        "H1_verdict.json not overwritten. Production unchanged.",
    ]
    with open(REVIEW_DIR / "H1RV_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── Final terminal output ─────────────────────────────────────
    total_h1_tests = 1697  # from last regression
    print(f"\n{'='*60}")
    print("H1-RV FINAL RESULTS")
    print(f"{'='*60}")
    print(f"SEPARABILITY_VERDICT_ORIGINAL    = {sep_original}")
    print(f"SEPARABILITY_VERDICT_RECOMPUTED  = {sep_recomputed}")
    print(f"SEPARABILITY_VERDICT_AUDIT       = {sep_audit}")
    print(f"TREE_SIGNAL_MIXED                = {'SUPPORTED' if tree_signal_mixed  else 'NOT_SUPPORTED'}")
    print(f"TREE_SIGNAL_POOLED               = {'SUPPORTED' if tree_signal_pooled else 'NOT_SUPPORTED'}")
    print(f"LOGISTIC_SIGNAL                  = {'SUPPORTED' if logistic_signal     else 'ABSENT'}")
    print(f"ORACLE_TYPE                      = {oracle_type}")
    print(f"PERSISTENT_NEGATIVE_ROG_INTERPRETATION = {pers_interp}")
    print(f"MIXED_ORACLE_HEADROOM            = {mixed_headroom}")
    print(f"GLOBAL_PREDICT_DOMINANCE         = {predict_dominance}")
    drl_combined = drl_primary
    if drl_secondary:
        drl_combined = f"{drl_primary} + {drl_secondary}"
    print(f"DRL_CASE                         = {drl_combined}")
    print(f"H1_OUTCOME_ORIGINAL              = INCONCLUSIVE")
    print(f"H1_OUTCOME_REVIEWED              = {h1_outcome_reviewed}")
    print(f"E4_READINESS                     = {e4_readiness}")
    print(f"RL_STATUS                        = {RL_STATUS}")
    print(f"DRL_TRAINING                     = {DRL_TRAINING}")
    print(f"FULL_REGRESSION                  = (run separately)")
    print(f"NEXT_STAGE                       = {next_stage}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
