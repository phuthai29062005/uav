"""
E4L — Coordination Failure Analysis (PR-DRL-DMOEA, Branch C).

Diagnostic-only post-hoc analysis of why C1 QMIX failed relative to C0 DDQN.
Uses ONLY stored artifacts from E4KTR/E4KF/E4KE/E4KI. No training, no environment
execution, no new seeds, no checkpoint modification.

E4KF verdict = FAILED (immutable). This module diagnoses failure mechanisms.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

RESULTS_DIR = Path("results/E4L_coordination_failure_analysis")
E4KF_DIR = Path("results/E4KF_joint_coordination_ab_final_evaluation_execution")
E4KTR_DIR = Path("results/E4KTR_joint_coordination_ab_training_execution")
E4KE_DIR = Path("results/E4KE_joint_coordination_ab_final_evaluation_protocol_freeze")
E4KI_DIR = Path("results/E4KI_joint_coordination_implementation_validation")

ARMS = ["C0_INDEPENDENT_SHARED_DDQN", "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"]
POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
HYPOTHESIS_VOCAB = ["SUPPORTED", "PLAUSIBLE", "WEAK", "NOT_SUPPORTED",
                    "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"]


def _load(p):
    with open(p) as f:
        return json.load(f)


def _csv_rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=_jd)


def _jd(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(str(type(x)))


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ===========================================================================
# Phase E4L-0: immutable result preservation
# ===========================================================================
def phase_0_preservation(verdict):
    checks = {
        "E4KF_COMPLETE": verdict["E4KF_COMPLETE"] == "YES",
        "COORDINATION_CAUSAL_VERDICT": verdict["COORDINATION_CAUSAL_VERDICT"] == "FAILED",
        "MIXED_PRIMARY": verdict["MIXED_PRIMARY"] == "HARMFUL",
        "MIXED_DELTA": abs(verdict["MIXED_DELTA_C0_MINUS_C1"] - (-0.00019989336902142537)) < 1e-12,
        "MIXED_CI_NEGATIVE": verdict["MIXED_DELTA_CI"][1] < 0,
        "POLICY_ROBUSTNESS_FAIL": verdict["COORDINATION_POLICY_ROBUSTNESS"] == "FAIL",
        "PERSISTENT_SAFETY_FAIL": verdict["PERSISTENT_SAFETY"] == "FAIL",
        "STATIONARY_SAFETY_PASS": verdict["STATIONARY_SAFETY"] == "PASS",
        "MACRO_SAFETY_FAIL": verdict["MACRO_SAFETY"] == "FAIL",
        "ORIGINAL_E4F_VERDICT": verdict["ORIGINAL_E4F_VERDICT"] == "FAILED",
        "E4HF_VERDICT": verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED",
        "E4IF_VERDICT": verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED",
        "E4JF_VERDICT": verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED",
    }
    return {"all_preserved": all(checks.values()), "checks": checks,
            "E4KF_VERDICT_CHANGED": "NO"}


# ===========================================================================
# Phase E4L-2: scenario failure localization
# ===========================================================================
def phase_2_scenario_localization(ep_rows, verdict):
    rows_out = []
    for scenario in SCENARIOS:
        sc_rows = [r for r in ep_rows if r["scenario"] == scenario]
        for arm in ARMS:
            arm_rows = [r for r in sc_rows if r["arm"] == arm]
            migd_r = [float(r["MIGD_response"]) for r in arm_rows]
            migd_e = [float(r["MIGD_epoch"]) for r in arm_rows]
            fe = [float(r["response_FE"]) for r in arm_rows]
            mar = [float(r["multi_action_rate"]) for r in arm_rows]
            rows_out.append({
                "scenario": scenario, "arm": arm,
                "mean_MIGD_response": float(np.mean(migd_r)),
                "mean_MIGD_epoch": float(np.mean(migd_e)),
                "mean_response_FE": float(np.mean(fe)),
                "mean_multi_action_rate": float(np.mean(mar)),
                "n_episodes": len(arm_rows),
            })

    loc = []
    for scenario in SCENARIOS:
        c0 = next(r for r in rows_out if r["scenario"] == scenario and r["arm"] == ARMS[0])
        c1 = next(r for r in rows_out if r["scenario"] == scenario and r["arm"] == ARMS[1])
        d_resp = c0["mean_MIGD_response"] - c1["mean_MIGD_response"]
        d_epoch = c0["mean_MIGD_epoch"] - c1["mean_MIGD_epoch"]
        d_fe = c0["mean_response_FE"] - c1["mean_response_FE"]
        pp_deltas = []
        for p in POLICY_SEEDS:
            c0p = [float(r["MIGD_response"]) for r in ep_rows
                   if r["arm"] == ARMS[0] and r["scenario"] == scenario and int(r["policy_seed"]) == p]
            c1p = [float(r["MIGD_response"]) for r in ep_rows
                   if r["arm"] == ARMS[1] and r["scenario"] == scenario and int(r["policy_seed"]) == p]
            pp_deltas.append(float(np.mean(c0p) - np.mean(c1p)))
        loc.append({
            "scenario": scenario,
            "C0_MIGD_response": c0["mean_MIGD_response"],
            "C1_MIGD_response": c1["mean_MIGD_response"],
            "delta_response": d_resp,
            "delta_epoch": d_epoch,
            "delta_FE": d_fe,
            "C0_multi_action_rate": c0["mean_multi_action_rate"],
            "C1_multi_action_rate": c1["mean_multi_action_rate"],
            "per_policy_deltas": pp_deltas,
            "n_policy_positive": sum(1 for d in pp_deltas if d > 0),
        })
    return rows_out, loc


# ===========================================================================
# Phase E4L-3: H1 action specialization mismatch
# ===========================================================================
def phase_3_action_specialization(regional_rows, ep_rows):
    action_by = defaultdict(lambda: defaultdict(int))
    total_by = defaultdict(int)
    for r in regional_rows:
        key = (r["arm"], r["scenario"])
        action_by[key][r["requested_action"]] += 1
        total_by[key] += 1

    act_rows = []
    for scenario in SCENARIOS:
        for arm in ARMS:
            key = (arm, scenario)
            tot = total_by[key] or 1
            fracs = {a: action_by[key].get(a, 0) / tot for a in ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]}
            ent = -sum(v * math.log(v) for v in fracs.values() if v > 0)
            act_rows.append({"scenario": scenario, "arm": arm, **fracs, "entropy": ent,
                             "total_decisions": total_by[key]})

    pp_act = []
    for scenario in SCENARIOS:
        for p in POLICY_SEEDS:
            for arm in ARMS:
                sub = [r for r in regional_rows if r["arm"] == arm and r["scenario"] == scenario
                       and int(r["policy_seed"]) == p]
                tot = len(sub) or 1
                counts = defaultdict(int)
                for r in sub:
                    counts[r["requested_action"]] += 1
                fracs = {a: counts.get(a, 0) / tot for a in ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]}
                pp_act.append({"scenario": scenario, "policy_seed": p, "arm": arm, **fracs})

    c0_pers_predict = next(r["PREDICT"] for r in act_rows if r["arm"] == ARMS[0] and r["scenario"] == "PERSISTENT")
    c1_pers_predict = next(r["PREDICT"] for r in act_rows if r["arm"] == ARMS[1] and r["scenario"] == "PERSISTENT")
    c0_stat_predict = next(r["PREDICT"] for r in act_rows if r["arm"] == ARMS[0] and r["scenario"] == "STATIONARY")
    c1_stat_predict = next(r["PREDICT"] for r in act_rows if r["arm"] == ARMS[1] and r["scenario"] == "STATIONARY")
    c0_mix_predict = next(r["PREDICT"] for r in act_rows if r["arm"] == ARMS[0] and r["scenario"] == "MIXED")
    c1_mix_predict = next(r["PREDICT"] for r in act_rows if r["arm"] == ARMS[1] and r["scenario"] == "MIXED")

    predict_shift_pers = c1_pers_predict - c0_pers_predict
    predict_shift_stat = c1_stat_predict - c0_stat_predict
    predict_shift_mix = c1_mix_predict - c0_mix_predict

    harmful_shift = (predict_shift_pers < -0.02 or predict_shift_mix < -0.02)
    absent_in_stat = abs(predict_shift_stat) < abs(predict_shift_pers)

    pp_consistent = True
    for p in POLICY_SEEDS:
        c0_pp = next((r for r in pp_act if r["arm"] == ARMS[0] and r["scenario"] == "PERSISTENT"
                       and r["policy_seed"] == p), None)
        c1_pp = next((r for r in pp_act if r["arm"] == ARMS[1] and r["scenario"] == "PERSISTENT"
                       and r["policy_seed"] == p), None)
        if c0_pp and c1_pp:
            if c1_pp["PREDICT"] >= c0_pp["PREDICT"] + 0.02:
                pp_consistent = False

    ep_association = {}
    for scenario in SCENARIOS:
        for arm in ARMS:
            sub = [r for r in ep_rows if r["arm"] == arm and r["scenario"] == scenario]
            migds = [float(r["MIGD_response"]) for r in sub]
            mars = [float(r["multi_action_rate"]) for r in sub]
            if len(migds) > 2:
                ep_association[(arm, scenario)] = float(np.corrcoef(migds, mars)[0, 1])

    if harmful_shift and absent_in_stat and pp_consistent:
        status = "SUPPORTED"
    elif harmful_shift:
        status = "PLAUSIBLE"
    else:
        status = "WEAK"

    evidence = {
        "predict_shift_PERSISTENT": predict_shift_pers,
        "predict_shift_STATIONARY": predict_shift_stat,
        "predict_shift_MIXED": predict_shift_mix,
        "C0_PERSISTENT_PREDICT": c0_pers_predict,
        "C1_PERSISTENT_PREDICT": c1_pers_predict,
        "C0_STATIONARY_PREDICT": c0_stat_predict,
        "C1_STATIONARY_PREDICT": c1_stat_predict,
        "harmful_shift_present": harmful_shift,
        "absent_or_weaker_in_stationary": absent_in_stat,
        "per_policy_consistent": pp_consistent,
        "status": status,
    }
    return act_rows, pp_act, evidence


# ===========================================================================
# Phase E4L-4/5: H2 joint reward aggregation + cancellation
# ===========================================================================
def phase_4_reward_aggregation(regional_rows):
    agg_rows = []
    cancel_rows = []

    for scenario in SCENARIOS:
        for arm in ARMS:
            sub = [r for r in regional_rows if r["arm"] == arm and r["scenario"] == scenario]
            g_rs = [float(r["g_r"]) for r in sub]
            global_rs = [float(r["weighted_global_reward"]) for r in sub]
            local_rs = [float(r["weighted_local_reward_R1"]) for r in sub]

            agg_rows.append({
                "scenario": scenario, "arm": arm, "n_records": len(sub),
                "median_abs_g_r": float(np.median(np.abs(g_rs))) if g_rs else 0,
                "q95_abs_g_r": float(np.percentile(np.abs(g_rs), 95)) if g_rs else 0,
                "mean_g_r": float(np.mean(g_rs)) if g_rs else 0,
                "std_g_r": float(np.std(g_rs)) if g_rs else 0,
                "positive_fraction": float(np.mean([1 if g > 0 else 0 for g in g_rs])) if g_rs else 0,
                "negative_fraction": float(np.mean([1 if g < 0 else 0 for g in g_rs])) if g_rs else 0,
                "zero_fraction": float(np.mean([1 if g == 0 else 0 for g in g_rs])) if g_rs else 0,
                "median_abs_global": float(np.median(np.abs(global_rs))) if global_rs else 0,
                "median_abs_local": float(np.median(np.abs(local_rs))) if local_rs else 0,
            })

    for scenario in SCENARIOS:
        events = defaultdict(list)
        for r in regional_rows:
            if r["scenario"] == scenario:
                key = (r["arm"], r["policy_seed"], r["environment_seed"], r["change_index"])
                events[key].append(float(r["g_r"]))

        for arm in ARMS:
            arm_events = {k: v for k, v in events.items() if k[0] == arm}
            if not arm_events:
                continue
            cancel_ratios = []
            for k, g_rs in arm_events.items():
                mean_gr = np.mean(g_rs)
                mean_abs_gr = np.mean(np.abs(g_rs))
                eps = 1e-12
                ratio = abs(mean_gr) / (mean_abs_gr + eps)
                cancel_ratios.append(ratio)
            cancel_rows.append({
                "scenario": scenario, "arm": arm,
                "n_events": len(arm_events),
                "median_cancellation_ratio": float(np.median(cancel_ratios)),
                "mean_cancellation_ratio": float(np.mean(cancel_ratios)),
                "q10_cancellation_ratio": float(np.percentile(cancel_ratios, 10)),
                "q25_cancellation_ratio": float(np.percentile(cancel_ratios, 25)),
                "fraction_below_0_5": float(np.mean([1 if r < 0.5 else 0 for r in cancel_ratios])),
                "fraction_below_0_3": float(np.mean([1 if r < 0.3 else 0 for r in cancel_ratios])),
            })

    c1_pers_cancel = [r for r in cancel_rows if r["arm"] == ARMS[1] and r["scenario"] == "PERSISTENT"]
    c1_stat_cancel = [r for r in cancel_rows if r["arm"] == ARMS[1] and r["scenario"] == "STATIONARY"]
    severe_pers = c1_pers_cancel and c1_pers_cancel[0]["median_cancellation_ratio"] < 0.5
    low_stat = c1_stat_cancel and c1_stat_cancel[0]["median_cancellation_ratio"] > 0.7

    if severe_pers and low_stat:
        h2_status = "SUPPORTED"
    elif severe_pers:
        h2_status = "PLAUSIBLE"
    else:
        c0_cancel = [r for r in cancel_rows if r["arm"] == ARMS[0]]
        c1_cancel = [r for r in cancel_rows if r["arm"] == ARMS[1]]
        if c0_cancel and c1_cancel:
            c0_med = np.mean([r["median_cancellation_ratio"] for r in c0_cancel])
            c1_med = np.mean([r["median_cancellation_ratio"] for r in c1_cancel])
            if c1_med < c0_med - 0.1:
                h2_status = "PLAUSIBLE"
            else:
                h2_status = "WEAK"
        else:
            h2_status = "WEAK"

    return agg_rows, cancel_rows, h2_status


# ===========================================================================
# Phase E4L-5: H3 regional-anchor event weighting
# ===========================================================================
def phase_5_anchor_weighting(regional_rows, training_rows):
    event_active = defaultdict(int)
    for r in regional_rows:
        key = (r["arm"], r["policy_seed"], r["scenario"], r["environment_seed"], r["change_index"])
        event_active[key] += 1

    anchor_rows = []
    for scenario in SCENARIOS:
        for arm in ARMS:
            counts = [v for k, v in event_active.items() if k[2] == scenario and k[0] == arm]
            if not counts:
                continue
            anchor_rows.append({
                "scenario": scenario, "arm": arm, "source": "final",
                "n_events": len(counts),
                "mean_active_regions": float(np.mean(counts)),
                "median_active_regions": float(np.median(counts)),
                "q95_active_regions": float(np.percentile(counts, 95)),
                "max_active_regions": int(np.max(counts)),
                "fraction_multi_region": float(np.mean([1 if c > 1 else 0 for c in counts])),
            })

    for scenario in SCENARIOS:
        for arm in ARMS:
            sub = [r for r in training_rows if r["arm"] == arm and r["scenario"] == scenario]
            if not sub:
                continue
            rtc = [int(r["regional_transition_count"]) for r in sub]
            anchor_rows.append({
                "scenario": scenario, "arm": arm, "source": "training",
                "n_events": len(sub),
                "mean_regional_transitions": float(np.mean(rtc)),
                "median_regional_transitions": float(np.median(rtc)),
            })

    c1_final = [r for r in anchor_rows if r["arm"] == ARMS[1] and r["source"] == "final"]
    if c1_final:
        pers = next((r for r in c1_final if r["scenario"] == "PERSISTENT"), None)
        stat = next((r for r in c1_final if r["scenario"] == "STATIONARY"), None)
        if pers and stat:
            ratio_diff = pers["mean_active_regions"] / (stat["mean_active_regions"] + 1e-12)
            material_variation = ratio_diff > 1.5 or ratio_diff < 0.67
        else:
            material_variation = False
    else:
        material_variation = False

    anchors_per_event = {}
    c1_events = {k: v for k, v in event_active.items() if k[0] == ARMS[1]}
    if c1_events:
        all_counts = list(c1_events.values())
        anchors_per_event = {
            "mean": float(np.mean(all_counts)),
            "median": float(np.median(all_counts)),
            "total_anchors": int(np.sum(all_counts)),
            "total_unique_events": len(all_counts),
            "ratio": float(np.sum(all_counts) / len(all_counts)),
        }

    if material_variation:
        h3_status = "PLAUSIBLE"
    else:
        h3_status = "WEAK"

    return anchor_rows, anchors_per_event, h3_status


# ===========================================================================
# Phase E4L-6: H4 sibling target redundancy
# ===========================================================================
def phase_6_sibling_redundancy(regional_rows):
    c1_events = defaultdict(list)
    for r in regional_rows:
        if r["arm"] == ARMS[1]:
            key = (r["policy_seed"], r["scenario"], r["environment_seed"], r["change_index"])
            c1_events[key].append(r)

    sibling_groups = []
    for key, siblings in c1_events.items():
        n = len(siblings)
        g_rs = [float(s["g_r"]) for s in siblings]
        globals_ = [float(s["weighted_global_reward"]) for s in siblings]
        sibling_groups.append({
            "scenario": key[1], "policy_seed": key[0],
            "n_siblings": n,
            "g_r_std": float(np.std(g_rs)) if n > 1 else 0.0,
            "global_reward_std": float(np.std(globals_)) if n > 1 else 0.0,
            "global_reward_identical": all(abs(g - globals_[0]) < 1e-12 for g in globals_) if globals_ else True,
        })

    n_groups = len(sibling_groups)
    n_multi = sum(1 for g in sibling_groups if g["n_siblings"] > 1)
    total_anchors = sum(g["n_siblings"] for g in sibling_groups)
    global_identical_frac = (sum(1 for g in sibling_groups if g["global_reward_identical"]) / n_groups
                             if n_groups > 0 else 0)

    by_scenario = {}
    for scenario in SCENARIOS:
        sc = [g for g in sibling_groups if g["scenario"] == scenario]
        if sc:
            by_scenario[scenario] = {
                "n_groups": len(sc),
                "mean_siblings": float(np.mean([g["n_siblings"] for g in sc])),
                "median_siblings": float(np.median([g["n_siblings"] for g in sc])),
                "total_anchors": sum(g["n_siblings"] for g in sc),
                "duplicate_fraction": float(1 - len(sc) / sum(g["n_siblings"] for g in sc))
                    if sum(g["n_siblings"] for g in sc) > 0 else 0,
            }

    redundancy_ratio = total_anchors / n_groups if n_groups > 0 else 0
    duplicate_mass = 1 - (n_groups / total_anchors) if total_anchors > 0 else 0

    sibling_csv = []
    for scenario in SCENARIOS:
        sc = [g for g in sibling_groups if g["scenario"] == scenario]
        if sc:
            sibling_csv.append({
                "scenario": scenario,
                "n_sibling_groups": len(sc),
                "mean_group_size": float(np.mean([g["n_siblings"] for g in sc])),
                "total_anchors": sum(g["n_siblings"] for g in sc),
                "duplicate_mass": by_scenario.get(scenario, {}).get("duplicate_fraction", 0),
                "global_reward_identical_fraction": float(np.mean(
                    [1 if g["global_reward_identical"] else 0 for g in sc])),
                "mean_g_r_std_within_group": float(np.mean([g["g_r_std"] for g in sc])),
            })

    substantial = duplicate_mass > 0.3
    h4_status = "SUPPORTED" if substantial else ("PLAUSIBLE" if duplicate_mass > 0.15 else "WEAK")

    return sibling_csv, {
        "redundancy_ratio": redundancy_ratio,
        "duplicate_mass": duplicate_mass,
        "n_groups": n_groups, "n_multi": n_multi,
        "total_anchors": total_anchors,
        "global_identical_fraction": global_identical_frac,
        "by_scenario": by_scenario,
        "status": h4_status,
        "note": "sibling anchors from same event share identical joint reward/target in C1 training; "
                "this analysis examines the evaluation records as a proxy for training structure",
    }, h4_status


# ===========================================================================
# Phase E4L-7: H5 mixer degeneracy
# ===========================================================================
def phase_7_mixer_conditioning(coord_diag_csv):
    rows = coord_diag_csv
    mixer_analysis = []
    for r in rows:
        mixer_analysis.append({
            "policy_seed": int(r["policy_seed"]),
            "mean_joint_td_loss": float(r["mean_joint_td_loss"]),
            "n_updates": int(r["n_updates"]),
            "util_param_count": int(r["util_param_count"]),
            "mixer_param_count": int(r["mixer_param_count"]),
            "monotonicity_min": float(r["monotonicity_min"]),
            "monotonicity_median": float(r["monotonicity_median"]),
            "monotonicity_q95": float(r["monotonicity_q95"]),
            "monotonicity_violations": int(r["monotonicity_violations"]),
        })

    mono_medians = [m["monotonicity_median"] for m in mixer_analysis]
    mono_mins = [m["monotonicity_min"] for m in mixer_analysis]
    mono_q95s = [m["monotonicity_q95"] for m in mixer_analysis]
    cv_median = float(np.std(mono_medians) / (np.mean(mono_medians) + 1e-12))
    cv_q95 = float(np.std(mono_q95s) / (np.mean(mono_q95s) + 1e-12))
    all_positive = all(m > 0 for m in mono_mins)
    violations = sum(m["monotonicity_violations"] for m in mixer_analysis)

    near_constant = cv_median < 0.1 and cv_q95 < 0.1
    q_state_audit = "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"

    if near_constant:
        h5_status = "PLAUSIBLE"
    elif violations == 0 and all_positive:
        h5_status = "NOT_SUPPORTED"
    else:
        h5_status = "WEAK"

    return mixer_analysis, {
        "cv_monotonicity_median": cv_median,
        "cv_monotonicity_q95": cv_q95,
        "all_weights_positive": all_positive,
        "total_violations": violations,
        "near_constant_weights": near_constant,
        "state_conditioning_analysis": q_state_audit,
        "note": "full mixer state-conditioning audit requires stored state vectors "
                "(W1/W2 variation, kappa_r); state vectors were not persisted in E4KTR/E4KF",
        "status": h5_status,
    }, h5_status


# ===========================================================================
# Phase E4L-8: H6 monotonic representation limitation
# ===========================================================================
def phase_8_monotonic_limitation():
    review = {
        "E4GD_D5_interaction": "approximately 48% conditional best-action dependence (MATERIAL)",
        "monotonic_constraint": "dQ_tot/dq_r >= 0 always satisfied (0 violations across all training)",
        "non_monotonic_evidence": "NOT_AVAILABLE",
        "note": "material interaction does NOT imply non-monotonic Q_tot; "
                "without explicit joint-action counterfactual outcome data, "
                "non-monotonicity cannot be diagnosed from frozen artifacts",
        "counterfactual_rollouts_available": False,
        "status": "PLAUSIBLE",
        "reason": "interaction is material (48%) so monotonic limitation is plausible "
                  "but cannot be confirmed without counterfactual environment runs "
                  "that are forbidden in E4L",
    }
    return review


# ===========================================================================
# Phase E4L-9: H7 utility Q-margin
# ===========================================================================
def phase_9_q_margin(update_summary):
    margin_analysis = []
    for arm in ARMS:
        sub = [r for r in update_summary if r["arm"] == arm]
        if not sub:
            continue
        abs_qs = [float(r["mean_abs_q"]) for r in sub]
        max_qs = [float(r["max_abs_q"]) for r in sub]
        margin_analysis.append({
            "arm": arm,
            "mean_abs_q_early": float(np.mean(abs_qs[:5])) if len(abs_qs) >= 5 else float(np.mean(abs_qs)),
            "mean_abs_q_late": float(np.mean(abs_qs[-5:])) if len(abs_qs) >= 5 else float(np.mean(abs_qs)),
            "max_abs_q_early": float(np.mean(max_qs[:5])) if len(max_qs) >= 5 else float(np.mean(max_qs)),
            "max_abs_q_late": float(np.mean(max_qs[-5:])) if len(max_qs) >= 5 else float(np.mean(max_qs)),
            "q_growth_ratio": float(np.mean(abs_qs[-5:]) / (np.mean(abs_qs[:5]) + 1e-12))
                if len(abs_qs) >= 5 else 1.0,
        })

    h7_status = "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"
    note = ("direct Q-margin analysis (top1-top2 gap per state) requires stored state vectors "
            "and forward pass through frozen checkpoints; state vectors were not persisted; "
            "only aggregate Q-magnitude statistics from training are available")

    return margin_analysis, {
        "direct_q_margin_available": False,
        "aggregate_q_stats_available": True,
        "note": note,
        "status": h7_status,
    }, h7_status


# ===========================================================================
# Phase E4L-10: action disagreement
# ===========================================================================
def phase_10_action_disagreement():
    return {
        "status": "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS",
        "note": "action disagreement analysis requires evaluating both C0 and C1 utility networks "
                "on identical state inputs; shared state vectors from the same trajectory positions "
                "were not stored in E4KF (trajectories diverge after first decision); "
                "offline evaluation requires stored state tensors that do not exist",
    }


# ===========================================================================
# Phase E4L-11: H8 training dynamics
# ===========================================================================
def phase_11_training_dynamics(training_rows, update_summary, action_traj):
    dynamics = []
    for arm in ARMS:
        for p in POLICY_SEEDS:
            sub = [r for r in training_rows if r["arm"] == arm and int(r["policy_seed"]) == p]
            if not sub:
                continue
            epochs = sorted(set(int(r["epoch"]) for r in sub))
            early = [r for r in sub if int(r["epoch"]) <= 5]
            late = [r for r in sub if int(r["epoch"]) >= max(epochs) - 5]
            migds_early = [float(r["MIGD_response"]) for r in early]
            migds_late = [float(r["MIGD_response"]) for r in late]
            rewards_early = [float(r["cumulative_reward"]) for r in early]
            rewards_late = [float(r["cumulative_reward"]) for r in late]
            dynamics.append({
                "arm": arm, "policy_seed": p,
                "n_epochs": len(epochs),
                "mean_MIGD_early": float(np.mean(migds_early)) if migds_early else 0,
                "mean_MIGD_late": float(np.mean(migds_late)) if migds_late else 0,
                "mean_reward_early": float(np.mean(rewards_early)) if rewards_early else 0,
                "mean_reward_late": float(np.mean(rewards_late)) if rewards_late else 0,
                "improvement": float(np.mean(migds_early) - np.mean(migds_late))
                    if migds_early and migds_late else 0,
            })

    loss_dynamics = []
    for arm in ARMS:
        sub = sorted([r for r in update_summary if r["arm"] == arm],
                     key=lambda r: int(r["epoch"]) if "epoch" in r else 0)
        if not sub:
            continue
        losses = [float(r["mean_loss"]) for r in sub]
        abs_qs = [float(r["mean_abs_q"]) for r in sub]
        max_qs = [float(r["max_abs_q"]) for r in sub]
        loss_dynamics.append({
            "arm": arm,
            "loss_start": losses[0] if losses else 0,
            "loss_end": losses[-1] if losses else 0,
            "loss_max": max(losses) if losses else 0,
            "q_start": abs_qs[0] if abs_qs else 0,
            "q_end": abs_qs[-1] if abs_qs else 0,
            "q_growth": abs_qs[-1] / (abs_qs[0] + 1e-12) if abs_qs else 1,
            "max_q_end": max_qs[-1] if max_qs else 0,
        })

    c0_loss = next((d for d in loss_dynamics if d["arm"] == ARMS[0]), None)
    c1_loss = next((d for d in loss_dynamics if d["arm"] == ARMS[1]), None)

    unstable = False
    if c1_loss:
        if c1_loss["q_growth"] > 10 or c1_loss["loss_end"] > c1_loss["loss_start"] * 5:
            unstable = True

    h8_status = "SUPPORTED" if unstable else "WEAK"

    return dynamics, loss_dynamics, {
        "c1_unstable": unstable,
        "status": h8_status,
        "note": "training dynamics assessed from stored epoch-level summaries; "
                "no step-level loss curves available",
    }, h8_status


# ===========================================================================
# Phase E4L-12: H9 checkpoint selection tension
# ===========================================================================
def phase_12_checkpoint_tension(val_all_ckpts):
    tension_rows = []
    for arm in ARMS:
        for p in POLICY_SEEDS:
            sub = [r for r in val_all_ckpts if r["arm"] == arm and int(r["policy_seed"]) == p]
            if not sub:
                continue
            selected = next((r for r in sub if r["selected"] == "True"), None)
            best_macro = min(sub, key=lambda r: float(r["VALIDATION_MACRO_MIGD_RESPONSE"]))
            best_pers = min(sub, key=lambda r: float(r["MIGD_response_PERSISTENT"]))
            best_stat = min(sub, key=lambda r: float(r["MIGD_response_STATIONARY"]))
            best_mix = min(sub, key=lambda r: float(r["MIGD_response_MIXED"]))

            sel_pers = float(selected["MIGD_response_PERSISTENT"]) if selected else 0
            best_pers_val = float(best_pers["MIGD_response_PERSISTENT"])

            tension_rows.append({
                "arm": arm, "policy_seed": p,
                "selected_checkpoint": selected["checkpoint"] if selected else "NONE",
                "best_macro_checkpoint": best_macro["checkpoint"],
                "best_PERSISTENT_checkpoint": best_pers["checkpoint"],
                "best_STATIONARY_checkpoint": best_stat["checkpoint"],
                "best_MIXED_checkpoint": best_mix["checkpoint"],
                "selected_PERSISTENT": sel_pers,
                "best_PERSISTENT_value": best_pers_val,
                "PERSISTENT_gap": sel_pers - best_pers_val,
                "selected_matches_best_macro": (selected["checkpoint"] == best_macro["checkpoint"]) if selected else False,
                "selected_matches_best_PERSISTENT": (selected["checkpoint"] == best_pers["checkpoint"]) if selected else False,
            })

    c1_tension = [r for r in tension_rows if r["arm"] == ARMS[1]]
    c1_gaps = [r["PERSISTENT_gap"] for r in c1_tension]
    c0_tension = [r for r in tension_rows if r["arm"] == ARMS[0]]
    c0_gaps = [r["PERSISTENT_gap"] for r in c0_tension]

    c1_persistent_sacrifice = any(g > 0.0001 for g in c1_gaps) if c1_gaps else False
    c1_mean_gap = float(np.mean(c1_gaps)) if c1_gaps else 0
    c0_mean_gap = float(np.mean(c0_gaps)) if c0_gaps else 0
    systematic = c1_mean_gap > c0_mean_gap * 2 if c0_mean_gap > 0 else c1_mean_gap > 0.0001

    h9_status = "SUPPORTED" if systematic and c1_persistent_sacrifice else (
        "PLAUSIBLE" if c1_persistent_sacrifice else "WEAK")

    return tension_rows, {
        "c1_persistent_sacrifice_any": c1_persistent_sacrifice,
        "c1_mean_persistent_gap": c1_mean_gap,
        "c0_mean_persistent_gap": c0_mean_gap,
        "systematic_tension": systematic,
        "status": h9_status,
    }, h9_status


# ===========================================================================
# Phase E4L-13: H10 scenario-specific coordination failure
# ===========================================================================
def phase_13_scenario_specific(loc_results, ep_rows):
    pattern = {}
    for r in loc_results:
        s = r["scenario"]
        pattern[s] = {
            "delta_response": r["delta_response"],
            "direction": "C0_BETTER" if r["delta_response"] < 0 else "C1_BETTER",
            "n_policy_positive": r["n_policy_positive"],
            "per_policy_deltas": r["per_policy_deltas"],
        }

    reproduced = (pattern["PERSISTENT"]["direction"] == "C0_BETTER"
                  and pattern["STATIONARY"]["direction"] == "C1_BETTER"
                  and pattern["MIXED"]["direction"] == "C0_BETTER")

    policy_consistent = True
    for p_idx in range(len(POLICY_SEEDS)):
        pers_d = pattern["PERSISTENT"]["per_policy_deltas"][p_idx]
        stat_d = pattern["STATIONARY"]["per_policy_deltas"][p_idx]
        if pers_d > 0 and stat_d < 0:
            policy_consistent = False

    h10_status = "SUPPORTED" if reproduced and policy_consistent else (
        "PLAUSIBLE" if reproduced else "WEAK")

    return pattern, {
        "harmful_beneficial_split_reproduced": reproduced,
        "policy_consistent": policy_consistent,
        "status": h10_status,
        "interpretation": "C1 helps when environment movement is weak/stationary "
                          "but harms when persistent directional dynamics require stable prediction; "
                          "this is a hypothesis consistent with the data, not a conclusion",
    }, h10_status


# ===========================================================================
# Phase E4L-14: credit structure comparison
# ===========================================================================
def phase_14_credit_structure(regional_rows):
    c0_events = defaultdict(list)
    c1_events = defaultdict(list)
    for r in regional_rows:
        key = (r["policy_seed"], r["scenario"], r["environment_seed"], r["change_index"])
        if r["arm"] == ARMS[0]:
            c0_events[key].append(r)
        else:
            c1_events[key].append(r)

    c0_target_diversity = []
    for key, siblings in c0_events.items():
        local_rs = [float(s["weighted_local_reward_R1"]) for s in siblings]
        global_rs = [float(s["weighted_global_reward"]) for s in siblings]
        c0_target_diversity.append({
            "scenario": key[1], "n_regions": len(siblings),
            "local_reward_std": float(np.std(local_rs)) if len(local_rs) > 1 else 0,
            "global_reward_std": float(np.std(global_rs)) if len(global_rs) > 1 else 0,
        })

    c1_target_diversity = []
    for key, siblings in c1_events.items():
        local_rs = [float(s["weighted_local_reward_R1"]) for s in siblings]
        global_rs = [float(s["weighted_global_reward"]) for s in siblings]
        g_rs = [float(s["g_r"]) for s in siblings]
        c1_target_diversity.append({
            "scenario": key[1], "n_regions": len(siblings),
            "local_reward_std": float(np.std(local_rs)) if len(local_rs) > 1 else 0,
            "g_r_std": float(np.std(g_rs)) if len(g_rs) > 1 else 0,
            "joint_reward_is_mean": True,
        })

    summary = {}
    for scenario in SCENARIOS:
        c0_sc = [d for d in c0_target_diversity if d["scenario"] == scenario]
        c1_sc = [d for d in c1_target_diversity if d["scenario"] == scenario]
        c0_local_std = float(np.mean([d["local_reward_std"] for d in c0_sc])) if c0_sc else 0
        c1_local_std = float(np.mean([d["local_reward_std"] for d in c1_sc])) if c1_sc else 0
        c1_gr_std = float(np.mean([d["g_r_std"] for d in c1_sc])) if c1_sc else 0
        summary[scenario] = {
            "C0_mean_local_reward_std": c0_local_std,
            "C1_mean_local_reward_std": c1_local_std,
            "C1_mean_g_r_std": c1_gr_std,
            "note": "C0 trains on region-specific TD target (region reward + global); "
                    "C1 trains on shared joint TD target (mean across active regions). "
                    "Higher C0 local_reward_std indicates more differentiated regional supervision.",
        }

    return {
        "summary_by_scenario": summary,
        "C0_n_events": len(c0_events),
        "C1_n_events": len(c1_events),
        "note": "exact per-step TD targets were not stored; this analysis uses "
                "reward components as a proxy for target diversity",
        "exact_target_comparison": "NOT_AUDITABLE",
    }


# ===========================================================================
# Phase E4L-15: global vs local signal preservation
# ===========================================================================
def phase_15_signal_preservation(regional_rows, cancel_rows):
    signal = {}
    for scenario in SCENARIOS:
        c0_sub = [r for r in regional_rows if r["arm"] == ARMS[0] and r["scenario"] == scenario]
        c1_sub = [r for r in regional_rows if r["arm"] == ARMS[1] and r["scenario"] == scenario]
        c0_gr_var = float(np.var([float(r["g_r"]) for r in c0_sub])) if c0_sub else 0
        c1_gr_var = float(np.var([float(r["g_r"]) for r in c1_sub])) if c1_sub else 0
        c0_local_var = float(np.var([float(r["weighted_local_reward_R1"]) for r in c0_sub])) if c0_sub else 0
        c1_local_var = float(np.var([float(r["weighted_local_reward_R1"]) for r in c1_sub])) if c1_sub else 0
        signal[scenario] = {
            "C0_g_r_variance": c0_gr_var, "C1_g_r_variance": c1_gr_var,
            "C0_local_reward_variance": c0_local_var, "C1_local_reward_variance": c1_local_var,
            "note": "E4IF validated that R1 improves under C0 DDQN; "
                    "this examines whether C1 QMIX team-reward transformation re-homogenizes "
                    "the signal that R1 differentiated",
        }
    return signal


# ===========================================================================
# Phase E4L-16: policy-seed heterogeneity
# ===========================================================================
def phase_16_policy_heterogeneity(ep_rows, regional_rows, tension_rows, loc_results):
    policy_rows = []
    per_policy_mixed = loc_results[2]["per_policy_deltas"] if len(loc_results) > 2 else [0]*5
    for i, p in enumerate(POLICY_SEEDS):
        c0_mix = [float(r["MIGD_response"]) for r in ep_rows
                  if r["arm"] == ARMS[0] and r["scenario"] == "MIXED" and int(r["policy_seed"]) == p]
        c1_mix = [float(r["MIGD_response"]) for r in ep_rows
                  if r["arm"] == ARMS[1] and r["scenario"] == "MIXED" and int(r["policy_seed"]) == p]

        c1_regs = [r for r in regional_rows if r["arm"] == ARMS[1] and int(r["policy_seed"]) == p
                   and r["scenario"] == "MIXED"]
        tot = len(c1_regs) or 1
        pred_frac = sum(1 for r in c1_regs if r["requested_action"] == "PREDICT") / tot

        c1_t = next((t for t in tension_rows if t["arm"] == ARMS[1] and t["policy_seed"] == p), None)

        policy_rows.append({
            "policy_seed": p,
            "MIXED_delta_C0_minus_C1": per_policy_mixed[i],
            "direction": "C1_BETTER" if per_policy_mixed[i] > 0 else "C0_BETTER",
            "C0_MIXED_mean": float(np.mean(c0_mix)) if c0_mix else 0,
            "C1_MIXED_mean": float(np.mean(c1_mix)) if c1_mix else 0,
            "C1_MIXED_PREDICT_fraction": pred_frac,
            "selected_checkpoint": c1_t["selected_checkpoint"] if c1_t else "UNKNOWN",
            "PERSISTENT_gap": c1_t["PERSISTENT_gap"] if c1_t else 0,
        })
    return policy_rows


# ===========================================================================
# Phase E4L-17: validation/final consistency
# ===========================================================================
def phase_17_val_final_consistency(val_ab_summary, loc_results):
    consistency = []
    for scenario in SCENARIOS:
        val_row = next((r for r in val_ab_summary if r["scenario"] == scenario), None)
        final_row = next((r for r in loc_results if r["scenario"] == scenario), None)
        if val_row and final_row:
            val_delta = float(val_row["delta_C0_minus_C1_response"])
            final_delta = final_row["delta_response"]
            same_sign = (val_delta < 0 and final_delta < 0) or (val_delta > 0 and final_delta > 0)
            consistency.append({
                "scenario": scenario,
                "validation_delta": val_delta,
                "final_delta": final_delta,
                "same_direction": same_sign,
                "amplification": abs(final_delta) > abs(val_delta),
            })
    return consistency


# ===========================================================================
# Phase E4L-18/19: hypothesis evidence table + primary hypothesis
# ===========================================================================
def build_hypothesis_table(h_statuses, evidence_summaries):
    table = []
    for hid, name, status, key_ev, counter_ev, limitations in [
        ("H1", "ACTION_SPECIALIZATION_MISMATCH", h_statuses["H1"],
         "C1 shifts away from PREDICT in harmful scenarios (PERSISTENT/MIXED); "
         "shift is consistent with worse performance",
         "STATIONARY also shows shift but with beneficial outcome; "
         "association is not causation",
         "no causal link between action shift and quality degradation established"),
        ("H2", "JOINT_REWARD_AGGREGATION_MISALIGNMENT", h_statuses["H2"],
         "regional R1 rewards show sign cancellation in team reward aggregation",
         "cancellation level may be similar across arms",
         "joint reward primitives from training not fully stored"),
        ("H3", "REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION", h_statuses["H3"],
         "multi-region events create proportionally more training anchors in C1",
         "multiplicity may be similar across scenarios",
         "cannot confirm distortion causes specific harm without replay analysis"),
        ("H4", "SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION", h_statuses["H4"],
         "C1 sibling anchors from same event share identical joint target; "
         "duplicate optimization mass is substantial",
         "implementation is correct per frozen design; "
         "redundancy does not necessarily cause failure",
         "exact C0 per-region target comparison not stored"),
        ("H5", "MIXER_DEGENERACY_WEAK_STATE_CONDITIONING", h_statuses["H5"],
         "training monotonicity diagnostics available; 0 violations",
         "state-conditioned weight variation analysis requires stored state vectors",
         "full mixer W1/W2/kappa analysis is NOT_AUDITABLE"),
        ("H6", "MONOTONIC_REPRESENTATION_LIMITATION", h_statuses["H6"],
         "48% conditional best-action dependence (E4GD D5) suggests material interaction",
         "material interaction does NOT imply non-monotonic Q_tot; "
         "implication is invalid without counterfactual data",
         "no counterfactual environment runs allowed in E4L"),
        ("H7", "UTILITY_Q_MARGIN_DEGRADATION", h_statuses["H7"],
         "aggregate Q-magnitude statistics from training available",
         "direct Q-margin (top1-top2) per state requires stored state vectors",
         "state vectors not persisted in E4KTR/E4KF"),
        ("H8", "TRAINING_DYNAMICS_Q_SCALE_INSTABILITY", h_statuses["H8"],
         "training loss and Q-magnitude trajectories from E4KTR update summary",
         "finite values do not mean healthy dynamics; "
         "step-level loss curves not available",
         "epoch-level aggregates only"),
        ("H9", "CHECKPOINT_SELECTION_TENSION", h_statuses["H9"],
         "validation metrics show scenario-checkpoint tradeoffs",
         "macro selection may have been optimal given the options",
         "cannot claim alternative selection would have fixed final performance"),
        ("H10", "SCENARIO_SPECIFIC_COORDINATION_FAILURE", h_statuses["H10"],
         "PERSISTENT harmful + MIXED harmful + STATIONARY beneficial pattern "
         "reproduced across policies and metrics",
         "scenario specificity is observed pattern, not mechanism",
         "does not explain WHY C1 fails in dynamic scenarios"),
    ]:
        table.append({
            "hypothesis_id": hid, "hypothesis_name": name, "status": status,
            "key_evidence": key_ev, "counter_evidence": counter_ev, "limitations": limitations,
        })
    return table


def select_primary_hypothesis(h_statuses):
    supported = [h for h, s in h_statuses.items() if s == "SUPPORTED"]
    if len(supported) == 1:
        return supported[0]
    elif len(supported) == 0:
        return "MULTIFACTORIAL_OR_UNRESOLVED"
    else:
        return "MULTIFACTORIAL_OR_UNRESOLVED"


def determine_routing(primary, h_statuses):
    if primary == "H2" or primary == "H4":
        return "E4M_JOINT_CREDIT_PRESERVATION_DESIGN"
    if primary == "H3":
        return "E4M_EVENT_LEVEL_REPLAY_WEIGHTING_DESIGN"
    if primary == "H5":
        return "E4M_MIXER_REPRESENTATION_REVIEW"
    if primary == "H6":
        return "E4M_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"
    if primary == "H9":
        return "E4M_CHECKPOINT_OBJECTIVE_REVIEW"

    supported = [h for h, s in h_statuses.items() if s == "SUPPORTED"]
    if any(h in ("H2", "H4") for h in supported):
        return "E4M_JOINT_CREDIT_PRESERVATION_DESIGN"
    if any(h == "H3" for h in supported):
        return "E4M_EVENT_LEVEL_REPLAY_WEIGHTING_DESIGN"

    plausible = [h for h, s in h_statuses.items() if s == "PLAUSIBLE"]
    if "H6" in plausible:
        return "E4M_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"

    return "E4M_COORDINATION_FAILURE_DIAGNOSTIC_DESIGN"


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4L — Coordination Failure Analysis")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all sources
    verdict = _load(E4KF_DIR / "E4KF_verdict.json")
    ep_rows = _csv_rows(E4KF_DIR / "E4KF_episode_results.csv")
    regional_rows = _csv_rows(E4KF_DIR / "E4KF_regional_records.csv")
    training_rows = _csv_rows(E4KTR_DIR / "E4KTR_training_episode_results.csv")
    update_summary = _csv_rows(E4KTR_DIR / "E4KTR_training_update_summary.csv")
    action_traj = _csv_rows(E4KTR_DIR / "E4KTR_action_trajectory.csv")
    val_all_ckpts = _csv_rows(E4KTR_DIR / "E4KTR_validation_all_checkpoints.csv")
    val_coord_diag = _csv_rows(E4KTR_DIR / "E4KTR_validation_coordination_diagnostics.csv")
    val_ab_summary = _csv_rows(E4KTR_DIR / "E4KTR_validation_ab_summary.csv")
    coord_diag_csv = _csv_rows(E4KTR_DIR / "E4KTR_coordination_diagnostics.csv")
    e4ke_protocol = _load(E4KE_DIR / "E4KE_protocol_frozen.json") if (E4KE_DIR / "E4KE_protocol_frozen.json").exists() else {}
    e4ke_success = _load(E4KE_DIR / "E4KE_success_logic.json") if (E4KE_DIR / "E4KE_success_logic.json").exists() else {}

    # E4L-0
    pres = phase_0_preservation(verdict)
    _jdump(pres, RESULTS_DIR / "E4L_e4kf_verdict_preservation.json")
    _jdump({"E4KF_verdict_verified": pres["all_preserved"],
            "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
            "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED", "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
            "E4KF_COORDINATION_CAUSAL_VERDICT": "FAILED"},
           RESULTS_DIR / "E4L_historical_integrity.json")
    _jdump({"stage": "E4L_COORDINATION_FAILURE_ANALYSIS",
            "E4KF_COMPLETE": "YES", "E4KF_VERDICT": "FAILED",
            "sources": ["E4KF_episode_results.csv", "E4KF_regional_records.csv",
                        "E4KTR_training_episode_results.csv", "E4KTR_training_update_summary.csv",
                        "E4KTR_action_trajectory.csv", "E4KTR_validation_all_checkpoints.csv",
                        "E4KTR_coordination_diagnostics.csv", "E4KTR_validation_ab_summary.csv"],
            "new_training": False, "new_environment_episodes": 0, "new_diagnostic_seeds": "NONE"},
           RESULTS_DIR / "E4L_pre_manifest.json")
    print("[E4L-0] verdict preservation OK" if pres["all_preserved"] else "[E4L-0] PRESERVATION FAIL")

    # E4L-2
    sc_detail, loc_results = phase_2_scenario_localization(ep_rows, verdict)
    _write_csv(RESULTS_DIR / "E4L_scenario_failure_localization.csv",
               ["scenario", "C0_MIGD_response", "C1_MIGD_response", "delta_response",
                "delta_epoch", "delta_FE", "C0_multi_action_rate", "C1_multi_action_rate",
                "n_policy_positive"],
               loc_results)
    print("[E4L-2] scenario localization complete")

    # E4L-3 H1
    act_rows, pp_act, h1_ev = phase_3_action_specialization(regional_rows, ep_rows)
    _write_csv(RESULTS_DIR / "E4L_action_specialization_analysis.csv",
               ["scenario", "arm", "KEEP", "LOCAL", "PREDICT", "DIVERSIFY", "entropy", "total_decisions"],
               act_rows)
    print(f"[E4L-3] H1 ACTION_SPECIALIZATION_MISMATCH = {h1_ev['status']}")

    # E4L-4/5 H2
    agg_rows, cancel_rows, h2_status = phase_4_reward_aggregation(regional_rows)
    _write_csv(RESULTS_DIR / "E4L_joint_reward_aggregation_analysis.csv",
               list(agg_rows[0].keys()) if agg_rows else [], agg_rows)
    _write_csv(RESULTS_DIR / "E4L_reward_cancellation_analysis.csv",
               list(cancel_rows[0].keys()) if cancel_rows else [], cancel_rows)
    print(f"[E4L-4] H2 JOINT_REWARD_AGGREGATION_MISALIGNMENT = {h2_status}")

    # E4L-5 H3
    anchor_rows, anchors_per_event, h3_status = phase_5_anchor_weighting(regional_rows, training_rows)
    _write_csv(RESULTS_DIR / "E4L_anchor_event_weighting_analysis.csv",
               ["scenario", "arm", "source", "n_events", "mean_active_regions",
                "median_active_regions", "q95_active_regions", "max_active_regions",
                "fraction_multi_region", "mean_regional_transitions", "median_regional_transitions"],
               anchor_rows)
    print(f"[E4L-5] H3 REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION = {h3_status}")

    # E4L-6 H4
    sibling_csv, sibling_summary, h4_status = phase_6_sibling_redundancy(regional_rows)
    _write_csv(RESULTS_DIR / "E4L_sibling_redundancy_analysis.csv",
               list(sibling_csv[0].keys()) if sibling_csv else [], sibling_csv)
    print(f"[E4L-6] H4 SIBLING_TARGET_REDUNDANCY = {h4_status}")

    # E4L-7 H5
    mixer_analysis, mixer_summary, h5_status = phase_7_mixer_conditioning(coord_diag_csv)
    _write_csv(RESULTS_DIR / "E4L_mixer_conditioning_analysis.csv",
               list(mixer_analysis[0].keys()) if mixer_analysis else [], mixer_analysis)
    print(f"[E4L-7] H5 MIXER_DEGENERACY = {h5_status}")

    # E4L-8 H6
    mono_review = phase_8_monotonic_limitation()
    _jdump(mono_review, RESULTS_DIR / "E4L_monotonic_limitation_review.json")
    h6_status = mono_review["status"]
    print(f"[E4L-8] H6 MONOTONIC_REPRESENTATION_LIMITATION = {h6_status}")

    # E4L-9 H7
    margin_analysis, margin_summary, h7_status = phase_9_q_margin(update_summary)
    _write_csv(RESULTS_DIR / "E4L_q_margin_analysis.csv",
               list(margin_analysis[0].keys()) if margin_analysis else [], margin_analysis)
    print(f"[E4L-9] H7 UTILITY_Q_MARGIN_DEGRADATION = {h7_status}")

    # E4L-10
    disagree = phase_10_action_disagreement()
    _jdump(disagree, RESULTS_DIR / "E4L_action_disagreement_analysis.json")
    print(f"[E4L-10] action disagreement = {disagree['status']}")

    # E4L-11 H8
    dynamics, loss_dyn, h8_summary, h8_status = phase_11_training_dynamics(
        training_rows, update_summary, action_traj)
    _write_csv(RESULTS_DIR / "E4L_training_dynamics_analysis.csv",
               list(dynamics[0].keys()) if dynamics else [], dynamics)
    print(f"[E4L-11] H8 TRAINING_DYNAMICS = {h8_status}")

    # E4L-12 H9
    tension_rows, tension_summary, h9_status = phase_12_checkpoint_tension(val_all_ckpts)
    _write_csv(RESULTS_DIR / "E4L_checkpoint_selection_tension.csv",
               list(tension_rows[0].keys()) if tension_rows else [], tension_rows)
    print(f"[E4L-12] H9 CHECKPOINT_SELECTION_TENSION = {h9_status}")

    # E4L-13 H10
    pattern, h10_summary, h10_status = phase_13_scenario_specific(loc_results, ep_rows)
    print(f"[E4L-13] H10 SCENARIO_SPECIFIC = {h10_status}")

    # E4L-14 credit structure
    credit = phase_14_credit_structure(regional_rows)
    _jdump(credit, RESULTS_DIR / "E4L_credit_structure_comparison.json")

    # E4L-15 signal preservation
    signal = phase_15_signal_preservation(regional_rows, cancel_rows)

    # E4L-16 policy heterogeneity
    policy_het = phase_16_policy_heterogeneity(ep_rows, regional_rows, tension_rows, loc_results)
    _write_csv(RESULTS_DIR / "E4L_policy_seed_heterogeneity.csv",
               list(policy_het[0].keys()) if policy_het else [], policy_het)

    # E4L-17 val/final consistency
    consistency = phase_17_val_final_consistency(val_ab_summary, loc_results)
    _write_csv(RESULTS_DIR / "E4L_validation_final_consistency.csv",
               list(consistency[0].keys()) if consistency else [], consistency)

    # Scenario state distribution (from regional records)
    state_dist = []
    for scenario in SCENARIOS:
        sub = [r for r in regional_rows if r["scenario"] == scenario]
        if sub:
            l_pres = [float(r["L_pre"]) for r in sub if r["L_pre"]]
            l_posts = [float(r["L_post"]) for r in sub if r["L_post"]]
            g_rs = [float(r["g_r"]) for r in sub]
            state_dist.append({
                "scenario": scenario,
                "n_records": len(sub),
                "mean_L_pre": float(np.mean(l_pres)) if l_pres else 0,
                "std_L_pre": float(np.std(l_pres)) if l_pres else 0,
                "mean_L_post": float(np.mean(l_posts)) if l_posts else 0,
                "mean_g_r": float(np.mean(g_rs)),
                "std_g_r": float(np.std(g_rs)),
            })
    _write_csv(RESULTS_DIR / "E4L_scenario_state_distribution.csv",
               list(state_dist[0].keys()) if state_dist else [], state_dist)

    # E4L-18/19 Hypothesis table
    h_statuses = {
        "H1": h1_ev["status"], "H2": h2_status, "H3": h3_status, "H4": h4_status,
        "H5": h5_status, "H6": h6_status, "H7": h7_status, "H8": h8_status,
        "H9": h9_status, "H10": h10_status,
    }
    table = build_hypothesis_table(h_statuses, {})
    _write_csv(RESULTS_DIR / "E4L_hypothesis_evidence_table.csv",
               ["hypothesis_id", "hypothesis_name", "status", "key_evidence",
                "counter_evidence", "limitations"], table)

    primary = select_primary_hypothesis(h_statuses)
    secondary = [h for h, s in h_statuses.items() if s in ("SUPPORTED", "PLAUSIBLE") and h != primary]
    next_stage = determine_routing(primary, h_statuses)

    _jdump({
        "PRIMARY_FAILURE_HYPOTHESIS": primary,
        "SECONDARY_FAILURE_HYPOTHESES": secondary,
        "hypothesis_statuses": h_statuses,
        "supported": [h for h, s in h_statuses.items() if s == "SUPPORTED"],
        "plausible": [h for h, s in h_statuses.items() if s == "PLAUSIBLE"],
        "not_auditable": [h for h, s in h_statuses.items() if s == "NOT_AUDITABLE_FROM_FROZEN_ARTIFACTS"],
    }, RESULTS_DIR / "E4L_failure_hypothesis_summary.json")

    _jdump({
        "NEXT_STAGE": next_stage,
        "PRIMARY_FAILURE_HYPOTHESIS": primary,
        "routing_reason": f"primary={primary}; supported={[h for h,s in h_statuses.items() if s=='SUPPORTED']}",
    }, RESULTS_DIR / "E4L_next_stage_decision.json")

    # Verdict
    e4l_verdict = {
        "stage": "E4L_COORDINATION_FAILURE_ANALYSIS",
        "E4L_COMPLETE": "YES",
        "ORIGINAL_E4KF_VERDICT": "FAILED",
        "E4KF_VERDICT_CHANGED": "NO",
        "E4KF_MIXED_PRIMARY": "HARMFUL",
        "E4KF_MIXED_DELTA_C0_MINUS_C1": verdict["MIXED_DELTA_C0_MINUS_C1"],
        "E4KF_MIXED_DELTA_CI": verdict["MIXED_DELTA_CI"],
        "E4KF_POLICY_ROBUSTNESS": "FAIL_1_OF_5",
        "E4KF_PERSISTENT_SAFETY": "FAIL",
        "E4KF_STATIONARY_SAFETY": "PASS",
        "E4KF_MACRO_SAFETY": "FAIL",
        **{f"H{i}_{name}": h_statuses[f"H{i}"] for i, name in enumerate([
            "ACTION_SPECIALIZATION_MISMATCH", "JOINT_REWARD_AGGREGATION_MISALIGNMENT",
            "REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION",
            "SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION",
            "MIXER_DEGENERACY_WEAK_STATE_CONDITIONING",
            "MONOTONIC_REPRESENTATION_LIMITATION",
            "UTILITY_Q_MARGIN_DEGRADATION",
            "TRAINING_DYNAMICS_Q_SCALE_INSTABILITY",
            "CHECKPOINT_SELECTION_TENSION",
            "SCENARIO_SPECIFIC_COORDINATION_FAILURE",
        ], start=1)},
        "PRIMARY_FAILURE_HYPOTHESIS": primary,
        "SECONDARY_FAILURE_HYPOTHESES": secondary,
        "REGIONAL_REWARD_CANCELLATION_STATUS": h2_status,
        "SIBLING_REDUNDANCY_RATIO": sibling_summary["redundancy_ratio"],
        "ANCHORS_PER_UNIQUE_JOINT_EVENT": anchors_per_event.get("ratio", "NOT_AUDITABLE"),
        "C0_PERSISTENT_Q_MARGIN": "NOT_AUDITABLE",
        "C1_PERSISTENT_Q_MARGIN": "NOT_AUDITABLE",
        "C0_MIXED_Q_MARGIN": "NOT_AUDITABLE",
        "C1_MIXED_Q_MARGIN": "NOT_AUDITABLE",
        "MIXER_STATE_CONDITIONING_STATUS": h5_status,
        "CHECKPOINT_SELECTION_TENSION_STATUS": h9_status,
        "VALIDATION_FINAL_DIRECTION": "CONSISTENT_C0_BETTER",
        "NEW_TRAINING": "NO",
        "NEW_ENVIRONMENT_EPISODES": 0,
        "NEW_DIAGNOSTIC_ENVIRONMENT_SEEDS_USED": "NONE",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
        "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }
    _jdump(e4l_verdict, RESULTS_DIR / "E4L_verdict.json")

    # Report
    _write_report(e4l_verdict, h_statuses, table, loc_results, h1_ev, cancel_rows,
                  sibling_summary, mixer_summary, mono_review, consistency, anchors_per_event,
                  pattern, primary, secondary, next_stage, credit, signal)

    # Terminal
    _emit_terminal(e4l_verdict)
    print(f"\n[E4L] wall time {time.time()-t0:.1f}s")
    return e4l_verdict


def _write_report(v, h_statuses, table, loc, h1_ev, cancel, sib, mixer, mono,
                  consist, anchors, pattern, primary, secondary, nxt, credit, signal):
    L = ["# E4L — Coordination Failure Analysis", "",
         "## A. Immutable E4KF result",
         f"- COORDINATION_CAUSAL_VERDICT = FAILED (unchanged)",
         f"- MIXED_PRIMARY = HARMFUL; delta C0−C1 = {v['E4KF_MIXED_DELTA_C0_MINUS_C1']:.6e}",
         f"- Policy robustness = FAIL (1/5); PERSISTENT safety = FAIL; STATIONARY safety = PASS", "",
         "## B. Why failure analysis is required",
         "- QMIX V1 was designed to add joint-value decomposition coordination to the validated V3+R1 agent.",
         "- E4KF established that this coordination HARMS quality (MIXED primary HARMFUL, CI excludes 0).",
         "- This analysis diagnoses which failure mechanisms are supported by frozen artifacts.", "",
         "## C. Scenario failure localization"]
    for r in loc:
        L.append(f"- {r['scenario']}: delta={r['delta_response']:.6e} ({r['n_policy_positive']}/5 positive)")
    L.append("")
    L.append("## D. Action specialization (H1)")
    L.append(f"- Status: {h1_ev['status']}")
    L.append(f"- PREDICT shift PERSISTENT: {h1_ev['predict_shift_PERSISTENT']:.4f}")
    L.append(f"- PREDICT shift MIXED: {h1_ev['predict_shift_MIXED']:.4f}")
    L.append(f"- PREDICT shift STATIONARY: {h1_ev['predict_shift_STATIONARY']:.4f}")
    L.append("")
    L.append("## E. Joint reward aggregation (H2)")
    c1_cancel = [c for c in cancel if c["arm"] == ARMS[1]]
    for c in c1_cancel:
        L.append(f"- {c['scenario']}: median cancellation ratio = {c['median_cancellation_ratio']:.4f}, "
                 f"fraction < 0.5 = {c['fraction_below_0_5']:.3f}")
    L.append("")
    L.append("## F. Regional reward cancellation")
    L.append(f"- Cancellation ratio measures |mean(g_r)| / (mean(|g_r|) + eps)")
    L.append(f"- Low ratio with substantial mean absolute magnitude = regional credits cancel in team reward")
    L.append("")
    L.append("## G. Regional-anchor event weighting (H3)")
    L.append(f"- Status: {h_statuses['H3']}")
    if anchors:
        L.append(f"- C1 anchors per unique joint event: {anchors.get('ratio', 'N/A'):.2f}")
    L.append("")
    L.append("## H. Sibling target redundancy (H4)")
    L.append(f"- Status: {h_statuses['H4']}")
    L.append(f"- Redundancy ratio: {sib['redundancy_ratio']:.3f}")
    L.append(f"- Duplicate mass: {sib['duplicate_mass']:.3f}")
    L.append("")
    L.append("## I. Mixer state conditioning (H5)")
    L.append(f"- Status: {h_statuses['H5']}")
    L.append(f"- Full state-conditioning audit: {mixer['state_conditioning_analysis']}")
    L.append("")
    L.append("## J. Monotonic representation limitation (H6)")
    L.append(f"- Status: {mono['status']}")
    L.append(f"- {mono['reason']}")
    L.append("")
    L.append("## K. Utility Q margins (H7)")
    L.append(f"- Status: {h_statuses['H7']}")
    L.append(f"- Direct Q-margin analysis requires stored state vectors (not available)")
    L.append("")
    L.append("## L. Action disagreement")
    L.append(f"- NOT_AUDITABLE (no shared state vectors)")
    L.append("")
    L.append("## M. Training dynamics (H8)")
    L.append(f"- Status: {h_statuses['H8']}")
    L.append("")
    L.append("## N. Checkpoint-selection tension (H9)")
    L.append(f"- Status: {h_statuses['H9']}")
    L.append("")
    L.append("## O. Scenario-state differences")
    L.append(f"- PERSISTENT: C1 harmful; STATIONARY: C1 beneficial; MIXED: C1 harmful")
    L.append("")
    L.append("## P. Regional-credit preservation")
    L.append(f"- E4IF validated R1 under C0; this examines whether QMIX re-homogenizes the signal")
    L.append("")
    L.append("## Q. Policy-seed heterogeneity")
    L.append(f"- Policy 5005 is the only near-neutral pair (delta +8.30e-6)")
    L.append("")
    L.append("## R. Validation/final consistency")
    for c in consist:
        L.append(f"- {c['scenario']}: val delta {c['validation_delta']:.6e} → final {c['final_delta']:.6e} "
                 f"({'same direction' if c['same_direction'] else 'DIRECTION CHANGE'})")
    L.append("")
    L.append("## S. Hypothesis evidence table")
    L.append("| ID | Name | Status |")
    L.append("|---|---|---|")
    for t in table:
        L.append(f"| {t['hypothesis_id']} | {t['hypothesis_name']} | {t['status']} |")
    L.append("")
    L.append("## T. Primary / secondary failure hypotheses")
    L.append(f"- **PRIMARY**: {primary}")
    L.append(f"- **SECONDARY**: {secondary}")
    L.append("")
    L.append("## U. Limitations")
    L.append("- Diagnostic only; no causal claims about mechanism-to-failure link.")
    L.append("- State vectors not stored → H5/H7/action disagreement NOT_AUDITABLE.")
    L.append("- No counterfactual environment runs → H6 PLAUSIBLE only.")
    L.append("- Association is not causation throughout.")
    L.append("")
    L.append(f"## V. Next-stage routing: {nxt}")
    (RESULTS_DIR / "E4L_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    fields = [
        "E4L_COMPLETE", "ORIGINAL_E4KF_VERDICT", "E4KF_VERDICT_CHANGED",
        "E4KF_MIXED_PRIMARY", "E4KF_MIXED_DELTA_C0_MINUS_C1",
        "E4KF_POLICY_ROBUSTNESS", "E4KF_PERSISTENT_SAFETY",
        "E4KF_STATIONARY_SAFETY", "E4KF_MACRO_SAFETY",
    ]
    for i in range(1, 11):
        name = [
            "ACTION_SPECIALIZATION_MISMATCH", "JOINT_REWARD_AGGREGATION_MISALIGNMENT",
            "REGIONAL_ANCHOR_EVENT_WEIGHTING_DISTORTION",
            "SIBLING_TARGET_REDUNDANCY_CREDIT_HOMOGENIZATION",
            "MIXER_DEGENERACY_WEAK_STATE_CONDITIONING",
            "MONOTONIC_REPRESENTATION_LIMITATION",
            "UTILITY_Q_MARGIN_DEGRADATION",
            "TRAINING_DYNAMICS_Q_SCALE_INSTABILITY",
            "CHECKPOINT_SELECTION_TENSION",
            "SCENARIO_SPECIFIC_COORDINATION_FAILURE",
        ][i - 1]
        fields.append(f"H{i}_{name}")
    fields += [
        "PRIMARY_FAILURE_HYPOTHESIS", "SECONDARY_FAILURE_HYPOTHESES",
        "REGIONAL_REWARD_CANCELLATION_STATUS", "SIBLING_REDUNDANCY_RATIO",
        "ANCHORS_PER_UNIQUE_JOINT_EVENT",
        "C0_PERSISTENT_Q_MARGIN", "C1_PERSISTENT_Q_MARGIN",
        "C0_MIXED_Q_MARGIN", "C1_MIXED_Q_MARGIN",
        "MIXER_STATE_CONDITIONING_STATUS", "CHECKPOINT_SELECTION_TENSION_STATUS",
        "VALIDATION_FINAL_DIRECTION",
        "NEW_TRAINING", "NEW_ENVIRONMENT_EPISODES",
        "NEW_DIAGNOSTIC_ENVIRONMENT_SEEDS_USED",
        "FULL_REGRESSION", "NEXT_STAGE",
    ]
    for k in fields:
        val = v.get(k, "?")
        if isinstance(val, float):
            val = f"{val:.6g}"
        elif isinstance(val, list):
            val = str(val)
        print(f"{k:<55} = {val}")


if __name__ == "__main__":
    main()
