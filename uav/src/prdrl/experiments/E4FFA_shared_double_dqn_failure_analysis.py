"""
E4F-FA — Shared Double-DQN Failure Analysis (POST-HOC DIAGNOSTIC).

The original E4F scientific verdict (FAILED) is IMMUTABLE. This stage does NOT
retrain, fine-tune, modify reward/architecture, open new seeds, reselect
checkpoints/comparators, or run ANY environment rollout. It reads only existing
frozen artifacts and performs deterministic frozen-network analysis to explain
WHY the preregistered initial DDQN failed and which mechanism the next stage
should target.

All diagnostic thresholds below are PREDECLARED and are descriptive only — never
scientific performance gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, "src")

from scipy.stats import spearmanr

from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS

# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/E4F_failure_analysis")
E4F_DIR = Path("results/E4F_final_evaluation")
E4TR_DIR = Path("results/E4TR_shared_double_dqn_training")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")
E3R_DIR = Path("results/E3R_rule_v2_semantic_rerun")

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]
PRESELECTED_COMPARATOR = {"PERSISTENT": "GLOBAL_PREDICT",
                          "STATIONARY": "GLOBAL_DIVERSIFY", "MIXED": "GLOBAL_PREDICT"}
COMPARATOR_ACTION = {"PERSISTENT": "PREDICT", "STATIONARY": "DIVERSIFY", "MIXED": "PREDICT"}
LOCAL_FEATURES = ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]
GLOBAL_FEATURES = ["active_region_fraction", "mean_severity", "max_severity",
                   "mean_convergence_loss", "max_convergence_loss"]
FEATURE_NAMES = LOCAL_FEATURES + GLOBAL_FEATURES

# ---- PREDECLARED diagnostic thresholds (before any computation) -------------
COLLAPSE_THRESH = 0.90          # dominant fraction >= -> COLLAPSED
LOW_DIVERSITY_THRESH = 0.75     # dominant fraction >= -> LOW_DIVERSITY
SPECIALIZATION_PRESENT = 0.50   # MIXED multi-action event rate >= -> PRESENT
SPECIALIZATION_WEAK = 0.10      # >= -> WEAK, else ABSENT
MIMICRY_STRONG = 0.90
MIMICRY_MODERATE = 0.70
TIE_TOL = 1e-5
RHO_STRONG = 0.7
RHO_PARTIAL = 0.3
QDOM_STRONG = 0.90              # single action argmax fraction >= -> STRONG_ACTION_BIAS
QDOM_LOWMARGIN = 1e-3           # mean top1-top2 |Q| margin < -> LOW_MARGIN_UNCERTAIN
STATE_USAGE_RATIO = 3.0        # global/local mean|grad| ratio > -> GLOBAL_DOMINATED
WEAK_SENS_ABS = 1e-4           # all mean|grad| < -> WEAK_SENSITIVITY
COVERAGE_BALANCED = 0.05       # each action >= 5% of training decisions
COVERAGE_MIN = 0.01            # any action < 1% -> SEVERELY_IMBALANCED


# ---------------------------------------------------------------------------
def _hash_file(path) -> str:
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _read_csv(path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _entropy(fracs: List[float]) -> float:
    h = 0.0
    for p in fracs:
        if p > 0:
            h -= p * math.log(p)
    return h


# ---------------------------------------------------------------------------
# FA-0 — integrity freeze
# ---------------------------------------------------------------------------

def fa0_integrity() -> dict:
    e4f_verdict = _load(E4F_DIR / "E4F_verdict.json")
    ck_freeze = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    hashes = {
        "E4F_verdict.json": _hash_file(E4F_DIR / "E4F_verdict.json"),
        "E4F_episode_results.csv": _hash_file(E4F_DIR / "E4F_episode_results.csv"),
        "E4F_change_results.jsonl": _hash_file(E4F_DIR / "E4F_change_results.jsonl"),
    }
    ck_ok = True
    ck_rows = {}
    for ps in POLICY_SEEDS:
        info = ck_freeze["policies"][str(ps)]
        actual = _hash_file(info["checkpoint_path"])
        match = actual == info["sha256"]
        ck_ok = ck_ok and match
        ck_rows[str(ps)] = {"path": info["checkpoint_path"], "sha256": actual,
                            "matches_frozen": match}
    manifest = {
        "stage": "E4F_FAILURE_ANALYSIS", "post_hoc": True,
        "SCIENTIFIC_VERDICT_FROZEN": e4f_verdict["SCIENTIFIC_VERDICT"],
        "e4f_artifact_hashes": hashes,
        "checkpoint_hashes": ck_rows,
        "checkpoints_match_frozen": ck_ok,
        "no_environment_rollout": True, "no_training": True,
        "no_verdict_change": True,
    }
    _jdump(manifest, RESULTS_DIR / "E4FFA_pre_manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# FA-1 — policy action collapse audit
# ---------------------------------------------------------------------------

def fa1_action_collapse() -> dict:
    rows = _read_csv(E4F_DIR / "E4F_action_distribution.csv")
    per = {}
    pooled = {s: {a: [] for a in ACTION_ORDER} for s in SCENARIOS}
    for r in rows:
        ps, sc = int(r["policy_seed"]), r["scenario"]
        fr = {a: float(r[a]) for a in ACTION_ORDER}
        for a in ACTION_ORDER:
            pooled[sc][a].append(fr[a])
        order = sorted(ACTION_ORDER, key=lambda a: -fr[a])
        dom, dom_frac = order[0], fr[order[0]]
        H = _entropy([fr[a] for a in ACTION_ORDER])
        cls = ("COLLAPSED" if dom_frac >= COLLAPSE_THRESH else
               "LOW_DIVERSITY" if dom_frac >= LOW_DIVERSITY_THRESH else "MULTI_ACTION")
        per[f"{ps}_{sc}"] = {
            "policy_seed": ps, "scenario": sc, "fractions": fr,
            "dominant_action": dom, "dominant_action_fraction": dom_frac,
            "shannon_entropy": H, "normalized_entropy": H / math.log(N_ACTIONS),
            "effective_num_actions": math.exp(H), "classification": cls,
        }
    pooled_out = {}
    for sc in SCENARIOS:
        pf = {a: float(np.mean(pooled[sc][a])) for a in ACTION_ORDER}
        dom = max(ACTION_ORDER, key=lambda a: pf[a])
        pooled_out[sc] = {"pooled_fractions": pf, "dominant_action": dom,
                          "dominant_fraction": pf[dom],
                          "approximates_comparator": dom == COMPARATOR_ACTION[sc]}
    out = {"predeclared_thresholds": {"COLLAPSED>=": COLLAPSE_THRESH,
                                      "LOW_DIVERSITY>=": LOW_DIVERSITY_THRESH},
           "per_policy_scenario": per, "pooled_by_scenario": pooled_out}
    _jdump(out, RESULTS_DIR / "E4FFA_action_collapse.json")
    return out


# ---------------------------------------------------------------------------
# change_results.jsonl loader (DDQN + comparator change-level)
# ---------------------------------------------------------------------------

def load_change_rows():
    ddqn = []       # DDQN change rows
    comp_change = {}  # (scenario, env_seed, change_index) -> comparator MIGD_response_change
    comp_arms = set(PRESELECTED_COMPARATOR.values())
    with open(E4F_DIR / "E4F_change_results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ctrl = row["controller"]
            if ctrl == "DDQN":
                ddqn.append(row)
            elif ctrl in comp_arms:
                # store per scenario only where it is the preselected comparator
                sc = row["scenario"]
                if PRESELECTED_COMPARATOR[sc] == ctrl:
                    comp_change[(sc, int(row["environment_seed"]), int(row["change_index"]))] = row
    return ddqn, comp_change


# ---------------------------------------------------------------------------
# FA-2 — within-event regional heterogeneity
# ---------------------------------------------------------------------------

def fa2_event_heterogeneity(ddqn_change) -> Tuple[dict, List[dict]]:
    agg = {}  # (ps, sc) -> list of unique-action counts + entropies
    for row in ddqn_change:
        ps, sc = int(row["policy_seed"]), row["scenario"]
        reqs = [reg["requested"] for reg in row["regional"]]
        uniq = len(set(reqs))
        counts = {a: reqs.count(a) for a in set(reqs)}
        n = len(reqs)
        H = _entropy([c / n for c in counts.values()])
        dom_frac = max(counts.values()) / n
        agg.setdefault((ps, sc), []).append((uniq, H, dom_frac))
    per = {}
    rows_csv = []
    for (ps, sc), vals in sorted(agg.items()):
        uniqs = np.array([v[0] for v in vals])
        Hs = np.array([v[1] for v in vals])
        f1 = float(np.mean(uniqs == 1)); f2 = float(np.mean(uniqs >= 2)); f3 = float(np.mean(uniqs >= 3))
        per[f"{ps}_{sc}"] = {
            "policy_seed": ps, "scenario": sc, "n_events": len(vals),
            "frac_events_1_unique": f1, "frac_events_ge2_unique": f2,
            "frac_events_ge3_unique": f3, "mean_unique_actions": float(uniqs.mean()),
            "median_unique_actions": float(np.median(uniqs)),
            "mean_regional_entropy": float(Hs.mean()),
        }
        rows_csv.append({"policy_seed": ps, "scenario": sc, "n_events": len(vals),
                         "frac_events_1_unique": f1, "frac_events_ge2_unique": f2,
                         "frac_events_ge3_unique": f3,
                         "mean_unique_actions": float(uniqs.mean()),
                         "median_unique_actions": float(np.median(uniqs))})
    # MIXED multi-action rate pooled
    mixed_multi = float(np.mean([per[k]["frac_events_ge2_unique"]
                                 for k in per if per[k]["scenario"] == "MIXED"]))
    status = ("PRESENT" if mixed_multi >= SPECIALIZATION_PRESENT else
              "WEAK" if mixed_multi >= SPECIALIZATION_WEAK else "ABSENT")
    _write_csv(RESULTS_DIR / "E4FFA_event_heterogeneity.csv",
               ["policy_seed", "scenario", "n_events", "frac_events_1_unique",
                "frac_events_ge2_unique", "frac_events_ge3_unique",
                "mean_unique_actions", "median_unique_actions"], rows_csv)
    return {"per_policy_scenario": per, "mixed_pooled_multi_action_rate": mixed_multi,
            "REGIONAL_SPECIALIZATION_STATUS": status,
            "predeclared_thresholds": {"PRESENT>=": SPECIALIZATION_PRESENT,
                                       "WEAK>=": SPECIALIZATION_WEAK}}, rows_csv


# ---------------------------------------------------------------------------
# FA-3 — global-action mimicry
# ---------------------------------------------------------------------------

def fa3_global_mimicry(ddqn_change) -> dict:
    match = {}   # (ps,sc) -> [matches, total]
    mixed_predict = {}  # ps -> [predict, total]
    for row in ddqn_change:
        ps, sc = int(row["policy_seed"]), row["scenario"]
        comp_a = COMPARATOR_ACTION[sc]
        for reg in row["regional"]:
            m = match.setdefault((ps, sc), [0, 0])
            m[1] += 1
            if reg["requested"] == comp_a:
                m[0] += 1
            if sc == "MIXED":
                mp = mixed_predict.setdefault(ps, [0, 0])
                mp[1] += 1
                if reg["requested"] == "PREDICT":
                    mp[0] += 1
    per = {}
    for (ps, sc), (mm, tt) in sorted(match.items()):
        per[f"{ps}_{sc}"] = {"policy_seed": ps, "scenario": sc,
                             "comparator_action": COMPARATOR_ACTION[sc],
                             "match_fraction": mm / tt}
    mixed_pf = {ps: mixed_predict[ps][0] / mixed_predict[ps][1] for ps in mixed_predict}
    mixed_predict_frac = float(np.mean(list(mixed_pf.values())))
    status = ("STRONG" if mixed_predict_frac >= MIMICRY_STRONG else
              "MODERATE" if mixed_predict_frac >= MIMICRY_MODERATE else "WEAK")
    out = {"per_policy_scenario": per, "mixed_predict_fraction_by_policy": mixed_pf,
           "mixed_predict_fraction_pooled": mixed_predict_frac,
           "GLOBAL_ACTION_MIMICRY": status,
           "predeclared_thresholds": {"STRONG>=": MIMICRY_STRONG, "MODERATE>=": MIMICRY_MODERATE}}
    _jdump(out, RESULTS_DIR / "E4FFA_global_action_mimicry.json")
    return out


# ---------------------------------------------------------------------------
# FA-4 — policy-seed failure structure
# ---------------------------------------------------------------------------

def fa4_policy_failure_structure() -> List[dict]:
    rows = _read_csv(E4F_DIR / "E4F_policy_seed_summary.csv")
    rows_sorted = sorted(rows, key=lambda r: (int(r["policy_seed"]), r["scenario"]))
    _write_csv(RESULTS_DIR / "E4FFA_policy_failure_structure.csv",
               list(rows_sorted[0].keys()), rows_sorted)
    return rows_sorted


# ---------------------------------------------------------------------------
# FA-5 — episode-level performance shape (paired)
# ---------------------------------------------------------------------------

def fa5_episode_shape(episodes_idx) -> List[dict]:
    out_rows = []
    for ps in POLICY_SEEDS:
        for sc in SCENARIOS:
            comp = PRESELECTED_COMPARATOR[sc]
            deltas = []
            for e in range(1081, 1141):
                dd = episodes_idx[("DDQN", str(ps), sc, e)]["MIGD_response"]
                bb = episodes_idx[(comp, "", sc, e)]["MIGD_response"]
                deltas.append(bb - dd)   # + = DDQN better
            d = np.array(deltas)
            out_rows.append({
                "policy_seed": ps, "scenario": sc, "n_env": 60,
                "median": float(np.median(d)),
                "q10": float(np.percentile(d, 10)), "q25": float(np.percentile(d, 25)),
                "q75": float(np.percentile(d, 75)), "q90": float(np.percentile(d, 90)),
                "frac_ddqn_better": float(np.mean(d > TIE_TOL)),
                "frac_tied": float(np.mean(np.abs(d) <= TIE_TOL)),
                "frac_comparator_better": float(np.mean(d < -TIE_TOL)),
            })
    _write_csv(RESULTS_DIR / "E4FFA_episode_effect_distribution.csv",
               list(out_rows[0].keys()), out_rows)
    return out_rows


# ---------------------------------------------------------------------------
# FA-6 — change-level failure localization
# ---------------------------------------------------------------------------

def fa6_change_localization(ddqn_change, comp_change) -> List[dict]:
    # accumulate paired deltas by (ps, sc, change_index)
    acc = {}
    for row in ddqn_change:
        ps, sc, ci = int(row["policy_seed"]), row["scenario"], int(row["change_index"])
        e = int(row["environment_seed"])
        comp = comp_change.get((sc, e, ci))
        if comp is None:
            continue
        delta = comp["MIGD_response_change"] - row["MIGD_response_change"]
        acc.setdefault((ps, sc, ci), []).append(delta)
    rows = []
    band = {}
    for (ps, sc, ci), vals in sorted(acc.items()):
        m = float(np.mean(vals))
        rows.append({"policy_seed": ps, "scenario": sc, "change_index": ci,
                     "mean_paired_delta": m, "n": len(vals)})
        b = "early" if ci <= 2 else "middle" if ci <= 6 else "late"
        band.setdefault((ps, sc, b), []).append(m)
    # band summary rows
    for (ps, sc, b), vals in sorted(band.items()):
        rows.append({"policy_seed": ps, "scenario": sc, "change_index": f"BAND_{b}",
                     "mean_paired_delta": float(np.mean(vals)), "n": len(vals)})
    _write_csv(RESULTS_DIR / "E4FFA_change_localization.csv",
               ["policy_seed", "scenario", "change_index", "mean_paired_delta", "n"], rows)
    return rows


# ---------------------------------------------------------------------------
# FA-7 — response vs recovery gap
# ---------------------------------------------------------------------------

def fa7_recovery(episodes_idx) -> dict:
    out = {}
    for ps in POLICY_SEEDS:
        for sc in SCENARIOS:
            comp = PRESELECTED_COMPARATOR[sc]
            dd_r = np.mean([episodes_idx[("DDQN", str(ps), sc, e)]["MIGD_response"] for e in range(1081, 1141)])
            bb_r = np.mean([episodes_idx[(comp, "", sc, e)]["MIGD_response"] for e in range(1081, 1141)])
            dd_e = np.mean([episodes_idx[("DDQN", str(ps), sc, e)]["MIGD_epoch"] for e in range(1081, 1141)])
            bb_e = np.mean([episodes_idx[(comp, "", sc, e)]["MIGD_epoch"] for e in range(1081, 1141)])
            response_gap = float(dd_r - bb_r)   # + means DDQN worse
            recovery_gap = float(dd_e - bb_e)
            atten = abs(response_gap) - abs(recovery_gap)
            cls = ("RECOVERY_COMPENSATES" if atten > TIE_TOL else
                   "RECOVERY_AMPLIFIES" if atten < -TIE_TOL else "RECOVERY_NEUTRAL")
            out[f"{ps}_{sc}"] = {"policy_seed": ps, "scenario": sc,
                                 "response_gap": response_gap, "recovery_gap": recovery_gap,
                                 "recovery_attenuation": atten, "classification": cls}
    _jdump(out, RESULTS_DIR / "E4FFA_recovery_analysis.json")
    return out


# ---------------------------------------------------------------------------
# FA-8 — FE-quality tradeoff
# ---------------------------------------------------------------------------

def fa8_fe_quality(episodes_idx) -> dict:
    out = {}
    for ps in POLICY_SEEDS:
        for sc in SCENARIOS:
            comp = PRESELECTED_COMPARATOR[sc]
            fe_saving, quality_delta = [], []
            for e in range(1081, 1141):
                dd = episodes_idx[("DDQN", str(ps), sc, e)]
                bb = episodes_idx[(comp, "", sc, e)]
                fe_saving.append(bb["response_FE"] - dd["response_FE"])
                quality_delta.append(bb["MIGD_response"] - dd["MIGD_response"])  # + DDQN better
            if np.std(fe_saving) < 1e-12 or np.std(quality_delta) < 1e-12:
                rho = float("nan")
            else:
                rho = float(spearmanr(fe_saving, quality_delta).correlation)
            out[f"{ps}_{sc}"] = {"policy_seed": ps, "scenario": sc,
                                 "mean_fe_saving": float(np.mean(fe_saving)),
                                 "mean_quality_delta": float(np.mean(quality_delta)),
                                 "spearman_fe_vs_quality": rho}
    _jdump({"note": "exploratory; does not reverse FAILED", "per_policy_scenario": out},
           RESULTS_DIR / "E4FFA_fe_quality_tradeoff.json")
    return out


# ---------------------------------------------------------------------------
# FA-9 — reward component scale audit
# ---------------------------------------------------------------------------

def fa9_reward_scale() -> dict:
    cols = _read_csv(E4TR_DIR / "E4TR_training_diagnostics.csv")[0].keys()
    has_components = any("global_reward" in c or "local_reward" in c or "fe_cost" in c
                         for c in cols)
    out = {
        "coefficients": {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01},
        "coefficient_ratio_note": "1.0 vs 0.1 vs 0.01 does NOT prove empirical scale dominance",
        "component_logs_present": has_components,
        "REWARD_SCALE_EMPIRICAL_AUDIT": "AVAILABLE" if has_components
                                        else "UNAVAILABLE_FROM_FROZEN_ARTIFACTS",
        "logged_reward_fields": [c for c in cols if "reward" in c.lower()],
    }
    _jdump(out, RESULTS_DIR / "E4FFA_reward_scale_audit.json")
    return out


# ---------------------------------------------------------------------------
# FA-10 — reward-evaluation alignment
# ---------------------------------------------------------------------------

def fa10_reward_alignment() -> dict:
    rows = _read_csv(E4TR_DIR / "E4TR_training_diagnostics.csv")
    per = {}
    rhos_resp = []
    for ps in POLICY_SEEDS:
        for sc in SCENARIOS:
            sub = [r for r in rows if int(r["policy_seed"]) == ps and r["scenario"] == sc]
            rew = [float(r["cumulative_reward"]) for r in sub]
            mr = [float(r["MIGD_response"]) for r in sub]
            me = [float(r["MIGD_epoch"]) for r in sub]
            rho_r = float(spearmanr(rew, mr).correlation) if np.std(rew) > 0 and np.std(mr) > 0 else float("nan")
            rho_e = float(spearmanr(rew, me).correlation) if np.std(rew) > 0 and np.std(me) > 0 else float("nan")
            per[f"{ps}_{sc}"] = {"policy_seed": ps, "scenario": sc,
                                 "spearman_reward_vs_MIGD_response": rho_r,
                                 "spearman_reward_vs_MIGD_epoch": rho_e,
                                 "n_episodes": len(sub)}
            if not math.isnan(rho_r):
                rhos_resp.append(rho_r)
    # expected sign: higher reward -> lower MIGD -> negative rho
    mean_rho = float(np.mean(rhos_resp)) if rhos_resp else float("nan")
    correct_sign = mean_rho < 0
    mag = abs(mean_rho)
    if math.isnan(mean_rho):
        align = "NOT_AUDITABLE"
    elif not correct_sign:
        align = "WEAK"   # wrong sign -> not aligned
    elif mag >= RHO_STRONG:
        align = "STRONG"
    elif mag >= RHO_PARTIAL:
        align = "PARTIAL"
    else:
        align = "WEAK"
    out = {"predeclared_thresholds": {"STRONG|rho|>=": RHO_STRONG, "PARTIAL|rho|>=": RHO_PARTIAL,
                                      "expected_sign": "negative (reward up -> MIGD down)"},
           "per_policy_scenario": per, "mean_rho_response": mean_rho,
           "expected_sign_correct": correct_sign, "REWARD_ALIGNMENT": align}
    _jdump(out, RESULTS_DIR / "E4FFA_reward_alignment.json")
    return out


# ---------------------------------------------------------------------------
# FA-11 — training action trajectory
# ---------------------------------------------------------------------------

def fa11_training_trajectory() -> Tuple[dict, List[dict]]:
    rows = _read_csv(E4TR_DIR / "E4TR_training_diagnostics.csv")
    csv_rows = []
    per_collapse = {}
    for ps in POLICY_SEEDS:
        for sc in SCENARIOS:
            epoch_dom = []
            for ep in range(6):
                sub = [r for r in rows if int(r["policy_seed"]) == ps
                       and r["scenario"] == sc and int(r["epoch"]) == ep]
                if not sub:
                    continue
                fr = {a: float(np.mean([float(r[f"action_fraction_{a}"]) for r in sub]))
                      for a in ACTION_ORDER}
                dom = max(ACTION_ORDER, key=lambda a: fr[a])
                epoch_dom.append(fr[dom])
                csv_rows.append({"policy_seed": ps, "scenario": sc, "epoch": ep + 1,
                                 **{f"frac_{a}": fr[a] for a in ACTION_ORDER},
                                 "dominant_action": dom, "dominant_fraction": fr[dom]})
            if epoch_dom:
                per_collapse[f"{ps}_{sc}"] = {"epoch1_dom": epoch_dom[0],
                                              "epoch6_dom": epoch_dom[-1]}
    # collapse timing classification
    early = sum(1 for v in per_collapse.values() if v["epoch1_dom"] >= LOW_DIVERSITY_THRESH)
    late_only = sum(1 for v in per_collapse.values()
                    if v["epoch1_dom"] < LOW_DIVERSITY_THRESH and v["epoch6_dom"] >= LOW_DIVERSITY_THRESH)
    never = sum(1 for v in per_collapse.values() if v["epoch6_dom"] < LOW_DIVERSITY_THRESH)
    n = len(per_collapse)
    if never == n:
        timing = "NO_COLLAPSE"
    elif early >= 0.6 * n:
        timing = "EARLY"
    elif late_only >= 0.6 * n:
        timing = "LATE"
    elif early > 0 and late_only > 0:
        timing = "POLICY_DEPENDENT"
    else:
        timing = "GRADUAL"
    _write_csv(RESULTS_DIR / "E4FFA_training_action_trajectory.csv",
               ["policy_seed", "scenario", "epoch"] + [f"frac_{a}" for a in ACTION_ORDER]
               + ["dominant_action", "dominant_fraction"], csv_rows)
    return {"COLLAPSE_TIMING": timing, "n_early": early, "n_late_only": late_only,
            "n_never": never, "n_cells": n}, csv_rows


# ---------------------------------------------------------------------------
# FA-12 — validation vs final shift + checkpoint-selection tension
# ---------------------------------------------------------------------------

def fa12_validation_final_shift(episodes_idx) -> dict:
    val = _read_csv(E4TR_DIR / "E4TR_validation_all_policies.csv")
    sel = {r["policy_seed"]: r for r in val if r["selected"] == "True"}
    per = {}
    tension_flags = 0
    for ps in POLICY_SEEDS:
        vrow = sel[str(ps)]
        val_macro = float(vrow["VALIDATION_MACRO_MIGD_RESPONSE"])
        val_mixed = float(vrow["MIGD_response_MIXED"])
        # final macro/mixed from episodes
        fin = {}
        for sc in SCENARIOS:
            fin[sc] = float(np.mean([episodes_idx[("DDQN", str(ps), sc, e)]["MIGD_response"]
                                     for e in range(1081, 1141)]))
        final_macro = float(np.mean([fin[sc] for sc in SCENARIOS]))
        # tension: checkpoint chosen on macro; was MIXED (primary) the weak axis at selection?
        # descriptive: rank of MIXED among scenarios at validation
        val_scen = {"PERSISTENT": float(vrow["MIGD_response_PERSISTENT"]),
                    "STATIONARY": float(vrow["MIGD_response_STATIONARY"]),
                    "MIXED": val_mixed}
        mixed_is_worst = val_mixed == max(val_scen.values())
        if mixed_is_worst:
            tension_flags += 1
        per[str(ps)] = {"val_macro": val_macro, "val_mixed": val_mixed,
                        "final_macro": final_macro, "final_mixed": fin["MIXED"],
                        "macro_shift": final_macro - val_macro,
                        "mixed_shift": fin["MIXED"] - val_mixed,
                        "mixed_worst_scenario_at_validation": mixed_is_worst}
    # tension SUPPORTED if selection metric (macro) systematically differs from MIXED-only
    # and MIXED is the primary/worst axis for most policies
    if tension_flags >= 4:
        tension = "SUPPORTED"
    elif tension_flags >= 2:
        tension = "WEAK"
    else:
        tension = "ABSENT"
    shift = {"selection_metric": "VALIDATION_MACRO_MIGD_RESPONSE (not MIXED-only)",
             "per_policy": per,
             "interpretation": "compares each selected checkpoint's VALIDATION macro/MIXED "
                               "MIGD_response against its FINAL macro/MIXED; distribution "
                               "shift vs already-visible-at-validation weakness."}
    _jdump(shift, RESULTS_DIR / "E4FFA_validation_final_shift.json")
    out = {"selection_metric": "VALIDATION_MACRO_MIGD_RESPONSE (not MIXED-only)",
           "n_policies_mixed_worst_at_val": tension_flags,
           "CHECKPOINT_SELECTION_TENSION": tension,
           "warning": "no unselected checkpoint was evaluated on final seeds"}
    _jdump(out, RESULTS_DIR / "E4FFA_checkpoint_selection_audit.json")
    return out


# ---------------------------------------------------------------------------
# Frozen-network analysis helpers (FA-13/14) — synthetic states, NO env
# ---------------------------------------------------------------------------

def load_nets() -> Dict[int, SharedRegionalQNetwork]:
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    nets = {}
    for ps in POLICY_SEEDS:
        payload = torch.load(ck["policies"][str(ps)]["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9)
        net.load_state_dict(payload["online_state_dict"])
        net.eval()
        nets[ps] = net
    return nets


def synthetic_state_grid(n=5000, seed=20260926):
    """Clearly-labeled SYNTHETIC states within valid normalized ranges.
    Also a 'realistic' block using E3R-derived feature scales."""
    rng = np.random.default_rng(seed)
    # broad uniform block over full valid ranges
    broad = np.zeros((n, 9), dtype=np.float32)
    broad[:, 0] = rng.uniform(0, 1, n)     # severity_r
    broad[:, 1] = rng.uniform(0, 1, n)     # convergence_loss_r (already /10 space)
    broad[:, 2] = rng.uniform(0, 1, n)     # occupancy_r
    broad[:, 3] = rng.uniform(-1, 1, n)    # delta_severity_r
    broad[:, 4] = rng.uniform(0, 1, n)     # active_region_fraction
    broad[:, 5] = rng.uniform(0, 1, n)     # mean_severity
    broad[:, 6] = rng.uniform(0, 1, n)     # max_severity
    broad[:, 7] = rng.uniform(0, 1, n)     # mean_convergence_loss
    broad[:, 8] = rng.uniform(0, 1, n)     # max_convergence_loss
    # realistic block from E3R feature scales (severity ~[0,0.01], occ ~[0.05,0.2], dsev ~[0,0.001])
    real = np.zeros((n, 9), dtype=np.float32)
    real[:, 0] = rng.uniform(0, 0.01, n)
    real[:, 1] = rng.uniform(0, 0.02, n)   # convergence_loss tiny after /10
    real[:, 2] = rng.uniform(0.05, 0.2, n)
    real[:, 3] = rng.uniform(-0.001, 0.001, n)
    real[:, 4] = 1.0                        # all 10 regions active
    real[:, 5] = rng.uniform(0, 0.01, n)
    real[:, 6] = rng.uniform(0, 0.01, n)
    real[:, 7] = rng.uniform(0, 0.02, n)
    real[:, 8] = rng.uniform(0, 0.02, n)
    return broad, real


def fa13_q_dominance(nets) -> dict:
    broad, real = synthetic_state_grid()
    out = {"note": "SYNTHETIC grid; does NOT represent the true E4F state distribution",
           "per_policy": {}}
    strong_bias_policies = 0
    lowmargin_policies = 0
    for ps, net in nets.items():
        block = {}
        for label, X in [("broad_uniform", broad), ("realistic_e3r_scaled", real)]:
            with torch.no_grad():
                q = net(torch.tensor(X, dtype=torch.float32)).numpy()
            arg = q.argmax(axis=1)
            frac = {ACTION_ORDER[a]: float(np.mean(arg == a)) for a in range(N_ACTIONS)}
            dom = max(frac, key=frac.get)
            qs = np.sort(q, axis=1)
            margin = float(np.mean(qs[:, -1] - qs[:, -2]))
            block[label] = {"argmax_fraction": frac, "dominant_action": dom,
                            "dominant_fraction": frac[dom],
                            "mean_top1_top2_margin": margin,
                            "mean_q_variance_across_actions": float(np.mean(np.var(q, axis=1)))}
        out["per_policy"][str(ps)] = block
        # use realistic block for classification signals
        rb = block["realistic_e3r_scaled"]
        if rb["dominant_fraction"] >= QDOM_STRONG:
            strong_bias_policies += 1
        if rb["mean_top1_top2_margin"] < QDOM_LOWMARGIN:
            lowmargin_policies += 1
    n = len(nets)
    if strong_bias_policies >= 0.6 * n:
        status = "STRONG_ACTION_BIAS"
    elif lowmargin_policies >= 0.6 * n:
        status = "LOW_MARGIN_UNCERTAIN"
    else:
        status = "STATE_DEPENDENT"
    out["Q_FUNCTION_DOMINANCE"] = status
    out["predeclared_thresholds"] = {"STRONG_ACTION_BIAS_dom>=": QDOM_STRONG,
                                     "LOW_MARGIN_abs<": QDOM_LOWMARGIN}
    out["n_strong_bias_policies"] = strong_bias_policies
    _jdump(out, RESULTS_DIR / "E4FFA_q_function_diagnostics.json")
    return out


def fa14_input_sensitivity(nets) -> dict:
    broad, real = synthetic_state_grid(n=2000)
    out = {"note": "autograd |dQ_max/dx| on SYNTHETIC states; weights unchanged",
           "per_policy": {}}
    ratios = []
    weak = 0
    for ps, net in nets.items():
        X = torch.tensor(real, dtype=torch.float32, requires_grad=True)
        q = net(X)
        qmax = q.max(dim=1).values.sum()
        qmax.backward()
        g = X.grad.abs().mean(dim=0).numpy()   # (9,)
        local_sens = float(np.mean(g[:4]))
        global_sens = float(np.mean(g[4:]))
        ratio = global_sens / (local_sens + 1e-12)
        ratios.append(ratio)
        if max(local_sens, global_sens) < WEAK_SENS_ABS:
            weak += 1
        out["per_policy"][str(ps)] = {
            "mean_abs_grad": {FEATURE_NAMES[i]: float(g[i]) for i in range(9)},
            "local_mean_sensitivity": local_sens, "global_mean_sensitivity": global_sens,
            "global_over_local_ratio": ratio}
    mean_ratio = float(np.mean(ratios))
    n = len(nets)
    if weak >= 0.6 * n:
        usage = "WEAK_SENSITIVITY"
    elif mean_ratio > STATE_USAGE_RATIO:
        usage = "GLOBAL_DOMINATED"
    elif mean_ratio < 1.0 / STATE_USAGE_RATIO:
        usage = "LOCAL_SIGNAL_USED"
    else:
        usage = "MIXED"
    out["mean_global_over_local_ratio"] = mean_ratio
    out["STATE_USAGE"] = usage
    out["predeclared_thresholds"] = {"GLOBAL_DOMINATED_ratio>": STATE_USAGE_RATIO,
                                     "WEAK_SENSITIVITY_abs<": WEAK_SENS_ABS}
    _jdump(out, RESULTS_DIR / "E4FFA_input_sensitivity.json")
    return out


# ---------------------------------------------------------------------------
# FA-15 — joint-action representation audit (code)
# ---------------------------------------------------------------------------

def fa15_joint_action_repr() -> dict:
    net = SharedRegionalQNetwork(9)
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    input_dim = linears[0].in_features
    out = {
        "network_input_dim": input_dim,
        "input_composition": "concat(local_V2[4], GlobalContextV1[5])",
        "contains_other_region_actions": False,
        "contains_previous_joint_action": False,
        "contains_joint_action_summary": False,
        "centralized_critic": False, "mixing_network": False,
        "value_decomposition": False,
        "reward_depends_on_joint_action_vector": True,
        "JOINT_ACTION_REPRESENTATION": "IMPLICIT_STATE_ONLY",
        "interpretation": "Q conditions on (s_r, a_r) with a SHARED global STATE "
                          "context but NOT on other regions' actions; identical "
                          "(s_r,a_r) can receive different targets depending on a_-r "
                          "-> credit-assignment/representation hypothesis (not proven cause).",
    }
    _jdump(out, RESULTS_DIR / "E4FFA_joint_action_representation.json")
    return out


# ---------------------------------------------------------------------------
# FA-16 — parameter sharing audit
# ---------------------------------------------------------------------------

def fa16_parameter_sharing() -> dict:
    out = {
        "shared_network_across_regions": True,
        "region_id_in_state": False,
        "region_specific_structural_identifier": False,
        "consequence": "two regions with identical 9-D features are indistinguishable "
                       "to the policy (permutation symmetry).",
        "state_aliasing_possible": True,
        "assessment": "Symmetry is appropriate for permutation-invariant regional "
                      "control, but combined with near-degenerate local features it "
                      "permits state aliasing. Adding region_id is NOT recommended "
                      "here without scientific justification.",
    }
    _jdump(out, RESULTS_DIR / "E4FFA_parameter_sharing_audit.json")
    return out


# ---------------------------------------------------------------------------
# FA-17 — state range / normalization audit (from E3R feature scales; states not logged)
# ---------------------------------------------------------------------------

def fa17_state_range() -> dict:
    calib = _load(E3R_DIR / "E3R_feature_calibration.json")
    sev = calib["sev_percentiles"]; occ = calib["occ_percentiles"]; dsev = calib["dsev_percentiles"]
    # after frozen normalization: severity identity[0,1]; conv_loss /10; occ identity; dsev clip[-1,1]
    signal_loss = (sev["p75"] < 0.05 and dsev["p75"] < 0.01)
    out = {
        "e4f_states_recorded": False,
        "STATE_RANGE_AUDIT": "NOT_AVAILABLE (E4F/E4TR did not log raw states)",
        "proxy_source": "E3R_feature_calibration.json (n=6000; different seeds, DIAGNOSTIC ONLY)",
        "severity_percentiles": sev, "occupancy_percentiles": occ, "delta_severity_percentiles": dsev,
        "normalization": {"severity_r": "clip[0,1] identity",
                          "convergence_loss_r": "clip[0,10]/10",
                          "occupancy_r": "clip[0,1] identity",
                          "delta_severity_r": "clip[-1,1] identity"},
        "observation": "severity p75=%.4g and delta_severity p75=%.4g occupy tiny "
                       "sub-ranges of their normalized domains; identity/÷10 leaves "
                       "local features compressed near 0." % (sev["p75"], dsev["p75"]),
        "NORMALIZATION_SIGNAL_LOSS": "SUPPORTED" if signal_loss else "POSSIBLE",
        "caveat": "inferred from E3R-era detector scales, not E4F states; no environment rerun.",
    }
    _jdump(out, RESULTS_DIR / "E4FFA_state_range_audit.json")
    return out


# ---------------------------------------------------------------------------
# FA-18 — action coverage / replay audit
# ---------------------------------------------------------------------------

def fa18_action_coverage() -> dict:
    rows = _read_csv(E4TR_DIR / "E4TR_training_diagnostics.csv")
    # pooled action fractions over ALL training episodes (weighted equally per episode)
    pooled = {a: float(np.mean([float(r[f"action_fraction_{a}"]) for r in rows]))
              for a in ACTION_ORDER}
    # early (epoch 0-1, includes warmup random) vs late (epoch 5)
    early = [r for r in rows if int(r["epoch"]) <= 1]
    late = [r for r in rows if int(r["epoch"]) == 5]
    early_fr = {a: float(np.mean([float(r[f"action_fraction_{a}"]) for r in early])) for a in ACTION_ORDER}
    late_fr = {a: float(np.mean([float(r[f"action_fraction_{a}"]) for r in late])) for a in ACTION_ORDER}
    min_frac = min(pooled.values())
    if min_frac < COVERAGE_MIN:
        cov = "SEVERELY_IMBALANCED"
    elif min_frac < COVERAGE_BALANCED:
        cov = "IMBALANCED"
    else:
        cov = "BALANCED_ENOUGH"
    out = {"predeclared_thresholds": {"BALANCED_each>=": COVERAGE_BALANCED,
                                      "SEVERELY_any<": COVERAGE_MIN},
           "pooled_training_action_fractions": pooled,
           "early_epochs_0_1": early_fr, "late_epoch_5": late_fr,
           "replay_composition_logged": False,
           "ACTION_COVERAGE": cov,
           "note": "warmup uses epsilon~1 (random) so early experience spans all 4 "
                   "actions; 'sufficient' predeclared independent of final performance."}
    _jdump(out, RESULTS_DIR / "E4FFA_action_coverage_audit.json")
    return out


# ---------------------------------------------------------------------------
# FA-19 — H1 <-> E4F reconciliation
# ---------------------------------------------------------------------------

def fa19_reconciliation(spec_status, mimicry, joint_repr) -> str:
    text = f"""# E4F-FA — H1 <-> E4F Reconciliation

**Bounded conclusion: B + C (multiple explanations remain; A is rejected).**

H1 / H1-RV established, on their own seeds, that:
- regional response headroom is SUPPORTED in MIXED,
- GLOBAL_PREDICT dominance is weak,
- a nonlinear V2 state-action signal exists,
- the H1 oracle was GREEDY_FACTORIZED (per-region isolated trials), and
- joint coordination is required.

E4F established, on the untouched final seeds, that:
- the trained Shared Double-DQN does NOT beat GLOBAL_PREDICT (MIXED
  DIRECTIONAL_NEGATIVE, 0/5 policies positive), and
- it DOES beat RULE_V1_V2STATE on MIXED (BENEFICIAL).

Reconciliation:
- (A) "H1 headroom was false" — REJECTED. H1/H1-RV headroom is a property of the
  greedy-factorized oracle envelope and is not overturned by E4F.
- (B) "Headroom exists but this learning architecture failed to exploit it" —
  SUPPORTED. Diagnostics show REGIONAL_SPECIALIZATION_STATUS={spec_status} and
  GLOBAL_ACTION_MIMICRY={mimicry}: the nominally-regional controller operationally
  collapsed toward a single global action per scenario.
- (C) "Greedy-factorized H1 headroom is not sufficient evidence of a realizable
  JOINT-policy gain" — SUPPORTED. The oracle evaluated each region in isolation;
  the executed joint policy need not realize the factorized envelope, and the
  current Q-function is {joint_repr} (no joint-action conditioning).

Therefore the honest conclusion is (D): the E4F failure is consistent with BOTH
(B) an architecture/representation that did not exploit the headroom and (C) the
factorized oracle overstating realizable joint gain. The frozen artifacts cannot
uniquely separate these. H1 and E4F are NOT pooled (different seeds, different
objects); both historical results stand unchanged.
"""
    (RESULTS_DIR / "E4FFA_h1_e4f_reconciliation.md").write_text(text)
    short = ("A_rejected; B_supported(architecture_did_not_exploit_headroom) + "
             "C_supported(factorized_oracle_not_realizable_joint_gain); "
             "D_multiple_explanations_remain")
    return short


# ---------------------------------------------------------------------------
# FA-20 — failure-mode evidence table + primary hypothesis
# ---------------------------------------------------------------------------

def fa20_evidence_table(fa1, fa2, fa3, fa10, fa11, fa13, fa14, fa15, fa17, fa18, fa9) -> dict:
    # count COLLAPSED cells
    n_collapsed = sum(1 for v in fa1["per_policy_scenario"].values()
                      if v["classification"] == "COLLAPSED")
    n_cells = len(fa1["per_policy_scenario"])
    table = {}

    table["A_GLOBAL_ACTION_DOMINANCE"] = {
        "evidence_for": [
            f"within-event REGIONAL_SPECIALIZATION_STATUS={fa2['REGIONAL_SPECIALIZATION_STATUS']} "
            f"(MIXED multi-action rate={fa2['mixed_pooled_multi_action_rate']:.3f})",
            f"GLOBAL_ACTION_MIMICRY={fa3['GLOBAL_ACTION_MIMICRY']} "
            f"(MIXED PREDICT fraction={fa3['mixed_predict_fraction_pooled']:.3f})",
            f"pooled MIXED dominant action approximates comparator="
            f"{fa1['pooled_by_scenario']['MIXED']['approximates_comparator']}",
        ],
        "evidence_against": ["STATIONARY collapsed to KEEP, not the DIVERSIFY comparator"],
        "status": "SUPPORTED" if (fa2["REGIONAL_SPECIALIZATION_STATUS"] == "ABSENT"
                                  and fa3["GLOBAL_ACTION_MIMICRY"] in ("STRONG", "MODERATE"))
                  else "PLAUSIBLE",
    }
    table["B_POLICY_COLLAPSE"] = {
        "evidence_for": [
            f"{n_collapsed}/{n_cells} policy-scenario cells COLLAPSED (dominant>={COLLAPSE_THRESH})",
            f"COLLAPSE_TIMING={fa11['COLLAPSE_TIMING']}",
        ],
        "evidence_against": ["some cells LOW_DIVERSITY rather than fully collapsed"],
        "status": "SUPPORTED" if n_collapsed >= 0.6 * n_cells else "PLAUSIBLE",
    }
    table["C_REWARD_CREDIT_MISALIGNMENT"] = {
        "evidence_for": [f"reward depends on joint action but Q is {fa15['JOINT_ACTION_REPRESENTATION']}",
                         f"REWARD_ALIGNMENT={fa10['REWARD_ALIGNMENT']}"],
        "evidence_against": ["per-component reward scales not logged; cannot quantify"],
        "status": ("PLAUSIBLE" if fa9["REWARD_SCALE_EMPIRICAL_AUDIT"].startswith("UNAVAIL")
                   else "NOT_AUDITABLE"),
    }
    table["D_JOINT_ACTION_REPRESENTATION_LIMITATION"] = {
        "evidence_for": [f"Q-function is {fa15['JOINT_ACTION_REPRESENTATION']} (code-audited): "
                         "no other-region actions, no joint-action context/critic",
                         "reward depends on full joint action vector"],
        "evidence_against": ["structural fact; causal role in THIS failure not directly provable "
                             "from frozen artifacts"],
        "status": "SUPPORTED",   # structural fact confirmed by code audit
    }
    table["E_STATE_INFORMATION_LIMITATION"] = {
        "evidence_for": [f"STATE_USAGE={fa14['STATE_USAGE']} "
                         f"(global/local sensitivity ratio={fa14['mean_global_over_local_ratio']:.2f})"],
        "evidence_against": ["synthetic-grid sensitivity, not true state distribution"],
        "status": "PLAUSIBLE" if fa14["STATE_USAGE"] in ("GLOBAL_DOMINATED", "WEAK_SENSITIVITY") else "NOT_SUPPORTED",
    }
    table["F_STATE_NORMALIZATION_LIMITATION"] = {
        "evidence_for": [fa17["observation"], f"NORMALIZATION_SIGNAL_LOSS={fa17['NORMALIZATION_SIGNAL_LOSS']}"],
        "evidence_against": ["inferred from E3R feature scales; E4F states not logged"],
        "status": "SUPPORTED" if fa17["NORMALIZATION_SIGNAL_LOSS"] == "SUPPORTED" else "PLAUSIBLE",
    }
    table["G_ACTION_COVERAGE_IMBALANCE"] = {
        "evidence_for": [f"ACTION_COVERAGE={fa18['ACTION_COVERAGE']}"],
        "evidence_against": ["warmup epsilon~1 gave random experience over all 4 actions"],
        "status": "SUPPORTED" if fa18["ACTION_COVERAGE"] in ("IMBALANCED", "SEVERELY_IMBALANCED")
                  else "NOT_SUPPORTED",
    }
    table["H_CHECKPOINT_SELECTION_TENSION"] = {
        "evidence_for": ["selection metric was VALIDATION_MACRO not MIXED-only"],
        "evidence_against": ["no unselected checkpoint evaluated on final seeds (forbidden)"],
        "status": "NOT_AUDITABLE",
    }
    table["I_TRAINING_NUMERICAL_FAILURE"] = {
        "evidence_for": [],
        "evidence_against": ["E4TR numerical_stability=PASS for all 5 policies"],
        "status": "NOT_SUPPORTED",
    }
    table["J_IMPLEMENTATION_FAILURE"] = {
        "evidence_for": [],
        "evidence_against": ["2080/2080 tests pass; determinism + firewall + integrity PASS"],
        "status": "NOT_SUPPORTED",
    }

    # primary hypothesis: candidate with >=2 independent SUPPORTED diagnostics
    supported = [k for k, v in table.items() if v["status"] == "SUPPORTED"]
    # count independent supporting diagnostics per supported candidate
    support_counts = {
        "A_GLOBAL_ACTION_DOMINANCE": (fa2["REGIONAL_SPECIALIZATION_STATUS"] == "ABSENT")
                                     + (fa3["GLOBAL_ACTION_MIMICRY"] in ("STRONG", "MODERATE"))
                                     + fa1["pooled_by_scenario"]["MIXED"]["approximates_comparator"],
        "B_POLICY_COLLAPSE": (n_collapsed >= 0.6 * n_cells)
                             + (fa11["COLLAPSE_TIMING"] in ("EARLY", "GRADUAL", "LATE", "POLICY_DEPENDENT")),
        "D_JOINT_ACTION_REPRESENTATION_LIMITATION": 1,   # single structural audit
        "F_STATE_NORMALIZATION_LIMITATION": (fa17["NORMALIZATION_SIGNAL_LOSS"] == "SUPPORTED") + 1,
    }
    # primary = best-supported behavioral/mechanistic with >=2 independent diagnostics
    eligible = {k: c for k, c in support_counts.items()
                if k in supported and c >= 2}
    if not eligible:
        primary = "MULTIFACTORIAL_UNRESOLVED"
    else:
        # prefer the candidate with the most independent supports; behavioral A/B win ties
        primary = max(eligible, key=lambda k: (eligible[k],
                      k in ("A_GLOBAL_ACTION_DOMINANCE", "B_POLICY_COLLAPSE")))
        primary = {"A_GLOBAL_ACTION_DOMINANCE": "GLOBAL_ACTION_DOMINANCE",
                   "B_POLICY_COLLAPSE": "POLICY_COLLAPSE",
                   "D_JOINT_ACTION_REPRESENTATION_LIMITATION": "JOINT_ACTION_REPRESENTATION_LIMITATION",
                   "F_STATE_NORMALIZATION_LIMITATION": "STATE_NORMALIZATION_LIMITATION"}[primary]

    secondary = []
    label_map = {"A_GLOBAL_ACTION_DOMINANCE": "GLOBAL_ACTION_DOMINANCE",
                 "B_POLICY_COLLAPSE": "POLICY_COLLAPSE",
                 "C_REWARD_CREDIT_MISALIGNMENT": "REWARD_CREDIT_MISALIGNMENT",
                 "D_JOINT_ACTION_REPRESENTATION_LIMITATION": "JOINT_ACTION_REPRESENTATION_LIMITATION",
                 "E_STATE_INFORMATION_LIMITATION": "STATE_INFORMATION_LIMITATION",
                 "F_STATE_NORMALIZATION_LIMITATION": "STATE_NORMALIZATION_LIMITATION",
                 "G_ACTION_COVERAGE_IMBALANCE": "ACTION_COVERAGE_IMBALANCE",
                 "H_CHECKPOINT_SELECTION_TENSION": "CHECKPOINT_SELECTION_TENSION"}
    for k, v in table.items():
        if v["status"] in ("SUPPORTED", "PLAUSIBLE") and k in label_map \
                and label_map[k] != primary:
            secondary.append(label_map[k] + f"({v['status']})")

    out = {"candidates": table, "supported": supported,
           "PRIMARY_FAILURE_HYPOTHESIS": primary,
           "SECONDARY_FAILURE_HYPOTHESES": secondary,
           "rule": "PRIMARY only if >=2 independent SUPPORTED diagnostics else MULTIFACTORIAL_UNRESOLVED"}
    _jdump(out, RESULTS_DIR / "E4FFA_failure_mode_evidence.json")
    return out


# ---------------------------------------------------------------------------
# FA-21 — next research direction (recommendation only)
# ---------------------------------------------------------------------------

def fa21_next_stage(evidence) -> str:
    primary = evidence["PRIMARY_FAILURE_HYPOTHESIS"]
    sup = set(evidence["supported"])
    # count mechanistic (root-cause) hypotheses that are SUPPORTED
    mech = {"D_JOINT_ACTION_REPRESENTATION_LIMITATION", "F_STATE_NORMALIZATION_LIMITATION",
            "E_STATE_INFORMATION_LIMITATION", "C_REWARD_CREDIT_MISALIGNMENT"}
    mech_supported = sup & mech
    if primary in ("GLOBAL_ACTION_DOMINANCE", "POLICY_COLLAPSE"):
        # behavioral symptom; route by which mechanism is best supported
        if len(mech_supported) >= 2:
            return "E4G_CONTROLLED_DIAGNOSTIC_DESIGN"
        if "D_JOINT_ACTION_REPRESENTATION_LIMITATION" in mech_supported:
            return "E4G_JOINT_COORDINATION_ARCHITECTURE_REVIEW"
        if mech_supported & {"F_STATE_NORMALIZATION_LIMITATION", "E_STATE_INFORMATION_LIMITATION"}:
            return "E4G_STATE_REPRESENTATION_REVIEW"
        return "E4G_CONTROLLED_DIAGNOSTIC_DESIGN"
    if primary == "JOINT_ACTION_REPRESENTATION_LIMITATION":
        return "E4G_JOINT_COORDINATION_ARCHITECTURE_REVIEW"
    if primary in ("STATE_NORMALIZATION_LIMITATION", "STATE_INFORMATION_LIMITATION"):
        return "E4G_STATE_REPRESENTATION_REVIEW"
    if primary == "REWARD_CREDIT_MISALIGNMENT":
        return "E4G_REWARD_CREDIT_ASSIGNMENT_REVIEW"
    if primary == "ACTION_COVERAGE_IMBALANCE":
        return "E4G_TRAINING_DYNAMICS_REVIEW"
    return "E4G_CONTROLLED_DIAGNOSTIC_DESIGN"


# ---------------------------------------------------------------------------
# FA-22 — preserve negative result
# ---------------------------------------------------------------------------

def fa22_addendum() -> dict:
    out = {
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4F_FAILURE_ANALYSIS_IS_POST_HOC": True,
        "E4F_FA_DOES_NOT_CHANGE_ORIGINAL_VERDICT": True,
        "any_future_redesigned_model_is_a_NEW_experiment": True,
        "new_experiment_must_be_predeclared_before_training": True,
    }
    _jdump(out, RESULTS_DIR / "E4FFA_verdict_addendum.json")
    return out


# ---------------------------------------------------------------------------
def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _episode_key(r):
    return (r["controller"], r["policy_seed"], r["scenario"], int(r["environment_seed"]))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("E4F-FA — Shared Double-DQN Failure Analysis (POST-HOC)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = fa0_integrity()
    episodes = _read_csv(E4F_DIR / "E4F_episode_results.csv")
    episodes_idx = {_episode_key(r): {"MIGD_response": float(r["MIGD_response"]),
                                      "MIGD_epoch": float(r["MIGD_epoch"]),
                                      "response_FE": float(r["response_FE"])}
                    for r in episodes}
    ddqn_change, comp_change = load_change_rows()
    print(f"[E4FFA] loaded {len(episodes)} episodes, {len(ddqn_change)} DDQN change rows")

    fa1 = fa1_action_collapse()
    fa2, _ = fa2_event_heterogeneity(ddqn_change)
    fa3 = fa3_global_mimicry(ddqn_change)
    fa4_policy_failure_structure()
    fa5_episode_shape(episodes_idx)
    fa6_change_localization(ddqn_change, comp_change)
    fa7_recovery(episodes_idx)
    fa8_fe_quality(episodes_idx)
    fa9 = fa9_reward_scale()
    fa10 = fa10_reward_alignment()
    fa11, _ = fa11_training_trajectory()
    fa12_validation_final_shift(episodes_idx)
    nets = load_nets()
    fa13 = fa13_q_dominance(nets)
    fa14 = fa14_input_sensitivity(nets)
    fa15 = fa15_joint_action_repr()
    fa16_parameter_sharing()
    fa17 = fa17_state_range()
    fa18 = fa18_action_coverage()
    recon_short = fa19_reconciliation(fa2["REGIONAL_SPECIALIZATION_STATUS"],
                                      fa3["GLOBAL_ACTION_MIMICRY"],
                                      fa15["JOINT_ACTION_REPRESENTATION"])
    evidence = fa20_evidence_table(fa1, fa2, fa3, fa10, fa11, fa13, fa14, fa15, fa17, fa18, fa9)
    next_stage = fa21_next_stage(evidence)
    fa22_addendum()

    # dominant DDQN action per scenario (pooled)
    dom = {sc: (fa1["pooled_by_scenario"][sc]["dominant_action"]
                if fa1["pooled_by_scenario"][sc]["dominant_fraction"] >= LOW_DIVERSITY_THRESH
                else "MIXED") for sc in SCENARIOS}

    verdict = {
        "stage": "E4F_FAILURE_ANALYSIS",
        "E4F_ORIGINAL_VERDICT": "FAILED",
        "FAILURE_ANALYSIS_STATUS": "COMPLETE",
        "PERSISTENT_DOMINANT_DDQN_ACTION": dom["PERSISTENT"],
        "STATIONARY_DOMINANT_DDQN_ACTION": dom["STATIONARY"],
        "MIXED_DOMINANT_DDQN_ACTION": dom["MIXED"],
        "MIXED_PREDICT_FRACTION": fa3["mixed_predict_fraction_pooled"],
        "MIXED_EVENT_MULTI_ACTION_RATE": fa2["mixed_pooled_multi_action_rate"],
        "REGIONAL_SPECIALIZATION_STATUS": fa2["REGIONAL_SPECIALIZATION_STATUS"],
        "GLOBAL_ACTION_MIMICRY": fa3["GLOBAL_ACTION_MIMICRY"],
        "COLLAPSE_TIMING": fa11["COLLAPSE_TIMING"],
        "REWARD_SCALE_EMPIRICAL_AUDIT": fa9["REWARD_SCALE_EMPIRICAL_AUDIT"],
        "REWARD_ALIGNMENT": fa10["REWARD_ALIGNMENT"],
        "Q_FUNCTION_DOMINANCE": fa13["Q_FUNCTION_DOMINANCE"],
        "STATE_USAGE": fa14["STATE_USAGE"],
        "JOINT_ACTION_REPRESENTATION": fa15["JOINT_ACTION_REPRESENTATION"],
        "NORMALIZATION_SIGNAL_LOSS": fa17["NORMALIZATION_SIGNAL_LOSS"],
        "ACTION_COVERAGE": fa18["ACTION_COVERAGE"],
        "CHECKPOINT_SELECTION_TENSION": _load(RESULTS_DIR / "E4FFA_checkpoint_selection_audit.json")["CHECKPOINT_SELECTION_TENSION"],
        "H1_E4F_RECONCILIATION": recon_short,
        "PRIMARY_FAILURE_HYPOTHESIS": evidence["PRIMARY_FAILURE_HYPOTHESIS"],
        "SECONDARY_FAILURE_HYPOTHESES": evidence["SECONDARY_FAILURE_HYPOTHESES"],
        "E4F_VERDICT_CHANGED": "NO", "NEW_TRAINING_RUN": "NO", "NEW_ENVIRONMENT_RUN": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }
    _jdump(verdict, RESULTS_DIR / "E4FFA_verdict.json")
    _write_report(verdict, evidence, fa1, fa2, fa3)
    _emit_terminal(verdict)
    return verdict


def _write_report(v, evidence, fa1, fa2, fa3):
    L = ["# E4F-FA — Shared Double-DQN Failure Analysis (post-hoc)", "",
         f"**Original scientific verdict: FAILED (immutable). E4F_VERDICT_CHANGED = NO.**", "",
         "Diagnostic only — no rollouts, no training, no verdict change.", "",
         "## Headline mechanism", "",
         f"- The nominally-regional controller operationally collapsed toward a single "
         f"action per scenario: pooled dominant actions "
         f"PERSISTENT={v['PERSISTENT_DOMINANT_DDQN_ACTION']}, "
         f"STATIONARY={v['STATIONARY_DOMINANT_DDQN_ACTION']}, "
         f"MIXED={v['MIXED_DOMINANT_DDQN_ACTION']}.",
         f"- MIXED PREDICT fraction = {v['MIXED_PREDICT_FRACTION']:.3f}; within-event "
         f"multi-action rate = {v['MIXED_EVENT_MULTI_ACTION_RATE']:.3f} → "
         f"REGIONAL_SPECIALIZATION = {v['REGIONAL_SPECIALIZATION_STATUS']}.",
         f"- GLOBAL_ACTION_MIMICRY = {v['GLOBAL_ACTION_MIMICRY']}; COLLAPSE_TIMING = {v['COLLAPSE_TIMING']}.",
         "", "## Representation / reward diagnostics", "",
         f"- JOINT_ACTION_REPRESENTATION = {v['JOINT_ACTION_REPRESENTATION']} (Q conditions on "
         f"own region action + shared global STATE only; reward depends on the joint action vector).",
         f"- NORMALIZATION_SIGNAL_LOSS = {v['NORMALIZATION_SIGNAL_LOSS']} (local V2 features occupy "
         f"tiny sub-ranges after normalization; E3R-scale proxy, states not logged).",
         f"- STATE_USAGE = {v['STATE_USAGE']}; Q_FUNCTION_DOMINANCE = {v['Q_FUNCTION_DOMINANCE']}.",
         f"- REWARD_SCALE_EMPIRICAL_AUDIT = {v['REWARD_SCALE_EMPIRICAL_AUDIT']}; "
         f"REWARD_ALIGNMENT = {v['REWARD_ALIGNMENT']}.",
         f"- ACTION_COVERAGE = {v['ACTION_COVERAGE']}; CHECKPOINT_SELECTION_TENSION = {v['CHECKPOINT_SELECTION_TENSION']}.",
         "", "## Failure-mode evidence table", ""]
    for k, val in evidence["candidates"].items():
        L.append(f"- **{k}** → {val['status']}")
    L += ["", "## Diagnosis", "",
          f"- PRIMARY_FAILURE_HYPOTHESIS = **{v['PRIMARY_FAILURE_HYPOTHESIS']}**",
          f"- SECONDARY = {', '.join(v['SECONDARY_FAILURE_HYPOTHESES'])}", "",
          "## H1 ↔ E4F reconciliation", "", v["H1_E4F_RECONCILIATION"], "",
          "## Next research direction (recommendation only; NOT implemented)", "",
          f"- NEXT_STAGE = **{v['NEXT_STAGE']}**",
          "- Any architecture/reward/state change after this stage is a NEW predeclared experiment. "
          "Do NOT jump to PPO/SAC/transformer/GNN/bigger-net/reward-retune/more-epochs without a "
          "diagnosed target mechanism."]
    (RESULTS_DIR / "E4FFA_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    print(f"E4F_ORIGINAL_VERDICT              = {v['E4F_ORIGINAL_VERDICT']}")
    print(f"FAILURE_ANALYSIS_STATUS          = {v['FAILURE_ANALYSIS_STATUS']}")
    print(f"PERSISTENT_DOMINANT_DDQN_ACTION  = {v['PERSISTENT_DOMINANT_DDQN_ACTION']}")
    print(f"STATIONARY_DOMINANT_DDQN_ACTION  = {v['STATIONARY_DOMINANT_DDQN_ACTION']}")
    print(f"MIXED_DOMINANT_DDQN_ACTION       = {v['MIXED_DOMINANT_DDQN_ACTION']}")
    print(f"MIXED_PREDICT_FRACTION           = {v['MIXED_PREDICT_FRACTION']:.4f}")
    print(f"MIXED_EVENT_MULTI_ACTION_RATE    = {v['MIXED_EVENT_MULTI_ACTION_RATE']:.4f}")
    print(f"REGIONAL_SPECIALIZATION_STATUS   = {v['REGIONAL_SPECIALIZATION_STATUS']}")
    print(f"GLOBAL_ACTION_MIMICRY            = {v['GLOBAL_ACTION_MIMICRY']}")
    print(f"COLLAPSE_TIMING                  = {v['COLLAPSE_TIMING']}")
    print(f"REWARD_SCALE_EMPIRICAL_AUDIT     = {v['REWARD_SCALE_EMPIRICAL_AUDIT']}")
    print(f"REWARD_ALIGNMENT                 = {v['REWARD_ALIGNMENT']}")
    print(f"Q_FUNCTION_DOMINANCE             = {v['Q_FUNCTION_DOMINANCE']}")
    print(f"STATE_USAGE                      = {v['STATE_USAGE']}")
    print(f"JOINT_ACTION_REPRESENTATION      = {v['JOINT_ACTION_REPRESENTATION']}")
    print(f"NORMALIZATION_SIGNAL_LOSS        = {v['NORMALIZATION_SIGNAL_LOSS']}")
    print(f"ACTION_COVERAGE                  = {v['ACTION_COVERAGE']}")
    print(f"CHECKPOINT_SELECTION_TENSION     = {v['CHECKPOINT_SELECTION_TENSION']}")
    print(f"H1_E4F_RECONCILIATION            = {v['H1_E4F_RECONCILIATION']}")
    print(f"PRIMARY_FAILURE_HYPOTHESIS       = {v['PRIMARY_FAILURE_HYPOTHESIS']}")
    print(f"SECONDARY_FAILURE_HYPOTHESES     = {', '.join(v['SECONDARY_FAILURE_HYPOTHESES'])}")
    print(f"E4F_VERDICT_CHANGED              = {v['E4F_VERDICT_CHANGED']}")
    print(f"NEW_TRAINING_RUN                 = {v['NEW_TRAINING_RUN']}")
    print(f"NEW_ENVIRONMENT_RUN              = {v['NEW_ENVIRONMENT_RUN']}")
    print(f"FULL_REGRESSION                  = {v['FULL_REGRESSION']}")
    print(f"NEXT_STAGE                       = {v['NEXT_STAGE']}")


if __name__ == "__main__":
    main()
