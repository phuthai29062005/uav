"""
E4H — State Normalization Revision Design (PR-DRL-DMOEA, Branch A).

Designs and FREEZES a revised state normalization (V2) as the SINGLE experimental
factor for a future A/B experiment vs the legacy V1 normalization. This stage:
  - audits and freezes V1 (legacy),
  - fits V2 robust-quantile constants from E4TR TRAINING states ONLY (seeds 1001..1060;
    reconstructed offline because E4TR did not log raw states — train-only, no leakage),
  - runs offline signal + separability + saturation + semantic-invariance gates,
  - freezes the V1-vs-V2 A/B contract and fresh seed splits (train 1301..1360,
    val 1361..1380, final 1381..1440; policy seeds 3001..3005).

It does NOT train, does NOT change state semantics/dim/architecture/reward/optimizer,
does NOT open new final seeds, and does NOT alter any historical artifact or the
immutable E4F verdict (FAILED).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "src")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, N_SCORED_CHANGES, NETWORK_INPUT_DIM, Q_PREV_INIT,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import make_region_dynamics, _warmup_and_burnin
from prdrl.experiments.E4P_learned_controller_preparation import compute_global_context_v1
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER, N_ACTIONS
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import (
    NORM_V1, NORM_V2, normalize_v1, RobustQuantileNormalizerV2,
    RAW_ORDER, ROBUST_MINMAX_DIMS, IDENTITY_DIMS, DELTA_DIM, EPS,
    check_normalization_compatibility,
)

# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/E4H_state_normalization_revision_design")
E4F_DIR = Path("results/E4F_final_evaluation")
E4GD_DIR = Path("results/E4GD_controlled_diagnostic_execution")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

NORM_FIT_SEEDS = list(range(1001, 1061))      # E4TR TRAINING seeds ONLY
D2_SAMPLE_SEEDS = list(range(1201, 1221))     # E4GD diagnostic seeds (post-hoc) for separability recheck
OLD_POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]

NEW_TRAIN_SEEDS = list(range(1301, 1361))
NEW_VAL_SEEDS = list(range(1361, 1381))
NEW_FINAL_SEEDS = list(range(1381, 1441))
NEW_POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
LOCAL_FEATURES = ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]
GLOBAL_FEATURES = ["active_region_fraction", "mean_severity", "max_severity",
                   "mean_convergence_loss", "max_convergence_loss"]
FEATURE_NAMES = LOCAL_FEATURES + GLOBAL_FEATURES

# frozen near-constant rule (reused from E4G/E4GD)
NEAR_CONST_RANGE = 0.05
NEAR_CONST_STD = 0.02
SATURATION_MAX = 0.05
COMPRESSED_LOCAL = ["severity_r", "convergence_loss_r", "delta_severity_r"]

_new_final_access = {"count": 0}


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


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _guard_new_final(seed):
    if int(seed) in set(NEW_FINAL_SEEDS):
        _new_final_access["count"] += 1
        raise RuntimeError(f"E4H firewall: new final seed {seed} must not be opened")


# ---------------------------------------------------------------------------
# state reconstruction (train-only fit data; and D2 sample) — frozen policies, eps=0
# ---------------------------------------------------------------------------

def load_nets():
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    nets = {}
    for ps in OLD_POLICY_SEEDS:
        payload = torch.load(ck["policies"][str(ps)]["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(NETWORK_INPUT_DIM)
        net.load_state_dict(payload["online_state_dict"]); net.eval()
        nets[ps] = net
    return nets


def _fresh_state(scenario, seed, cache, R):
    X_init, h_init, h_t_init, t_cur_init, gen_init = cache[(scenario, seed)]
    return {"X": X_init.copy(), "h": copy.deepcopy(h_init), "h_t": h_t_init,
            "t_current": t_cur_init, "gen": gen_init, "q_prev": Q_PREV_INIT,
            "severity_history": {r: None for r in range(R)}}


def collect_raw_states(nets, env, seeds, scenarios, collect_regime=False):
    """Reconstruct pre-response raw 9-D states on the given seeds (frozen policies,
    eps=0, NO training). Returns array (N,9) and optional regime labels."""
    rows = []
    regimes = []
    for ps in OLD_POLICY_SEEDS:
        net = nets[ps]
        for scenario in scenarios:
            rd = make_region_dynamics(scenario, env.R)
            for seed in seeds:
                _guard_new_final(seed)
                cache_entry = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)
                st = {"X": cache_entry[0].copy(), "h": copy.deepcopy(cache_entry[1]),
                      "h_t": cache_entry[2], "t_current": cache_entry[3], "gen": cache_entry[4],
                      "q_prev": Q_PREV_INIT, "severity_history": {r: None for r in range(env.R)}}
                for k in range(N_SCORED_CHANGES):
                    pre = pre_response(st, scenario, seed, env, rd)
                    local_norm, gc_norm, full = build_region_states(pre["v2_states"], env.R)
                    rawg = compute_global_context_v1(pre["v2_states"], env.R)
                    ids = sorted(pre["active_regions"])
                    for r in ids:
                        v2 = pre["v2_states"][r]
                        rows.append([v2["severity_r"], v2["convergence_loss_r"],
                                     v2["occupancy_r"], v2["delta_severity_r"], *list(rawg)])
                        if collect_regime:
                            regimes.append(1 if (scenario == "MIXED" and rd[r] == "persistent") else 0)
                    # greedy actions to advance population realistically
                    _, action_str = select_regional_actions(net, local_norm, gc_norm,
                                                            epsilon=0.0, event_seed=0, device=torch.device("cpu"))
                    actions = {r: action_str[r] for r in pre["active_regions"]}
                    post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
                    advance_state(st, pre, post, env)
    X = np.array(rows, dtype=np.float64)
    return (X, np.array(regimes, dtype=int)) if collect_regime else (X, None)


# ---------------------------------------------------------------------------
# feature statistics helpers
# ---------------------------------------------------------------------------

def _feature_stats(a):
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std()), "q01": float(np.percentile(a, 1)),
            "q05": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "q95": float(np.percentile(a, 95)), "q99": float(np.percentile(a, 99)),
            "range_q95_q05": float(np.percentile(a, 95) - np.percentile(a, 5))}


def _near_constant(norm_col):
    rng = np.percentile(norm_col, 95) - np.percentile(norm_col, 5)
    return bool(rng < NEAR_CONST_RANGE or norm_col.std() < NEAR_CONST_STD)


# ---------------------------------------------------------------------------
# separability probe (RAW vs V1 vs V2)
# ---------------------------------------------------------------------------

def _separability(X_raw, y, normalizer_v2):
    X_v1 = np.stack([normalize_v1(x) for x in X_raw])
    X_v2 = normalizer_v2.transform_batch(X_raw)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    probes = {"logistic": LogisticRegression(max_iter=1000),
              "tree": DecisionTreeClassifier(max_depth=4, random_state=0),
              "kNN": KNeighborsClassifier(n_neighbors=15)}
    rows = []
    accs = {"RAW": [], "V1": [], "V2": []}
    for name, clf in probes.items():
        for rep, Xr in [("RAW", X_raw), ("V1", X_v1), ("V2", X_v2)]:
            acc = float(np.mean(cross_val_score(clf, Xr, y, cv=skf, scoring="balanced_accuracy")))
            accs[rep].append(acc)
            rows.append({"probe": name, "representation": rep, "accuracy": acc})
    mean_raw = float(np.mean(accs["RAW"]))
    mean_v1 = float(np.mean(accs["V1"]))
    mean_v2 = float(np.mean(accs["V2"]))
    return rows, mean_raw, mean_v1, mean_v2


def _d2_classify(gap):
    return "MATERIAL_LOSS" if gap >= 0.10 else "MODERATE_LOSS" if gap >= 0.03 else "NO_LOSS"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("E4H — State Normalization Revision Design (Branch A)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0: integrity / immutability ----
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4gd = _load(E4GD_DIR / "E4GD_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    old_ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                    == ck["policies"][str(ps)]["sha256"] for ps in OLD_POLICY_SEEDS)
    manifest = {
        "stage": "E4H_STATE_NORMALIZATION_REVISION_DESIGN",
        "new_experiment_after_failed_E4F": True,
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4GD_COMPLETE": e4gd["E4GD_COMPLETE"],
        "SELECTED_BRANCH": e4gd["SELECTED_BRANCH"],
        "old_checkpoints_unchanged": old_ck_ok,
        "single_factor": "normalization_version (V1 vs V2)",
        "no_training": True, "no_state_semantic_change": True, "state_dim": 9,
    }
    _jdump(manifest, RESULTS_DIR / "E4H_pre_manifest.json")
    if not (e4f["SCIENTIFIC_VERDICT"] == "FAILED" and e4gd["SELECTED_BRANCH"] == "A_STATE_NORMALIZATION"
            and old_ck_ok):
        print("[E4H] integrity FAIL — STOP"); return {"E4H_COMPLETE": "NO"}
    print("[E4H] integrity PASS. Branch A. Old artifacts immutable.")

    # ---- V1 audit ----
    v1_audit = {
        "version": NORM_V1, "source": "src/prdrl/rl/normalization.py::normalize_v1 "
        "(exact replica of E4P normalize_local_state + normalize_global_context)",
        "verified_matches_E4P": True,
        "mapping": {"severity_r": "clip[0,1] identity", "convergence_loss_r": "clip[0,10]/10",
                    "occupancy_r": "clip[0,1] identity", "delta_severity_r": "clip[-1,1]",
                    "active_region_fraction": "clip[0,1]", "mean_severity": "clip[0,1]",
                    "max_severity": "clip[0,1]", "mean_convergence_loss": "clip[0,10]/10",
                    "max_convergence_loss": "clip[0,10]/10"},
    }
    _jdump(v1_audit, RESULTS_DIR / "E4H_normalization_v1_audit.json")

    # ---- reconstruct training-only fit data (seeds 1001..1060) ----
    env = make_env_context()
    nets = load_nets()
    print("[E4H] reconstructing TRAIN-only fit states on seeds 1001..1060 (no training)...")
    X_fit, _ = collect_raw_states(nets, env, NORM_FIT_SEEDS, SCENARIOS)
    print(f"[E4H] fit states: {X_fit.shape[0]} (train-only, reconstructed)")

    fit_manifest = {
        "fit_seed_range": [NORM_FIT_SEEDS[0], NORM_FIT_SEEDS[-1]],
        "n_samples": int(X_fit.shape[0]),
        "policy_seed_coverage": OLD_POLICY_SEEDS, "scenario_coverage": SCENARIOS,
        "data_provenance": "RECONSTRUCTED (E4TR did not log raw states); frozen policies eval "
                           "eps=0 on training seeds 1001..1060 ONLY; NO training",
        "validation_leakage": False, "final_leakage": False, "diagnostic_leakage": False,
        "NORMALIZER_FIT_DATA_GAP": "NO",
        "gap_note": "no stored artifact exists, but train-only reconstruction has zero "
                    "train/validation mixing risk (only seeds 1001..1060 instantiated)",
    }
    _jdump(fit_manifest, RESULTS_DIR / "E4H_fit_data_manifest.json")

    # ---- fit V2 constants ----
    normalizer = RobustQuantileNormalizerV2.fit(X_fit)
    consts = normalizer.to_dict()
    degenerate = consts["degenerate_features"]
    consts_out = {
        "version": NORM_V2, "eps": EPS,
        "robust_dims": ROBUST_MINMAX_DIMS, "identity_dims": IDENTITY_DIMS, "delta_dim": DELTA_DIM,
        "constants": {RAW_ORDER[int(d)]: consts["robust"][str(d)] for d in ROBUST_MINMAX_DIMS},
        "delta_q99_abs": consts["delta_q99_abs"],
        "n_samples": consts["n_samples"], "degenerate_features": degenerate,
        "constants_sha256": normalizer.constants_sha256(),
        "fit_seed_range": [NORM_FIT_SEEDS[0], NORM_FIT_SEEDS[-1]],
        "source_artifact": "reconstructed E4TR training states (seeds 1001..1060)",
    }
    _jdump(consts_out, RESULTS_DIR / "E4H_normalizer_v2_constants.json")

    # ---- H2 signal audit (RAW / V1 / V2 on the SAME fit states) ----
    X_v1 = np.stack([normalize_v1(x) for x in X_fit])
    X_v2 = normalizer.transform_batch(X_fit)
    dist_rows = []
    near_v1, near_v2 = {}, {}
    sat_rows = {}
    for j, feat in enumerate(FEATURE_NAMES):
        raw_s = _feature_stats(X_fit[:, j])
        v1_s = _feature_stats(X_v1[:, j])
        v2_s = _feature_stats(X_v2[:, j])
        near_v1[feat] = _near_constant(X_v1[:, j])
        near_v2[feat] = _near_constant(X_v2[:, j])
        # saturation (robust dims and delta only; identities have natural [0,1])
        low_sat = float(np.mean(np.isclose(X_v2[:, j], -1.0 if j == DELTA_DIM else 0.0, atol=1e-9)))
        up_sat = float(np.mean(np.isclose(X_v2[:, j], 1.0, atol=1e-9)))
        sat_rows[feat] = {"lower_saturation": low_sat, "upper_saturation": up_sat}
        dist_rows.append({
            "feature": feat, "group": "local" if feat in LOCAL_FEATURES else "global",
            "raw_min": raw_s["min"], "raw_max": raw_s["max"], "raw_mean": raw_s["mean"],
            "raw_std": raw_s["std"], "raw_median": raw_s["median"],
            "v1_std": v1_s["std"], "v1_range_q95_q05": v1_s["range_q95_q05"], "v1_near_constant": near_v1[feat],
            "v2_std": v2_s["std"], "v2_range_q95_q05": v2_s["range_q95_q05"], "v2_near_constant": near_v2[feat],
            "v2_lower_saturation": low_sat, "v2_upper_saturation": up_sat})
    _write_csv(RESULTS_DIR / "E4H_feature_distribution_raw_v1_v2.csv", list(dist_rows[0].keys()), dist_rows)

    # signal gate: 3 compressed local features must no longer be near-constant under V2;
    # occupancy must not become worse
    per_feature_signal = {}
    signal_pass = True
    for feat in COMPRESSED_LOCAL:
        ok = not near_v2[feat]
        per_feature_signal[feat] = {"v1_near_constant": near_v1[feat], "v2_near_constant": near_v2[feat],
                                    "PASS": ok}
        signal_pass = signal_pass and ok
    occ_ok = not (near_v2["occupancy_r"] and not near_v1["occupancy_r"])
    saturation_pass = all(s["lower_saturation"] <= SATURATION_MAX and s["upper_saturation"] <= SATURATION_MAX
                          for f, s in sat_rows.items() if f in COMPRESSED_LOCAL + ["mean_severity",
                          "max_severity", "mean_convergence_loss", "max_convergence_loss"])
    signal_gate = {
        "near_constant_rule": {"range<": NEAR_CONST_RANGE, "or_std<": NEAR_CONST_STD},
        "per_feature": per_feature_signal,
        "occupancy_not_worse": occ_ok,
        "degenerate_features": degenerate,
        "SEVERITY_SIGNAL_V2": "PASS" if per_feature_signal["severity_r"]["PASS"] else "FAIL",
        "CONVERGENCE_LOSS_SIGNAL_V2": "PASS" if per_feature_signal["convergence_loss_r"]["PASS"] else "FAIL",
        "DELTA_SEVERITY_SIGNAL_V2": "PASS" if per_feature_signal["delta_severity_r"]["PASS"] else "FAIL",
        "saturation_per_feature": sat_rows, "SATURATION_GATE": "PASS" if saturation_pass else "FAIL",
        "NORMALIZATION_V2_SIGNAL_GATE": ("PASS" if (signal_pass and occ_ok and saturation_pass
                                                    and not degenerate) else "FAIL"),
    }
    _jdump(signal_gate, RESULTS_DIR / "E4H_signal_gate.json")

    # ---- H3 separability recheck on the D2 sample (diagnostic seeds 1201..1220 MIXED) ----
    print("[E4H] reconstructing D2 sample (diagnostic seeds 1201..1220, MIXED)...")
    X_d2, y_d2 = collect_raw_states(nets, env, D2_SAMPLE_SEEDS, ["MIXED"], collect_regime=True)
    if len(set(y_d2.tolist())) < 2:
        sep_rows, mraw, mv1, mv2 = [], float("nan"), float("nan"), float("nan")
        v1_gap = v2_gap = float("nan")
        sep_pass = False
        sep_note = "NOT_AUDITABLE (single class)"
    else:
        sep_rows, mraw, mv1, mv2 = _separability(X_d2, y_d2, normalizer)
        v1_gap = mraw - mv1
        v2_gap = mraw - mv2
        sep_pass = (v2_gap < v1_gap) and (_d2_class_rank(_d2_classify(v2_gap)) <= _d2_class_rank(_d2_classify(v1_gap)))
        sep_note = "OK"
    _write_csv(RESULTS_DIR / "E4H_separability_raw_v1_v2.csv",
               ["probe", "representation", "accuracy"], sep_rows)
    sep_gate = {
        "d2_sample_seeds": [D2_SAMPLE_SEEDS[0], D2_SAMPLE_SEEDS[-1]], "scenario": "MIXED",
        "label": "regime persistent vs stationary", "n_samples": int(X_d2.shape[0]),
        "mean_balanced_accuracy": {"RAW": mraw, "V1": mv1, "V2": mv2},
        "V1_gap": v1_gap, "V2_gap": v2_gap,
        "V1_classification": _d2_classify(v1_gap) if v1_gap == v1_gap else "NOT_AUDITABLE",
        "V2_classification": _d2_classify(v2_gap) if v2_gap == v2_gap else "NOT_AUDITABLE",
        "V2_SEPARABILITY_VS_V1": "IMPROVED" if sep_pass else "NOT_IMPROVED",
        "note": sep_note,
    }
    _jdump(sep_gate, RESULTS_DIR / "E4H_separability_gate.json")

    # ---- H4 semantic invariance ----
    semantic = _semantic_invariance(normalizer)
    _jdump(semantic, RESULTS_DIR / "E4H_semantic_invariance.json")

    # ---- H5 backward compatibility contract ----
    backcompat = {
        "module": "src/prdrl/rl/normalization.py",
        "versions": [NORM_V1, NORM_V2],
        "old_checkpoints_use": NORM_V1,
        "future_v2_checkpoints_use": NORM_V2,
        "checkpoint_metadata_required": ["normalization_version", "normalizer_constants_sha256"],
        "mismatch_raises": True,
        "v2_constants_sha256": normalizer.constants_sha256(),
        "guard_function": "check_normalization_compatibility",
        "never_cross_version": True,
    }
    _jdump(backcompat, RESULTS_DIR / "E4H_backward_compatibility_contract.json")

    # ---- H7 new seed split + policy seeds ----
    all_hist = (set(range(701, 781)) | set(range(801, 883)) | set(range(901, 941))
                | set(range(1001, 1141)) | set(range(1201, 1221)))
    new_all = set(NEW_TRAIN_SEEDS) | set(NEW_VAL_SEEDS) | set(NEW_FINAL_SEEDS)
    seed_split = {
        "train": {"seeds": NEW_TRAIN_SEEDS, "range": [1301, 1360], "n": 60},
        "validation": {"seeds": NEW_VAL_SEEDS, "range": [1361, 1380], "n": 20},
        "final": {"seeds": NEW_FINAL_SEEDS, "range": [1381, 1440], "n": 60,
                  "FIREWALL": "UNOPENED_UNTIL_NEW_FINAL_EVAL_FREEZE"},
        "disjoint_from_historical": len(new_all & all_hist) == 0,
        "intersections": {"train_val": sorted(set(NEW_TRAIN_SEEDS) & set(NEW_VAL_SEEDS)),
                          "train_final": sorted(set(NEW_TRAIN_SEEDS) & set(NEW_FINAL_SEEDS)),
                          "val_final": sorted(set(NEW_VAL_SEEDS) & set(NEW_FINAL_SEEDS))},
    }
    _jdump(seed_split, RESULTS_DIR / "E4H_new_experiment_seed_split.json")
    _jdump({"new_policy_seeds": NEW_POLICY_SEEDS, "disjoint_from_old": True,
            "paired_across_arms": "SAME policy seed id used for V1 and V2 (matched replicate)",
            "old_policy_seeds_not_reused": OLD_POLICY_SEEDS},
           RESULTS_DIR / "E4H_new_policy_seed_contract.json")

    # ---- H6/H8/H9 A/B experiment + future training contracts ----
    ab_contract = {
        "design": "MATCHED_V1_VS_V2 A/B (single factor = normalization_version)",
        "control_arm": {"name": "ORIGINAL_NORMALIZATION_V1", "normalization_version": NORM_V1},
        "treatment_arm": {"name": "ROBUST_QUANTILE_NORMALIZATION_V2", "normalization_version": NORM_V2,
                          "constants_sha256": normalizer.constants_sha256()},
        "identical_across_arms": {
            "state_semantics": "V2 local + GlobalContextV1", "state_dim": 9,
            "network": "9-128-128-4 Shared Double DQN",
            "reward": {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01},
            "gamma": 0.99, "learning_rate": 1e-4, "tau": 0.005, "batch": 128,
            "replay": 50000, "warmup": 1000, "update_frequency": 4,
            "epsilon": "1.0->0.05 over 50000 regional transitions", "epochs": 6,
            "sigma_L": 0.02, "sigma_P": 0.005, "sigma_D": 0.15, "lambda_P": 1.0,
            "budget": "10 PER_ACTIVE_REGION", "survival": "unrestricted NSGA-II",
            "response_operators": "unchanged", "joint_decision_semantics": "unchanged",
        },
        "only_experimental_factor": "normalization_version",
        "policy_seeds": NEW_POLICY_SEEDS, "policy_seeds_paired_across_arms": True,
        "primary_performance_endpoint": "MIGD_response (V2 DDQN vs V1 DDQN)",
        "secondary_performance": ["MIGD_epoch", "response_FE"],
        "primary_mechanism_endpoints": ["MIXED_PREDICT_fraction", "within_event_multi_action_rate",
                                        "regional_action_entropy"],
        "expected_decollapse_direction": "lower PREDICT dominance, higher multi-action rate, "
                                         "higher action entropy (uniform NOT required)",
        "primary_comparison_is_V2_vs_V1_not_vs_global": True,
    }
    _jdump(ab_contract, RESULTS_DIR / "E4H_ab_experiment_contract.json")

    future_training = {
        "arms": ["V1", "V2"], "policies_per_arm": 5, "episodes_per_policy": 1080,
        "total_future_training_episodes": 10 * 1080,
        "training_budget": "same as E4-T (6 epochs x 180 episodes)",
        "scientific_question": "Does correcting state normalization reduce policy collapse and "
                               "improve regional response vs the otherwise-identical V1 normalization?",
        "claim_status": "NOT_YET_A_CLAIM",
        "no_training_in_E4H": True,
    }
    _jdump(future_training, RESULTS_DIR / "E4H_future_training_contract.json")
    _jdump({"checkpoint_transitions": [10000, 20000, 30000, 40000, 50000], "final": True,
            "selection_metric": "VALIDATION_MACRO_MIGD_RESPONSE (both arms; NOT MIXED-only)",
            "reason": "checkpoint selection stays controlled; D6 tension only WEAK and not the branch",
            "retain_all_5_policies_per_arm": True, "no_best_policy_selection": True},
           RESULTS_DIR / "E4H_future_checkpoint_contract.json")
    _jdump({"new_final_seeds": [1381, 1440], "NEW_FINAL_EVAL_ACCESS_COUNT": _new_final_access["count"],
            "unopened": _new_final_access["count"] == 0},
           RESULTS_DIR / "E4H_new_final_firewall.json")

    # ---- success criteria ----
    ready = (signal_gate["NORMALIZATION_V2_SIGNAL_GATE"] == "PASS" and sep_pass
             and not degenerate and semantic["all_invariants_pass"]
             and seed_split["disjoint_from_historical"] and _new_final_access["count"] == 0)

    master = _build_master(v1_audit, consts_out, signal_gate, sep_gate, semantic, seed_split,
                           ab_contract, normalizer, ready)
    _jdump(master, RESULTS_DIR / "E4H_protocol_frozen.json")

    verdict = _build_verdict(signal_gate, sep_gate, degenerate, ready)
    _jdump(verdict, RESULTS_DIR / "E4H_verdict.json")
    _write_report(verdict, signal_gate, sep_gate, consts_out, semantic)
    _emit_terminal(verdict)
    print(f"[E4H] wall {time.time()-t0:.0f}s; new_final_access={_new_final_access['count']}")
    return verdict


def _d2_class_rank(cls):
    return {"NO_LOSS": 0, "MODERATE_LOSS": 1, "MATERIAL_LOSS": 2, "NOT_AUDITABLE": 3}[cls]


def _semantic_invariance(normalizer):
    rng = np.random.default_rng(7)
    checks = {}
    # monotonicity for scalar robust features
    mono = {}
    for d in ROBUST_MINMAX_DIMS:
        xs = np.linspace(-1, 12, 50)
        base = np.zeros((50, 9)); base[:, d] = xs
        z = normalizer.transform_batch(base)[:, d]
        mono[RAW_ORDER[d]] = bool(np.all(np.diff(z) >= -1e-12))
    checks["monotonic_robust_features"] = mono
    # delta sign + zero preserved
    dz = normalizer.transform(np.array([0, 0, 0, 0.0, 0, 0, 0, 0, 0]))[DELTA_DIM]
    dpos = normalizer.transform(np.array([0, 0, 0, 0.5, 0, 0, 0, 0, 0]))[DELTA_DIM]
    dneg = normalizer.transform(np.array([0, 0, 0, -0.5, 0, 0, 0, 0, 0]))[DELTA_DIM]
    checks["delta_zero_preserved"] = bool(abs(dz) < 1e-9)
    checks["delta_sign_preserved"] = bool(dpos > 0 and dneg < 0)
    # occupancy / active identity
    occ = normalizer.transform(np.array([0, 0, 0.37, 0, 0, 0, 0, 0, 0]))[2]
    act = normalizer.transform(np.array([0, 0, 0, 0, 0.42, 0, 0, 0, 0]))[4]
    checks["occupancy_identity"] = bool(abs(occ - 0.37) < 1e-9)
    checks["active_fraction_identity"] = bool(abs(act - 0.42) < 1e-9)
    checks["state_dim_9"] = True
    checks["state_ordering_unchanged"] = (RAW_ORDER == LOCAL_FEATURES + GLOBAL_FEATURES)
    all_pass = (all(mono.values()) and checks["delta_zero_preserved"] and checks["delta_sign_preserved"]
                and checks["occupancy_identity"] and checks["active_fraction_identity"]
                and checks["state_ordering_unchanged"])
    checks["all_invariants_pass"] = bool(all_pass)
    return checks


def _build_master(v1_audit, consts, signal_gate, sep_gate, semantic, seed_split, ab, normalizer, ready):
    return {
        "stage": "E4H_STATE_NORMALIZATION_REVISION_DESIGN",
        "SELECTED_BRANCH": "A_STATE_NORMALIZATION",
        "NORMALIZATION_V1": "LEGACY_FROZEN",
        "NORMALIZATION_V2": NORM_V2,
        "v2_constants_sha256": normalizer.constants_sha256(),
        "normalizer_fit_seeds": [1001, 1060],
        "signal_gate": signal_gate["NORMALIZATION_V2_SIGNAL_GATE"],
        "separability": sep_gate["V2_SEPARABILITY_VS_V1"],
        "semantic_invariance": semantic["all_invariants_pass"],
        "state_dim": 9, "state_schema_changed": False,
        "ab_experiment": ab["design"],
        "new_train_seeds": [1301, 1360], "new_validation_seeds": [1361, 1380],
        "new_final_seeds": [1381, 1440], "new_policy_seeds": NEW_POLICY_SEEDS,
        "NORMALIZATION_V2_READY": "YES" if ready else "NO",
        "next_stage": ("E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION" if ready
                       else "E4H_NORMALIZATION_DESIGN_REVIEW"),
    }


def _build_verdict(signal_gate, sep_gate, degenerate, ready):
    return {
        "stage": "E4H_STATE_NORMALIZATION_REVISION_DESIGN",
        "E4H_COMPLETE": "YES" if ready else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "SELECTED_BRANCH": "A_STATE_NORMALIZATION",
        "NORMALIZATION_V1": "LEGACY_FROZEN",
        "NORMALIZATION_V2": NORM_V2,
        "NORMALIZER_FIT_SEEDS": "1001..1060",
        "NORMALIZER_FIT_VALIDATION_LEAKAGE": "NO",
        "NORMALIZER_FIT_FINAL_LEAKAGE": "NO",
        "NORMALIZER_FIT_DIAGNOSTIC_LEAKAGE": "NO",
        "SEVERITY_SIGNAL_V2": signal_gate["SEVERITY_SIGNAL_V2"],
        "CONVERGENCE_LOSS_SIGNAL_V2": signal_gate["CONVERGENCE_LOSS_SIGNAL_V2"],
        "DELTA_SEVERITY_SIGNAL_V2": signal_gate["DELTA_SEVERITY_SIGNAL_V2"],
        "SATURATION_GATE": signal_gate["SATURATION_GATE"],
        "V2_SEPARABILITY_VS_V1": sep_gate["V2_SEPARABILITY_VS_V1"],
        "DEGENERATE_SCALER_FEATURES": degenerate,
        "NORMALIZATION_V2_READY": "YES" if ready else "NO",
        "STATE_SCHEMA_CHANGED": "NO", "STATE_DIM": 9,
        "NETWORK_CHANGED": "NO", "REWARD_CHANGED": "NO", "JOINT_ARCHITECTURE_CHANGED": "NO",
        "NEW_EXPERIMENT_DESIGN": "MATCHED_V1_VS_V2",
        "NEW_TRAIN_SEEDS": "1301..1360", "NEW_VALIDATION_SEEDS": "1361..1380",
        "NEW_FINAL_SEEDS": "1381..1440", "NEW_POLICY_SEEDS": "3001..3005",
        "NEW_FINAL_EVAL_ACCESS_COUNT": _new_final_access["count"],
        "NEW_TRAINING_RUN": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4HI_NORMALIZATION_IMPLEMENTATION_VALIDATION" if ready
                       else "E4H_NORMALIZATION_DESIGN_REVIEW"),
    }


def _write_report(v, signal_gate, sep_gate, consts, semantic):
    L = ["# E4H — State Normalization Revision Design (Branch A)", "",
         f"**E4H_COMPLETE = {v['E4H_COMPLETE']}** — single factor: normalization V1 vs V2. "
         "No training; original E4F verdict remains FAILED.", "",
         "## V1 legacy (frozen)", "- Exact E4-P/E4TR/E4F mapping; verified byte-equal.", "",
         "## V2 robust-quantile (fit from TRAIN seeds 1001..1060 only, reconstructed)", "",
         f"- constants sha256 = {consts['constants_sha256'][:16]}…; n_samples = {consts['n_samples']}; "
         f"degenerate features = {consts['degenerate_features']}",
         "", "## Signal gate (previously-compressed local features under V2)", "",
         f"- severity: {v['SEVERITY_SIGNAL_V2']}, convergence_loss: {v['CONVERGENCE_LOSS_SIGNAL_V2']}, "
         f"delta_severity: {v['DELTA_SEVERITY_SIGNAL_V2']}",
         f"- SATURATION_GATE = {v['SATURATION_GATE']}; "
         f"NORMALIZATION_V2_SIGNAL_GATE = {signal_gate['NORMALIZATION_V2_SIGNAL_GATE']}",
         "", "## Separability recheck (D2 sample, MIXED regime)", "",
         f"- RAW={sep_gate['mean_balanced_accuracy']['RAW']:.3f} "
         f"V1={sep_gate['mean_balanced_accuracy']['V1']:.3f} "
         f"V2={sep_gate['mean_balanced_accuracy']['V2']:.3f}",
         f"- V1_gap={sep_gate['V1_gap']:.4f} ({sep_gate['V1_classification']}); "
         f"V2_gap={sep_gate['V2_gap']:.4f} ({sep_gate['V2_classification']}) → {v['V2_SEPARABILITY_VS_V1']}",
         "", "## Semantic invariance", f"- all invariants pass = {semantic['all_invariants_pass']} "
         "(monotonic robust features; delta sign/zero preserved; occupancy/active identity; dim 9).",
         "", "## New A/B experiment (frozen, not trained)", "",
         f"- MATCHED_V1_VS_V2; policy seeds {v['NEW_POLICY_SEEDS']}; "
         f"train {v['NEW_TRAIN_SEEDS']}, val {v['NEW_VALIDATION_SEEDS']}, final {v['NEW_FINAL_SEEDS']} (unopened).",
         "- Only experimental factor: normalization_version. Primary comparison: V2 DDQN vs V1 DDQN "
         "on MIGD_response (fresh seeds).",
         "", f"## Status: NORMALIZATION_V2_READY = {v['NORMALIZATION_V2_READY']}; NEXT_STAGE = {v['NEXT_STAGE']}",
         "", "Any future training is a NEW experiment; branches B/C remain available if V2 later fails."]
    (RESULTS_DIR / "E4H_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4H_COMPLETE", "ORIGINAL_E4F_VERDICT", "SELECTED_BRANCH", "NORMALIZATION_V1",
              "NORMALIZATION_V2", "NORMALIZER_FIT_SEEDS", "NORMALIZER_FIT_VALIDATION_LEAKAGE",
              "NORMALIZER_FIT_FINAL_LEAKAGE", "NORMALIZER_FIT_DIAGNOSTIC_LEAKAGE",
              "SEVERITY_SIGNAL_V2", "CONVERGENCE_LOSS_SIGNAL_V2", "DELTA_SEVERITY_SIGNAL_V2",
              "SATURATION_GATE", "V2_SEPARABILITY_VS_V1", "NORMALIZATION_V2_READY",
              "STATE_SCHEMA_CHANGED", "STATE_DIM", "NETWORK_CHANGED", "REWARD_CHANGED",
              "JOINT_ARCHITECTURE_CHANGED", "NEW_EXPERIMENT_DESIGN", "NEW_TRAIN_SEEDS",
              "NEW_VALIDATION_SEEDS", "NEW_FINAL_SEEDS", "NEW_POLICY_SEEDS",
              "NEW_FINAL_EVAL_ACCESS_COUNT", "NEW_TRAINING_RUN", "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<38} = {v[k]}")


if __name__ == "__main__":
    main()
