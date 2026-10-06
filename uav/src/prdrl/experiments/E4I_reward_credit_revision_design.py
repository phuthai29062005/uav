"""
E4I — Reward / Credit Revision Design (PR-DRL-DMOEA, Branch B).

DESIGN / AUDIT / FREEZE ONLY. No training, no environment instantiation, no weight
modification, no access to future final seeds. E4HF showed V3 normalization fixes the
collapse mechanism (mechanism de-collapse SUPPORTED 3/3) but does NOT improve — and in
safety scenarios harms — performance. Branch B (reward/credit) is selected: with state
information restored (V3 fixed in BOTH future arms), does a non-degenerate region-specific
quality-improvement reward improve final MIXED performance?

This stage: (0) preserve history, (1) audit the exact ORIGINAL_REWARD_R0 source + its
local-reward inertness, (2) verify the local quality metric supports R1 with no extra FE,
(3-4) freeze the R1 candidate, (5) offline scale audit from pre-existing E4GD artifacts,
(6-25) freeze the single-factor V3+R0 vs V3+R1 future experiment and the readiness logic.

Because only the local-reward DIFFERENCE (L_pre - L_post) was ever persisted (not L_pre
and L_post separately), g_r's magnitude is NOT exactly reconstructable offline: the R1
ACTIVITY gate is auditable (g_r=0 iff raw_local=0) but the R1 SCALE gate is NOT_AUDITABLE.
Per the frozen protocol this yields R1_REWARD_READY=NO and NEXT_STAGE=E4ID (a small
diagnostic-only rollout to measure g_r magnitude) — NOT a guessed coefficient.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

from prdrl.rl.reward import (
    ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT, GAMMA_DEFAULT, BUDGET_PER_REGION, _EPS,
)
from prdrl.rl.reward_r1 import (
    REWARD_R0_NAME, REWARD_R1_NAME, EPS_R1, ALPHA_G_R1, ALPHA_LOCAL_R1_INITIAL, BETA_R1,
    ALPHA_LOCAL_MIN, ALPHA_LOCAL_MAX, regional_quality_gain_r1,
)
from prdrl.rl.normalization import NORM_V3

RESULTS_DIR = Path("results/E4I_reward_credit_revision_design")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4HE_DIR = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4F_DIR = Path("results/E4F_final_evaluation")
E4GD_DIR = Path("results/E4GD_controlled_diagnostic_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

# Future single-factor reward experiment (frozen here, NOT run)
NEW_TRAIN_SEEDS = list(range(1501, 1561))       # 60
NEW_VALIDATION_SEEDS = list(range(1561, 1581))  # 20
NEW_FINAL_SEEDS = list(range(1581, 1641))       # 60 — FIREWALLED
NEW_POLICY_SEEDS = [4001, 4002, 4003, 4004, 4005]
REWARD_BOOTSTRAP_MASTER_SEED = 66001
PRACTICAL_MARGIN = 1e-5

# All historically consumed environment seeds (for disjointness proof)
CONSUMED_SEED_SETS = {
    "E3_701_780": list(range(701, 781)),
    "E3R_801_882": list(range(801, 883)),
    "H1_901_940": list(range(901, 941)),
    "E4_train_1001_1060": list(range(1001, 1061)),
    "E4_val_1061_1080": list(range(1061, 1081)),
    "E4_final_1081_1140": list(range(1081, 1141)),
    "E4GD_diag_1201_1220": list(range(1201, 1221)),
    "E4H_train_1301_1360": list(range(1301, 1361)),
    "E4H_val_1361_1380": list(range(1361, 1381)),
    "E4H_final_1381_1440": list(range(1381, 1441)),
}
CONSUMED_POLICY_SEEDS = {"E4TR": [2001, 2002, 2003, 2004, 2005],
                         "E4HTR": [3001, 3002, 3003, 3004, 3005]}

_reward_final_access = {"count": 0}


def _hash_file(p) -> str:
    p = Path(p)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _json_default(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(str(type(x)))


def _jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=_json_default)


def _load(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase E4I-1 — R0 audit + local inertness (from source + pre-existing D4 artifacts)
# ---------------------------------------------------------------------------

def audit_r0_and_inertness() -> Dict:
    """Reconstruct R0 local-reward stats from the pre-existing E4GD D4 CSV (5000
    reconstructable rows) and echo the D4 json aggregate (30000 transitions)."""
    rows = list(csv.DictReader(open(E4GD_DIR / "E4GD_D4_reward_components.csv")))
    rl = np.array([float(r["raw_local_reward"]) for r in rows])
    wl = np.abs(np.array([float(r["weighted_local_reward"]) for r in rows]))
    wg = np.abs(np.array([float(r["weighted_global_reward"]) for r in rows]))
    wfe = np.abs(np.array([float(r["weighted_fe_penalty"]) for r in rows]))
    n = len(rl)
    nz = rl != 0.0
    zero_frac = float(np.mean(~nz))
    nonzero_frac = float(np.mean(nz))
    d4 = _load(E4GD_DIR / "E4GD_D4_reward_scale_results.json")

    # Inertness cause (evidence-bounded):
    #  A local metric never changes?  -> NO: nonzero fraction ~0.50
    #  C scale mismatch?              -> YES (dominant): local is an UNNORMALIZED raw PBI
    #     difference (~1e-3) vs global a NORMALIZED IGD ratio (~3e-2); even the nonzero
    #     local magnitudes are 2-3 orders below global.
    #  D coefficient (0.1)?           -> secondary (another 10x), not the root cause.
    #  E implementation (NaN->0)?     -> contributes to zero fraction (inactive regions).
    cause = ("C_SCALE_MISMATCH_DOMINANT: local reward is an UNNORMALIZED raw PBI-loss "
             "difference (median|raw_local| among nonzero ~9e-4) while global is a "
             "NORMALIZED IGD-improvement ratio (median|weighted_global|~3e-2); global "
             "overwhelms local by orders of magnitude. NOT cause A (local signal varies "
             "~50%% of transitions). Coefficient alpha_l=0.1 (D) and NaN->0 for inactive "
             "regions (E) are secondary contributors.")

    return {
        "source_of_truth": "src/prdrl/rl/reward.py (code authoritative)",
        "reconstruction_sample": {"file": "E4GD_D4_reward_components.csv", "n_rows": n},
        "d4_json_aggregate_n_transitions": d4["n_transitions"],
        "LOCAL_REWARD_ZERO_FRACTION": zero_frac,
        "LOCAL_REWARD_NONZERO_FRACTION": nonzero_frac,
        "median_abs_raw_local": float(np.median(np.abs(rl))),
        "median_abs_raw_local_among_nonzero": float(np.median(np.abs(rl[nz]))) if nz.any() else 0.0,
        "raw_local_q05_q50_q95": np.percentile(np.abs(rl), [5, 50, 95]).tolist(),
        "median_abs_weighted_local_csv": float(np.median(wl)),
        "median_abs_weighted_global_csv": float(np.median(wg)),
        "median_abs_weighted_fe_csv": float(np.median(wfe)),
        "d4_weighted_global_abs_median": d4["weighted_global_abs"]["median"],
        "d4_weighted_local_abs_median": d4["weighted_local_abs"]["median"],
        "d4_weighted_fe_abs_median": d4["weighted_fe_abs"]["median"],
        "d4_median_ratio_global_over_local": d4["median_ratio_global_over_local"],
        "d4_dominance_threshold_10x": d4["dominance_threshold_10x"],
        "D4_REWARD_SCALE": d4["D4_REWARD_SCALE"],
        "R0_LOCAL_INERTNESS_CAUSE": cause,
    }


# ---------------------------------------------------------------------------
# Phase E4I-5 — offline R1 scale audit (activity auditable; magnitude NOT)
# ---------------------------------------------------------------------------

def offline_r1_scale_audit(r0: Dict) -> Dict:
    """g_r = 0 iff raw_local = 0, so the ACTIVITY (nonzero fraction) is EXACTLY the
    R0 raw_local nonzero fraction. But median|g_r| = median|L_pre-L_post|/(L_pre+L_post+eps)
    requires L_pre and L_post SEPARATELY, which were never persisted (only their
    difference). Hence SCALE is NOT auditable offline."""
    nonzero = r0["LOCAL_REWARD_NONZERO_FRACTION"]
    activity_pass = bool(nonzero >= 0.20 and r0["median_abs_raw_local_among_nonzero"] > 0.0)
    return {
        "reconstructable_from": "E4GD D4 raw_local_reward (== L_pre - L_post)",
        "g_r_equals_zero_iff_raw_local_zero": True,
        "R1_LOCAL_SIGNAL_NONZERO_FRACTION": nonzero,
        "R1_ACTIVITY_GATE_RULE": "nonzero_fraction>=0.20 AND median|g_r|>0",
        "R1_ACTIVITY_GATE": "PASS" if activity_pass else "FAIL",
        "median_abs_g_r": "NOT_AUDITABLE",
        "reason_scale_not_auditable": (
            "g_r magnitude needs L_pre and L_post separately; only the difference "
            "(raw_local_reward) was persisted in D4, and no L_pre/L_post arrays exist "
            "in any pre-existing artifact. Reconstructing them requires re-running the "
            "environment, which E4I forbids."),
        "R1_WEIGHTED_LOCAL_MEDIAN_ABS": "NOT_AUDITABLE",
        "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO": "NOT_AUDITABLE",
        "R1_SCALE_GATE_RULE": "median|weighted_global| < 10 x median|alpha_local*g_r|",
        "R1_SCALE_GATE": "NOT_AUDITABLE",
    }


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4I — Reward / Credit Revision Design (Branch B, DESIGN/FREEZE ONLY)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase E4I-0 historical integrity ----
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    frz = _load(E4HE_DIR / "E4HE_checkpoint_freeze.json")["checkpoints"]
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    ck_hashes = {k: {"path": v["checkpoint_path"], "frozen_sha256": v["sha256"],
                     "recomputed_sha256": _hash_file(v["checkpoint_path"]),
                     "match": _hash_file(v["checkpoint_path"]) == v["sha256"]}
                 for k, v in frz.items()}
    ck_all_match = all(v["match"] for v in ck_hashes.values())
    hist = {
        "E4HF_COMPLETE": e4hf["E4HF_COMPLETE"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "E4HF_mechanism_decollapse": e4hf["MECHANISM_DECOLLAPSE_SUPPORTED"],
        "checkpoint_hashes": ck_hashes, "checkpoint_all_match": ck_all_match,
        "v3_constants_sha256": v3_sha,
        "no_post_e4hf_retraining": True, "no_checkpoint_reselection": True,
        "historical_artifact_hashes": {
            "E4HF_verdict.json": _hash_file(E4HF_DIR / "E4HF_verdict.json"),
            "E4F_verdict.json": _hash_file(E4F_DIR / "E4F_verdict.json"),
            "E4GD_D4_reward_components.csv": _hash_file(E4GD_DIR / "E4GD_D4_reward_components.csv"),
            "E4GD_D4_reward_scale_results.json": _hash_file(E4GD_DIR / "E4GD_D4_reward_scale_results.json"),
            "reward.py": _hash_file("src/prdrl/rl/reward.py"),
            "quality_loss.py": _hash_file("src/prdrl/detectors/quality_loss.py"),
        },
    }
    _jdump(hist, RESULTS_DIR / "E4I_historical_result_preservation.json")

    integrity_ok = (e4hf["E4HF_COMPLETE"] == "YES"
                    and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
                    and e4f["SCIENTIFIC_VERDICT"] == "FAILED" and ck_all_match)
    _jdump({"stage": "E4I_REWARD_CREDIT_REVISION_DESIGN", "design_only": True,
            "no_training": True, "no_env_instantiation": True, "no_weight_modification": True,
            "integrity_ok": integrity_ok, "selected_branch": "B_REWARD_CREDIT"},
           RESULTS_DIR / "E4I_pre_manifest.json")
    if not integrity_ok:
        print("[E4I] integrity FAIL — STOP")
        _jdump({"E4I_COMPLETE": "NO", "NEXT_STAGE": "E4I_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4I_verdict.json")
        return {"E4I_COMPLETE": "NO"}
    print("[E4I] integrity PASS (E4HF FAILED preserved, E4F FAILED preserved, 10 ckpts match).")

    # ---- Phase E4I-1 R0 audit + inertness ----
    r0 = audit_r0_and_inertness()
    _jdump({
        "reward_version": REWARD_R0_NAME,
        "formula": "r_r = alpha_g*r_global + alpha_l*r_local,r - beta*FE_cost,r",
        "alpha_g": ALPHA_G_DEFAULT, "alpha_l": ALPHA_L_DEFAULT, "beta": BETA_DEFAULT,
        "gamma": GAMMA_DEFAULT, "eps": _EPS, "budget_per_region": BUDGET_PER_REGION,
        "global_reward": "compute_global_reward=(q_after-q_before)/(|q_before|+eps), q=-IGD "
                         "of the ACTUAL joint executed population (normalized).",
        "local_reward": "compute_regional_pbi_reward=-(pbi_after-pbi_before)=pbi_before-pbi_after "
                        "= L_pre - L_post (UNNORMALIZED raw regional PBI quality-loss difference); "
                        "NaN (inactive region) -> 0.0.",
        "fe_cost": "compute_fe_cost=actual_response_fe/(budget_per_region*n_active); KEEP=0.",
        "units": {"global": "dimensionless normalized IGD ratio", "local": "raw PBI-loss units",
                  "fe": "fraction of per-region budget"},
        "clipping": "none on reward terms",
        "empty_region_semantics": "NaN loss -> local reward 0.0 (existing convention)",
        "active_region_semantics": "reward computed per active region from joint outcome",
        "pre_post_populations": "L_pre from pre-response population; L_post from executed "
                                "response population (both already evaluated).",
        "timing": "pre-response state, executed response, immediate post-response only",
        "depends_on": "realized outcome (global) + requested/realized region losses (local) + "
                      "realized FE (penalty)",
        "source": "src/prdrl/rl/reward.py",
    }, RESULTS_DIR / "E4I_reward_r0_audit.json")
    _jdump(r0, RESULTS_DIR / "E4I_r0_local_inertness_analysis.json")
    print(f"[E4I] R0 local zero_frac={r0['LOCAL_REWARD_ZERO_FRACTION']:.4f} "
          f"nonzero={r0['LOCAL_REWARD_NONZERO_FRACTION']:.4f}; cause={r0['R0_LOCAL_INERTNESS_CAUSE'][:40]}...")

    # ---- Phase E4I-2 local quality metric contract ----
    _jdump({
        "metric": "regional_PBI_quality_loss",
        "source": "src/prdrl/detectors/quality_loss.py::quality_loss",
        "L_r_definition": "per reference-direction region r: median over region members of "
                          "max(0, g_plus - g_minus), g = PBI(d1 + theta*d2). L_r >= 0, lower=better.",
        "L_pre": "quality_loss(F_tm_minus, F_tm_plus, rids, V, THETA_PBI)[r] == R0 pbi_before "
                 "== the V2 state convergence_loss_r feature.",
        "L_post": "quality_loss(F_tm_rm, F_tm_rp, rids_resp, V, THETA_PBI)[r] == R0 pbi_after, "
                  "computed on the executed response population.",
        "same_region_id_pre_post": True,
        "region_identity": "FIXED reference-direction Pareto regions (r indexes V[r]); "
                           "partitions are NOT redefined after seeing reward.",
        "extra_objective_evaluations_required": 0,
        "already_evaluated_populations": True,
        "R1_LOCAL_METRIC_FEASIBLE": "YES",
        "empty_or_degenerate_region": "region active pre but no valid post members -> L_post NaN "
                                      "-> g_r = 0.0 (REUSED existing R0 NaN->0 convention; not an "
                                      "invented constant penalty).",
    }, RESULTS_DIR / "E4I_local_quality_metric_contract.json")

    # ---- Phase E4I-3/-4 R1 formula freeze ----
    # sanity: bounded + sign + tie + eps + NaN (eps in denominator => ±1 boundary approached)
    assert abs(regional_quality_gain_r1(1.0, 0.0) - 1.0) < 1e-9   # 1/(1+eps)
    assert abs(regional_quality_gain_r1(0.0, 1.0) + 1.0) < 1e-9   # -1/(1+eps)
    assert regional_quality_gain_r1(0.5, 0.5) == 0.0             # exact tie
    assert regional_quality_gain_r1(0.0, 0.0) == 0.0             # double zero
    assert regional_quality_gain_r1(float("nan"), 0.1) == 0.0    # degenerate -> 0
    assert regional_quality_gain_r1(1.0, 3.0) < 0.0             # worsened
    assert -1.0 <= regional_quality_gain_r1(3.0, 1.0) <= 1.0
    _jdump({
        "reward_version": REWARD_R1_NAME,
        "local_credit_term": "SYMMETRIC_REGIONAL_QUALITY_GAIN_R1",
        "g_r_formula": "if L_pre==0 and L_post==0: 0 else (L_pre-L_post)/(L_pre+L_post+eps)",
        "eps": EPS_R1, "bounds": [-1.0, 1.0], "sign_convention": ">0 improved, 0 tie, <0 worsened",
        "reward_formula": "r_r^R1 = 1.0*r_global + alpha_local*g_r - 0.01*FE_cost,r",
        "alpha_g": ALPHA_G_R1, "alpha_local_initial": ALPHA_LOCAL_R1_INITIAL, "beta": BETA_R1,
        "fitted_constants": "none", "extra_counterfactual_rollout": False,
        "extra_objective_evaluation": 0,
        "NOT_a_difference_reward": True, "NOT_counterfactual_marginal_contribution": True,
        "note": "direct regional quality-change reward; global term + FE penalty identical to R0.",
        "source": "src/prdrl/rl/reward_r1.py",
    }, RESULTS_DIR / "E4I_reward_r1_formula.json")

    # ---- Phase E4I-5 offline scale audit ----
    scale = offline_r1_scale_audit(r0)
    _jdump(scale, RESULTS_DIR / "E4I_r1_offline_scale_audit.json")
    _jdump({"R1_ACTIVITY_GATE": scale["R1_ACTIVITY_GATE"],
            "nonzero_fraction": scale["R1_LOCAL_SIGNAL_NONZERO_FRACTION"],
            "rule": scale["R1_ACTIVITY_GATE_RULE"]}, RESULTS_DIR / "E4I_r1_activity_gate.json")
    _jdump({"R1_SCALE_GATE": scale["R1_SCALE_GATE"], "rule": scale["R1_SCALE_GATE_RULE"],
            "reason": scale["reason_scale_not_auditable"],
            "alpha_local_initial": ALPHA_LOCAL_R1_INITIAL,
            "scale_match_rule_if_needed": "alpha_local = median_abs_global / median_abs_g_r "
                                          "(deterministic, magnitude-only, NEVER performance-tuned)",
            "alpha_local_min": ALPHA_LOCAL_MIN, "alpha_local_max": ALPHA_LOCAL_MAX,
            "coefficient_frozen_without_performance_tuning": True},
           RESULTS_DIR / "E4I_r1_scale_gate.json")

    # ---- Phase E4I-7/-8 causality + FE ----
    _jdump({"uses_pre_response_state": True, "uses_executed_response": True,
            "uses_immediate_post_response_population": True,
            "uses_future_environment_state": False, "uses_next_change_info": False,
            "uses_future_pf_beyond_legitimate_eval": False, "R1_CAUSALITY": "PASS"},
           RESULTS_DIR / "E4I_r1_causality.json")
    _jdump({"EXTRA_FE_FOR_REWARD_R1": 0, "computed_from_already_evaluated_populations": True,
            "R1_reward_introduces_no_new_objective_evaluation": True},
           RESULTS_DIR / "E4I_r1_fe_accounting.json")

    # ---- reward versioning + single-factor + future contracts ----
    _jdump({"R0": REWARD_R0_NAME, "R1": REWARD_R1_NAME,
            "checkpoints_must_store": "reward_version",
            "backward_compat": "historical E4TR/E4HF checkpoints keep their reward metadata; "
                               "R0 checkpoint requires ORIGINAL_REWARD_R0, R1 requires "
                               "REWARD_R1_REGIONAL_QUALITY_GAIN; mismatch = hard error.",
            "R0_source": "src/prdrl/rl/reward.py (UNMODIFIED)",
            "R1_source": "src/prdrl/rl/reward_r1.py (ADDED; R0 byte-stable)"},
           RESULTS_DIR / "E4I_reward_versioning_contract.json")
    _jdump({"experimental_factor": "REWARD_ONLY",
            "CONTROL": {"arm_name": "R0_CONTROL", "normalization": NORM_V3, "reward": REWARD_R0_NAME},
            "TREATMENT": {"arm_name": "R1_TREATMENT", "normalization": NORM_V3, "reward": REWARD_R1_NAME},
            "normalization_fixed_both_arms": NORM_V3, "v3_refit": False, "v1_arm": False, "v2_arm": False,
            "forbidden_comparison": "V1+old_reward vs V3+new_reward (would change 2 factors)",
            "everything_else_identical": True},
           RESULTS_DIR / "E4I_single_factor_contract.json")

    # ---- Phase E4I-15 fresh seeds + disjointness ----
    all_consumed = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    new_env = set(NEW_TRAIN_SEEDS) | set(NEW_VALIDATION_SEEDS) | set(NEW_FINAL_SEEDS)
    env_disjoint = len(new_env & all_consumed) == 0
    all_consumed_pol = set().union(*[set(v) for v in CONSUMED_POLICY_SEEDS.values()])
    pol_disjoint = len(set(NEW_POLICY_SEEDS) & all_consumed_pol) == 0
    internal_disjoint = (len(set(NEW_TRAIN_SEEDS) & set(NEW_VALIDATION_SEEDS)) == 0
                         and len(set(NEW_TRAIN_SEEDS) & set(NEW_FINAL_SEEDS)) == 0
                         and len(set(NEW_VALIDATION_SEEDS) & set(NEW_FINAL_SEEDS)) == 0)
    _jdump({"train": [1501, 1560], "validation": [1561, 1580], "final": [1581, 1640],
            "policy_seeds": NEW_POLICY_SEEDS, "n_train": 60, "n_val": 20, "n_final": 60,
            "env_disjoint_from_all_history": env_disjoint,
            "policy_disjoint_from_history": pol_disjoint, "internal_split_disjoint": internal_disjoint,
            "consumed_env_sets": list(CONSUMED_SEED_SETS.keys()),
            "final_firewall": {"seeds": [1581, 1640], "untouched_until_reward_final_protocol_frozen": True,
                               "REWARD_FINAL_EVAL_ACCESS_COUNT": _reward_final_access["count"]}},
           RESULTS_DIR / "E4I_future_seed_contract.json")
    _jdump({"final_seeds": [1581, 1640], "REWARD_FINAL_EVAL_ACCESS_COUNT": _reward_final_access["count"],
            "FUTURE_FINAL_STATUS": "UNTOUCHED"}, RESULTS_DIR / "E4I_final_firewall.json")

    # ---- Phase E4I-16/-17 pairing + training ----
    _jdump({"principle": "R0_<p> and R1_<p> start from identical network weights (paired init as E4HT)",
            "policy_seeds": NEW_POLICY_SEEDS, "shared": ["train_schedule", "scenario_schedule",
            "initialization", "training_budget", "validation_criterion", "checkpoint_schedule"],
            "separate": ["replay", "optimizer", "controller_rng", "networks"],
            "only_reward_differs": True}, RESULTS_DIR / "E4I_policy_pairing_contract.json")
    _jdump({"state_dim": 9, "normalization": NORM_V3, "network": "9-128-128-4",
            "algorithm": "SHARED_DOUBLE_DQN", "centralized_critic": False,
            "joint_action_representation_change": False, "qmix": False, "vdn": False,
            "autoregressive_policy": False,
            "hyperparameters": {"gamma": 0.99, "lr": 1e-4, "optimizer": "Adam", "loss": "SmoothL1",
                                "batch": 128, "replay": 50000, "warmup": 1000, "update_every": 4,
                                "grad_steps": 1, "tau": 0.005, "grad_clip": None,
                                "epsilon": "1.0->0.05 over 50000 transitions", "epochs": 6,
                                "episodes_per_policy": 1080},
            "operators": {"KEEP_FE": 0, "sigma_L": 0.02, "sigma_P": 0.005, "lambda_P": 1.0,
                          "predict_fallback": "LOCAL", "sigma_D": 0.15, "budget": "10 PER_ACTIVE_REGION",
                          "survival": "unrestricted NSGA-II"},
            "no_hyperparameter_retuning": True}, RESULTS_DIR / "E4I_future_training_contract.json")
    _jdump({"selection_metric": "VALIDATION_MACRO_MIGD_RESPONSE",
            "tie_breakers": ["macro_MIGD_epoch", "mean_response_fe", "earlier_transition_count"],
            "reward_value_not_used_for_selection": True, "same_for_both_arms": True},
           RESULTS_DIR / "E4I_future_checkpoint_contract.json")

    # ---- Phase E4I-19..-23 mechanism + statistics ----
    _jdump({"primary_question": "With V3 normalization fixed in both arms, does replacing the "
            "inert local credit term with a non-degenerate region-specific quality-improvement "
            "reward improve final MIXED response quality?",
            "primary_endpoint": "MIGD_response", "primary_scenario": "MIXED",
            "effect": "R0 - R1 (positive => R1 better)",
            "mechanism_endpoints": {
                "1_LOCAL_REWARD_NONZERO_FRACTION": "expected R1 > R0",
                "2_GLOBAL_TO_LOCAL_WEIGHTED_MAGNITUDE_RATIO": "expected R1 < R0 (median abs weighted)",
                "3_REGIONAL_REWARD_DISPERSION_WITHIN_EVENT": "expected R1 > R0 (std of regional "
                    "reward across active regions per event)"},
            "primary_mechanism": "credit differentiation (NOT action entropy; V3 already restored it)",
            "descriptive_only": ["PREDICT_fraction", "multi_action_rate", "action_entropy"],
            "mechanism_source": "training/validation reward logs",
            "REWARD_CREDIT_MECHANISM_SUPPORTED": "YES iff >=2 of 3 endpoints move as expected",
            "mechanism_cannot_rescue_harmful_final_MIGD": True},
           RESULTS_DIR / "E4I_future_mechanism_contract.json")
    _jdump({"design": "PAIRED_CROSSED_TWO_AXIS_BOOTSTRAP", "policy_axis": 5, "environment_axis": 60,
            "N_BOOTSTRAP": 50000, "CI_LEVEL": 0.95, "CI_METHOD": "percentile",
            "BOOTSTRAP_MASTER_SEED": REWARD_BOOTSTRAP_MASTER_SEED, "effect": "R0 - R1",
            "positive_favors": "R1", "PRACTICAL_MARGIN": PRACTICAL_MARGIN,
            "policy_robustness": ">=4 of 5 policy pairs with final MIXED R0-R1 > 0",
            "safety_scenarios": ["PERSISTENT", "STATIONARY"], "safety_endpoints": ["MIGD_response", "MIGD_epoch"]},
           RESULTS_DIR / "E4I_future_statistical_contract.json")
    _jdump({"REWARD_CAUSAL_VALIDATED_requires": ["MIXED R0-R1 response BENEFICIAL",
            "reward mechanism supported", ">=4/5 policy robustness", "PERSISTENT safety PASS",
            "STATIONARY safety PASS", "macro safety PASS", "integrity PASS"],
            "PARTIAL": "directional-positive performance without full support, OR practically-neutral "
                       "performance with clear reward-mechanism improvement and no safety harm "
                       "(bounded mechanistic partial)",
            "FAILED": "MIXED directional-negative/harmful OR safety failure OR integrity failure",
            "mechanism_cannot_rescue_harmful_quality": True},
           RESULTS_DIR / "E4I_future_success_logic.json")

    # ---- Phase E4I-25 R1 readiness ----
    readiness_items = {
        "1_R0_audited": True,
        "2_R1_local_metric_feasible": True,
        "3_no_extra_FE": True,
        "4_R1_formula_defined": True,
        "5_degenerate_semantics_defined": True,
        "6_causality_pass": True,
        "7_activity_gate_pass": scale["R1_ACTIVITY_GATE"] == "PASS",
        "8_scale_gate_pass": scale["R1_SCALE_GATE"] == "PASS",       # NOT_AUDITABLE -> False
        "9_coefficient_frozen_without_perf_tuning": True,
        "10_single_factor_frozen": True,
        "11_fresh_seed_split_frozen": env_disjoint and pol_disjoint and internal_disjoint,
        "12_no_training": True,
        "13_full_regression_pass": None,   # verified separately
    }
    r1_ready = all(v for k, v in readiness_items.items() if v is not None)
    if r1_ready:
        next_stage = "E4II_REWARD_IMPLEMENTATION_VALIDATION"
    elif scale["R1_SCALE_GATE"] == "NOT_AUDITABLE":
        next_stage = "E4ID_REWARD_SCALE_DIAGNOSTIC_DESIGN"
    else:
        next_stage = "E4I_REWARD_METRIC_REVIEW"

    verdict = build_verdict(hist, r0, scale, readiness_items, r1_ready, next_stage,
                            env_disjoint, pol_disjoint, internal_disjoint)
    _jdump(verdict, RESULTS_DIR / "E4I_verdict.json")

    # protocol frozen master
    _jdump({"stage": "E4I_REWARD_CREDIT_REVISION_DESIGN", "selected_branch": "B_REWARD_CREDIT",
            "fixed_normalization": NORM_V3, "control": "V3_PLUS_R0", "treatment": "V3_PLUS_R1",
            "future_factor": "REWARD_ONLY", "R1_REWARD_READY": "YES" if r1_ready else "NO",
            "bootstrap_master_seed": REWARD_BOOTSTRAP_MASTER_SEED, "practical_margin": PRACTICAL_MARGIN,
            "components": ["E4I_reward_r0_audit.json", "E4I_r0_local_inertness_analysis.json",
                           "E4I_local_quality_metric_contract.json", "E4I_reward_r1_formula.json",
                           "E4I_r1_offline_scale_audit.json", "E4I_single_factor_contract.json",
                           "E4I_future_seed_contract.json", "E4I_future_training_contract.json",
                           "E4I_future_mechanism_contract.json", "E4I_future_statistical_contract.json",
                           "E4I_future_success_logic.json"],
            "next_stage": next_stage}, RESULTS_DIR / "E4I_protocol_frozen.json")

    write_report(verdict, hist, r0, scale, readiness_items)
    emit_terminal(verdict)
    print(f"[E4I] wall {time.time()-t0:.1f}s")
    return verdict


def build_verdict(hist, r0, scale, readiness_items, r1_ready, next_stage,
                  env_disjoint, pol_disjoint, internal_disjoint):
    return {
        "stage": "E4I_REWARD_CREDIT_REVISION_DESIGN",
        "E4I_COMPLETE": "YES",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": hist["E4HF_NORMALIZATION_CAUSAL_VERDICT"],
        "SELECTED_BRANCH": "B_REWARD_CREDIT",
        "FIXED_NORMALIZATION": NORM_V3,
        "R0_REWARD": REWARD_R0_NAME,
        "R0_LOCAL_REWARD_ZERO_FRACTION": r0["LOCAL_REWARD_ZERO_FRACTION"],
        "R0_LOCAL_REWARD_NONZERO_FRACTION": r0["LOCAL_REWARD_NONZERO_FRACTION"],
        "R0_WEIGHTED_GLOBAL_MEDIAN_ABS": r0["d4_weighted_global_abs_median"],
        "R0_WEIGHTED_LOCAL_MEDIAN_ABS": r0["d4_weighted_local_abs_median"],
        "R0_GLOBAL_LOCAL_MAGNITUDE_RATIO": r0["d4_median_ratio_global_over_local"],
        "R0_LOCAL_INERTNESS_CAUSE": r0["R0_LOCAL_INERTNESS_CAUSE"],
        "R1_REWARD": REWARD_R1_NAME,
        "R1_LOCAL_METRIC_FEASIBLE": "YES",
        "R1_LOCAL_SIGNAL_NONZERO_FRACTION": scale["R1_LOCAL_SIGNAL_NONZERO_FRACTION"],
        "R1_WEIGHTED_LOCAL_MEDIAN_ABS": scale["R1_WEIGHTED_LOCAL_MEDIAN_ABS"],
        "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO": scale["R1_GLOBAL_LOCAL_MAGNITUDE_RATIO"],
        "R1_ALPHA_LOCAL": ALPHA_LOCAL_R1_INITIAL,
        "R1_ACTIVITY_GATE": scale["R1_ACTIVITY_GATE"],
        "R1_SCALE_GATE": scale["R1_SCALE_GATE"],
        "R1_CAUSALITY": "PASS",
        "R1_EXTRA_FE": 0,
        "R1_REWARD_READY": "YES" if r1_ready else "NO",
        "R1_READINESS_ITEMS": readiness_items,
        "STATE_CHANGED": "NO",
        "NORMALIZATION_CHANGED": "NO",
        "NETWORK_CHANGED": "NO",
        "JOINT_ARCHITECTURE_CHANGED": "NO",
        "FUTURE_EXPERIMENT_FACTOR": "REWARD_ONLY",
        "FUTURE_CONTROL": "V3_PLUS_R0",
        "FUTURE_TREATMENT": "V3_PLUS_R1",
        "NEW_TRAIN_SEEDS": "1501..1560",
        "NEW_VALIDATION_SEEDS": "1561..1580",
        "NEW_FINAL_SEEDS": "1581..1640",
        "NEW_POLICY_SEEDS": "4001..4005",
        "SEED_SETS_DISJOINT": env_disjoint and pol_disjoint and internal_disjoint,
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _reward_final_access["count"],
        "REWARD_BOOTSTRAP_MASTER_SEED": REWARD_BOOTSTRAP_MASTER_SEED,
        "PRACTICAL_MARGIN": PRACTICAL_MARGIN,
        "NEW_TRAINING": "NO",
        "CHECKPOINT_INTEGRITY": "PASS" if hist["checkpoint_all_match"] else "FAIL",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": next_stage,
    }


def write_report(v, hist, r0, scale, readiness_items):
    L = ["# E4I — Reward / Credit Revision Design (Branch B)", "",
         f"**R1_REWARD_READY = {v['R1_REWARD_READY']} → NEXT_STAGE = {v['NEXT_STAGE']}**",
         "", "Design/audit/freeze only. No training, no environment instantiation, no weight change, "
         "future final seeds 1581..1640 untouched.", "",
         "## A. Historical failures preserved",
         f"- ORIGINAL_E4F_VERDICT = FAILED; E4HF_NORMALIZATION_CAUSAL_VERDICT = "
         f"{v['E4HF_NORMALIZATION_CAUSAL_VERDICT']}; 10 checkpoints hash-match (integrity "
         f"{v['CHECKPOINT_INTEGRITY']}); V3 constants unchanged; no post-E4HF retraining/reselection.",
         "", "## B. Why the normalization branch is complete",
         "- E4HF proved V3 fixes the collapse MECHANISM (de-collapse 3/3) but does not improve — "
         "and in PERSISTENT/STATIONARY harms — performance. State information is restored, so the "
         "remaining question is credit, not state. V3 is retained in BOTH future arms.", "",
         "## C. Exact original reward R0",
         "- `r_r = 1.0*r_global + 0.1*r_local - 0.01*FE_cost`. Global = normalized IGD improvement "
         "of the joint executed population; local = `pbi_before - pbi_after` (UNNORMALIZED raw "
         "regional PBI-loss difference); FE = response_fe/(10*n_active). Source authoritative: "
         "src/prdrl/rl/reward.py.", "",
         "## D. Why local reward R0 is inert",
         f"- NOT because it never changes: nonzero fraction = {r0['LOCAL_REWARD_NONZERO_FRACTION']:.4f} "
         f"(zero {r0['LOCAL_REWARD_ZERO_FRACTION']:.4f}). It is a SCALE mismatch: median|weighted_global| "
         f"= {r0['d4_weighted_global_abs_median']:.4g} vs median|weighted_local| = "
         f"{r0['d4_weighted_local_abs_median']:.4g} (D4 = {r0['D4_REWARD_SCALE']}). The unnormalized raw "
         "PBI difference is 2-3 orders of magnitude below the normalized global term; alpha_l=0.1 and "
         "NaN->0 are secondary. → R0_LOCAL_INERTNESS_CAUSE = C_SCALE_MISMATCH_DOMINANT.", "",
         "## E. Regional quality metric audit",
         "- L_pre = R0 pbi_before = the V2 state convergence_loss_r; L_post = R0 pbi_after; both are "
         "`quality_loss` (PBI, >=0, lower=better) on the SAME reference-direction region, from "
         "already-evaluated populations → R1_LOCAL_METRIC_FEASIBLE = YES, EXTRA_FE = 0.", "",
         "## F. Proposed R1 local credit term",
         "- SYMMETRIC_REGIONAL_QUALITY_GAIN: `g_r = (L_pre - L_post)/(L_pre + L_post + 1e-12)`, "
         "double-zero → 0; NaN → 0. `r_r^R1 = 1.0*r_global + 0.1*g_r - 0.01*FE_cost`. Global & FE "
         "identical to R0 (single-factor local-signal replacement).", "",
         "## G. R1 boundedness / semantics",
         "- g_r ∈ [-1, 1]; >0 improved, 0 tie, <0 worsened. No fitted constants, no counterfactual "
         "rollout. NOT a difference reward / marginal-contribution — a direct regional quality-change reward.", "",
         "## H. Offline signal activity",
         f"- g_r = 0 ⟺ raw_local = 0, so R1 nonzero fraction = {scale['R1_LOCAL_SIGNAL_NONZERO_FRACTION']:.4f} "
         f"≥ 0.20 → R1_ACTIVITY_GATE = {scale['R1_ACTIVITY_GATE']}.", "",
         "## I. Offline global/local scale",
         f"- R1_SCALE_GATE = {scale['R1_SCALE_GATE']}: median|g_r| needs L_pre and L_post separately, "
         "but only their difference was persisted (D4 raw_local_reward). No pre-existing artifact stores "
         "the two loss terms, and E4I forbids new environment runs.", "",
         "## J. Coefficient decision",
         "- Initial alpha_local = 0.1 preserved. Any scale-match (`alpha_local = median|global|/median|g_r|`, "
         "bounds [0.01, 10.0]) is deterministic and magnitude-only, NEVER performance-tuned. Cannot be "
         "finalized offline because the scale gate is NOT_AUDITABLE.", "",
         "## K. Causality and FE accounting",
         "- R1_CAUSALITY = PASS (pre-response state + executed response + immediate post-response only). "
         "EXTRA_FE_FOR_REWARD_R1 = 0.", "",
         "## L. Single-factor experiment",
         "- CONTROL = V3 + R0; TREATMENT = V3 + R1. Only reward differs; V3 fixed both arms; no V1/V2 arm; "
         "joint architecture unchanged.", "",
         "## M. Fresh seed firewall",
         "- train 1501..1560, val 1561..1580, final 1581..1640, policy 4001..4005; disjoint from all "
         f"history ({v['SEED_SETS_DISJOINT']}); REWARD_FINAL_EVAL_ACCESS_COUNT = 0.", "",
         "## N. Future mechanism endpoints",
         "- (1) local reward nonzero fraction ↑, (2) global/local weighted magnitude ratio ↓, "
         "(3) within-event regional reward dispersion ↑; ≥2/3 ⇒ REWARD_CREDIT_MECHANISM_SUPPORTED. "
         "Action entropy is descriptive only (V3 already restored diversity).", "",
         "## O. Future statistical design",
         f"- Paired crossed two-axis bootstrap (5×60, N=50000, 95% percentile), master seed "
         f"{REWARD_BOOTSTRAP_MASTER_SEED}, effect R0−R1, margin 1e-5, robustness ≥4/5.", "",
         "## P. Future verdict logic",
         "- VALIDATED / PARTIAL / FAILED mirroring prior discipline; mechanism cannot rescue harmful MIGD.", "",
         "## Q. R1 readiness",
         f"- R1_REWARD_READY = {v['R1_REWARD_READY']} (scale gate NOT_AUDITABLE offline). Items: "
         + ", ".join(f"{k}={val}" for k, val in readiness_items.items() if val is not None) + ".", "",
         f"## R. Next stage: {v['NEXT_STAGE']}",
         "- A small diagnostic-only rollout (E4ID) will measure median|g_r| to complete the scale gate "
         "and freeze alpha_local, before any reward training. R0 stays byte-stable; V3 and joint "
         "architecture are not undone."]
    (RESULTS_DIR / "E4I_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    fields = ["E4I_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
              "SELECTED_BRANCH", "FIXED_NORMALIZATION", "R0_REWARD", "R0_LOCAL_REWARD_ZERO_FRACTION",
              "R0_WEIGHTED_GLOBAL_MEDIAN_ABS", "R0_WEIGHTED_LOCAL_MEDIAN_ABS",
              "R0_GLOBAL_LOCAL_MAGNITUDE_RATIO", "R0_LOCAL_INERTNESS_CAUSE", "R1_REWARD",
              "R1_LOCAL_METRIC_FEASIBLE", "R1_LOCAL_SIGNAL_NONZERO_FRACTION", "R1_WEIGHTED_LOCAL_MEDIAN_ABS",
              "R1_GLOBAL_LOCAL_MAGNITUDE_RATIO", "R1_ALPHA_LOCAL", "R1_ACTIVITY_GATE", "R1_SCALE_GATE",
              "R1_CAUSALITY", "R1_EXTRA_FE", "R1_REWARD_READY", "STATE_CHANGED", "NORMALIZATION_CHANGED",
              "NETWORK_CHANGED", "JOINT_ARCHITECTURE_CHANGED", "FUTURE_EXPERIMENT_FACTOR", "FUTURE_CONTROL",
              "FUTURE_TREATMENT", "NEW_TRAIN_SEEDS", "NEW_VALIDATION_SEEDS", "NEW_FINAL_SEEDS",
              "NEW_POLICY_SEEDS", "REWARD_FINAL_EVAL_ACCESS_COUNT", "NEW_TRAINING", "FULL_REGRESSION",
              "NEXT_STAGE"]
    for k in fields:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        if k == "R0_LOCAL_INERTNESS_CAUSE":
            val = str(val).split(":")[0]
        print(f"{k:<38} = {val}")


if __name__ == "__main__":
    main()
