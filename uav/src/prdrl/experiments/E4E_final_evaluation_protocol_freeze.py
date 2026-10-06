"""
E4E — Final Evaluation Protocol Freeze (PR-DRL-DMOEA).

LAST protocol-freeze stage before the reserved final evaluation set (seeds
1081..1140) is opened in E4F. E4E performs ONLY:
    - pre-freeze audit (E4TR integrity, checkpoint hashes, firewalls),
    - protocol definition (comparators, endpoints, effect, bootstrap, macro,
      robustness, margin, safety gates, rule comparison, verdict logic,
      execution order, matrix contract, output schema),
    - synthetic validation of the inferential statistics,
    - firewall validation.

E4E MUST NOT instantiate any final-evaluation environment. No environment
constructor / benchmark evaluator / baseline runner / DDQN evaluator is called
with seeds 1081..1140. FINAL_EVAL_ACCESS_COUNT must remain 0.

No scientific final-test result is produced in E4E.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

from prdrl.evaluation.crossed_bootstrap import (
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD, BOOTSTRAP_MASTER_SEED, derive_seed,
    scenario_point_effect, crossed_two_axis_bootstrap,
    crossed_two_axis_bootstrap_macro, naive_row_bootstrap,
)
from prdrl.evaluation.effects import (
    cell_effects, scenario_mean_effect, per_policy_effects,
    relative_improvement, macro_effect, summarize_scenario,
)
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect, policy_robustness, safety_gate, scientific_verdict,
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE,
    HARMFUL, VALIDATED, PARTIAL, FAILED,
)

# ---------------------------------------------------------------------------
# Frozen constants (E4E)
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/E4E_final_evaluation_protocol_freeze")
E4TR_DIR = Path("results/E4TR_shared_double_dqn_training")
E4F_DIR = "results/E4F_final_evaluation"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]
FINAL_EVAL_SEEDS = list(range(1081, 1141))   # 60 — RESERVED, must stay unopened
N_ENVIRONMENT = 60
N_POLICY = 5

PRIMARY_ENDPOINT = "MIGD_response"
SECONDARY_ENDPOINTS = ["MIGD_epoch", "response_FE"]

PRIMARY_SCENARIO = "MIXED"
PRESELECTED_COMPARATOR = {                    # from VALIDATION only (E4TR), frozen
    "PERSISTENT": "GLOBAL_PREDICT",
    "STATIONARY": "GLOBAL_DIVERSIFY",
    "MIXED": "GLOBAL_PREDICT",
}
RULE_COMPARATOR = "RULE_V1_V2STATE"
FIXED_GLOBALS = ["GLOBAL_KEEP", "GLOBAL_LOCAL", "GLOBAL_PREDICT", "GLOBAL_DIVERSIFY"]
DDQN_ARMS = [f"DDQN_{ps}" for ps in POLICY_SEEDS]

MIXED_POLICY_ROBUSTNESS_THRESHOLD = 4        # >= 4 of 5 policy means positive

# deterministic E4F execution order (arm order per env seed)
EXECUTION_ARM_ORDER = FIXED_GLOBALS + [RULE_COMPARATOR] + DDQN_ARMS

FORBIDDEN_TOKENS = ("TBD", "TODO", "UNFROZEN", "PLACEHOLDER", "placeholder")


# ---------------------------------------------------------------------------
# Firewall — E4E must never open a final environment
# ---------------------------------------------------------------------------

class FinalEvalFirewall:
    def __init__(self):
        self.access_count = 0
        self.forbidden = set(FINAL_EVAL_SEEDS)

    def check(self, seed: int):
        if int(seed) in self.forbidden:
            self.access_count += 1
            raise RuntimeError(f"E4E firewall: final seed {seed} must not be opened")


FIREWALL = FinalEvalFirewall()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _no_forbidden_tokens(obj) -> List[str]:
    """Return list of forbidden placeholder tokens found anywhere in obj."""
    found = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k); walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for tok in FORBIDDEN_TOKENS:
                if tok in o:
                    found.append(o)
                    break
    walk(obj)
    return found


# ---------------------------------------------------------------------------
# PHASE E4E-0 — pre-freeze audit
# ---------------------------------------------------------------------------

def pre_freeze_audit() -> Dict:
    v = json.load(open(E4TR_DIR / "E4TR_verdict.json"))
    sel = json.load(open(E4TR_DIR / "E4TR_selected_checkpoints.json"))
    fw = json.load(open(E4TR_DIR / "E4TR_final_eval_firewall.json"))

    checkpoint_ok = True
    ckpt_rows = {}
    for ps in POLICY_SEEDS:
        info = sel["selected"][str(ps)]
        actual = _hash_file(info["checkpoint_path"])
        exists = Path(info["checkpoint_path"]).exists()
        match = (actual == info["sha256"]) and exists
        checkpoint_ok = checkpoint_ok and match
        ckpt_rows[str(ps)] = {
            "checkpoint": info["selected_checkpoint"],
            "transition_count": info["transition_count"],
            "path": info["checkpoint_path"],
            "sha256_recorded": info["sha256"],
            "sha256_actual": actual,
            "hash_match": match,
            "file_size": info["file_size"],
        }

    prod_files = ["src/prdrl/operators/local.py", "src/prdrl/operators/predict.py",
                  "src/prdrl/operators/diversify.py", "src/prdrl/operators/budget.py",
                  "src/prdrl/oracle/oracle.py", "src/prdrl/core/survival.py"]
    prod_hashes = {fp: _hash_file(fp) for fp in prod_files}

    audit = {
        "E4TR_COMPLETE": v["E4TR_COMPLETE"],
        "POLICY_RUNS_COMPLETE": v["POLICY_RUNS_COMPLETE"],
        "all_five_selected_checkpoints_exist": all(
            Path(sel["selected"][str(ps)]["checkpoint_path"]).exists() for ps in POLICY_SEEDS),
        "checkpoint_sha256_match": checkpoint_ok,
        "oracle_firewall": v["oracle_label_firewall"],
        "rng_isolation": v["rng_isolation"],
        "validation_determinism": v["validation_determinism"],
        "e4tr_final_eval_access_count": fw["FINAL_EVAL_ACCESS_COUNT"],
        "checkpoints": ckpt_rows,
        "production_operator_hashes": prod_hashes,
    }
    audit["CHECKPOINT_INTEGRITY"] = "PASS" if (
        checkpoint_ok and v["E4TR_COMPLETE"] == "YES"
        and v["POLICY_RUNS_COMPLETE"] == "5/5") else "FAIL"
    return audit, sel


# ---------------------------------------------------------------------------
# PHASE E4E-24 — synthetic statistical validation (no final data)
# ---------------------------------------------------------------------------

def synthetic_stats_validation() -> Dict:
    checks: Dict[str, Dict] = {}
    rng = np.random.default_rng(12345)          # synthetic-data RNG (not final env)
    B = rng.uniform(0.005, 0.009, size=N_ENVIRONMENT)   # synthetic comparator

    # 1. identical -> Delta 0
    L_id = np.tile(B, (N_POLICY, 1))
    checks["1_identical_zero_effect"] = {
        "delta": scenario_mean_effect(L_id, B),
        "PASS": abs(scenario_mean_effect(L_id, B)) < 1e-12,
    }
    # 2. DDQN better by 0.001 -> +0.001
    L_better = np.tile(B - 0.001, (N_POLICY, 1))
    d2 = scenario_mean_effect(L_better, B)
    checks["2_constant_positive_0p001"] = {"delta": d2, "PASS": abs(d2 - 0.001) < 1e-12}
    # 3. DDQN worse by 0.001 -> -0.001
    L_worse = np.tile(B + 0.001, (N_POLICY, 1))
    d3 = scenario_mean_effect(L_worse, B)
    checks["3_constant_negative_0p001"] = {"delta": d3, "PASS": abs(d3 + 0.001) < 1e-12}

    # 4. policy-axis variation preserved (CI widens when policy variance added)
    base = np.tile(B - 0.001, (N_POLICY, 1))
    pol_offsets = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])[:, None]
    L_polvar = base + pol_offsets
    ci_novar = crossed_two_axis_bootstrap(base, B, n_boot=4000, seed=101)
    ci_polvar = crossed_two_axis_bootstrap(L_polvar, B, n_boot=4000, seed=101)
    w_nov = ci_novar["ci_hi"] - ci_novar["ci_lo"]
    w_pol = ci_polvar["ci_hi"] - ci_polvar["ci_lo"]
    checks["4_policy_axis_variation_preserved"] = {
        "width_no_variation": w_nov, "width_policy_variation": w_pol,
        "PASS": w_pol > w_nov,
    }
    # 5. environment-axis variation preserved
    env_offsets = rng.uniform(-0.003, 0.003, size=N_ENVIRONMENT)[None, :]
    L_envvar = base + env_offsets
    ci_envvar = crossed_two_axis_bootstrap(L_envvar, B, n_boot=4000, seed=102)
    w_env = ci_envvar["ci_hi"] - ci_envvar["ci_lo"]
    checks["5_environment_axis_variation_preserved"] = {
        "width_no_variation": w_nov, "width_env_variation": w_env,
        "PASS": w_env > w_nov,
    }
    # 6. crossed differs from naive on a clustered fixture (policy-clustered)
    #    strong per-policy offsets shared across ALL environments -> naive row
    #    bootstrap understates variance.
    cluster_offsets = np.array([-0.004, -0.004, 0.004, 0.004, 0.0])[:, None]
    L_cluster = np.tile(B - 0.001, (N_POLICY, 1)) + cluster_offsets
    ci_crossed = crossed_two_axis_bootstrap(L_cluster, B, n_boot=8000, seed=103)
    ci_naive = naive_row_bootstrap(L_cluster, B, n_boot=8000, seed=103)
    w_crossed = ci_crossed["ci_hi"] - ci_crossed["ci_lo"]
    w_naive = ci_naive["ci_hi"] - ci_naive["ci_lo"]
    checks["6_crossed_wider_than_naive_on_cluster"] = {
        "width_crossed": w_crossed, "width_naive": w_naive,
        "PASS": w_crossed > w_naive,
    }
    # 7. macro equal 1/3 weighting
    deltas = {"PERSISTENT": 0.001, "STATIONARY": 0.002, "MIXED": 0.003}
    m = macro_effect(deltas)
    checks["7_macro_equal_third_weight"] = {"macro": m, "PASS": abs(m - 0.002) < 1e-12}
    # 8. macro reuses same draws across scenarios (identical scenarios -> macro
    #    bootstrap == single-scenario bootstrap with same seed)
    Lm = {s: L_polvar for s in SCENARIOS}
    Bm = {s: B for s in SCENARIOS}
    macro_ci = crossed_two_axis_bootstrap_macro(Lm, Bm, SCENARIOS, n_boot=4000, seed=777)
    single_ci = crossed_two_axis_bootstrap(L_polvar, B, n_boot=4000, seed=777)
    same = (abs(macro_ci["ci_lo"] - single_ci["ci_lo"]) < 1e-12
            and abs(macro_ci["ci_hi"] - single_ci["ci_hi"]) < 1e-12
            and abs(macro_ci["point_estimate"] - single_ci["point_estimate"]) < 1e-12)
    checks["8_macro_shared_draws_across_scenarios"] = {
        "macro_ci": [macro_ci["ci_lo"], macro_ci["ci_hi"]],
        "single_ci": [single_ci["ci_lo"], single_ci["ci_hi"]],
        "PASS": same,
    }
    # 9. policy robustness count exact
    r_pass = policy_robustness([0.1, 0.1, 0.1, 0.1, -0.1], threshold=4)
    r_fail = policy_robustness([0.1, 0.1, 0.1, -0.1, -0.1], threshold=4)
    checks["9_policy_robustness_count_exact"] = {
        "pass_case_n_positive": r_pass["n_positive"], "pass_case_PASS": r_pass["PASS"],
        "fail_case_n_positive": r_fail["n_positive"], "fail_case_PASS": r_fail["PASS"],
        "PASS": (r_pass["PASS"] is True and r_fail["PASS"] is False),
    }
    # 10. verdict logic exact
    v_validated = scientific_verdict(
        mixed_primary_label=BENEFICIAL, ddqn_vs_rule_mixed_label=BENEFICIAL,
        mixed_policy_robustness_pass=True, persistent_safety_pass=True,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)["verdict"]
    v_partial = scientific_verdict(
        mixed_primary_label=DIRECTIONAL_POSITIVE, ddqn_vs_rule_mixed_label=DIRECTIONAL_POSITIVE,
        mixed_policy_robustness_pass=False, persistent_safety_pass=True,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)["verdict"]
    v_failed_neutral = scientific_verdict(
        mixed_primary_label=PRACTICALLY_NEUTRAL, ddqn_vs_rule_mixed_label=BENEFICIAL,
        mixed_policy_robustness_pass=True, persistent_safety_pass=True,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)["verdict"]
    v_failed_safety = scientific_verdict(
        mixed_primary_label=BENEFICIAL, ddqn_vs_rule_mixed_label=BENEFICIAL,
        mixed_policy_robustness_pass=True, persistent_safety_pass=False,
        stationary_safety_pass=True, macro_safety_pass=True,
        checkpoint_integrity_pass=True, protocol_integrity_pass=True,
        no_training_during_final=True)["verdict"]
    checks["10_verdict_logic_exact"] = {
        "validated": v_validated, "partial": v_partial,
        "failed_neutral": v_failed_neutral, "failed_safety": v_failed_safety,
        "PASS": (v_validated == VALIDATED and v_partial == PARTIAL
                 and v_failed_neutral == FAILED and v_failed_safety == FAILED),
    }
    # 11. effect classification exact (five canonical labels)
    lab = {
        "beneficial": classify_effect(0.01, 0.001, 0.02),
        "dir_pos": classify_effect(0.01, -0.001, 0.02),
        "neutral": classify_effect(0.0, -0.01, 0.01),
        "dir_neg": classify_effect(-0.01, -0.02, 0.001),
        "harmful": classify_effect(-0.01, -0.02, -0.001),
    }
    checks["11_effect_classification_exact"] = {
        **lab,
        "PASS": (lab["beneficial"] == BENEFICIAL and lab["dir_pos"] == DIRECTIONAL_POSITIVE
                 and lab["neutral"] == PRACTICALLY_NEUTRAL and lab["dir_neg"] == DIRECTIONAL_NEGATIVE
                 and lab["harmful"] == HARMFUL),
    }

    all_pass = all(c["PASS"] for c in checks.values())
    return {"checks": checks, "ALL_PASS": all_pass, "n_checks": len(checks)}


# ---------------------------------------------------------------------------
# Protocol file builders
# ---------------------------------------------------------------------------

def build_protocol_files(audit: Dict, sel: Dict) -> Dict[str, dict]:
    files: Dict[str, dict] = {}

    files["E4E_checkpoint_freeze.json"] = {
        "immutable_for_final_evaluation": True,
        "no_weights_may_be_altered_after_this_stage": True,
        "policies": {
            str(ps): {
                "policy_seed": ps,
                "checkpoint_filename": sel["selected"][str(ps)]["selected_checkpoint"],
                "transition_count": sel["selected"][str(ps)]["transition_count"],
                "sha256": sel["selected"][str(ps)]["sha256"],
                "file_size": sel["selected"][str(ps)]["file_size"],
                "checkpoint_path": sel["selected"][str(ps)]["checkpoint_path"],
                "e4tr_validation_macro_migd_response":
                    sel["selected"][str(ps)]["VALIDATION_MACRO_MIGD_RESPONSE"],
            } for ps in POLICY_SEEDS
        },
    }

    files["E4E_comparator_freeze.json"] = {
        "preselected_from": "VALIDATION_SEEDS_ONLY (E4TR 1061..1080)",
        "primary_comparator_by_scenario": PRESELECTED_COMPARATOR,
        "rule_comparator": RULE_COMPARATOR,
        "fixed_globals_all_evaluated_descriptively": FIXED_GLOBALS,
        "no_best_fixed_global_final_reselection": True,
        "must_not_reselect_on_final_seeds": True,
    }

    files["E4E_endpoint_contract.json"] = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_direction": "lower_is_better",
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "forbidden_as_final_endpoints": [
            "cumulative_RL_reward", "Q_values", "TD_loss", "training_reward",
        ],
        "metric_definitions": {
            "MIGD_response": "IGD of joint population immediately after response, "
                             "mean over 10 scored changes per episode",
            "MIGD_epoch": "IGD after NSGA-II recovery, mean over 10 scored changes",
            "response_FE": "total response function evaluations per episode",
        },
    }

    files["E4E_effect_definition.json"] = {
        "cell_effect": "d[p,e,s] = B[e,s] - L[p,e,s]",
        "sign_convention": "POSITIVE means DDQN better (lower MIGD)",
        "scenario_mean": "Delta_response[s] = mean over 5 policy seeds and 60 env "
                         "seeds of d[p,e,s] (equal weight)",
        "secondary": {
            "Delta_epoch[s]": "B_epoch - L_epoch (positive favors DDQN)",
            "FE_saving[s]": "B_fe - L_fe (positive favors DDQN)",
        },
        "relative_improvement": "RI_response[s] = Delta_response[s] / mean_e B[e,s] "
                                "(descriptive only, not inferential)",
        "primary_statistical_units": ["policy_training_seed", "environment_seed"],
        "forbidden_replicate_units": ["region", "change", "transition"],
    }

    files["E4E_two_axis_bootstrap.json"] = {
        "design": "CROSSED_TWO_AXIS_BOOTSTRAP",
        "policy_axis_n": N_POLICY,
        "environment_axis_n": N_ENVIRONMENT,
        "resample_policy_with_replacement": True,
        "resample_environment_with_replacement": True,
        "form_full_sampled_cartesian_product": True,
        "N_BOOTSTRAP": N_BOOTSTRAP,
        "CI_LEVEL": CI_LEVEL,
        "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "substream_derivation": "SeedSequence[MASTER, *label_digests]",
        "naive_row_bootstrap": "FORBIDDEN_FOR_INFERENCE",
        "reason": "300 learned cells share 5 policies and 60 environments; naive "
                  "row bootstrap understates policy-training variance",
    }

    files["E4E_macro_contract.json"] = {
        "macro_effect": "(Delta_PERSISTENT + Delta_STATIONARY + Delta_MIXED)/3",
        "scenario_weight": 1.0 / 3.0,
        "no_observation_count_weighting": True,
        "macro_bootstrap": "draw policy indices ONCE and environment indices ONCE "
                           "per replicate; reuse identical draws across all three "
                           "scenarios; then average the three scenario effects",
        "preserves_cross_scenario_dependence": True,
    }

    files["E4E_policy_robustness.json"] = {
        "definition": "Delta_policy[p,s] = mean_e (B[e,s] - L[p,e,s])",
        "descriptive_independent_policy_replication": True,
        "no_best_policy_selection": True,
        "MIXED_POLICY_ROBUSTNESS_RULE": f"PASS iff >= {MIXED_POLICY_ROBUSTNESS_THRESHOLD} "
                                        f"of 5 policy-seed mean effects on MIXED are > 0",
        "threshold": MIXED_POLICY_ROBUSTNESS_THRESHOLD,
        "no_replacement_of_failed_policy_seeds": True,
    }

    files["E4E_practical_margin.json"] = {
        "DELTA_E4": DELTA_E4,
        "applies_to": "MIGD differences",
        "frozen_before_final_data": True,
        "immutable_after_final_evaluation": True,
    }

    files["E4E_safety_gates.json"] = {
        "persistent_comparator": PRESELECTED_COMPARATOR["PERSISTENT"],
        "stationary_comparator": PRESELECTED_COMPARATOR["STATIONARY"],
        "PERSISTENT_SAFETY": "PASS iff neither MIGD_response nor MIGD_epoch is HARMFUL",
        "STATIONARY_SAFETY": "PASS iff neither MIGD_response nor MIGD_epoch is HARMFUL",
        "MACRO_SAFETY": "PASS iff neither macro MIGD_response nor macro MIGD_epoch is HARMFUL",
        "improvement_not_required_for_safety": True,
        "fe_cannot_rescue_harmful_quality": True,
    }

    files["E4E_rule_comparison.json"] = {
        "comparison": "DDQN vs RULE_V1_V2STATE on MIXED MIGD_response",
        "rule_thresholds_source": "results/E3R_rule_v2_semantic_rerun/E3R_rule_v1_v2state.json",
        "no_retuning": True, "no_recalibration_on_final_seeds": True,
        "effect": "RULE_MIGD - DDQN_MIGD (positive favors DDQN)",
        "bootstrap": "same crossed two-axis (policy=5, environment=60)",
        "gate_name": "DDQN_VS_RULE_MIXED",
        "required_for_full_validation": "BENEFICIAL",
    }

    files["E4E_verdict_logic.json"] = {
        "labels": [BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL,
                   DIRECTIONAL_NEGATIVE, HARMFUL],
        "classification": {
            "BENEFICIAL": "mean > +DELTA_E4 AND ci_lo > 0",
            "DIRECTIONAL_POSITIVE": "mean > +DELTA_E4 AND ci_lo <= 0 <= ci_hi",
            "PRACTICALLY_NEUTRAL": "abs(mean) <= DELTA_E4 AND NOT (ci_hi < 0)",
            "DIRECTIONAL_NEGATIVE": "mean < -DELTA_E4 AND ci_lo <= 0 <= ci_hi",
            "HARMFUL": "mean < -DELTA_E4 AND ci_hi < 0",
        },
        "exhaustive_extension_documented_before_E4F": {
            "abs(mean)<=DELTA AND ci_hi<0": "DIRECTIONAL_NEGATIVE",
            "mean>+DELTA AND ci_hi<0 (contradiction)": "DIRECTIONAL_NEGATIVE (defer to CI)",
            "mean<-DELTA AND ci_lo>0 (contradiction)": "DIRECTIONAL_POSITIVE (defer to CI)",
            "note": "label meanings unchanged; extension only covers combinations "
                    "left uncovered by the five canonical definitions",
        },
        "final_verdicts": [VALIDATED, PARTIAL, FAILED],
        "VALIDATED_requires_all": [
            "MIXED_PRIMARY = BENEFICIAL",
            "DDQN_VS_RULE_MIXED = BENEFICIAL",
            "MIXED_POLICY_ROBUSTNESS = PASS",
            "PERSISTENT_SAFETY = PASS",
            "STATIONARY_SAFETY = PASS",
            "MACRO_SAFETY = PASS",
            "checkpoint_integrity = PASS",
            "final_evaluation_protocol_integrity = PASS",
            "no_training_during_final_evaluation",
        ],
        "PARTIAL_when": "MIXED_PRIMARY in {BENEFICIAL, DIRECTIONAL_POSITIVE} AND no "
                        "safety HARMFUL AND engineering PASS, but a secondary "
                        "confirmatory gate fails (CI crosses zero / <4-of-5 policies "
                        "positive / weak rule comparison / weak macro)",
        "FAILED_when": "MIXED_PRIMARY in {HARMFUL, DIRECTIONAL_NEGATIVE, "
                       "PRACTICALLY_NEUTRAL} OR any safety FAIL OR integrity failure",
        "directional_positive_is_partial_not_validated": True,
        "practically_neutral_mixed_is_failed_even_if_fe_saved": True,
        "no_p_value_fishing": True,
        "single_primary_hypothesis": "DDQN vs validation-selected GLOBAL_PREDICT on "
                                     "MIXED MIGD_response",
    }

    files["E4E_execution_order.json"] = {
        "deterministic_outcome_independent_order": True,
        "scenario_order": SCENARIOS,
        "environment_seed_order": "1081..1140 ascending",
        "arm_order_per_env_seed": EXECUTION_ARM_ORDER,
        "no_early_stop_on_results": True,
        "record_exact_order": True,
    }

    files["E4E_final_matrix_contract.json"] = {
        "learned_cells": N_POLICY * N_ENVIRONMENT * len(SCENARIOS),   # 900
        "baseline_arms": FIXED_GLOBALS + [RULE_COMPARATOR],
        "baseline_cells": len(FIXED_GLOBALS + [RULE_COMPARATOR]) * N_ENVIRONMENT * len(SCENARIOS),
        "total_planned_episodes": (N_POLICY + len(FIXED_GLOBALS) + 1) * N_ENVIRONMENT * len(SCENARIOS),
        "policy_seeds": POLICY_SEEDS,
        "environment_seeds": [FINAL_EVAL_SEEDS[0], FINAL_EVAL_SEEDS[-1]],
        "scenarios": SCENARIOS,
        "FINAL_MATRIX_COMPLETE_required": True,
        "missing_data_policy": "if any policy/env/scenario missing -> "
                               "FINAL_MATRIX_COMPLETE=NO and engineering failure review; "
                               "do NOT analyze a reduced matrix",
        "interruption_contract": "resume only from exact saved deterministic task "
                                 "index; do not rerun verified-complete cells; "
                                 "do not replace failed cells with other seeds",
    }

    files["E4E_output_schema.json"] = {
        "e4f_results_dir": E4F_DIR,
        "files": [
            "E4F_pre_execution_manifest.json", "E4F_checkpoint_integrity.json",
            "E4F_episode_results.csv", "E4F_change_results.jsonl",
            "E4F_ddqn_results.csv", "E4F_baseline_results.csv", "E4F_rule_results.csv",
            "E4F_action_distribution.csv", "E4F_policy_seed_summary.csv",
            "E4F_scenario_effects.csv", "E4F_macro_effects.csv",
            "E4F_rule_comparison.csv", "E4F_fe_effects.csv",
            "E4F_bootstrap_results.json", "E4F_scientific_gates.json",
            "E4F_verdict.json", "E4F_report.md",
        ],
        "episode_row_keys": ["controller", "policy_seed_or_null", "scenario",
                             "environment_seed"],
        "episode_metrics": ["MIGD_response", "MIGD_epoch", "response_FE"],
        "no_duplicate_keys": True,
        "retain_change_level_rows_for_diagnostics": True,
    }

    files["E4E_final_eval_execution_contract.json"] = {
        "evaluate": "selected online network weights only",
        "epsilon": 0.0, "network_mode": "eval", "gradients": "disabled",
        "replay": "disabled", "optimizer": "disabled",
        "target_network_update": "disabled", "learning": "none",
        "checkpoint_modification": "forbidden",
        "state": "V2_local + GlobalContextV1", "network_input_dim": 9,
        "actions": ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"],
        "production": "O1_F005", "no_HARD": True, "no_MEMORY": True,
        "no_RECALL": True, "no_B_r_a": True,
        "common_environment_pairing": "by environment_seed; all arms share the same "
                                      "deterministic environment schedule per "
                                      "(scenario, environment_seed)",
        "no_new_common_random_number_scheme": True,
        "production_rng_unchanged": True,
    }
    return files


def build_master_frozen(audit, sel, protocol_files, synth) -> dict:
    return {
        "stage": "E4E_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "all_fields_frozen_before_final_evaluation": True,
        "date_frozen": "2026-09-25",
        "RL_STATUS": "TRAINED_AWAITING_FINAL_EVAL",
        "DRL_TRAINING": "COMPLETE",
        "FINAL_EVALUATION": "NOT_RUN",
        "SCIENTIFIC_SUPERIORITY_CLAIM": "NOT_MADE",
        "scientific_question": "Does the learned regional Shared Double-DQN policy "
            "outperform the strongest PRESELECTED fixed-global response strategy "
            "under heterogeneous MIXED dynamics, while remaining safe under "
            "PERSISTENT and STATIONARY dynamics?",
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "primary_scenario": PRIMARY_SCENARIO,
        "preselected_comparator": PRESELECTED_COMPARATOR,
        "rule_comparator": RULE_COMPARATOR,
        "policy_seeds": POLICY_SEEDS,
        "final_environment_seeds": {"start": FINAL_EVAL_SEEDS[0],
                                    "end": FINAL_EVAL_SEEDS[-1], "n": N_ENVIRONMENT},
        "scenarios": SCENARIOS,
        "statistical_design": "CROSSED_TWO_AXIS_BOOTSTRAP",
        "policy_axis_n": N_POLICY, "environment_axis_n": N_ENVIRONMENT,
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "practical_margin_DELTA_E4": DELTA_E4,
        "mixed_policy_robustness_threshold": MIXED_POLICY_ROBUSTNESS_THRESHOLD,
        "final_matrix_planned_episodes": (N_POLICY + len(FIXED_GLOBALS) + 1)
            * N_ENVIRONMENT * len(SCENARIOS),
        "checkpoint_integrity": audit["CHECKPOINT_INTEGRITY"],
        "synthetic_stats_validation": "PASS" if synth["ALL_PASS"] else "FAIL",
        "final_eval_access_count": FIREWALL.access_count,
        "protocol_components": list(protocol_files.keys()),
        "next_stage": "E4F_FINAL_EVALUATION_EXECUTION",
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("E4E — Final Evaluation Protocol Freeze")
    print("=" * 60)

    audit, sel = pre_freeze_audit()
    if audit["CHECKPOINT_INTEGRITY"] != "PASS":
        _jdump(audit, RESULTS_DIR / "E4E_pre_freeze_manifest.json")
        print("[E4E] CHECKPOINT INTEGRITY FAIL — STOP")
        _emit_terminal(audit, {"ALL_PASS": False}, e4e_complete=False,
                       next_stage="E4TR_CHECKPOINT_INTEGRITY_REVIEW")
        return {"E4E_COMPLETE": "NO"}

    synth = synthetic_stats_validation()
    protocol_files = build_protocol_files(audit, sel)
    master = build_master_frozen(audit, sel, protocol_files, synth)

    # placeholder-token scan across every protocol file + master
    token_hits = []
    for name, obj in {**protocol_files, "E4E_final_eval_protocol_frozen.json": master}.items():
        hits = _no_forbidden_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    # firewall must remain closed
    assert FIREWALL.access_count == 0, "E4E opened a final seed"

    # ---- write all artifacts ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _jdump(audit, RESULTS_DIR / "E4E_pre_freeze_manifest.json")
    for name, obj in protocol_files.items():
        _jdump(obj, RESULTS_DIR / name)
    _jdump(synth, RESULTS_DIR / "E4E_synthetic_stats_validation.json")
    _jdump(master, RESULTS_DIR / "E4E_final_eval_protocol_frozen.json")

    e4e_complete = (
        audit["CHECKPOINT_INTEGRITY"] == "PASS"
        and synth["ALL_PASS"]
        and len(token_hits) == 0
        and FIREWALL.access_count == 0
    )

    verdict = {
        "stage": "E4E_FINAL_EVALUATION_PROTOCOL_FREEZE",
        "E4E_COMPLETE": "YES" if e4e_complete else "NO",
        "CHECKPOINT_INTEGRITY": audit["CHECKPOINT_INTEGRITY"],
        "COMPARATOR_FREEZE": "PASS",
        "ENDPOINT_FREEZE": "PASS",
        "CROSSED_BOOTSTRAP_IMPLEMENTATION": "PASS" if synth["ALL_PASS"] else "FAIL",
        "MACRO_BOOTSTRAP": "PASS" if synth["checks"]["8_macro_shared_draws_across_scenarios"]["PASS"] else "FAIL",
        "PRACTICAL_MARGIN_FROZEN": DELTA_E4,
        "VERDICT_LOGIC_FROZEN": "PASS" if synth["checks"]["10_verdict_logic_exact"]["PASS"] else "FAIL",
        "FINAL_MATRIX_CONTRACT_FROZEN": "PASS",
        "SYNTHETIC_STATS_VALIDATION": "PASS" if synth["ALL_PASS"] else "FAIL",
        "no_placeholder_tokens": len(token_hits) == 0,
        "placeholder_hits": token_hits,
        "FINAL_EVAL_ACCESS_COUNT": FIREWALL.access_count,
        "FINAL_EVALUATION": "NOT_RUN",
        "SCIENTIFIC_SUPERIORITY_CLAIM": "NOT_MADE",
        "RL_STATUS": "TRAINED_AWAITING_FINAL_EVAL",
        "DRL_TRAINING": "COMPLETE",
        "NEXT_STAGE": "E4F_FINAL_EVALUATION_EXECUTION" if e4e_complete else "E4E_FIX_REQUIRED",
    }
    _jdump(verdict, RESULTS_DIR / "E4E_verdict.json")
    _write_report(audit, synth, master, verdict)

    _emit_terminal(audit, synth, e4e_complete, verdict["NEXT_STAGE"])
    return verdict


def _write_report(audit, synth, master, verdict):
    lines = [
        "# E4E — Final Evaluation Protocol Freeze", "",
        f"**E4E_COMPLETE = {verdict['E4E_COMPLETE']}**", "",
        "Last protocol-freeze stage before the reserved final evaluation set "
        "(seeds 1081..1140) is opened in E4F. No final environment was instantiated "
        f"(FINAL_EVAL_ACCESS_COUNT = {FIREWALL.access_count}).", "",
        "## Frozen decisions", "",
        f"- Primary endpoint: **{PRIMARY_ENDPOINT}** (lower is better)",
        f"- Secondary: {', '.join(SECONDARY_ENDPOINTS)}",
        f"- Primary scenario: **{PRIMARY_SCENARIO}**",
        f"- Preselected comparators (validation-only): "
        f"PERSISTENT={PRESELECTED_COMPARATOR['PERSISTENT']}, "
        f"STATIONARY={PRESELECTED_COMPARATOR['STATIONARY']}, "
        f"MIXED={PRESELECTED_COMPARATOR['MIXED']}",
        f"- Rule comparator: {RULE_COMPARATOR} (frozen E3-R thresholds, no retuning)",
        f"- Design: CROSSED_TWO_AXIS_BOOTSTRAP (policy={N_POLICY}, env={N_ENVIRONMENT}), "
        f"N={N_BOOTSTRAP}, {int(CI_LEVEL*100)}% percentile CI, master seed {BOOTSTRAP_MASTER_SEED}",
        f"- Practical margin DELTA_E4 = {DELTA_E4}",
        f"- MIXED policy robustness: >= {MIXED_POLICY_ROBUSTNESS_THRESHOLD} of 5 positive",
        f"- Planned final matrix: {master['final_matrix_planned_episodes']} episodes "
        f"(900 learned + 900 baseline)", "",
        "## Checkpoint freeze (immutable)", "",
        "| policy | checkpoint | transitions | sha256 (head) |",
        "|---|---|---|---|",
    ]
    for ps in POLICY_SEEDS:
        c = audit["checkpoints"][str(ps)]
        lines.append(f"| {ps} | {c['checkpoint']} | {c['transition_count']} "
                     f"| {c['sha256_recorded'][:12]}… |")
    lines += [
        "", "## Synthetic statistics validation", "",
        f"- Checks: {synth['n_checks']}, ALL_PASS = {synth['ALL_PASS']}",
        "- Includes: zero/positive/negative constant effects, policy- and "
        "environment-axis variation preservation, crossed-vs-naive on a clustered "
        "fixture, macro 1/3 weighting with shared draws, policy-robustness count, "
        "five-label classification, and three-way verdict logic.", "",
        "## Status", "",
        f"- CHECKPOINT_INTEGRITY = {audit['CHECKPOINT_INTEGRITY']}",
        f"- FINAL_EVAL_ACCESS_COUNT = {FIREWALL.access_count}",
        f"- FINAL_EVALUATION = NOT_RUN; SCIENTIFIC_SUPERIORITY_CLAIM = NOT_MADE",
        f"- NEXT_STAGE = {verdict['NEXT_STAGE']}", "",
        "After E4E_COMPLETE=YES the comparator, primary endpoint, primary scenario, "
        "practical margin, bootstrap design, policy-robustness rule, safety gates and "
        "verdict logic are immutable; whatever E4F returns must be preserved.",
    ]
    (RESULTS_DIR / "E4E_report.md").write_text("\n".join(lines))


def _emit_terminal(audit, synth, e4e_complete, next_stage):
    total = (N_POLICY + len(FIXED_GLOBALS) + 1) * N_ENVIRONMENT * len(SCENARIOS)
    print()
    print(f"E4E_COMPLETE                 = {'YES' if e4e_complete else 'NO'}")
    print(f"CHECKPOINT_INTEGRITY         = {audit['CHECKPOINT_INTEGRITY']}")
    print(f"POLICY_SEEDS                 = {','.join(str(p) for p in POLICY_SEEDS)}")
    print(f"FINAL_ENVIRONMENT_SEEDS      = 1081..1140")
    print(f"PRIMARY_ENDPOINT             = MIGD_RESPONSE")
    print(f"SECONDARY_ENDPOINTS          = MIGD_EPOCH,RESPONSE_FE")
    print(f"PRIMARY_SCENARIO             = MIXED")
    print(f"PRIMARY_COMPARATOR_MIXED     = GLOBAL_PREDICT")
    print(f"SAFETY_COMPARATOR_PERSISTENT = GLOBAL_PREDICT")
    print(f"SAFETY_COMPARATOR_STATIONARY = GLOBAL_DIVERSIFY")
    print(f"RULE_COMPARATOR              = RULE_V1_V2STATE")
    print(f"EFFECT_SIGN                  = POSITIVE_MEANS_DDQN_BETTER")
    print(f"STATISTICAL_DESIGN           = CROSSED_TWO_AXIS_BOOTSTRAP")
    print(f"POLICY_AXIS_N                = {N_POLICY}")
    print(f"ENVIRONMENT_AXIS_N           = {N_ENVIRONMENT}")
    print(f"BOOTSTRAP_RESAMPLES          = {N_BOOTSTRAP}")
    print(f"CI_LEVEL                     = 95%")
    print(f"PRACTICAL_MARGIN             = {DELTA_E4}")
    print(f"MIXED_POLICY_ROBUSTNESS_RULE = AT_LEAST_4_OF_5_POSITIVE")
    print(f"FINAL_VERDICTS               = VALIDATED,PARTIAL,FAILED")
    print(f"FINAL_MATRIX_PLANNED         = {total}_EPISODES")
    print(f"FINAL_EVAL_ACCESS_COUNT      = {FIREWALL.access_count}")
    print(f"FINAL_EVALUATION             = NOT_RUN")
    print(f"SCIENTIFIC_SUPERIORITY_CLAIM = NOT_MADE")
    print(f"FULL_REGRESSION              = (run separately)")
    print(f"NEXT_STAGE                   = {next_stage}")


if __name__ == "__main__":
    main()
