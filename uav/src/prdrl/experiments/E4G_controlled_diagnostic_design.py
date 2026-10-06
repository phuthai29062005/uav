"""
E4G — Controlled Diagnostic Design (PR-DRL-DMOEA).

DESIGNS and FREEZES the next diagnostic experiments (D1..D8) that will isolate
WHY the frozen Shared Double-DQN collapsed toward a near-global action policy.

This stage does NOT:
  - train any policy or branch,
  - run any environment rollout,
  - modify any network weights / reward / normalization,
  - reuse final seeds 1081..1140,
  - change the immutable E4F verdict (FAILED).

It ONLY writes frozen diagnostic contracts, reserves fresh diagnostic-only seeds
(1201..1220, POST_HOC_DIAGNOSTIC_ONLY), and freezes branch-eligibility logic.
Execution of the diagnostics is deferred to E4GD_CONTROLLED_DIAGNOSTIC_EXECUTION.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "src")

RESULTS_DIR = Path("results/E4G_controlled_diagnostic_design")
E4F_DIR = Path("results/E4F_final_evaluation")
E4FFA_DIR = Path("results/E4F_failure_analysis")
E4TR_DIR = Path("results/E4TR_shared_double_dqn_training")
E4E_DIR = Path("results/E4E_final_evaluation_protocol_freeze")

DIAGNOSTIC_SEEDS = list(range(1201, 1221))     # 20, POST_HOC_DIAGNOSTIC_ONLY
POLICY_SEEDS = [2001, 2002, 2003, 2004, 2005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
LOCAL_FEATURES = ["severity_r", "convergence_loss_r", "occupancy_r", "delta_severity_r"]
GLOBAL_FEATURES = ["active_region_fraction", "mean_severity", "max_severity",
                   "mean_convergence_loss", "max_convergence_loss"]

# consumed seeds in the PR-DRL lineage (must stay disjoint from diagnostic seeds)
CONSUMED_SEEDS = {
    "E3_701_780": set(range(701, 781)),
    "E3R_801_882": set(range(801, 883)),
    "H1_901_940": set(range(901, 941)),
    "E4_train_1001_1060": set(range(1001, 1061)),
    "E4_validation_1061_1080": set(range(1061, 1081)),
    "E4_final_1081_1140": set(range(1081, 1141)),
}

FORBIDDEN_TOKENS = ("TBD", "TODO", "UNFROZEN", "placeholder", "PLACEHOLDER")


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


def _scan_tokens(obj) -> List[str]:
    hits = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k); walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for t in FORBIDDEN_TOKENS:
                if t in o:
                    hits.append(o); break
    walk(obj)
    return hits


# ---------------------------------------------------------------------------
# Pre-manifest / integrity
# ---------------------------------------------------------------------------

def pre_manifest() -> dict:
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4ffa = _load(E4FFA_DIR / "E4FFA_verdict.json")
    ck = _load(E4E_DIR / "E4E_checkpoint_freeze.json")
    ck_ok = all(_hash_file(ck["policies"][str(ps)]["checkpoint_path"])
                == ck["policies"][str(ps)]["sha256"] for ps in POLICY_SEEDS)
    return {
        "stage": "E4G_CONTROLLED_DIAGNOSTIC_DESIGN",
        "design_and_freeze_only": True,
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "PRIMARY_FAILURE_HYPOTHESIS": e4ffa["PRIMARY_FAILURE_HYPOTHESIS"],
        "e4f_verdict_hash": _hash_file(E4F_DIR / "E4F_verdict.json"),
        "e4f_episode_results_hash": _hash_file(E4F_DIR / "E4F_episode_results.csv"),
        "e4ffa_verdict_hash": _hash_file(E4FFA_DIR / "E4FFA_verdict.json"),
        "checkpoints_unchanged": ck_ok,
        "no_training": True, "no_environment_rollout": True,
        "no_model_modification": True, "no_final_seed_reuse": True,
    }


# ---------------------------------------------------------------------------
# Diagnostic contracts D1..D8
# ---------------------------------------------------------------------------

def d1_normalization_contract() -> dict:
    return {
        "diagnostic": "D1_STATE_NORMALIZATION",
        "goal": "test whether current normalization suppresses discriminative regional signal",
        "execution_stage": "E4GD",
        "run_environment": False,
        "data_source_priority": [
            "stored E4TR/E4F raw pre-normalization states (STATUS: not logged -> unavailable)",
            "E3R_feature_calibration.json proxy (existing, 6000 samples, DIAGNOSTIC ONLY)",
            "offline state reconstruction on diagnostic seeds 1201..1220 (no policy, "
            "environment stepping to collect states only)",
        ],
        "features_local": LOCAL_FEATURES,
        "features_global": GLOBAL_FEATURES,
        "normalization": {"severity_r": "clip[0,1] identity",
                          "convergence_loss_r": "clip[0,10]/10",
                          "occupancy_r": "clip[0,1] identity",
                          "delta_severity_r": "clip[-1,1] identity"},
        "statistics_per_feature": ["min", "max", "mean", "std", "q01", "q05",
                                   "median", "q95", "q99", "frac_zero",
                                   "frac_at_lower_clip", "frac_at_upper_clip"],
        "near_constant_definition": "normalized (q95 - q05) range < NEAR_CONST_RANGE "
                                    "OR normalized_std < NEAR_CONST_STD (documented before compute)",
        "NEAR_CONST_RANGE": 0.05,
        "NEAR_CONST_STD": 0.02,
        "classification_rule": {
            "SEVERELY_COMPRESSED": ">=3 local features near-constant",
            "COMPRESSED": "1-2 local features near-constant",
            "PRESERVED": "0 local features near-constant",
            "NOT_AUDITABLE": "no usable data source",
        },
        "returns": "D1_NORMALIZATION_SIGNAL in {PRESERVED, COMPRESSED, SEVERELY_COMPRESSED, NOT_AUDITABLE}",
    }


def d2_separability_contract() -> dict:
    return {
        "diagnostic": "D2_STATE_SEPARABILITY_AFTER_NORMALIZATION",
        "goal": "does normalization reduce nonlinear separability of regional regimes",
        "execution_stage": "E4GD",
        "run_environment": False,
        "probes_allowed": ["shallow_decision_tree(max_depth<=4)", "logistic_regression",
                           "nearest_neighbor_distance"],
        "no_neural_retraining": True,
        "not_a_controller": True,
        "label_source": "diagnostic regime/action-preference groups (offline label) — "
                        "used ONLY as a probe target, NEVER as a controller training target",
        "comparison": "RAW feature representation vs CURRENT NORMALIZED representation on "
                      "IDENTICAL cross-validation folds (k=5, fixed seed)",
        "metric": "mean cross-val balanced accuracy (or AUC) difference (raw - normalized)",
        "classification_rule": {
            "MATERIAL_LOSS": "normalized accuracy below raw by >= 0.10",
            "MODERATE_LOSS": "gap in [0.03, 0.10)",
            "NO_LOSS": "gap < 0.03",
            "NOT_AUDITABLE": "no usable labelled states",
        },
        "returns": "D2_NORMALIZATION_SEPARABILITY in {NO_LOSS, MODERATE_LOSS, MATERIAL_LOSS, NOT_AUDITABLE}",
    }


def d3_joint_representation_contract() -> dict:
    return {
        "diagnostic": "D3_JOINT_ACTION_REPRESENTATION",
        "goal": "specify candidate joint-coordination representations for a later controlled experiment",
        "execution_stage": "E4GD (conceptual + tensor-contract; NO training)",
        "current": "Q_theta(s_r, a_r); s_r = local V2 + shared global STATE context; "
                   "does NOT condition on other regions' actions",
        "reward_depends_on_joint_action_vector": True,
        "candidate_representations": {
            "J0_CURRENT": {"decision_time_inputs": "s_r (local+global state), a_r",
                           "training_time_inputs": "same",
                           "simultaneous_execution_preserved": True,
                           "decentralized_selection_preserved": True},
            "J1_LOCAL_PLUS_OTHER_ACTION_SUMMARY": {
                "decision_time_inputs": "s_r + counts(KEEP,LOCAL,PREDICT,DIVERSIFY) of OTHER regions",
                "circularity_warning": "other-region action counts are NOT available before all "
                                       "actions are selected under simultaneous decision; "
                                       "NOT usable at decision time without breaking simultaneity",
                "training_time_inputs": "usable ONLY as a centralized (training-time) critic input",
                "simultaneous_execution_preserved": "only if the summary is a training-time critic "
                                                    "(CTDE), not a decision-time policy input",
                "decentralized_selection_preserved": True},
            "J2_CENTRALIZED_JOINT_VALUE": {
                "decision_time_inputs": "regional Q_r(s_r,a_r) (decentralized)",
                "training_time_inputs": "joint Q_tot via mixing/value-decomposition over all regions",
                "simultaneous_execution_preserved": True,
                "decentralized_selection_preserved": True,
                "note": "NOT implemented/trained in E4G"},
        },
        "design_options": {
            "CURRENT_SHARED_Q": {"decision_time": "s_r,a_r", "training_time": "s_r,a_r",
                                 "simultaneous_execution": True, "decentralized_selection": True},
            "CTDE_CENTRALIZED_CRITIC": {"decision_time": "s_r,a_r (actor)",
                                        "training_time": "centralized critic sees joint state+actions",
                                        "simultaneous_execution": True, "decentralized_selection": True},
            "VALUE_DECOMPOSITION": {"decision_time": "per-region Q_r",
                                    "training_time": "Q_tot = f(Q_1..Q_R) (VDN/QMIX-style)",
                                    "simultaneous_execution": True, "decentralized_selection": True},
            "AUTOREGRESSIVE_JOINT_POLICY": {"decision_time": "sequential a_1..a_R conditioning on prior",
                                            "training_time": "joint autoregressive factorization",
                                            "simultaneous_execution": False,
                                            "decentralized_selection": False,
                                            "note": "breaks simultaneous decision requirement"},
        },
        "circularity_addressed": True,
        "winner_selected": False,
        "returns": "JOINT_COORDINATION_DESIGN_OPTIONS (no winner chosen unless evidence requires)",
    }


def d4_reward_logging_contract() -> dict:
    return {
        "diagnostic": "D4_REWARD_CREDIT_ASSIGNMENT_LOGGING",
        "goal": "measure EMPIRICAL reward-component scales (not from coefficients alone)",
        "execution_stage": "E4GD (minimal-logging re-run on diagnostic seeds 1201..1220)",
        "changes_only": "adds per-component logging",
        "must_not_change": ["policy_architecture", "optimizer", "reward_weights",
                            "normalization", "environment_protocol", "state", "action_set"],
        "reward_weights_frozen": {"alpha_g": 1.0, "alpha_l": 0.1, "beta": 0.01},
        "log_per_regional_transition": ["raw_global_reward", "weighted_global_reward",
                                        "raw_local_reward", "weighted_local_reward",
                                        "raw_fe_cost", "weighted_fe_penalty"],
        "statistics": ["mean", "std", "median", "q05", "q95", "mean_abs"],
        "dominance_metric": "median(|weighted_global|) / (median(|weighted_local|)+eps)",
        "dominance_rule": {
            "GLOBAL_OVERWHELMS_LOCAL": "median|weighted_global| >= 10x median|weighted_local|",
            "COMPARABLE": "ratio in [0.5, 10)",
            "LOCAL_COMPARABLE_OR_LARGER": "ratio < 0.5",
        },
        "returns": "empirical component scales + REWARD_COMPONENT_DOMINANCE classification",
    }


def d5_conditional_credit_contract() -> dict:
    return {
        "diagnostic": "D5_CONDITIONAL_COUNTERFACTUAL_CREDIT",
        "goal": "measure whether the chosen action is locally credit-consistent given actual joint context",
        "execution_stage": "E4GD",
        "seeds": {"range": [1201, 1220], "label": "POST_HOC_DIAGNOSTIC_ONLY"},
        "not_training": True, "not_a_supervision_oracle": True, "not_a_new_reward": True,
        "procedure": [
            "for selected MIXED events, freeze the same pre-response state",
            "for each region r, vary a_r over {KEEP,LOCAL,PREDICT,DIVERSIFY} while holding "
            "a_-r fixed to the ACTUAL policy choices",
            "execute the joint action vector (production response semantics unchanged) and "
            "measure the joint quality outcome U_conditional(r,a|a_-r)",
        ],
        "quality_convention": "U = -IGD (higher better), consistent with all prior stages",
        "chosen_action_regret": "max_a U_conditional(r,a|a_-r) - U_conditional(r,a_chosen|a_-r)",
        "report_by": ["policy", "scenario", "region", "action"],
        "classification_rule": {
            "ALIGNED": "median chosen_action_regret <= DELTA_E4 (1e-5)",
            "MISALIGNED": "median chosen_action_regret > 10*DELTA_E4",
            "MIXED": "otherwise",
        },
        "interaction_materiality": "compare U_conditional(r,a|a_-r=actual) against "
                                   "U_conditional(r,a|a_-r=all-KEEP) to see if best local action "
                                   "depends on other-region actions (materiality for BRANCH C)",
        "returns": "CONDITIONAL_CREDIT_STATUS in {ALIGNED, MISALIGNED, MIXED}",
    }


def d6_checkpoint_tension_contract() -> dict:
    return {
        "diagnostic": "D6_CHECKPOINT_SELECTION_TENSION",
        "goal": "quantify tension between MACRO-selected checkpoint and MIXED-min checkpoint",
        "execution_stage": "E4GD",
        "data_source": "results/E4TR_shared_double_dqn_training/E4TR_validation_all_policies.csv",
        "validation_only": True,
        "forbidden": "evaluating unselected checkpoints on final seeds 1081..1140",
        "per_policy_extract": ["macro_MIGD_response", "MIGD_response_MIXED",
                               "MIGD_response_PERSISTENT", "MIGD_response_STATIONARY"],
        "compare": "checkpoint selected by frozen MACRO criterion vs checkpoint minimizing "
                   "MIXED validation MIGD_response (DIAGNOSTIC ONLY)",
        "report": ["same_checkpoint(bool)", "transition_difference", "validation_MIXED_difference"],
        "classification_rule": {
            "STRONG": ">=4/5 policies differ AND mean |val MIXED diff| > 1e-4",
            "MODERATE": ">=2/5 policies differ",
            "WEAK": "1/5 policies differ",
            "ABSENT": "0 policies differ",
        },
        "must_never_claim": "the alternative checkpoint would have won the final test",
        "returns": "CHECKPOINT_SELECTION_TENSION in {STRONG, MODERATE, WEAK, ABSENT}",
    }


def d7_collapse_timeline_contract() -> dict:
    return {
        "diagnostic": "D7_COLLAPSE_MECHANISM_TIMELINE",
        "goal": "identify when action entropy decreased and what it coincides with",
        "execution_stage": "E4GD",
        "data_source": "results/E4TR_shared_double_dqn_training/E4TR_training_diagnostics.csv "
                       "(historical logs only)",
        "run_environment": False,
        "per_policy_epoch_metrics": ["action_fractions", "action_entropy", "dominant_action",
                                     "epsilon", "cumulative_reward", "MIGD_response(train)",
                                     "MIGD_epoch(train)"],
        "phases": ["before_collapse", "transition", "after_collapse"],
        "coincidence_checks": ["epsilon_reduction", "reward_improvement",
                               "validation_improvement", "validation_deterioration"],
        "no_causal_claim_unless_supported": True,
        "returns": "collapse onset epoch + coincidence summary (descriptive)",
    }


def d8_feature_usage_contract() -> dict:
    return {
        "diagnostic": "D8_LOCAL_VS_GLOBAL_FEATURE_USAGE",
        "goal": "does each action's Q depend mainly on local or global features",
        "execution_stage": "E4GD",
        "method": "input-gradient (autograd) or finite-difference sensitivity on FROZEN weights",
        "frozen_weights_only": True, "no_weight_modification": True, "run_environment": False,
        "state_source": "stored states if available else labelled SYNTHETIC grid in valid ranges",
        "group_local": LOCAL_FEATURES, "group_global": GLOBAL_FEATURES,
        "report_per_policy_and_action": ["total_local_sensitivity", "total_global_sensitivity",
                                         "ratio_local_over_global"],
        "classification_rule": {
            "LOCAL_DOMINANT": "mean ratio_local_over_global > 1.5",
            "GLOBAL_DOMINANT": "mean ratio_local_over_global < 0.67",
            "BALANCED": "ratio in [0.67, 1.5]",
            "POLICY_DEPENDENT": "classification differs across policies",
        },
        "returns": "FEATURE_USAGE_PATTERN in {LOCAL_DOMINANT, GLOBAL_DOMINANT, BALANCED, POLICY_DEPENDENT}",
    }


# ---------------------------------------------------------------------------
def diagnostic_seed_contract() -> dict:
    diag = set(DIAGNOSTIC_SEEDS)
    overlaps = {k: sorted(diag & v) for k, v in CONSUMED_SEEDS.items() if diag & v}
    return {
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "range": [DIAGNOSTIC_SEEDS[0], DIAGNOSTIC_SEEDS[-1]], "n": len(DIAGNOSTIC_SEEDS),
        "label": "POST_HOC_DIAGNOSTIC_ONLY",
        "used_by": ["D4_reward_logging", "D5_conditional_credit"],
        "never_use_as_confirmatory_test_seeds": True,
        "disjoint_from_consumed_prdrl_seeds": len(overlaps) == 0,
        "overlaps": overlaps,
        "consumed_seed_sets": {k: [min(v), max(v)] for k, v in CONSUMED_SEEDS.items()},
    }


def branch_logic() -> dict:
    return {
        "branches": {
            "BRANCH_A_STATE_NORMALIZATION_REVISION": {
                "eligible_iff": "D1_NORMALIZATION_SIGNAL in {COMPRESSED, SEVERELY_COMPRESSED} "
                                "AND D2_NORMALIZATION_SEPARABILITY in {MODERATE_LOSS, MATERIAL_LOSS}",
            },
            "BRANCH_B_REWARD_CREDIT_REVISION": {
                "eligible_iff": "D4 REWARD_COMPONENT_DOMINANCE == GLOBAL_OVERWHELMS_LOCAL "
                                "OR D5 CONDITIONAL_CREDIT_STATUS == MISALIGNED",
            },
            "BRANCH_C_JOINT_COORDINATION_ARCHITECTURE_REVISION": {
                "eligible_iff": "JOINT_ACTION_REPRESENTATION limitation remains SUPPORTED "
                                "AND D5 interaction_materiality shows best local action depends "
                                "on other-region actions (material)",
            },
        },
        "priority_rule": "prefer the SMALLEST scientific modification that targets the STRONGEST "
                         "diagnosed mechanism",
        "one_factor_at_a_time": True,
        "forbidden_simultaneous_change": ["state", "reward", "architecture", "optimizer"],
        "no_hyperparameter_search": True,
        "branch_selected_in_E4G": False,
        "branch_selection_stage": "E4GD (after diagnostics executed)",
    }


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("E4G — Controlled Diagnostic Design")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pre_manifest()
    contracts = {
        "E4G_D1_normalization_contract.json": d1_normalization_contract(),
        "E4G_D2_separability_contract.json": d2_separability_contract(),
        "E4G_D3_joint_representation_contract.json": d3_joint_representation_contract(),
        "E4G_D4_reward_logging_contract.json": d4_reward_logging_contract(),
        "E4G_D5_conditional_credit_contract.json": d5_conditional_credit_contract(),
        "E4G_D6_checkpoint_tension_contract.json": d6_checkpoint_tension_contract(),
        "E4G_D7_collapse_timeline_contract.json": d7_collapse_timeline_contract(),
        "E4G_D8_feature_usage_contract.json": d8_feature_usage_contract(),
    }
    seed_contract = diagnostic_seed_contract()
    branches = branch_logic()

    master = {
        "stage": "E4G_CONTROLLED_DIAGNOSTIC_DESIGN",
        "all_diagnostics_frozen": True,
        "design_and_freeze_only": True,
        "ORIGINAL_E4F_VERDICT": manifest["ORIGINAL_E4F_VERDICT"],
        "PRIMARY_FAILURE_HYPOTHESIS": manifest["PRIMARY_FAILURE_HYPOTHESIS"],
        "diagnostics": list(contracts.keys()),
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "diagnostic_seed_label": "POST_HOC_DIAGNOSTIC_ONLY",
        "branch_logic": branches,
        "one_factor_at_a_time": True,
        "no_training": True, "no_environment_rollout": True,
        "no_model_modification": True, "no_final_seed_reuse": True,
        "next_stage": "E4GD_CONTROLLED_DIAGNOSTIC_EXECUTION",
    }

    # placeholder-token scan
    token_hits = []
    for name, obj in {**contracts, "E4G_diagnostic_seed_contract.json": seed_contract,
                      "E4G_branch_logic.json": branches,
                      "E4G_protocol_frozen.json": master}.items():
        hits = _scan_tokens(obj)
        if hits:
            token_hits.append({name: hits})

    seed_ok = seed_contract["disjoint_from_consumed_prdrl_seeds"]
    e4e_complete = (manifest["ORIGINAL_E4F_VERDICT"] == "FAILED"
                    and manifest["checkpoints_unchanged"] and seed_ok
                    and len(token_hits) == 0)

    # write everything
    _jdump(manifest, RESULTS_DIR / "E4G_pre_manifest.json")
    for name, obj in contracts.items():
        _jdump(obj, RESULTS_DIR / name)
    _jdump(seed_contract, RESULTS_DIR / "E4G_diagnostic_seed_contract.json")
    _jdump(branches, RESULTS_DIR / "E4G_branch_logic.json")
    _jdump(master, RESULTS_DIR / "E4G_protocol_frozen.json")

    verdict = {
        "stage": "E4G_CONTROLLED_DIAGNOSTIC_DESIGN",
        "E4G_COMPLETE": "YES" if e4e_complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "PRIMARY_FAILURE_HYPOTHESIS": manifest["PRIMARY_FAILURE_HYPOTHESIS"],
        "D1_NORMALIZATION_DIAGNOSTIC": "FROZEN",
        "D2_SEPARABILITY_DIAGNOSTIC": "FROZEN",
        "D3_JOINT_COORDINATION_DIAGNOSTIC": "FROZEN",
        "D4_REWARD_SCALE_DIAGNOSTIC": "FROZEN",
        "D5_CONDITIONAL_CREDIT_DIAGNOSTIC": "FROZEN",
        "D6_CHECKPOINT_TENSION_DIAGNOSTIC": "FROZEN",
        "D7_COLLAPSE_TIMELINE_DIAGNOSTIC": "FROZEN",
        "D8_FEATURE_USAGE_DIAGNOSTIC": "FROZEN",
        "DIAGNOSTIC_SEEDS": f"{DIAGNOSTIC_SEEDS[0]}..{DIAGNOSTIC_SEEDS[-1]}",
        "diagnostic_seeds_disjoint": seed_ok,
        "no_placeholder_tokens": len(token_hits) == 0,
        "placeholder_hits": token_hits,
        "NEW_TRAINING": "NO", "FINAL_SEED_REUSE": "NO", "MODEL_MODIFICATION": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": "E4GD_CONTROLLED_DIAGNOSTIC_EXECUTION",
    }
    _jdump(verdict, RESULTS_DIR / "E4G_verdict.json")
    _write_report(verdict, contracts, branches, seed_contract)
    _emit_terminal(verdict)
    return verdict


def _write_report(v, contracts, branches, seed_contract):
    L = ["# E4G — Controlled Diagnostic Design", "",
         f"**E4G_COMPLETE = {v['E4G_COMPLETE']}** — design/freeze only; no training, no rollouts, "
         "no model change.", "",
         f"Original E4F verdict remains **FAILED** (immutable). Primary failure hypothesis: "
         f"**{v['PRIMARY_FAILURE_HYPOTHESIS']}**.", "",
         "## Frozen diagnostics", "",
         "| id | goal | executes env? | classification |",
         "|---|---|---|---|"]
    labels = {
        "E4G_D1_normalization_contract.json": ("D1", "normalization compresses signal?"),
        "E4G_D2_separability_contract.json": ("D2", "normalization hurts separability?"),
        "E4G_D3_joint_representation_contract.json": ("D3", "joint-action representation options"),
        "E4G_D4_reward_logging_contract.json": ("D4", "empirical reward-component scales"),
        "E4G_D5_conditional_credit_contract.json": ("D5", "conditional counterfactual credit"),
        "E4G_D6_checkpoint_tension_contract.json": ("D6", "macro vs MIXED checkpoint tension"),
        "E4G_D7_collapse_timeline_contract.json": ("D7", "collapse onset timeline"),
        "E4G_D8_feature_usage_contract.json": ("D8", "local vs global feature usage"),
    }
    for name, obj in contracts.items():
        did, goal = labels[name]
        env = obj.get("run_environment", "see contract")
        ret = obj.get("returns", "")
        L.append(f"| {did} | {goal} | {env} | {ret[:60]} |")
    L += ["", "## Diagnostic seeds", "",
          f"- {seed_contract['range'][0]}..{seed_contract['range'][1]} "
          f"(n={seed_contract['n']}), label **POST_HOC_DIAGNOSTIC_ONLY**, "
          f"disjoint from consumed PR-DRL seeds = {seed_contract['disjoint_from_consumed_prdrl_seeds']}.",
          "- Used only by D4 (reward logging) and D5 (conditional credit); never a confirmatory test seed.",
          "", "## Branch logic (one factor at a time; selected in E4GD, not here)", ""]
    for b, spec in branches["branches"].items():
        L.append(f"- **{b}** — eligible iff: {spec['eligible_iff']}")
    L += ["", f"- Priority: {branches['priority_rule']}.",
          f"- Forbidden simultaneous change: {branches['forbidden_simultaneous_change']}.",
          "", f"## Next stage: {v['NEXT_STAGE']}",
          "", "No branch is executed until its diagnostic evidence is produced in E4GD. "
          "Any redesign targets exactly one diagnosed mechanism and is a NEW predeclared experiment."]
    (RESULTS_DIR / "E4G_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    print(f"E4G_COMPLETE                     = {v['E4G_COMPLETE']}")
    print(f"ORIGINAL_E4F_VERDICT             = {v['ORIGINAL_E4F_VERDICT']}")
    print(f"PRIMARY_FAILURE_HYPOTHESIS       = {v['PRIMARY_FAILURE_HYPOTHESIS']}")
    print(f"D1_NORMALIZATION_DIAGNOSTIC      = {v['D1_NORMALIZATION_DIAGNOSTIC']}")
    print(f"D2_SEPARABILITY_DIAGNOSTIC       = {v['D2_SEPARABILITY_DIAGNOSTIC']}")
    print(f"D3_JOINT_COORDINATION_DIAGNOSTIC = {v['D3_JOINT_COORDINATION_DIAGNOSTIC']}")
    print(f"D4_REWARD_SCALE_DIAGNOSTIC       = {v['D4_REWARD_SCALE_DIAGNOSTIC']}")
    print(f"D5_CONDITIONAL_CREDIT_DIAGNOSTIC = {v['D5_CONDITIONAL_CREDIT_DIAGNOSTIC']}")
    print(f"D6_CHECKPOINT_TENSION_DIAGNOSTIC = {v['D6_CHECKPOINT_TENSION_DIAGNOSTIC']}")
    print(f"D7_COLLAPSE_TIMELINE_DIAGNOSTIC  = {v['D7_COLLAPSE_TIMELINE_DIAGNOSTIC']}")
    print(f"D8_FEATURE_USAGE_DIAGNOSTIC      = {v['D8_FEATURE_USAGE_DIAGNOSTIC']}")
    print(f"DIAGNOSTIC_SEEDS                 = {v['DIAGNOSTIC_SEEDS']}")
    print(f"NEW_TRAINING                     = {v['NEW_TRAINING']}")
    print(f"FINAL_SEED_REUSE                 = {v['FINAL_SEED_REUSE']}")
    print(f"MODEL_MODIFICATION               = {v['MODEL_MODIFICATION']}")
    print(f"FULL_REGRESSION                  = {v['FULL_REGRESSION']}")
    print(f"NEXT_STAGE                       = {v['NEXT_STAGE']}")


if __name__ == "__main__":
    main()
