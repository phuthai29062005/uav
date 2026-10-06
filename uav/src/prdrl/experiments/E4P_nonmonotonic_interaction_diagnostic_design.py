"""E4P — Nonmonotonic Interaction Diagnostic Design.

DESIGN ONLY. No environment execution, no training, no seed opening.
Freezes counterfactual surface contracts for testing whether true
environment response surfaces contain context-dependent regional
action-rank reversals incompatible with monotonic factorization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

STAGE = "E4P_NONMONOTONIC_INTERACTION_DIAGNOSTIC_DESIGN"
BASE = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = BASE / "results" / "E4P_nonmonotonic_interaction_diagnostic_design"
E4ODR_DIR = BASE / "results" / "E4ODR_e4od_report_artifact_reconciliation"
E4OD_DIR = BASE / "results" / "E4OD_scenario_tradeoff_diagnostic_execution"
E4MF_DIR = BASE / "results" / "E4MF_credit_preservation_ab_final_evaluation_execution"
E4N_DIR = BASE / "results" / "E4N_credit_preservation_failure_analysis"
E4O_DIR = BASE / "results" / "E4O_scenario_tradeoff_diagnostic_design"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]

ALL_HISTORICAL_ENV_SEEDS = set()
for lo, hi in [
    (1001, 1060), (1061, 1080), (1081, 1140),
    (1201, 1220),
    (1301, 1360), (1361, 1380), (1381, 1440),
    (1501, 1560), (1561, 1580), (1581, 1640),
    (1701, 1720),
    (1801, 1860),
    (1901, 1960), (1961, 1980), (1981, 2040),
    (2101, 2160), (2161, 2180), (2181, 2240),
    (2301, 2360),
]:
    ALL_HISTORICAL_ENV_SEEDS |= set(range(lo, hi + 1))

ALL_HISTORICAL_POLICY_SEEDS = {
    3001, 3002, 3003, 3004, 3005,
    4001, 4002, 4003, 4004, 4005,
    5001, 5002, 5003, 5004, 5005,
    6001, 6002, 6003, 6004, 6005,
}
ALL_HISTORICAL_BOOTSTRAP_SEEDS = {56001, 66001, 76001, 86001, 106001, 116001}

DIAGNOSTIC_SEEDS = list(range(2401, 2421))
N_DIAG_SEEDS = 20
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
BOOTSTRAP_SEED = 126001
N_BOOTSTRAP = 50000
COUNTERFACTUAL_RANDOM_TAPES = 4

PRIMARY_ACTIONS = ["LOCAL", "PREDICT", "DIVERSIFY"]
SECONDARY_ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]

OPERATOR_CONSTANTS = {
    "KEEP": {"FE": 0},
    "LOCAL": {"sigma_L": 0.02, "B_PER_ACTIVE_REGION": 10},
    "PREDICT": {"sigma_P": 0.005, "lambda_P": 1.0, "B_PER_ACTIVE_REGION": 10,
                "fallback": "LOCAL"},
    "DIVERSIFY": {"sigma_D": 0.15, "B_PER_ACTIVE_REGION": 10},
}
SURVIVAL = "UNRESTRICTED_STANDARD_NSGA_II"
EPSILON_RANK = 1e-12


def _load(p):
    with open(p) as f:
        return json.load(f)


def _jdump(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # PHASE E4P-0: HISTORICAL INTEGRITY
    # ==================================================================
    print(f"[{STAGE}] Phase 0: Historical integrity...")

    e4odr = _load(E4ODR_DIR / "E4ODR_verdict.json")
    e4od = _load(E4OD_DIR / "E4OD_verdict.json")
    e4mf = _load(E4MF_DIR / "E4MF_verdict.json")

    checks = {
        "E4ODR_COMPLETE": e4odr["E4ODR_COMPLETE"] == "YES",
        "E4OD_COMPLETE": e4od.get("E4OD_COMPLETE", "YES") == "YES",
        "E4MF_FAILED": e4mf["CREDIT_PRESERVATION_CAUSAL_VERDICT"] == "FAILED",
        "E4OD_PERF_NOT_COMPUTED": e4od.get("PERFORMANCE_CAUSAL_VERDICT", "NOT_COMPUTED") == "NOT_COMPUTED",
        "O1_NOT_SUPPORTED": e4od["O1_SUPPORT"] == "NOT_SUPPORTED",
        "N8_NOT_SUPPORTED": e4odr["N8_STATUS"] == "NOT_SUPPORTED",
        "N9_PLAUSIBLE": e4odr["N9_STATUS"] == "PLAUSIBLE",
        "N7_CORRECTED_NAME": e4odr["N7_AUTHORITATIVE_NAME"] == "N7_POLICY_SEED_HETEROGENEITY",
        "RAW_HASH_PASS": e4odr["RAW_HASH_INTEGRITY"] == "PASS",
        "E4MF_NOT_CHANGED": e4odr["E4MF_VERDICT_CHANGED"] == "NO",
    }
    all_pass = all(checks.values())

    _jdump({
        "stage": STAGE,
        "checks": checks,
        "all_pass": all_pass,
    }, RESULTS_DIR / "E4P_historical_integrity.json")

    if not all_pass:
        failed = [k for k, v in checks.items() if not v]
        print(f"    *** INTEGRITY FAIL: {failed} ***")
        _jdump({
            "stage": STAGE, "E4P_COMPLETE": "NO",
            "NEXT_STAGE": "E4P_INTEGRITY_REVIEW",
            "failed_checks": failed,
        }, RESULTS_DIR / "E4P_verdict.json")
        return

    # ==================================================================
    # PRE-MANIFEST
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "design_only": True,
        "new_environment_execution": 0,
        "new_training": "NO",
        "seeds_opened": "NO",
        "checkpoints_changed": "NO",
    }, RESULTS_DIR / "E4P_pre_manifest.json")

    # ==================================================================
    # E4ODR LOCK
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "E4ODR_COMPLETE": "YES",
        "O1_NOT_SUPPORTED": True,
        "N8_NOT_SUPPORTED": True,
        "N9_PLAUSIBLE": True,
        "E4MF_FAILED": True,
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
    }, RESULTS_DIR / "E4P_e4odr_lock.json")

    # ==================================================================
    # PHASE E4P-1: PRIMARY HYPOTHESIS
    # ==================================================================
    print(f"[{STAGE}] Phase 1: Primary hypothesis...")
    _jdump({
        "stage": STAGE,
        "hypothesis_id": "P1",
        "hypothesis_name": "P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL",
        "hypothesis_text": (
            "STATIONARY response surfaces contain stronger context-dependent "
            "regional action-rank reversals than PERSISTENT/MIXED response surfaces."
        ),
        "relevance": (
            "Tests whether the remaining monotonic-factorization limitation (N9) "
            "plausibly explains the observed scenario trade-off."
        ),
        "scenario_contrast_direction": (
            "G_REV > 0 means STATIONARY has more reversals, supporting the "
            "hypothesis that monotonic factorization is more limiting there."
        ),
    }, RESULTS_DIR / "E4P_hypothesis_contract.json")

    # ==================================================================
    # PHASE E4P-2: SCIENTIFIC QUESTION
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "question": (
            "For fixed pre-response state S and region i, does the preferred "
            "action between a_i and b_i change when another region j changes "
            "action? Formally, does there exist context c1, c2 such that "
            "Delta_i(a,b | c1) > 0 and Delta_i(a,b | c2) < 0?"
        ),
        "distinction": (
            "Nonzero interaction alone does NOT violate QMIX monotonicity. "
            "The primary diagnostic tests CONTEXT-DEPENDENT ACTION-RANK REVERSAL, "
            "not merely interaction != 0."
        ),
        "monotonicity_constraint": (
            "QMIX imposes dQ_tot/dq_r >= 0. For fixed state, if q_i(a) > q_i(b) "
            "then monotonic mixer cannot reverse that ordering. A TRUE environment "
            "preference reversal where action a preferred under context c1 but "
            "action b preferred under context c2 is direct evidence that a single "
            "context-independent local action ordering is insufficient."
        ),
    }, RESULTS_DIR / "E4P_scientific_question.json")

    # ==================================================================
    # PHASE E4P-2: ENVIRONMENT UTILITY
    # ==================================================================
    print(f"[{STAGE}] Phase 2: Environment utility contract...")
    _jdump({
        "stage": STAGE,
        "utility_definition": (
            "V_S(A) = Q_env(P_post(S,A)) - Q_env(P_post(S,KEEP_ALL)). "
            "Q_env is the existing population quality function: IGD(F, pf_star) "
            "where F = objective values of the population, pf_star = reference "
            "Pareto front. Lower IGD = better quality. Therefore V_S(A) = "
            "IGD_post(KEEP_ALL) - IGD_post(A) so that higher V = better response."
        ),
        "quality_function": "IGD (Inverted Generational Distance)",
        "quality_source": "src/prdrl/metrics/igd.py :: igd(A, P_star)",
        "V_sign_convention": "V_S(A) = IGD(KEEP_ALL) - IGD(A); positive = improvement",
        "KEEP_ALL_baseline": (
            "KEEP_ALL = all active regions request KEEP (FE=0). Same pre-response "
            "state S. Baseline reused for all cells of the same surface."
        ),
        "learned_Q_ground_truth": False,
        "note": (
            "Do NOT use learned Q values as ground truth. Use only environment "
            "IGD computed from actual population after response."
        ),
    }, RESULTS_DIR / "E4P_environment_utility_contract.json")

    # ==================================================================
    # PHASE E4P-3/4: COUNTERFACTUAL SURFACE
    # ==================================================================
    print(f"[{STAGE}] Phase 3-4: Counterfactual surface contracts...")
    _jdump({
        "stage": STAGE,
        "surface_type": "REGION_PAIR_COUNTERFACTUAL",
        "description": (
            "For a selected active region pair (i,j), hold all other regions "
            "under a fixed background context. Vary actions for region i and j."
        ),
        "primary_surface": {
            "actions": PRIMARY_ACTIONS,
            "grid_size": "3x3",
            "n_cells": 9,
            "equal_response_budget": True,
            "KEEP_excluded": True,
            "KEEP_exclusion_reason": (
                "KEEP has FE=0 while LOCAL/PREDICT/DIVERSIFY use B=10 per active "
                "region. Excluding KEEP avoids confounding rank reversal with "
                "unequal response budget."
            ),
        },
        "secondary_surface": {
            "actions": SECONDARY_ACTIONS,
            "grid_size": "4x4",
            "n_cells": 16,
            "equal_response_budget": False,
            "label": "SECONDARY_FE_CONFOUNDED_CONTEXT",
            "note": "KEEP cells have different FE. Cannot override primary verdict.",
        },
        "isolation": (
            "Every action cell forks from the EXACT SAME pre-response snapshot. "
            "After evaluating a cell, discard the resulting population. No "
            "cross-cell contamination."
        ),
    }, RESULTS_DIR / "E4P_counterfactual_surface_contract.json")

    _jdump({
        "stage": STAGE,
        "primary_actions": PRIMARY_ACTIONS,
        "n_primary": 3,
        "KEEP_excluded": True,
        "equal_budget": True,
        "budget_per_action": 10,
    }, RESULTS_DIR / "E4P_primary_action_set.json")

    _jdump({
        "stage": STAGE,
        "secondary_actions": SECONDARY_ACTIONS,
        "n_secondary": 4,
        "KEEP_included": True,
        "label": "SECONDARY_FE_CONFOUNDED_CONTEXT",
    }, RESULTS_DIR / "E4P_secondary_action_set.json")

    # ==================================================================
    # PHASE E4P-5: BACKGROUND CONTEXTS
    # ==================================================================
    print(f"[{STAGE}] Phase 5: Background context contracts...")
    _jdump({
        "stage": STAGE,
        "BACKGROUND_A": {
            "name": "KEEP_BACKGROUND",
            "definition": "All active regions other than (i,j) request KEEP.",
            "role": "PRIMARY",
        },
        "BACKGROUND_B": {
            "name": "SOURCE_POLICY_BACKGROUND",
            "definition": (
                "All active regions other than (i,j) use greedy actions from "
                "the source policy at the same pre-response state."
            ),
            "role": "ROBUSTNESS_DESCRIPTIVE",
        },
        "primary_verdict_based_on": "KEEP_BACKGROUND",
        "policy_background_cannot_override_primary": True,
    }, RESULTS_DIR / "E4P_background_context_contract.json")

    # ==================================================================
    # PHASE E4P-6/7: SOURCE TRAJECTORIES & CHECKPOINTS
    # ==================================================================
    print(f"[{STAGE}] Phase 6-7: Source trajectories & checkpoints...")
    _jdump({
        "stage": STAGE,
        "source_arms": ["C1", "C2"],
        "reason": (
            "Learned controllers induce different population trajectories. "
            "Representability claim should not rely only on states visited "
            "by one arm."
        ),
        "aggregation": "equal 1/2 + 1/2 weight within matched policy×env×scenario unit",
    }, RESULTS_DIR / "E4P_source_trajectory_contract.json")

    _jdump({
        "stage": STAGE,
        "C1_policies": POLICY_SEEDS,
        "C2_policies": POLICY_SEEDS,
        "total_checkpoints": 10,
        "source": "Frozen from E4MT/E4MTR training",
        "no_reselection": True,
        "no_retraining": True,
    }, RESULTS_DIR / "E4P_checkpoint_lock.json")

    # ==================================================================
    # PHASE E4P-8: DIAGNOSTIC SEEDS
    # ==================================================================
    print(f"[{STAGE}] Phase 8: Diagnostic seeds...")
    proposed = set(DIAGNOSTIC_SEEDS)
    overlap_env = proposed & ALL_HISTORICAL_ENV_SEEDS
    overlap_pol = proposed & ALL_HISTORICAL_POLICY_SEEDS
    overlap_bs = proposed & ALL_HISTORICAL_BOOTSTRAP_SEEDS
    bs_overlap = {BOOTSTRAP_SEED} & ALL_HISTORICAL_BOOTSTRAP_SEEDS
    disjoint = len(overlap_env) == 0 and len(overlap_pol) == 0 and len(overlap_bs) == 0 and len(bs_overlap) == 0

    _jdump({
        "stage": STAGE,
        "diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "n_seeds": N_DIAG_SEEDS,
        "range": "2401..2420",
    }, RESULTS_DIR / "E4P_diagnostic_seed_contract.json")

    _jdump({
        "stage": STAGE,
        "proposed_diagnostic_seeds": DIAGNOSTIC_SEEDS,
        "proposed_bootstrap_seed": BOOTSTRAP_SEED,
        "overlap_with_env_seeds": sorted(overlap_env),
        "overlap_with_policy_seeds": sorted(overlap_pol),
        "overlap_with_bootstrap_seeds": sorted(overlap_bs | bs_overlap),
        "disjoint": disjoint,
        "historical_env_seed_ranges": [
            "1001-1060", "1061-1080", "1081-1140", "1201-1220",
            "1301-1360", "1361-1380", "1381-1440",
            "1501-1560", "1561-1580", "1581-1640", "1701-1720",
            "1801-1860", "1901-1960", "1961-1980", "1981-2040",
            "2101-2160", "2161-2180", "2181-2240", "2301-2360",
        ],
        "historical_policy_seeds": sorted(ALL_HISTORICAL_POLICY_SEEDS),
        "historical_bootstrap_seeds": sorted(ALL_HISTORICAL_BOOTSTRAP_SEEDS),
    }, RESULTS_DIR / "E4P_seed_disjointness.json")

    if not disjoint:
        print(f"    *** SEED OVERLAP: env={overlap_env}, pol={overlap_pol} ***")
        _jdump({
            "stage": STAGE, "E4P_COMPLETE": "NO",
            "NEXT_STAGE": "E4P_SEED_REVIEW",
        }, RESULTS_DIR / "E4P_verdict.json")
        return

    # ==================================================================
    # PHASE E4P-9: FUTURE SOURCE MATRIX
    # ==================================================================
    n_source_episodes = 2 * len(POLICY_SEEDS) * N_DIAG_SEEDS * len(SCENARIOS)
    # 2 * 5 * 20 * 3 = 600

    # ==================================================================
    # PHASE E4P-10: SNAPSHOT ELIGIBILITY
    # ==================================================================
    print(f"[{STAGE}] Phase 10: Snapshot eligibility...")
    _jdump({
        "stage": STAGE,
        "selection_rule": "FIRST dynamic-change event satisfying all eligibility criteria",
        "eligibility_criteria": [
            "at least 2 active regions",
            "pre-response regional states available",
            "current global context available",
            "PREDICT history available for candidate selected regions",
        ],
        "deterministic": True,
        "outcome_blind": True,
        "no_eligible_mark": "NO_ELIGIBLE_SNAPSHOT",
        "no_substitution_from_outcomes": True,
    }, RESULTS_DIR / "E4P_snapshot_eligibility_contract.json")

    # ==================================================================
    # PHASE E4P-11: REGION-PAIR SELECTION
    # ==================================================================
    print(f"[{STAGE}] Phase 11: Region-pair selection...")
    _jdump({
        "stage": STAGE,
        "selection_method": "SHA256_DETERMINISTIC_INDEX",
        "steps": [
            "1. Construct all unordered eligible active region pairs",
            "2. Sort pairs lexicographically by (region_i, region_j)",
            "3. Compute SHA256 of '{source_arm}_{policy_seed}_{scenario}_{env_seed}'",
            "4. Convert first 8 bytes to int, mod by number of pairs",
            "5. Select pair at that index",
        ],
        "hash_function": "hashlib.sha256",
        "python_builtin_hash_forbidden": True,
        "outcome_blind": True,
        "inputs_for_hash": ["source_arm", "policy_seed", "scenario", "environment_seed"],
        "forbidden_inputs": ["performance", "rewards", "Q values", "future outcomes"],
    }, RESULTS_DIR / "E4P_pair_selection_contract.json")

    # ==================================================================
    # PHASE E4P-12: PRODUCTION OPERATORS
    # ==================================================================
    # Already defined in OPERATOR_CONSTANTS at module level

    # ==================================================================
    # PHASE E4P-13/14: COMMON RANDOM NUMBERS & TAPE COUNT
    # ==================================================================
    print(f"[{STAGE}] Phase 13-14: Random tape contracts...")
    _jdump({
        "stage": STAGE,
        "n_tapes": COUNTERFACTUAL_RANDOM_TAPES,
        "purpose": (
            "Reduce stochastic differences unrelated to action choice. "
            "Same random tape reused across every action cell in a surface."
        ),
        "tape_generation": (
            "Deterministic from source_arm, policy_seed, scenario, env_seed, "
            "tape_index using a seeded RNG."
        ),
        "primary_cell_utility": "mean across 4 tapes",
        "tape_level_retained": True,
        "no_post_hoc_tape_count_change": True,
    }, RESULTS_DIR / "E4P_random_tape_contract.json")

    # ==================================================================
    # PHASE E4P-15: COUNTERFACTUAL ISOLATION
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "isolation_rule": (
            "Every action cell forks from the EXACT SAME pre-response snapshot. "
            "After evaluating a cell, discard the resulting population. "
            "No cross-cell contamination. S -> one response -> evaluate -> discard."
        ),
    }, RESULTS_DIR / "E4P_counterfactual_isolation_contract.json")

    # ==================================================================
    # PHASE E4P-16: ACTION PREFERENCE
    # ==================================================================
    print(f"[{STAGE}] Phase 16: Action preference contract...")
    _jdump({
        "stage": STAGE,
        "definition": (
            "Delta_i(a,b | j=c) = V_S(i=a,j=c) - V_S(i=b,j=c). "
            "Positive: a preferred to b. Negative: b preferred to a."
        ),
        "symmetry": "Compute analogously for region j holding i fixed.",
    }, RESULTS_DIR / "E4P_action_preference_contract.json")

    # ==================================================================
    # PHASE E4P-17/18: RANK REVERSAL
    # ==================================================================
    print(f"[{STAGE}] Phase 17-18: Rank reversal contract...")
    _jdump({
        "stage": STAGE,
        "definition": (
            "For fixed region i and action pair (a,b), a context-dependent "
            "rank reversal occurs if there exist actions c,d of region j "
            "such that Delta_i(a,b|j=c) > +epsilon_rank AND "
            "Delta_i(a,b|j=d) < -epsilon_rank."
        ),
        "epsilon_rank": EPSILON_RANK,
        "epsilon_source": "numerical tolerance (no arbitrary practical margin)",
        "ties": "abs(Delta) <= epsilon_rank → TIE. Ties do NOT create reversals.",
        "symmetric_check": True,
    }, RESULTS_DIR / "E4P_rank_reversal_contract.json")

    # ==================================================================
    # PHASE E4P-20: BEST-ACTION CONTEXT
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "definition": (
            "For every action c of the other region j: "
            "a_i*(c) = argmax_a V_S(i=a,j=c) over {LOCAL,PREDICT,DIVERSIFY}. "
            "BEST_ACTION_CONTEXT_DEPENDENT_i = YES iff a_i*(c) is not constant "
            "across j's three action contexts, excluding exact numerical ties."
        ),
        "symmetric": True,
    }, RESULTS_DIR / "E4P_best_action_context_contract.json")

    # ==================================================================
    # PHASE E4P-21: CROSS-DIFFERENCE INTERACTION
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "formula": (
            "I_ij(a,b;c,d) = V(a,c) - V(b,c) - V(a,d) + V(b,d)"
        ),
        "records": "signed I and |I|",
        "WARNING": (
            "I != 0 demonstrates interaction but does NOT by itself demonstrate "
            "a QMIX monotonicity violation. Interaction magnitude is SECONDARY. "
            "Rank reversal is PRIMARY."
        ),
    }, RESULTS_DIR / "E4P_cross_difference_contract.json")

    # ==================================================================
    # PHASE E4P-23: SOURCE-ARM AGGREGATION
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "rule": (
            "Report SOURCE_C1 and SOURCE_C2 reversal rates separately. "
            "For primary inferential statistic, average the two source-arm "
            "estimates with equal 1/2 + 1/2 weight within matched "
            "policy × environment × scenario unit."
        ),
        "prevents": "trajectory-source duplication from becoming false independent axis",
    }, RESULTS_DIR / "E4P_source_aggregation_contract.json")

    # ==================================================================
    # PHASE E4P-24/25/26: SCENARIO CONTRASTS
    # ==================================================================
    print(f"[{STAGE}] Phase 24-26: Scenario contrasts...")
    _jdump({
        "stage": STAGE,
        "G_REV": {
            "formula": "G_REV = RR_STAT - 0.5*(RR_PER + RR_MIX)",
            "interpretation": "G_REV > 0 means STATIONARY has more reversals",
            "role": "PRIMARY",
        },
        "G_BAC": {
            "formula": "G_BAC = BAC_STAT - 0.5*(BAC_PER + BAC_MIX)",
            "interpretation": "G_BAC > 0 means more context-dependent best-action in STATIONARY",
            "role": "SECONDARY",
        },
        "G_INTERACTION": {
            "formula": "G_INTERACTION = IM_STAT - 0.5*(IM_PER + IM_MIX)",
            "normalization": "|I| / (median absolute nonzero V within same surface + epsilon)",
            "role": "SECONDARY_ONLY",
            "note": "interaction alone not sufficient for monotonicity violation",
        },
    }, RESULTS_DIR / "E4P_primary_scenario_contrast.json")

    # ==================================================================
    # PHASE E4P-31: STATISTICAL DESIGN
    # ==================================================================
    print(f"[{STAGE}] Phase 31: Statistical design...")
    _jdump({
        "stage": STAGE,
        "bootstrap": {
            "type": "paired_crossed_two_axis",
            "axis_1": {"name": "policy", "n": 5},
            "axis_2": {"name": "environment", "n": 20},
            "same_indices_all_scenarios": True,
            "both_source_arms_within_unit": True,
            "source_arm_weight": "equal 1/2 + 1/2",
            "N": N_BOOTSTRAP,
            "CI": "95% percentile",
            "seed": BOOTSTRAP_SEED,
        },
        "per_replicate_compute": ["RR_PER", "RR_STAT", "RR_MIX", "G_REV"],
        "random_tapes_NOT_inferential_axis": True,
        "tape_consistency": {
            "definition": "fraction of tapes agreeing with mean-surface reversal classification",
            "role": "descriptive only, no significance test, no gate",
        },
    }, RESULTS_DIR / "E4P_statistical_design.json")

    # ==================================================================
    # PHASE E4P-28: P1 SUPPORT LOGIC
    # ==================================================================
    print(f"[{STAGE}] Phase 28: P1 support logic...")
    _jdump({
        "stage": STAGE,
        "hypothesis": "P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL",
        "SUPPORTED_requires_ALL": [
            "C1: G_REV > 0",
            "C2: crossed bootstrap 95% CI lower(G_REV) > 0",
            "C3: >= 4/5 policy-level G_REV values > 0",
            "C4: G_BAC > 0",
            "C5: >= 4/5 policy-level G_BAC values > 0",
            "C6: SOURCE_C1 G_REV > 0 AND SOURCE_C2 G_REV > 0",
        ],
        "PLAUSIBLE": (
            "directional evidence exists but one or more support criteria fail"
        ),
        "NOT_SUPPORTED": (
            "no systematic positive scenario gap or opposite direction"
        ),
        "NOT_AUDITABLE": "insufficient valid surfaces",
        "n_criteria": 6,
    }, RESULTS_DIR / "E4P_p1_support_logic.json")

    # ==================================================================
    # PHASE E4P-29: GLOBAL INCOMPATIBILITY
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "global_reversal_rate": (
            "Computed across all three scenarios (not scenario-specific)."
        ),
        "MONOTONIC_INCOMPATIBILITY_OBSERVED": (
            "YES iff at least one robust non-tie rank reversal exists."
        ),
        "separation": (
            "This does NOT establish that monotonicity caused E4MF failure. "
            "Keep separate from P1."
        ),
    }, RESULTS_DIR / "E4P_global_incompatibility_contract.json")

    # ==================================================================
    # PHASE E4P-35/36: PREDICT FALLBACK & FE AUDIT
    # ==================================================================
    print(f"[{STAGE}] Phase 35-36: Audit contracts...")
    _jdump({
        "stage": STAGE,
        "record": "requested_action and realized_action for every surface cell",
        "primary_requirement": (
            "Primary PREDICT cells: requested=PREDICT, realized=PREDICT for "
            "selected regions i,j."
        ),
        "fallback_handling": (
            "If fallback occurs unexpectedly, mark surface INVALID for primary P1. "
            "Retain for engineering audit. Do NOT silently treat PREDICT fallback "
            "as PREDICT."
        ),
    }, RESULTS_DIR / "E4P_predict_fallback_audit_contract.json")

    _jdump({
        "stage": STAGE,
        "primary_requirement": (
            "Equal response FE across all 9 primary cells for a fixed "
            "background/random tape."
        ),
        "failure": "Primary surface invalid if FE differs unexpectedly.",
        "purpose": (
            "Prevent rank reversal from being confounded by unequal response budget."
        ),
    }, RESULTS_DIR / "E4P_fe_audit_contract.json")

    # ==================================================================
    # NO PERFORMANCE VERDICT
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "reason": (
            "Counterfactual utilities are mechanism diagnostic values. "
            "Do NOT compute C1 vs C2 superiority. E4MF = FAILED is authoritative."
        ),
        "E4MF_REMAINS_AUTHORITATIVE": True,
    }, RESULTS_DIR / "E4P_no_performance_verdict.json")

    # ==================================================================
    # E4PI VALIDATION CONTRACT
    # ==================================================================
    print(f"[{STAGE}] E4PI validation contract...")
    _jdump({
        "stage": STAGE,
        "next_validation_stage": "E4PI_NONMONOTONIC_INTERACTION_DIAGNOSTIC_IMPLEMENTATION_VALIDATION",
        "must_use_synthetic_only": True,
        "must_validate": [
            "snapshot_cloning",
            "no_cross_cell_contamination",
            "deterministic_pair_selection",
            "SHA256_pair_index",
            "common_random_tapes",
            "exact_production_operator_semantics",
            "primary_3x3_surface",
            "secondary_4x4_surface",
            "equal_FE_primary_cells",
            "true_utility_calculation",
            "Delta_action_preference",
            "rank_reversal_detection",
            "ties_do_not_count",
            "best_action_context_dependence",
            "cross_difference_interaction",
            "source_arm_aggregation",
            "G_REV",
            "G_BAC",
            "crossed_5x20_bootstrap",
            "P1_truth_table",
            "no_performance_verdict",
            "seed_firewall",
        ],
        "seeds_2401_2420_forbidden_during_E4PI": True,
    }, RESULTS_DIR / "E4P_e4pi_validation_contract.json")

    # ==================================================================
    # FUTURE ROUTING
    # ==================================================================
    print(f"[{STAGE}] Future routing...")
    _jdump({
        "stage": STAGE,
        "routing_rules": {
            "P1_SUPPORTED": {
                "condition": "scenario-specific rank reversal is SUPPORTED",
                "NEXT_STAGE": "E4Q_DECENTRALIZED_IGM_LIMITATION_REVISION_DESIGN",
                "reason": (
                    "Issue deeper than specific QMIX hypernetwork. Fixed "
                    "context-independent decentralized argmax ordering may "
                    "itself be insufficient at sampled states."
                ),
            },
            "GLOBAL_REVERSALS_BUT_P1_NOT_SUPPORTED": {
                "condition": (
                    "MONOTONIC_INCOMPATIBILITY_OBSERVED=YES but STATIONARY "
                    "does not have more reversal than dynamic scenarios"
                ),
                "NEXT_STAGE": "E4Q_GENERAL_INTERACTION_REPRESENTATION_REVIEW",
            },
            "P1_NOT_SUPPORTED_REVERSALS_RARE": {
                "condition": "P1 NOT_SUPPORTED and reversals rare/absent",
                "NEXT_STAGE": "E4Q_ACTION_STATE_REPRESENTATION_REVIEW",
                "focus": "N3 action specialization, N4 Q-margin, state representation",
            },
            "NOT_AUDITABLE": {
                "NEXT_STAGE": "E4P_DIAGNOSTIC_INTEGRITY_REVIEW",
            },
        },
    }, RESULTS_DIR / "E4P_future_routing.json")

    # ==================================================================
    # FUTURE FIREWALL
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "P_DIAGNOSTIC_ACCESS_COUNT": 0,
        "seeds_2401_2420_NOT_INSTANTIATED": True,
    }, RESULTS_DIR / "E4P_future_firewall.json")

    # ==================================================================
    # PROTOCOL FROZEN
    # ==================================================================
    _jdump({
        "stage": STAGE,
        "protocol_frozen": True,
        "frozen_contracts": [
            "E4P_hypothesis_contract.json",
            "E4P_scientific_question.json",
            "E4P_environment_utility_contract.json",
            "E4P_counterfactual_surface_contract.json",
            "E4P_primary_action_set.json",
            "E4P_secondary_action_set.json",
            "E4P_background_context_contract.json",
            "E4P_source_trajectory_contract.json",
            "E4P_checkpoint_lock.json",
            "E4P_diagnostic_seed_contract.json",
            "E4P_seed_disjointness.json",
            "E4P_snapshot_eligibility_contract.json",
            "E4P_pair_selection_contract.json",
            "E4P_random_tape_contract.json",
            "E4P_counterfactual_isolation_contract.json",
            "E4P_action_preference_contract.json",
            "E4P_rank_reversal_contract.json",
            "E4P_best_action_context_contract.json",
            "E4P_cross_difference_contract.json",
            "E4P_source_aggregation_contract.json",
            "E4P_primary_scenario_contrast.json",
            "E4P_statistical_design.json",
            "E4P_p1_support_logic.json",
            "E4P_global_incompatibility_contract.json",
            "E4P_predict_fallback_audit_contract.json",
            "E4P_fe_audit_contract.json",
            "E4P_no_performance_verdict.json",
            "E4P_e4pi_validation_contract.json",
            "E4P_future_routing.json",
            "E4P_future_firewall.json",
        ],
        "operator_constants": OPERATOR_CONSTANTS,
        "survival": SURVIVAL,
        "epsilon_rank": EPSILON_RANK,
    }, RESULTS_DIR / "E4P_protocol_frozen.json")

    # ==================================================================
    # VERDICT
    # ==================================================================
    print(f"[{STAGE}] Writing verdict...")
    _jdump({
        "stage": STAGE,
        "E4P_COMPLETE": "YES",
        "ORIGINAL_E4MF_VERDICT": "FAILED",
        "E4MF_VERDICT_CHANGED": "NO",
        "E4OD_O1_GRADIENT_CONFLICT": "NOT_SUPPORTED",
        "N8_JOINT_REGIONAL_SUPERVISION_TRADEOFF": "NOT_SUPPORTED",
        "N9_MONOTONIC_QMIX_LIMITATION": "PLAUSIBLE",
        "PRIMARY_DIAGNOSTIC_HYPOTHESIS": "P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL",
        "PRIMARY_SCIENTIFIC_METRIC": "NONKEEP_CONTEXT_DEPENDENT_REVERSAL_RATE",
        "PRIMARY_ACTION_SET": "LOCAL_PREDICT_DIVERSIFY",
        "PRIMARY_SURFACE": "3x3_EQUAL_RESPONSE_BUDGET",
        "SECONDARY_SURFACE": "4x4_KEEP_INCLUSIVE",
        "PRIMARY_BACKGROUND": "KEEP_BACKGROUND",
        "ROBUSTNESS_BACKGROUND": "SOURCE_POLICY_BACKGROUND",
        "SOURCE_TRAJECTORIES": "C1_AND_C2",
        "FROZEN_CHECKPOINTS": "C1_C2_POLICIES_6001_TO_6005",
        "DIAGNOSTIC_SEEDS": "2401..2420",
        "FUTURE_SOURCE_EPISODES": n_source_episodes,
        "COUNTERFACTUAL_RANDOM_TAPES": COUNTERFACTUAL_RANDOM_TAPES,
        "PRIMARY_SCENARIO_CONTRAST": "G_REV_STAT_MINUS_MEAN_PER_MIX",
        "SECONDARY_SCENARIO_CONTRAST": "G_BAC_STAT_MINUS_MEAN_PER_MIX",
        "INFERENTIAL_AXES": "POLICY_5_X_ENVIRONMENT_20",
        "BOOTSTRAP_RESAMPLES": N_BOOTSTRAP,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "P_DIAGNOSTIC_ACCESS_COUNT": 0,
        "NEW_TRAINING": "NO",
        "NEW_ENVIRONMENT_EXECUTION": "NO",
        "NEW_CONTROLLER_REVISION": "NO",
        "NEXT_STAGE": "E4PI_NONMONOTONIC_INTERACTION_DIAGNOSTIC_IMPLEMENTATION_VALIDATION",
    }, RESULTS_DIR / "E4P_verdict.json")

    # ==================================================================
    # REPORT
    # ==================================================================
    print(f"[{STAGE}] Writing report...")
    rpt = [
        "# E4P — Nonmonotonic Interaction Diagnostic Design",
        "",
        "## A. Immutable Prior Evidence",
        "",
        "- E4KF: QMIX C1 FAILED relative to C0",
        "- E4MF: C2 credit-preservation FAILED relative to C1",
        "- E4N: PRIMARY = MULTIFACTORIAL_SCENARIO_TRADEOFF",
        "- E4OD: O1 gradient-conflict = NOT_SUPPORTED",
        "- E4ODR: reconciled labels/wording, no verdict change",
        "",
        "## B. Why Gradient Conflict Was Rejected",
        "",
        "E4OD showed STATIONARY has the LEAST gradient conflict (cosine +0.44,",
        "aligned), while PERSISTENT has the MOST (cosine -0.73). The observed",
        "pattern is opposite to what O1 required. Gradient conflict between",
        "L_joint and L_regional does NOT explain STATIONARY harm.",
        "",
        "## C. Remaining N9/H6 Question",
        "",
        "N9 (MONOTONIC_QMIX_LIMITATION_REMAINS) is PLAUSIBLE. The question is",
        "whether the true environment response surface itself exhibits context-",
        "dependent regional action-rank reversals incompatible with monotonic",
        "factorization, and whether this is stronger in STATIONARY.",
        "",
        "## D. QMIX Monotonic Representability Constraint",
        "",
        "QMIX imposes dQ_tot/dq_r >= 0. If q_i(a) > q_i(b) for fixed state,",
        "a monotonic mixer cannot reverse that ordering regardless of other",
        "regions' actions. A true environment preference reversal where action a",
        "is preferred under one sibling context but action b under another is",
        "direct evidence that context-independent local ordering is insufficient.",
        "",
        "## E. Why Generic Interaction Is Insufficient Evidence",
        "",
        "Nonzero interaction (I != 0) does NOT demonstrate a QMIX monotonicity",
        "violation. QMIX can represent many nonlinear interactions while remaining",
        "monotonic. The primary diagnostic tests CONTEXT-DEPENDENT ACTION-RANK",
        "REVERSAL, not merely interaction != 0.",
        "",
        "## F. Environment Response Utility",
        "",
        "V_S(A) = IGD(KEEP_ALL) - IGD(A). Uses existing igd(F, pf_star) from",
        "src/prdrl/metrics/igd.py. Higher V = better response quality. KEEP_ALL",
        "baseline reused for all cells of the same surface. Learned Q values are",
        "NOT used as ground truth.",
        "",
        "## G. Fixed-State Counterfactual Surface",
        "",
        "For a selected active region pair (i,j), hold all other regions under a",
        "fixed background. Vary actions for i and j. Every cell forks from the",
        "EXACT SAME pre-response snapshot. No cross-cell contamination.",
        "",
        "## H. Equal-Budget Primary Action Set",
        "",
        "Primary: {LOCAL, PREDICT, DIVERSIFY} = 3x3 = 9 cells.",
        "KEEP excluded because KEEP has FE=0 while the others use B=10.",
        "This avoids confounding rank reversal with unequal response budget.",
        "",
        "## I. Secondary KEEP-Inclusive Surface",
        "",
        "Secondary: {KEEP, LOCAL, PREDICT, DIVERSIFY} = 4x4 = 16 cells.",
        "Label: SECONDARY_FE_CONFOUNDED_CONTEXT. Cannot override primary verdict.",
        "",
        "## J. Background Contexts",
        "",
        "- KEEP_BACKGROUND (primary): all other active regions request KEEP",
        "- SOURCE_POLICY_BACKGROUND (robustness): other regions use source policy actions",
        "- Primary P1 verdict based on KEEP_BACKGROUND only",
        "",
        "## K. Source Trajectory Design",
        "",
        "Both C1 and C2 controllers generate source trajectories. Equal 1/2+1/2",
        "weight within matched policy x env x scenario unit. 2x5x20x3 = 600 episodes.",
        "",
        "## L. Fresh Diagnostic Seeds",
        "",
        "Seeds 2401..2420 (n=20). Verified disjoint from all historical seeds.",
        "Bootstrap seed 126001 also disjoint.",
        "",
        "## M. Snapshot Eligibility",
        "",
        "First dynamic-change event with >= 2 active regions, available states,",
        "and PREDICT history for candidate regions. Deterministic, outcome-blind.",
        "",
        "## N. Region-Pair Selection",
        "",
        "SHA256-based deterministic index from (source_arm, policy_seed, scenario,",
        "env_seed). Python hash() forbidden. Outcome-blind.",
        "",
        "## O. Common Random Numbers",
        "",
        f"4 deterministic random tapes per surface. Same tape reused across all",
        "cells. Primary cell utility = mean across 4 tapes.",
        "",
        "## P. Action Preference Difference",
        "",
        "Delta_i(a,b | j=c) = V_S(i=a,j=c) - V_S(i=b,j=c).",
        "Positive: a preferred. Negative: b preferred.",
        "",
        "## Q. Rank-Reversal Definition",
        "",
        "Reversal iff exists c,d such that Delta_i(a,b|j=c) > +1e-12 AND",
        "Delta_i(a,b|j=d) < -1e-12. Ties (|Delta| <= 1e-12) are not reversals.",
        "",
        "## R. Context-Dependent Best Action",
        "",
        "a_i*(c) = argmax_a V_S(i=a,j=c). Context-dependent iff a_i*(c) varies",
        "across j's contexts (excluding ties).",
        "",
        "## S. Difference-in-Differences Interaction",
        "",
        "I_ij(a,b;c,d) = V(a,c) - V(b,c) - V(a,d) + V(b,d). SECONDARY ONLY.",
        "Interaction alone is NOT sufficient for monotonicity violation.",
        "",
        "## T. Primary Scenario Contrast G_REV",
        "",
        "G_REV = RR_STAT - 0.5*(RR_PER + RR_MIX). Positive = STATIONARY has more",
        "context-dependent rank reversals.",
        "",
        "## U. Statistical Design",
        "",
        "Paired crossed two-axis bootstrap: 5 policies x 20 environments.",
        "N=50000, seed=126001, 95% percentile CI. Same indices all scenarios.",
        "Both source arms within unit. Tapes not an inferential axis.",
        "",
        "## V. P1 Support Logic",
        "",
        "P1 SUPPORTED requires ALL 6 criteria:",
        "1. G_REV > 0",
        "2. Bootstrap CI lower(G_REV) > 0",
        "3. >= 4/5 policy G_REV > 0",
        "4. G_BAC > 0",
        "5. >= 4/5 policy G_BAC > 0",
        "6. Both SOURCE_C1 G_REV > 0 AND SOURCE_C2 G_REV > 0",
        "",
        "## W. Global Incompatibility Boundary",
        "",
        "MONOTONIC_INCOMPATIBILITY_OBSERVED = YES iff any robust non-tie reversal",
        "exists (across all scenarios). Separate from scenario-specific P1.",
        "",
        "## X. FE/PREDICT-Fallback Audits",
        "",
        "- Equal FE required across all 9 primary cells",
        "- PREDICT fallback invalidates primary surface",
        "- Requested vs realized action recorded for all cells",
        "",
        "## Y. Interpretation Limits",
        "",
        "- E4P is design only — no new scientific data generated",
        "- E4MF remains FAILED",
        "- E4OD gradient conflict remains NOT_SUPPORTED",
        "- Pairwise surfaces cannot rule out higher-order 3+ region interactions",
        "- One selected pair per episode samples rather than exhausts all pairs",
        "- 4 random tapes approximate stochastic response expectation",
        "- KEEP-background evidence may differ from realistic policy-background",
        "- Rank reversal at sampled states ≠ universal impossibility",
        "- Nonzero interaction alone is not enough to reject QMIX monotonicity",
        "- No real-world UAV validation",
        "",
        "## Z. Future E4PI / E4PD Stages",
        "",
        "- E4PI: implementation validation (synthetic only, no seeds 2401-2420)",
        "- E4PD: execution (opens seeds 2401-2420, only after E4PI passes)",
        "",
        "## AA. Next-Stage Routing",
        "",
        "- P1 SUPPORTED → E4Q_DECENTRALIZED_IGM_LIMITATION_REVISION_DESIGN",
        "- Global reversals but P1 NOT_SUPPORTED → E4Q_GENERAL_INTERACTION_REPRESENTATION_REVIEW",
        "- P1 NOT_SUPPORTED + reversals rare → E4Q_ACTION_STATE_REPRESENTATION_REVIEW",
        "- NOT_AUDITABLE → E4P_DIAGNOSTIC_INTEGRITY_REVIEW",
        "",
    ]
    with open(RESULTS_DIR / "E4P_report.md", "w") as f:
        f.write("\n".join(rpt))

    # ==================================================================
    # TERMINAL SUMMARY
    # ==================================================================
    print()
    print("=" * 60)
    print(f"E4P_COMPLETE = YES")
    print(f"ORIGINAL_E4MF_VERDICT = FAILED")
    print(f"E4MF_VERDICT_CHANGED = NO")
    print(f"E4OD_O1_GRADIENT_CONFLICT = NOT_SUPPORTED")
    print(f"N8_JOINT_REGIONAL_SUPERVISION_TRADEOFF = NOT_SUPPORTED")
    print(f"N9_MONOTONIC_QMIX_LIMITATION = PLAUSIBLE")
    print(f"PRIMARY_DIAGNOSTIC_HYPOTHESIS = P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL")
    print(f"PRIMARY_SCIENTIFIC_METRIC = NONKEEP_CONTEXT_DEPENDENT_REVERSAL_RATE")
    print(f"PRIMARY_ACTION_SET = LOCAL_PREDICT_DIVERSIFY")
    print(f"PRIMARY_SURFACE = 3x3_EQUAL_RESPONSE_BUDGET")
    print(f"SECONDARY_SURFACE = 4x4_KEEP_INCLUSIVE")
    print(f"PRIMARY_BACKGROUND = KEEP_BACKGROUND")
    print(f"ROBUSTNESS_BACKGROUND = SOURCE_POLICY_BACKGROUND")
    print(f"SOURCE_TRAJECTORIES = C1_AND_C2")
    print(f"FROZEN_CHECKPOINTS = C1_C2_POLICIES_6001_TO_6005")
    print(f"DIAGNOSTIC_SEEDS = 2401..2420")
    print(f"FUTURE_SOURCE_EPISODES = {n_source_episodes}")
    print(f"COUNTERFACTUAL_RANDOM_TAPES = {COUNTERFACTUAL_RANDOM_TAPES}")
    print(f"PRIMARY_SCENARIO_CONTRAST = G_REV_STAT_MINUS_MEAN_PER_MIX")
    print(f"SECONDARY_SCENARIO_CONTRAST = G_BAC_STAT_MINUS_MEAN_PER_MIX")
    print(f"INFERENTIAL_AXES = POLICY_5_X_ENVIRONMENT_20")
    print(f"BOOTSTRAP_RESAMPLES = {N_BOOTSTRAP}")
    print(f"BOOTSTRAP_SEED = {BOOTSTRAP_SEED}")
    print(f"PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED")
    print(f"P_DIAGNOSTIC_ACCESS_COUNT = 0")
    print(f"NEW_TRAINING = NO")
    print(f"NEW_ENVIRONMENT_EXECUTION = NO")
    print(f"NEW_CONTROLLER_REVISION = NO")
    print(f"NEXT_STAGE = E4PI_NONMONOTONIC_INTERACTION_DIAGNOSTIC_IMPLEMENTATION_VALIDATION")
    print("=" * 60)


if __name__ == "__main__":
    run()
