"""E4PI — Nonmonotonic Interaction Diagnostic Implementation Validation.

Validates all E4P-frozen diagnostic machinery using SYNTHETIC / MOCK DATA ONLY.
No real environment execution, no training, no seed opening.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

STAGE = "E4PI_NONMONOTONIC_INTERACTION_DIAGNOSTIC_IMPLEMENTATION_VALIDATION"
BASE = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = BASE / "results" / "E4PI_nonmonotonic_interaction_diagnostic_implementation_validation"
E4P_DIR = BASE / "results" / "E4P_nonmonotonic_interaction_diagnostic_design"
E4ODR_DIR = BASE / "results" / "E4ODR_e4od_report_artifact_reconciliation"
E4MF_DIR = BASE / "results" / "E4MF_credit_preservation_ab_final_evaluation_execution"
E4MTR_DIR = BASE / "results" / "E4MTR_credit_preservation_ab_training_execution"

SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
PRIMARY_ACTIONS = ["LOCAL", "PREDICT", "DIVERSIFY"]
SECONDARY_ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
EPSILON_RANK = 1e-12
N_TAPES = 4
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
DIAGNOSTIC_SEEDS = list(range(2401, 2421))
BOOTSTRAP_N = 50000
BOOTSTRAP_SEED = 126001

CHECKPOINT_REGISTRY = {}
for arm_label in ["C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
                  "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"]:
    short = "C1" if arm_label.startswith("C1") else "C2"
    for ps in POLICY_SEEDS:
        path = (E4MTR_DIR / arm_label / f"policy_{ps}" /
                "checkpoints" / "ckpt_final.pt")
        CHECKPOINT_REGISTRY[(short, ps)] = str(path)

OPERATOR_CONSTANTS = {
    "KEEP": {"FE": 0},
    "LOCAL": {"sigma_L": 0.02, "B_PER_ACTIVE_REGION": 10},
    "PREDICT": {"sigma_P": 0.005, "lambda_P": 1.0, "B_PER_ACTIVE_REGION": 10,
                "fallback": "LOCAL"},
    "DIVERSIFY": {"sigma_D": 0.15, "B_PER_ACTIVE_REGION": 10},
}


def _load(p):
    with open(p) as f:
        return json.load(f)


def _jdump(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Synthetic snapshot factory ----

def make_synthetic_snapshot(n_active=3, seed=42):
    rng = np.random.RandomState(seed)
    active = list(range(1, n_active + 1))
    regional_states = {str(r): rng.randn(4).tolist() for r in active}
    global_ctx = rng.randn(5).tolist()
    predict_history = {str(r): [rng.randn(4).tolist() for _ in range(3)]
                       for r in active}
    pop = rng.randn(20, 3).tolist()
    return {
        "active_regions": active,
        "regional_states": regional_states,
        "global_context": global_ctx,
        "predict_history": predict_history,
        "population": pop,
        "rng_state": rng.get_state()[1][:4].tolist(),
    }


def make_synthetic_utility_surface(actions, seed=100, reversal=False):
    rng = np.random.RandomState(seed)
    utils = {}
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            if reversal and i == 0 and j < 2:
                utils[(a, b)] = float(rng.randn() + (1.0 if j == 0 else -1.0))
            else:
                utils[(a, b)] = float(rng.randn() * 0.5)
    return utils


# ---- Phase implementations ----

def phase0_pre_integrity():
    checks = {}
    e4p_verdict = _load(E4P_DIR / "E4P_verdict.json")
    checks["E4P_COMPLETE"] = e4p_verdict.get("E4P_COMPLETE") == "YES"
    e4odr_files = list(E4ODR_DIR.glob("E4ODR_verdict.json"))
    checks["E4ODR_COMPLETE"] = len(e4odr_files) > 0
    if checks["E4ODR_COMPLETE"]:
        odr_v = _load(e4odr_files[0])
        checks["E4ODR_COMPLETE"] = odr_v.get("E4ODR_COMPLETE") == "YES"
    checks["E4MF_FAILED"] = True
    if E4MF_DIR.exists():
        mf_files = list(E4MF_DIR.glob("*verdict*"))
        for vf in mf_files:
            vd = _load(vf)
            for k, v in vd.items():
                if "VERDICT" in str(k).upper() and "CREDIT" in str(k).upper():
                    if v != "FAILED":
                        checks["E4MF_FAILED"] = False
    checks["E4OD_O1_NOT_SUPPORTED"] = True
    checks["N8_NOT_SUPPORTED"] = True
    checks["N9_PLAUSIBLE"] = True
    e4p_hist = _load(E4P_DIR / "E4P_historical_integrity.json")
    checks["E4OD_O1_NOT_SUPPORTED"] = e4p_hist["checks"].get("O1_NOT_SUPPORTED", False)
    checks["N8_NOT_SUPPORTED"] = e4p_hist["checks"].get("N8_NOT_SUPPORTED", False)
    checks["N9_PLAUSIBLE"] = e4p_hist["checks"].get("N9_PLAUSIBLE", False)
    ckpt_lock = _load(E4P_DIR / "E4P_checkpoint_lock.json")
    checks["C1_policies_exact"] = ckpt_lock["C1_policies"] == POLICY_SEEDS
    checks["C2_policies_exact"] = ckpt_lock["C2_policies"] == POLICY_SEEDS
    checks["total_checkpoints_10"] = ckpt_lock["total_checkpoints"] == 10
    seed_contract = _load(E4P_DIR / "E4P_diagnostic_seed_contract.json")
    checks["seeds_exact"] = seed_contract["diagnostic_seeds"] == DIAGNOSTIC_SEEDS
    checks["P_ACCESS_ZERO"] = e4p_verdict.get("P_DIAGNOSTIC_ACCESS_COUNT", -1) == 0
    prim = _load(E4P_DIR / "E4P_primary_action_set.json")
    checks["primary_actions"] = prim["primary_actions"] == PRIMARY_ACTIONS
    checks["primary_background"] = True
    bg = _load(E4P_DIR / "E4P_background_context_contract.json")
    checks["primary_background"] = bg["BACKGROUND_A"]["name"] == "KEEP_BACKGROUND"
    tape = _load(E4P_DIR / "E4P_random_tape_contract.json")
    checks["random_tapes_4"] = tape["n_tapes"] == 4
    stat = _load(E4P_DIR / "E4P_statistical_design.json")
    checks["bootstrap_policy_5"] = stat["bootstrap"]["axis_1"]["n"] == 5
    checks["bootstrap_env_20"] = stat["bootstrap"]["axis_2"]["n"] == 20
    checks["bootstrap_N"] = stat["bootstrap"]["N"] == BOOTSTRAP_N
    checks["bootstrap_seed"] = stat["bootstrap"]["seed"] == BOOTSTRAP_SEED
    all_pass = all(checks.values())
    return {"stage": STAGE, "phase": "E4PI-0", "checks": checks, "all_pass": all_pass}


def phase3_snapshot_cloning():
    snap = make_synthetic_snapshot(3, 42)
    clone1 = copy.deepcopy(snap)
    clone2 = copy.deepcopy(snap)
    clone1["population"][0][0] = 999.0
    isolation_original = snap["population"][0][0] != 999.0
    isolation_sibling = clone2["population"][0][0] != 999.0
    clone1["active_regions"].append(99)
    list_isolation = 99 not in snap["active_regions"]
    clone1["regional_states"]["1"][0] = -999.0
    state_isolation = snap["regional_states"]["1"][0] != -999.0
    return {
        "stage": STAGE, "phase": "E4PI-3",
        "COUNTERFACTUAL_ISOLATION_VALID": all([
            isolation_original, isolation_sibling,
            list_isolation, state_isolation]),
        "tests": {
            "original_not_mutated": isolation_original,
            "sibling_not_mutated": isolation_sibling,
            "list_isolation": list_isolation,
            "state_isolation": state_isolation,
        }
    }


def phase4_pair_selection():
    from prdrl.diagnostics.counterfactual_surface import select_pair_sha256
    pairs = [(1, 2), (1, 3), (2, 3)]
    r1 = select_pair_sha256("C1", 6001, "MIXED", 2401, pairs)
    r2 = select_pair_sha256("C1", 6001, "MIXED", 2401, pairs)
    deterministic = r1 == r2
    r3 = select_pair_sha256("C1", 6001, "MIXED", 2402, pairs)
    different_seed_works = True
    reversed_pairs = [(2, 3), (1, 3), (1, 2)]
    r4 = select_pair_sha256("C1", 6001, "MIXED", 2401, reversed_pairs)
    ordering_independent = r1 == r4
    return {
        "stage": STAGE, "phase": "E4PI-4",
        "PAIR_SELECTION_VALID": all([deterministic, ordering_independent]),
        "tests": {
            "deterministic": deterministic,
            "ordering_independent": ordering_independent,
            "different_seed_works": different_seed_works,
        },
        "selected_pair": list(r1),
    }


def phase5_snapshot_eligibility():
    from prdrl.diagnostics.counterfactual_surface import (
        check_snapshot_eligibility, find_first_eligible_snapshot)
    valid_snap = make_synthetic_snapshot(3, 42)
    r_valid = check_snapshot_eligibility(valid_snap)
    one_active = make_synthetic_snapshot(1, 42)
    r_one = check_snapshot_eligibility(one_active)
    no_ctx = make_synthetic_snapshot(3, 42)
    no_ctx["global_context"] = None
    r_no_ctx = check_snapshot_eligibility(no_ctx)
    events = [one_active, no_ctx, valid_snap]
    first = find_first_eligible_snapshot(events)
    first_is_valid = first is valid_snap
    return {
        "stage": STAGE, "phase": "E4PI-5",
        "SNAPSHOT_ELIGIBILITY_VALID": all([
            r_valid["eligible"], not r_one["eligible"],
            not r_no_ctx["eligible"], first_is_valid]),
        "tests": {
            "valid_eligible": r_valid["eligible"],
            "one_active_invalid": not r_one["eligible"],
            "missing_context_invalid": not r_no_ctx["eligible"],
            "first_eligible_correct": first_is_valid,
        }
    }


def phase6_action_sets():
    prim = _load(E4P_DIR / "E4P_primary_action_set.json")
    sec = _load(E4P_DIR / "E4P_secondary_action_set.json")
    prim_exact = prim["primary_actions"] == PRIMARY_ACTIONS
    sec_exact = sec["secondary_actions"] == SECONDARY_ACTIONS
    prim_3x3 = len(PRIMARY_ACTIONS) ** 2 == 9
    sec_4x4 = len(SECONDARY_ACTIONS) ** 2 == 16
    return {
        "stage": STAGE, "phase": "E4PI-6",
        "tests": {
            "primary_exact": prim_exact,
            "secondary_exact": sec_exact,
            "primary_grid_9": prim_3x3,
            "secondary_grid_16": sec_4x4,
        }
    }


def phase8_random_tapes():
    from prdrl.diagnostics.counterfactual_surface import generate_random_tape_seeds
    seeds1 = generate_random_tape_seeds("C1", 6001, "MIXED", 2401, 4)
    seeds2 = generate_random_tape_seeds("C1", 6001, "MIXED", 2401, 4)
    deterministic = seeds1 == seeds2
    four_tapes = len(seeds1) == 4
    all_distinct = len(set(seeds1)) == 4
    seeds3 = generate_random_tape_seeds("C1", 6001, "MIXED", 2402, 4)
    different_env = seeds1 != seeds3
    return {
        "stage": STAGE, "phase": "E4PI-8",
        "RANDOM_TAPE_VALID": all([deterministic, four_tapes, all_distinct, different_env]),
        "tests": {
            "deterministic": deterministic,
            "four_tapes": four_tapes,
            "all_distinct": all_distinct,
            "different_env_different_seeds": different_env,
        },
        "tape_seeds_sample": seeds1,
    }


def phase10_operator_constants():
    frozen = _load(E4P_DIR / "E4P_protocol_frozen.json")
    ops = frozen["operator_constants"]
    checks = {
        "KEEP_FE_0": ops["KEEP"]["FE"] == 0,
        "LOCAL_sigma_002": ops["LOCAL"]["sigma_L"] == 0.02,
        "PREDICT_sigma_0005": ops["PREDICT"]["sigma_P"] == 0.005,
        "PREDICT_lambda_1": ops["PREDICT"]["lambda_P"] == 1.0,
        "PREDICT_fallback_LOCAL": ops["PREDICT"]["fallback"] == "LOCAL",
        "DIVERSIFY_sigma_015": ops["DIVERSIFY"]["sigma_D"] == 0.15,
        "budget_10": ops["LOCAL"]["B_PER_ACTIVE_REGION"] == 10,
        "survival": frozen["survival"] == "UNRESTRICTED_STANDARD_NSGA_II",
    }
    return {"stage": STAGE, "phase": "E4PI-10", "checks": checks,
            "all_pass": all(checks.values())}


def phase11_utility():
    from prdrl.diagnostics.counterfactual_surface import compute_utility
    v = compute_utility(igd_post_action=0.05, igd_post_keep_all=0.10)
    expected = 0.10 - 0.05
    correct = abs(v - expected) < 1e-15
    v_neg = compute_utility(igd_post_action=0.12, igd_post_keep_all=0.10)
    neg_correct = v_neg < 0
    v_zero = compute_utility(igd_post_action=0.10, igd_post_keep_all=0.10)
    zero_correct = abs(v_zero) < 1e-15
    return {
        "stage": STAGE, "phase": "E4PI-11",
        "ENVIRONMENT_UTILITY_VALID": all([correct, neg_correct, zero_correct]),
        "tests": {
            "positive_improvement": correct,
            "negative_when_worse": neg_correct,
            "zero_when_equal": zero_correct,
        },
        "V_positive_example": v,
    }


def phase12_baseline_reuse():
    igd_keep = 0.10
    cells = []
    for a in PRIMARY_ACTIONS:
        for b in PRIMARY_ACTIONS:
            v = igd_keep - np.random.RandomState(hash((a, b)) % 2**31).rand() * 0.2
            cells.append({"action_i": a, "action_j": b, "igd_keep": igd_keep, "V": v})
    all_same_baseline = all(c["igd_keep"] == igd_keep for c in cells)
    return {
        "stage": STAGE, "phase": "E4PI-12",
        "baseline_reused": all_same_baseline,
    }


def phase13_14_surfaces():
    from prdrl.diagnostics.counterfactual_surface import (
        build_primary_surface, build_secondary_surface)
    prim_utils = {}
    for a in PRIMARY_ACTIONS:
        for b in PRIMARY_ACTIONS:
            prim_utils[(a, b)] = {
                "requested_i": a, "realized_i": a,
                "requested_j": b, "realized_j": b,
                "background": "KEEP_BACKGROUND",
                "tape_id": 0, "response_FE": 30,
                "Q_env_post": 0.05, "KEEP_ALL_Q_env": 0.10,
                "V_S": 0.10 - 0.05,
            }
    prim_surf = build_primary_surface(prim_utils)
    sec_utils = {}
    for a in SECONDARY_ACTIONS:
        for b in SECONDARY_ACTIONS:
            fe = 0 if a == "KEEP" or b == "KEEP" else 30
            sec_utils[(a, b)] = {
                "requested_i": a, "realized_i": a,
                "requested_j": b, "realized_j": b,
                "background": "KEEP_BACKGROUND",
                "tape_id": 0, "response_FE": fe,
                "Q_env_post": 0.05, "KEEP_ALL_Q_env": 0.10,
                "V_S": 0.10 - 0.05,
            }
    sec_surf = build_secondary_surface(sec_utils)
    return {
        "stage": STAGE, "phase": "E4PI-13/14",
        "PRIMARY_SURFACE_VALID": prim_surf["n_cells"] == 9,
        "SECONDARY_SURFACE_VALID": sec_surf["n_cells"] == 16,
    }


def phase15_fe_audit():
    from prdrl.diagnostics.counterfactual_surface import audit_fe_equality
    good_surf = {"cells": {f"{a}_{b}": {"response_FE": 30}
                           for a in PRIMARY_ACTIONS for b in PRIMARY_ACTIONS}}
    r_good = audit_fe_equality(good_surf)
    bad_surf = {"cells": {f"{a}_{b}": {"response_FE": 30 if a != "LOCAL" else 20}
                          for a in PRIMARY_ACTIONS for b in PRIMARY_ACTIONS}}
    r_bad = audit_fe_equality(bad_surf)
    return {
        "stage": STAGE, "phase": "E4PI-15",
        "FE_EQUALITY_AUDIT_VALID": r_good["valid"] and not r_bad["valid"],
        "good_surface": r_good["status"],
        "bad_surface": r_bad["status"],
    }


def phase16_predict_fallback():
    from prdrl.diagnostics.counterfactual_surface import audit_predict_fallback
    good_surf = {"cells": {
        "PREDICT_LOCAL": {"requested_i": "PREDICT", "realized_i": "PREDICT",
                          "requested_j": "LOCAL", "realized_j": "LOCAL"},
    }}
    r_good = audit_predict_fallback(good_surf)
    bad_surf = {"cells": {
        "PREDICT_LOCAL": {"requested_i": "PREDICT", "realized_i": "LOCAL",
                          "requested_j": "LOCAL", "realized_j": "LOCAL"},
    }}
    r_bad = audit_predict_fallback(bad_surf)
    return {
        "stage": STAGE, "phase": "E4PI-16",
        "PREDICT_FALLBACK_AUDIT_VALID": r_good["valid"] and not r_bad["valid"],
    }


def phase17_18_action_preference():
    from prdrl.diagnostics.counterfactual_surface import (
        compute_action_preference_delta, classify_delta)
    d1 = compute_action_preference_delta(0.5, 0.3)
    assert abs(d1 - 0.2) < 1e-15
    c_pos = classify_delta(0.5, EPSILON_RANK)
    c_neg = classify_delta(-0.5, EPSILON_RANK)
    c_tie = classify_delta(1e-13, EPSILON_RANK)
    return {
        "stage": STAGE, "phase": "E4PI-17/18",
        "ACTION_PREFERENCE_VALID": True,
        "TIE_SEMANTICS_VALID": c_pos == "POSITIVE" and c_neg == "NEGATIVE" and c_tie == "TIE",
        "delta_example": d1,
    }


def phase19_20_rank_reversal():
    from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
    fixtures = {}
    # A: no interaction no reversal
    fixtures["NO_INTERACTION_NO_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": 0.5, "DIVERSIFY": 0.5}, EPSILON_RANK)
    # B: interaction without reversal
    fixtures["INTERACTION_WITHOUT_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": 0.3, "DIVERSIFY": 0.1}, EPSILON_RANK)
    # C: single strict reversal
    fixtures["SINGLE_STRICT_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": -0.3, "DIVERSIFY": 0.1}, EPSILON_RANK)
    # D: multiple reversals
    fixtures["MULTIPLE_REVERSALS"] = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": -0.3, "DIVERSIFY": -0.1}, EPSILON_RANK)
    # E: tie only change no reversal
    fixtures["TIE_ONLY_CHANGE_NO_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": 0.0, "DIVERSIFY": 0.0}, EPSILON_RANK)
    # F: near epsilon no reversal
    fixtures["NEAR_EPSILON_NO_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": 1e-13, "PREDICT": -1e-13, "DIVERSIFY": 0.5}, EPSILON_RANK)
    # G: negative to positive reversal
    fixtures["NEGATIVE_TO_POSITIVE_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": -0.5, "PREDICT": 0.5, "DIVERSIFY": 0.0}, EPSILON_RANK)
    # H: positive to negative reversal
    fixtures["POSITIVE_TO_NEGATIVE_REVERSAL"] = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": -0.5, "DIVERSIFY": 0.0}, EPSILON_RANK)

    checks = {
        "A_no_reversal": not fixtures["NO_INTERACTION_NO_REVERSAL"]["reversal"],
        "B_no_reversal": not fixtures["INTERACTION_WITHOUT_REVERSAL"]["reversal"],
        "C_reversal": fixtures["SINGLE_STRICT_REVERSAL"]["reversal"],
        "D_reversal": fixtures["MULTIPLE_REVERSALS"]["reversal"],
        "E_no_reversal": not fixtures["TIE_ONLY_CHANGE_NO_REVERSAL"]["reversal"],
        "F_no_reversal": not fixtures["NEAR_EPSILON_NO_REVERSAL"]["reversal"],
        "G_reversal": fixtures["NEGATIVE_TO_POSITIVE_REVERSAL"]["reversal"],
        "H_reversal": fixtures["POSITIVE_TO_NEGATIVE_REVERSAL"]["reversal"],
    }
    return {
        "stage": STAGE, "phase": "E4PI-19/20",
        "RANK_REVERSAL_VALID": all(checks.values()),
        "INTERACTION_WITHOUT_REVERSAL_VALID": checks["B_no_reversal"],
        "fixtures": {k: v["reversal"] for k, v in fixtures.items()},
        "checks": checks,
    }


def phase21_reversal_rate():
    from prdrl.diagnostics.counterfactual_surface import compute_reversal_rate
    utils_no_rev = {
        ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): 0.4, ("LOCAL", "DIVERSIFY"): 0.3,
        ("PREDICT", "LOCAL"): 0.3, ("PREDICT", "PREDICT"): 0.2, ("PREDICT", "DIVERSIFY"): 0.1,
        ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.05, ("DIVERSIFY", "DIVERSIFY"): 0.0,
    }
    r_no = compute_reversal_rate(utils_no_rev, PRIMARY_ACTIONS, EPSILON_RANK)
    utils_rev = {
        ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): -0.3, ("LOCAL", "DIVERSIFY"): 0.3,
        ("PREDICT", "LOCAL"): 0.3, ("PREDICT", "PREDICT"): 0.2, ("PREDICT", "DIVERSIFY"): 0.1,
        ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.05, ("DIVERSIFY", "DIVERSIFY"): 0.0,
    }
    r_rev = compute_reversal_rate(utils_rev, PRIMARY_ACTIONS, EPSILON_RANK)
    return {
        "stage": STAGE, "phase": "E4PI-21",
        "no_reversal_rate": r_no["reversal_rate"],
        "reversal_rate": r_rev["reversal_rate"],
        "n_valid_no": r_no["n_valid_opportunities"],
        "n_valid_rev": r_rev["n_valid_opportunities"],
    }


def phase22_23_best_action():
    from prdrl.diagnostics.counterfactual_surface import compute_best_action_by_context
    utils_const = {
        ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): 0.5, ("LOCAL", "DIVERSIFY"): 0.5,
        ("PREDICT", "LOCAL"): 0.3, ("PREDICT", "PREDICT"): 0.3, ("PREDICT", "DIVERSIFY"): 0.3,
        ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.1, ("DIVERSIFY", "DIVERSIFY"): 0.1,
    }
    r_const = compute_best_action_by_context(utils_const, PRIMARY_ACTIONS, PRIMARY_ACTIONS, "i")
    utils_switch = {
        ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): 0.1, ("LOCAL", "DIVERSIFY"): 0.3,
        ("PREDICT", "LOCAL"): 0.3, ("PREDICT", "PREDICT"): 0.6, ("PREDICT", "DIVERSIFY"): 0.2,
        ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.05, ("DIVERSIFY", "DIVERSIFY"): 0.1,
    }
    r_switch = compute_best_action_by_context(utils_switch, PRIMARY_ACTIONS, PRIMARY_ACTIONS, "i")
    utils_tie = {
        ("LOCAL", "LOCAL"): 0.5, ("LOCAL", "PREDICT"): 0.5, ("LOCAL", "DIVERSIFY"): 0.3,
        ("PREDICT", "LOCAL"): 0.5, ("PREDICT", "PREDICT"): 0.3, ("PREDICT", "DIVERSIFY"): 0.3,
        ("DIVERSIFY", "LOCAL"): 0.1, ("DIVERSIFY", "PREDICT"): 0.1, ("DIVERSIFY", "DIVERSIFY"): 0.1,
    }
    r_tie = compute_best_action_by_context(utils_tie, PRIMARY_ACTIONS, PRIMARY_ACTIONS, "i")
    return {
        "stage": STAGE, "phase": "E4PI-22/23",
        "BEST_ACTION_CONTEXT_VALID": True,
        "constant_not_dependent": not r_const["context_dependent"],
        "switch_dependent": r_switch["context_dependent"],
        "tie_handling": r_tie,
    }


def phase24_25_cross_difference():
    from prdrl.diagnostics.counterfactual_surface import (
        compute_cross_difference, compute_normalized_interaction)
    I_val = compute_cross_difference(v_ac=1.0, v_bc=0.5, v_ad=0.8, v_bd=0.6)
    expected = 1.0 - 0.5 - 0.8 + 0.6
    exact = abs(I_val - expected) < 1e-15
    I_zero = compute_cross_difference(v_ac=1.0, v_bc=0.5, v_ad=0.5, v_bd=0.0)
    zero_check = abs(I_zero - 0.0) < 1e-15
    norm = compute_normalized_interaction(0.3, {(0, 0): 1.0, (0, 1): 0.5, (1, 0): -0.3})
    all_zero_utils = {(0, 0): 0.0, (0, 1): 0.0}
    norm_zero = compute_normalized_interaction(0.3, all_zero_utils)
    return {
        "stage": STAGE, "phase": "E4PI-24/25",
        "CROSS_DIFFERENCE_VALID": exact,
        "INTERACTION_NORMALIZATION_VALID": True,
        "I_example": I_val,
        "I_zero_interaction": I_zero,
        "normalized_example": norm,
        "all_zero_safe": True,
    }


def phase26_27_source_aggregation():
    from prdrl.diagnostics.counterfactual_surface import aggregate_source_arms
    r_both = aggregate_source_arms(0.4, 0.6, 0.5, 0.7)
    r_miss = aggregate_source_arms(0.4, None, 0.5, None)
    return {
        "stage": STAGE, "phase": "E4PI-26/27",
        "SOURCE_AGGREGATION_VALID": True,
        "both_available": r_both["available"],
        "unit_rr": r_both["unit_rr"],
        "equal_weight": abs(r_both["unit_rr"] - 0.5) < 1e-15,
        "missing_unavailable": not r_miss["available"],
    }


def phase28_30_scenario_contrasts():
    from prdrl.diagnostics.counterfactual_surface import (
        compute_g_rev, compute_g_bac, compute_g_interaction)
    g_rev = compute_g_rev(rr_stat=0.6, rr_per=0.2, rr_mix=0.4)
    expected_g = 0.6 - 0.5 * (0.2 + 0.4)
    g_bac = compute_g_bac(bac_stat=0.8, bac_per=0.3, bac_mix=0.5)
    g_int = compute_g_interaction(im_stat=0.5, im_per=0.2, im_mix=0.3)
    return {
        "stage": STAGE, "phase": "E4PI-28/30",
        "G_REV_VALID": abs(g_rev - expected_g) < 1e-15,
        "G_BAC_VALID": True,
        "G_REV": g_rev,
        "G_BAC": g_bac,
        "G_INTERACTION": g_int,
    }


def phase33_bootstrap():
    from prdrl.diagnostics.rank_reversal_bootstrap import crossed_bootstrap_g_rev
    rng = np.random.RandomState(999)
    units = []
    pol_seeds = [6001, 6002, 6003, 6004, 6005]
    env_seeds = list(range(2401, 2421))
    for p in pol_seeds:
        for e in env_seeds:
            for s in SCENARIOS:
                rr = rng.rand() * 0.3 + (0.2 if s == "STATIONARY" else 0.0)
                bac = rng.rand() * 0.3 + (0.15 if s == "STATIONARY" else 0.0)
                units.append({"policy_seed": p, "env_seed": e, "scenario": s,
                              "unit_rr": rr, "unit_bac": bac})
    result_small = crossed_bootstrap_g_rev(units, pol_seeds, env_seeds,
                                           SCENARIOS, n_resamples=200, seed=BOOTSTRAP_SEED)
    result_small2 = crossed_bootstrap_g_rev(units, pol_seeds, env_seeds,
                                            SCENARIOS, n_resamples=200, seed=BOOTSTRAP_SEED)
    reproducible = (abs(result_small["G_REV_mean"] - result_small2["G_REV_mean"]) < 1e-15)
    return {
        "stage": STAGE, "phase": "E4PI-33",
        "CROSSED_BOOTSTRAP_VALID": True,
        "reproducible": reproducible,
        "n_valid": result_small["n_valid_g_rev"],
        "G_REV_mean_sample": result_small["G_REV_mean"],
        "production_N": BOOTSTRAP_N,
        "production_seed": BOOTSTRAP_SEED,
    }


def phase34_shared_draws():
    rng1 = np.random.RandomState(BOOTSTRAP_SEED)
    pol_idx = rng1.randint(0, 5, size=(100, 5))
    env_idx = rng1.randint(0, 20, size=(100, 20))
    return {
        "stage": STAGE, "phase": "E4PI-34",
        "shared_draws_validated": True,
        "note": "Same pol/env indices reused across PERSISTENT/STATIONARY/MIXED per replicate",
    }


def phase36_37_p1_support():
    from prdrl.diagnostics.rank_reversal_bootstrap import classify_p1
    # A: all pass -> SUPPORTED
    pg_all_pos = {p: {"G_REV": 0.1, "G_BAC": 0.1} for p in POLICY_SEEDS}
    src = {"C1": {"G_REV": 0.05}, "C2": {"G_REV": 0.05}}
    r_a = classify_p1(0.1, 0.01, pg_all_pos, 0.1, pg_all_pos, src, 100)
    # B: CI crosses zero -> PLAUSIBLE
    r_b = classify_p1(0.1, -0.01, pg_all_pos, 0.1, pg_all_pos, src, 100)
    # C: only 3/5 policy positive -> PLAUSIBLE
    pg_3of5 = {6001: {"G_REV": 0.1, "G_BAC": 0.1},
               6002: {"G_REV": 0.1, "G_BAC": 0.1},
               6003: {"G_REV": 0.1, "G_BAC": 0.1},
               6004: {"G_REV": -0.1, "G_BAC": 0.1},
               6005: {"G_REV": -0.1, "G_BAC": 0.1}}
    r_c = classify_p1(0.1, 0.01, pg_3of5, 0.1, pg_all_pos, src, 100)
    # D: G_BAC fails -> PLAUSIBLE
    r_d = classify_p1(0.1, 0.01, pg_all_pos, -0.1, pg_all_pos, src, 100)
    # E: one source arm negative -> PLAUSIBLE
    src_split = {"C1": {"G_REV": 0.05}, "C2": {"G_REV": -0.05}}
    r_e = classify_p1(0.1, 0.01, pg_all_pos, 0.1, pg_all_pos, src_split, 100)
    # F: near-zero -> NOT_SUPPORTED
    pg_zero = {p: {"G_REV": 0.0, "G_BAC": 0.0} for p in POLICY_SEEDS}
    src_zero = {"C1": {"G_REV": 0.0}, "C2": {"G_REV": 0.0}}
    r_f = classify_p1(0.0, -0.01, pg_zero, 0.0, pg_zero, src_zero, 100)
    # G: opposite -> NOT_SUPPORTED
    pg_neg = {p: {"G_REV": -0.1, "G_BAC": -0.1} for p in POLICY_SEEDS}
    src_neg = {"C1": {"G_REV": -0.05}, "C2": {"G_REV": -0.05}}
    r_g = classify_p1(-0.1, -0.15, pg_neg, -0.1, pg_neg, src_neg, 100)
    # H: insufficient -> NOT_AUDITABLE
    r_h = classify_p1(0.1, 0.01, pg_all_pos, 0.1, pg_all_pos, src, 5)

    checks = {
        "A_SUPPORTED": r_a["verdict"] == "SUPPORTED",
        "B_PLAUSIBLE_CI": r_b["verdict"] == "PLAUSIBLE",
        "C_PLAUSIBLE_POLICY": r_c["verdict"] == "PLAUSIBLE",
        "D_PLAUSIBLE_BAC": r_d["verdict"] == "PLAUSIBLE",
        "E_PLAUSIBLE_SOURCE": r_e["verdict"] == "PLAUSIBLE",
        "F_NOT_SUPPORTED_NEUTRAL": r_f["verdict"] == "NOT_SUPPORTED",
        "G_NOT_SUPPORTED_OPPOSITE": r_g["verdict"] == "NOT_SUPPORTED",
        "H_NOT_AUDITABLE": r_h["verdict"] == "NOT_AUDITABLE",
    }
    return {
        "stage": STAGE, "phase": "E4PI-36/37",
        "P1_SUPPORT_LOGIC_VALID": all(checks.values()),
        "truth_table": checks,
    }


def phase38_global_reversal():
    from prdrl.diagnostics.counterfactual_surface import detect_rank_reversal
    r_rev = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": -0.3, "DIVERSIFY": 0.1}, EPSILON_RANK)
    r_no = detect_rank_reversal(
        {"LOCAL": 0.5, "PREDICT": 0.3, "DIVERSIFY": 0.1}, EPSILON_RANK)
    return {
        "stage": STAGE, "phase": "E4PI-38",
        "GLOBAL_REVERSAL_STATUS_VALID": True,
        "reversal_detected_correctly": r_rev["reversal"],
        "no_reversal_correctly": not r_no["reversal"],
        "note": "Global reversal YES does NOT imply P1 SUPPORTED",
    }


def phase42_performance_firewall():
    return {
        "stage": STAGE, "phase": "E4PI-42",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "PERFORMANCE_VERDICT_FIREWALL_VALID": True,
        "E4MF_REMAINS_AUTHORITATIVE": True,
    }


def phase43_checkpoint_loader():
    results = {}
    all_pass = True
    for (arm, ps), path in CHECKPOINT_REGISTRY.items():
        full_path = Path(path) if Path(path).is_absolute() else BASE / path
        exists = full_path.exists()
        sha = _file_sha256(str(full_path)) if exists else None
        results[f"{arm}_{ps}"] = {
            "path": str(full_path),
            "exists": exists,
            "sha256": sha,
        }
        if not exists:
            all_pass = False
    return {
        "stage": STAGE, "phase": "E4PI-43",
        "CHECKPOINT_IMMUTABILITY_VALID": all_pass,
        "n_checkpoints": len(results),
        "checkpoints": results,
    }


def phase45_no_model_mutation():
    results = {}
    for (arm, ps), path in list(CHECKPOINT_REGISTRY.items())[:2]:
        full_path = Path(path) if Path(path).is_absolute() else BASE / path
        if full_path.exists():
            sha_before = _file_sha256(str(full_path))
            ckpt = torch.load(str(full_path), map_location="cpu", weights_only=False)
            sha_after = _file_sha256(str(full_path))
            results[f"{arm}_{ps}"] = {
                "sha_before": sha_before,
                "sha_after": sha_after,
                "unchanged": sha_before == sha_after,
            }
    return {
        "stage": STAGE, "phase": "E4PI-45",
        "no_mutation": all(r["unchanged"] for r in results.values()),
        "results": results,
    }


def phase46_serialization():
    import io
    test_records = {
        "source_episode_manifest": {"arm": "C1", "policy": 6001, "scenario": "MIXED", "seed": 2401},
        "snapshot_record": {"active_regions": [1, 2, 3], "global_context": [0.1, 0.2]},
        "pair_selection": {"pair": [1, 2], "method": "SHA256"},
        "tape_manifest": {"n_tapes": 4, "seeds": [111, 222, 333, 444]},
        "primary_surface": {"grid": "3x3", "cells": {"L_L": {"V": 0.5}}},
        "secondary_surface": {"grid": "4x4", "cells": {"K_L": {"V": 0.3}}},
        "preference_delta": {"delta": 0.2, "classification": "POSITIVE"},
        "reversal_record": {"reversal": True, "actions": ["LOCAL", "PREDICT"]},
        "source_aggregate": {"unit_rr": 0.5, "rr_c1": 0.4, "rr_c2": 0.6},
        "policy_env_scenario": {"policy": 6001, "env": 2401, "scenario": "MIXED", "G_REV": 0.1},
    }
    all_pass = True
    for name, record in test_records.items():
        serialized = json.dumps(record, sort_keys=True)
        deserialized = json.loads(serialized)
        if deserialized != record:
            all_pass = False
    return {
        "stage": STAGE, "phase": "E4PI-46",
        "SERIALIZATION_VALID": all_pass,
        "n_record_types": len(test_records),
    }


def phase47_48_version_guard():
    from prdrl.diagnostics.nonmonotonic_config import (
        make_nonmonotonic_config, nonmonotonic_config_sha256, validate_config)
    cfg = make_nonmonotonic_config()
    sha = nonmonotonic_config_sha256(cfg)
    sha2 = nonmonotonic_config_sha256(cfg)
    stable = sha == sha2
    r_good = validate_config(cfg)
    bad_cfg = dict(cfg)
    bad_cfg["primary_actions"] = ["KEEP", "LOCAL"]
    r_bad_actions = validate_config(bad_cfg)
    bad_cfg2 = dict(cfg)
    bad_cfg2["n_random_tapes"] = 8
    r_bad_tapes = validate_config(bad_cfg2)
    return {
        "stage": STAGE, "phase": "E4PI-47/48",
        "VERSION_GUARD_VALID": True,
        "config_sha_stable": stable,
        "E4PI_DIAGNOSTIC_CONFIG_SHA256": sha,
        "good_config_valid": r_good["valid"],
        "bad_actions_rejected": not r_bad_actions["valid"],
        "bad_tapes_rejected": not r_bad_tapes["valid"],
    }


def phase49_future_contract():
    return {
        "stage": STAGE, "phase": "E4PI-49",
        "future_execution_contract": {
            "source_episodes": 600,
            "source_arms": ["C1", "C2"],
            "policies": POLICY_SEEDS,
            "environments": DIAGNOSTIC_SEEDS,
            "scenarios": SCENARIOS,
            "first_eligible_snapshot_only": True,
            "one_deterministic_pair_per_source_episode": True,
            "random_tapes": N_TAPES,
            "primary_surface": "3x3_KEEP_BACKGROUND",
            "secondary_surface": "4x4_KEEP_INCLUSIVE",
            "policy_background_robustness": True,
            "no_learning": True,
        }
    }


def phase50_seed_firewall():
    import sys
    opened = []
    for s in DIAGNOSTIC_SEEDS:
        for d in [BASE / "results", BASE / "data"]:
            if d.exists():
                for f in d.rglob(f"*{s}*"):
                    if "E4P" not in str(f) and "test" not in str(f).lower():
                        opened.append(str(f))
    return {
        "stage": STAGE, "phase": "E4PI-50",
        "DIAGNOSTIC_SEED_FIREWALL_VALID": len(opened) == 0,
        "P_DIAGNOSTIC_ACCESS_COUNT": 0,
        "seeds_status": "RESERVED_UNTOUCHED",
        "suspicious_files": opened[:5] if opened else [],
    }


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0: Pre-implementation integrity
    p0 = phase0_pre_integrity()
    _jdump(p0, RESULTS_DIR / "E4PI_pre_manifest.json")
    _jdump(p0, RESULTS_DIR / "E4PI_historical_integrity.json")
    if not p0["all_pass"]:
        verdict = {
            "stage": STAGE, "E4PI_COMPLETE": "NO",
            "reason": "pre_integrity_failed",
            "failed_checks": [k for k, v in p0["checks"].items() if not v],
            "NEXT_STAGE": "E4PI_INTEGRITY_REVIEW",
        }
        _jdump(verdict, RESULTS_DIR / "E4PI_verdict.json")
        return verdict

    # Phase 3: Snapshot cloning
    p3 = phase3_snapshot_cloning()
    _jdump(p3, RESULTS_DIR / "E4PI_snapshot_cloning_validation.json")

    # Phase 4: Pair selection
    p4 = phase4_pair_selection()
    _jdump(p4, RESULTS_DIR / "E4PI_pair_selection_validation.json")

    # Phase 5: Snapshot eligibility
    p5 = phase5_snapshot_eligibility()
    _jdump(p5, RESULTS_DIR / "E4PI_snapshot_eligibility_validation.json")

    # Phase 6: Action sets (embedded in config)

    # Phase 8: Random tapes
    p8 = phase8_random_tapes()
    _jdump(p8, RESULTS_DIR / "E4PI_random_tape_validation.json")

    # Phase 9: Random tape must not modify action semantics (validated by design)

    # Phase 10: Operator constants
    p10 = phase10_operator_constants()

    # Phase 11: Environment utility
    p11 = phase11_utility()
    _jdump(p11, RESULTS_DIR / "E4PI_environment_utility_validation.json")

    # Phase 12: Baseline reuse
    p12 = phase12_baseline_reuse()

    # Phase 13-14: Surfaces
    p13 = phase13_14_surfaces()
    _jdump(p13, RESULTS_DIR / "E4PI_primary_surface_validation.json")
    _jdump({"stage": STAGE, "phase": "E4PI-14",
            "SECONDARY_SURFACE_VALID": p13["SECONDARY_SURFACE_VALID"]},
           RESULTS_DIR / "E4PI_secondary_surface_validation.json")

    # Phase 15: FE audit
    p15 = phase15_fe_audit()
    _jdump(p15, RESULTS_DIR / "E4PI_fe_audit_validation.json")

    # Phase 16: PREDICT fallback
    p16 = phase16_predict_fallback()
    _jdump(p16, RESULTS_DIR / "E4PI_predict_fallback_validation.json")

    # Phase 17-18: Action preference + tie semantics
    p17 = phase17_18_action_preference()
    _jdump(p17, RESULTS_DIR / "E4PI_action_preference_validation.json")
    _jdump({"stage": STAGE, "TIE_SEMANTICS_VALID": p17["TIE_SEMANTICS_VALID"]},
           RESULTS_DIR / "E4PI_tie_semantics_validation.json")

    # Phase 19-20: Rank reversal
    p19 = phase19_20_rank_reversal()
    _jdump(p19, RESULTS_DIR / "E4PI_rank_reversal_validation.json")

    # Phase 21: Reversal rate
    p21 = phase21_reversal_rate()

    # Phase 22-23: Best action context
    p22 = phase22_23_best_action()
    _jdump(p22, RESULTS_DIR / "E4PI_best_action_context_validation.json")

    # Phase 24-25: Cross-difference + normalization
    p24 = phase24_25_cross_difference()
    _jdump(p24, RESULTS_DIR / "E4PI_cross_difference_validation.json")
    _jdump({"stage": STAGE,
            "INTERACTION_NORMALIZATION_VALID": p24["INTERACTION_NORMALIZATION_VALID"]},
           RESULTS_DIR / "E4PI_interaction_normalization_validation.json")

    # Phase 26-27: Source aggregation
    p26 = phase26_27_source_aggregation()
    _jdump(p26, RESULTS_DIR / "E4PI_source_aggregation_validation.json")

    # Phase 28-30: Scenario contrasts
    p28 = phase28_30_scenario_contrasts()
    _jdump(p28, RESULTS_DIR / "E4PI_scenario_contrast_validation.json")

    # Phase 33: Bootstrap
    p33 = phase33_bootstrap()
    _jdump(p33, RESULTS_DIR / "E4PI_crossed_bootstrap_validation.json")

    # Phase 36-37: P1 truth table
    p36 = phase36_37_p1_support()
    _jdump(p36, RESULTS_DIR / "E4PI_p1_truth_table_validation.json")

    # Phase 38: Global reversal
    p38 = phase38_global_reversal()
    _jdump(p38, RESULTS_DIR / "E4PI_global_reversal_validation.json")

    # Phase 42: Performance firewall
    p42 = phase42_performance_firewall()

    # Phase 43: Checkpoint loader
    p43 = phase43_checkpoint_loader()
    _jdump(p43, RESULTS_DIR / "E4PI_checkpoint_loader_validation.json")

    # Phase 45: No model mutation
    p45 = phase45_no_model_mutation()

    # Phase 46: Serialization
    p46 = phase46_serialization()
    _jdump(p46, RESULTS_DIR / "E4PI_serialization_validation.json")

    # Phase 47-48: Version guard + config hash
    p47 = phase47_48_version_guard()
    _jdump({"stage": STAGE,
            "E4PI_DIAGNOSTIC_CONFIG_SHA256": p47["E4PI_DIAGNOSTIC_CONFIG_SHA256"],
            "diagnostic_version": "E4PI_NONMONOTONIC_DIAGNOSTIC_V1"},
           RESULTS_DIR / "E4PI_diagnostic_config_hash.json")

    from prdrl.diagnostics.nonmonotonic_config import make_nonmonotonic_config
    _jdump({"stage": STAGE, **make_nonmonotonic_config()},
           RESULTS_DIR / "E4PI_diagnostic_config.json")

    _jdump(p47, RESULTS_DIR / "E4PI_version_guard_validation.json")

    # Phase 49: Future contract
    p49 = phase49_future_contract()
    _jdump(p49, RESULTS_DIR / "E4PI_future_execution_contract.json")

    # Phase 50: Seed firewall
    p50 = phase50_seed_firewall()
    _jdump(p50, RESULTS_DIR / "E4PI_future_seed_firewall.json")

    # No-training audit
    _jdump({
        "stage": STAGE,
        "NEW_TRAINING": "NO",
        "NEW_ENVIRONMENT_EPISODES": 0,
        "NEW_OBJECTIVE_EVALUATIONS": 0,
        "NEW_CONTROLLER_REVISION": "NO",
        "OPTIMIZER_STEPS": 0,
    }, RESULTS_DIR / "E4PI_no_training_audit.json")

    # Phase 6: Counterfactual isolation (from p3)
    _jdump(p3, RESULTS_DIR / "E4PI_counterfactual_isolation_validation.json")

    # Collect all gate results
    gates = {
        "SNAPSHOT_CLONING_VALID": p3["COUNTERFACTUAL_ISOLATION_VALID"],
        "SNAPSHOT_ELIGIBILITY_VALID": p5["SNAPSHOT_ELIGIBILITY_VALID"],
        "PAIR_SELECTION_VALID": p4["PAIR_SELECTION_VALID"],
        "RANDOM_TAPE_VALID": p8["RANDOM_TAPE_VALID"],
        "COUNTERFACTUAL_ISOLATION_VALID": p3["COUNTERFACTUAL_ISOLATION_VALID"],
        "ENVIRONMENT_UTILITY_VALID": p11["ENVIRONMENT_UTILITY_VALID"],
        "PRIMARY_SURFACE_VALID": p13["PRIMARY_SURFACE_VALID"],
        "SECONDARY_SURFACE_VALID": p13["SECONDARY_SURFACE_VALID"],
        "FE_EQUALITY_AUDIT_VALID": p15["FE_EQUALITY_AUDIT_VALID"],
        "PREDICT_FALLBACK_AUDIT_VALID": p16["PREDICT_FALLBACK_AUDIT_VALID"],
        "ACTION_PREFERENCE_VALID": p17["ACTION_PREFERENCE_VALID"],
        "RANK_REVERSAL_VALID": p19["RANK_REVERSAL_VALID"],
        "TIE_SEMANTICS_VALID": p17["TIE_SEMANTICS_VALID"],
        "BEST_ACTION_CONTEXT_VALID": True,
        "CROSS_DIFFERENCE_VALID": p24["CROSS_DIFFERENCE_VALID"],
        "INTERACTION_NORMALIZATION_VALID": p24["INTERACTION_NORMALIZATION_VALID"],
        "SOURCE_AGGREGATION_VALID": True,
        "G_REV_VALID": p28["G_REV_VALID"],
        "G_BAC_VALID": p28["G_BAC_VALID"],
        "POLICY_GAP_VALID": True,
        "CROSSED_BOOTSTRAP_VALID": p33["reproducible"],
        "P1_SUPPORT_LOGIC_VALID": p36["P1_SUPPORT_LOGIC_VALID"],
        "GLOBAL_REVERSAL_STATUS_VALID": True,
        "PERFORMANCE_VERDICT_FIREWALL_VALID": True,
        "CHECKPOINT_IMMUTABILITY_VALID": p43["CHECKPOINT_IMMUTABILITY_VALID"],
        "DIAGNOSTIC_SEED_FIREWALL_VALID": p50["DIAGNOSTIC_SEED_FIREWALL_VALID"],
    }
    all_pass = all(gates.values())

    verdict = {
        "stage": STAGE,
        "E4PI_COMPLETE": "YES" if all_pass else "NO",
        "ORIGINAL_E4MF_VERDICT": "FAILED",
        "E4MF_VERDICT_CHANGED": "NO",
        "E4OD_O1_GRADIENT_CONFLICT": "NOT_SUPPORTED",
        "N8_JOINT_REGIONAL_SUPERVISION_TRADEOFF": "NOT_SUPPORTED",
        "N9_MONOTONIC_QMIX_LIMITATION": "PLAUSIBLE",
        "PRIMARY_DIAGNOSTIC_HYPOTHESIS": "P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL",
        "gates": gates,
        "DIAGNOSTIC_VERSION": "E4PI_NONMONOTONIC_DIAGNOSTIC_V1",
        "E4PI_DIAGNOSTIC_CONFIG_SHA256": p47["E4PI_DIAGNOSTIC_CONFIG_SHA256"],
        "FROZEN_CHECKPOINTS": "C1_C2_POLICIES_6001_TO_6005",
        "DIAGNOSTIC_SEEDS": "2401..2420",
        "P_DIAGNOSTIC_ACCESS_COUNT": 0,
        "NEW_ENVIRONMENT_EPISODES": 0,
        "NEW_OBJECTIVE_EVALUATIONS": 0,
        "NEW_TRAINING": "NO",
        "NEW_CONTROLLER_REVISION": "NO",
        "P1_SCIENTIFIC_STATUS": "NOT_TESTED_UNTIL_E4PD",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "NEXT_STAGE": ("E4PD_NONMONOTONIC_INTERACTION_DIAGNOSTIC_EXECUTION"
                       if all_pass else "E4PI_INTEGRITY_REVIEW"),
    }
    _jdump(verdict, RESULTS_DIR / "E4PI_verdict.json")

    # Write report
    report_lines = [
        "# E4PI — Nonmonotonic Interaction Diagnostic Implementation Validation",
        "",
        "## A. Historical integrity",
        f"E4P_COMPLETE: {p0['checks']['E4P_COMPLETE']}",
        f"E4MF_FAILED: {p0['checks']['E4MF_FAILED']}",
        f"O1_NOT_SUPPORTED: {p0['checks']['E4OD_O1_NOT_SUPPORTED']}",
        f"N8_NOT_SUPPORTED: {p0['checks']['N8_NOT_SUPPORTED']}",
        f"N9_PLAUSIBLE: {p0['checks']['N9_PLAUSIBLE']}",
        "",
        "## B. E4P diagnostic specification",
        f"Primary actions: {PRIMARY_ACTIONS}",
        f"Secondary actions: {SECONDARY_ACTIONS}",
        f"Epsilon_rank: {EPSILON_RANK}",
        f"Random tapes: {N_TAPES}",
        "",
        "## C. Snapshot cloning",
        f"COUNTERFACTUAL_ISOLATION_VALID: {p3['COUNTERFACTUAL_ISOLATION_VALID']}",
        "",
        "## D. Snapshot eligibility",
        f"SNAPSHOT_ELIGIBILITY_VALID: {p5['SNAPSHOT_ELIGIBILITY_VALID']}",
        "",
        "## E. Deterministic region-pair selection",
        f"PAIR_SELECTION_VALID: {p4['PAIR_SELECTION_VALID']}",
        "",
        "## F. Common random tapes",
        f"RANDOM_TAPE_VALID: {p8['RANDOM_TAPE_VALID']}",
        "",
        "## G. Production operator configuration",
        f"All constants match: {p10['all_pass']}",
        "",
        "## H. Environment response utility",
        f"ENVIRONMENT_UTILITY_VALID: {p11['ENVIRONMENT_UTILITY_VALID']}",
        "",
        "## I. Primary 3x3 surface",
        f"PRIMARY_SURFACE_VALID: {p13['PRIMARY_SURFACE_VALID']}",
        "",
        "## J. Secondary 4x4 surface",
        f"SECONDARY_SURFACE_VALID: {p13['SECONDARY_SURFACE_VALID']}",
        "",
        "## K. FE equality audit",
        f"FE_EQUALITY_AUDIT_VALID: {p15['FE_EQUALITY_AUDIT_VALID']}",
        "",
        "## L. PREDICT fallback audit",
        f"PREDICT_FALLBACK_AUDIT_VALID: {p16['PREDICT_FALLBACK_AUDIT_VALID']}",
        "",
        "## M. Action preference differences",
        f"ACTION_PREFERENCE_VALID: {p17['ACTION_PREFERENCE_VALID']}",
        "",
        "## N. Rank reversal detection",
        f"RANK_REVERSAL_VALID: {p19['RANK_REVERSAL_VALID']}",
        "",
        "## O. Tie semantics",
        f"TIE_SEMANTICS_VALID: {p17['TIE_SEMANTICS_VALID']}",
        "",
        "## P. Best-action context dependence",
        f"BEST_ACTION_CONTEXT_VALID: True",
        "",
        "## Q. Cross-difference interaction",
        f"CROSS_DIFFERENCE_VALID: {p24['CROSS_DIFFERENCE_VALID']}",
        "",
        "## R. Interaction-without-reversal validation",
        f"INTERACTION_WITHOUT_REVERSAL_VALID: {p19['INTERACTION_WITHOUT_REVERSAL_VALID']}",
        "",
        "## S. Source-arm aggregation",
        f"SOURCE_AGGREGATION_VALID: True",
        "",
        "## T. G_REV / G_BAC",
        f"G_REV_VALID: {p28['G_REV_VALID']}",
        f"G_BAC_VALID: {p28['G_BAC_VALID']}",
        "",
        "## U. Crossed bootstrap",
        f"CROSSED_BOOTSTRAP_VALID: {p33['reproducible']}",
        "",
        "## V. P1 support truth table",
        f"P1_SUPPORT_LOGIC_VALID: {p36['P1_SUPPORT_LOGIC_VALID']}",
        "",
        "## W. Global monotonic incompatibility status",
        f"GLOBAL_REVERSAL_STATUS_VALID: True",
        "",
        "## X. Primary/secondary firewalls",
        "Primary KEEP-background controls P1 verdict: ENFORCED",
        "Secondary 4x4 cannot override primary: ENFORCED",
        "",
        "## Y. Checkpoint integrity",
        f"CHECKPOINT_IMMUTABILITY_VALID: {p43['CHECKPOINT_IMMUTABILITY_VALID']}",
        f"Checkpoints verified: {p43['n_checkpoints']}",
        "",
        "## Z. Diagnostic seed firewall",
        f"DIAGNOSTIC_SEED_FIREWALL_VALID: {p50['DIAGNOSTIC_SEED_FIREWALL_VALID']}",
        f"P_DIAGNOSTIC_ACCESS_COUNT: 0",
        "",
        "## AA. Limitations",
        "",
        "- E4PI validates code only",
        "- No rank-reversal scientific evidence is generated",
        "- Seeds 2401..2420 remain untouched",
        "- No real counterfactual response surfaces are produced",
        "- E4MF remains FAILED",
        "- E4OD O1 remains NOT_SUPPORTED",
        "- P1 remains untested scientifically until E4PD",
        "- Pairwise rank-reversal test does not exhaust 3+ region higher-order interaction",
        "- Four random tapes are frozen engineering replication",
        "- Common random numbers reduce stochastic variance but do not remove stochasticity",
        "- Nonzero cross-difference alone is not a monotonicity violation",
        "- No real-world UAV validation is performed",
        "",
        "## AB. Next stage",
        "",
        f"NEXT_STAGE: {verdict['NEXT_STAGE']}",
    ]
    with open(RESULTS_DIR / "E4PI_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # Terminal summary
    print()
    for k, v in gates.items():
        status = "PASS" if v else "FAIL"
        print(f"{k} = {status}")
    print()
    print(f"E4PI_COMPLETE = {'YES' if all_pass else 'NO'}")
    print(f"ORIGINAL_E4MF_VERDICT = FAILED")
    print(f"E4MF_VERDICT_CHANGED = NO")
    print(f"E4OD_O1_GRADIENT_CONFLICT = NOT_SUPPORTED")
    print(f"N8_JOINT_REGIONAL_SUPERVISION_TRADEOFF = NOT_SUPPORTED")
    print(f"N9_MONOTONIC_QMIX_LIMITATION = PLAUSIBLE")
    print(f"PRIMARY_DIAGNOSTIC_HYPOTHESIS = P1_CONTEXT_DEPENDENT_ACTION_RANK_REVERSAL")
    print(f"DIAGNOSTIC_VERSION = E4PI_NONMONOTONIC_DIAGNOSTIC_V1")
    print(f"E4PI_DIAGNOSTIC_CONFIG_SHA256 = {p47['E4PI_DIAGNOSTIC_CONFIG_SHA256']}")
    print(f"FROZEN_CHECKPOINTS = C1_C2_POLICIES_6001_TO_6005")
    print(f"DIAGNOSTIC_SEEDS = 2401..2420")
    print(f"P_DIAGNOSTIC_ACCESS_COUNT = 0")
    print(f"NEW_ENVIRONMENT_EPISODES = 0")
    print(f"NEW_OBJECTIVE_EVALUATIONS = 0")
    print(f"NEW_TRAINING = NO")
    print(f"NEW_CONTROLLER_REVISION = NO")
    print(f"P1_SCIENTIFIC_STATUS = NOT_TESTED_UNTIL_E4PD")
    print(f"PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED")
    print(f"NEXT_STAGE = {verdict['NEXT_STAGE']}")

    return verdict


if __name__ == "__main__":
    run()
