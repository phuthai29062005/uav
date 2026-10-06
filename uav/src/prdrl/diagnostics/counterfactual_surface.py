"""Counterfactual response surface construction and utility computation (E4P/E4PI).

Implements: snapshot cloning, primary 3x3 / secondary 4x4 surfaces,
environment utility V_S, baseline reuse, action preference deltas,
rank reversal detection, best-action context dependence, cross-difference
interaction, FE audit, PREDICT fallback audit.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

PRIMARY_ACTIONS = ["LOCAL", "PREDICT", "DIVERSIFY"]
SECONDARY_ACTIONS = ["KEEP", "LOCAL", "PREDICT", "DIVERSIFY"]
EPSILON_RANK = 1e-12


def clone_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(snapshot)


def select_pair_sha256(source_arm: str, policy_seed: int,
                       scenario: str, env_seed: int,
                       eligible_pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not eligible_pairs:
        raise ValueError("No eligible pairs")
    sorted_pairs = sorted(eligible_pairs, key=lambda p: (p[0], p[1]))
    key = f"{source_arm}_{policy_seed}_{scenario}_{env_seed}"
    h = hashlib.sha256(key.encode()).digest()
    idx = int.from_bytes(h[:8], "big") % len(sorted_pairs)
    return sorted_pairs[idx]


def check_snapshot_eligibility(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    active = snapshot.get("active_regions", [])
    if len(active) < 2:
        return {"eligible": False, "reason": "fewer_than_2_active_regions"}
    if snapshot.get("global_context") is None:
        return {"eligible": False, "reason": "missing_global_context"}
    regional_states = snapshot.get("regional_states", {})
    predict_history = snapshot.get("predict_history", {})
    for r in active:
        if str(r) not in regional_states and r not in regional_states:
            return {"eligible": False, "reason": f"missing_regional_state_{r}"}
    return {"eligible": True, "reason": "all_criteria_met"}


def check_pair_predict_eligibility(snapshot: Dict[str, Any],
                                   pair: Tuple[int, int]) -> bool:
    predict_history = snapshot.get("predict_history", {})
    for r in pair:
        key = str(r) if str(r) in predict_history else r
        if key not in predict_history or not predict_history[key]:
            return False
    return True


def find_first_eligible_snapshot(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for event in events:
        result = check_snapshot_eligibility(event)
        if result["eligible"]:
            return event
    return None


def generate_random_tape_seeds(source_arm: str, policy_seed: int,
                               scenario: str, env_seed: int,
                               n_tapes: int = 4) -> List[int]:
    seeds = []
    for tape_idx in range(n_tapes):
        key = f"tape_{source_arm}_{policy_seed}_{scenario}_{env_seed}_{tape_idx}"
        h = hashlib.sha256(key.encode()).digest()
        seeds.append(int.from_bytes(h[:8], "big") % (2**31))
    return seeds


def compute_utility(igd_post_action: float, igd_post_keep_all: float) -> float:
    return igd_post_keep_all - igd_post_action


def build_primary_surface(utility_matrix: Dict[Tuple[str, str], Dict]) -> Dict:
    surface = {"grid_size": "3x3", "n_cells": 9, "cells": {}}
    for i_act in PRIMARY_ACTIONS:
        for j_act in PRIMARY_ACTIONS:
            key = f"{i_act}_{j_act}"
            cell = utility_matrix.get((i_act, j_act), {})
            surface["cells"][key] = cell
    return surface


def build_secondary_surface(utility_matrix: Dict[Tuple[str, str], Dict]) -> Dict:
    surface = {"grid_size": "4x4", "n_cells": 16,
               "PRIMARY_EQUAL_BUDGET": False, "cells": {}}
    for i_act in SECONDARY_ACTIONS:
        for j_act in SECONDARY_ACTIONS:
            key = f"{i_act}_{j_act}"
            cell = utility_matrix.get((i_act, j_act), {})
            if "KEEP" in (i_act, j_act):
                cell["PRIMARY_EQUAL_BUDGET"] = False
            surface["cells"][key] = cell
    return surface


def audit_fe_equality(surface: Dict) -> Dict:
    fes = []
    for key, cell in surface.get("cells", {}).items():
        if "response_FE" in cell:
            fes.append(cell["response_FE"])
    if not fes:
        return {"valid": False, "reason": "no_FE_data"}
    all_equal = all(abs(fe - fes[0]) < 1e-15 for fe in fes)
    return {"valid": all_equal, "all_FE": fes,
            "status": "PASS" if all_equal else "INVALID_PRIMARY_SURFACE"}


def audit_predict_fallback(surface: Dict) -> Dict:
    fallbacks = []
    for key, cell in surface.get("cells", {}).items():
        req = cell.get("requested_i", "")
        real = cell.get("realized_i", "")
        if req == "PREDICT" and real != "PREDICT":
            fallbacks.append({"cell": key, "region": "i",
                              "requested": req, "realized": real})
        req_j = cell.get("requested_j", "")
        real_j = cell.get("realized_j", "")
        if req_j == "PREDICT" and real_j != "PREDICT":
            fallbacks.append({"cell": key, "region": "j",
                              "requested": req_j, "realized": real_j})
    valid = len(fallbacks) == 0
    return {"valid": valid, "fallbacks": fallbacks,
            "status": "PASS" if valid else "INVALID_PRIMARY_SURFACE"}


def compute_action_preference_delta(v_a_c: float, v_b_c: float) -> float:
    return v_a_c - v_b_c


def classify_delta(delta: float, epsilon: float = EPSILON_RANK) -> str:
    if delta > epsilon:
        return "POSITIVE"
    elif delta < -epsilon:
        return "NEGATIVE"
    else:
        return "TIE"


def detect_rank_reversal(deltas_by_context: Dict[str, float],
                         epsilon: float = EPSILON_RANK) -> Dict:
    has_positive = False
    has_negative = False
    classifications = {}
    for ctx, delta in deltas_by_context.items():
        cls = classify_delta(delta, epsilon)
        classifications[ctx] = cls
        if cls == "POSITIVE":
            has_positive = True
        elif cls == "NEGATIVE":
            has_negative = True
    reversal = has_positive and has_negative
    return {"reversal": reversal, "classifications": classifications}


def compute_reversal_rate(surface_utilities: Dict[Tuple[str, str], float],
                          actions: List[str],
                          epsilon: float = EPSILON_RANK) -> Dict:
    n_reversals = 0
    n_valid = 0
    reversal_details = []
    action_pairs = list(itertools.combinations(actions, 2))
    for a, b in action_pairs:
        deltas_i = {}
        for c in actions:
            v_ac = surface_utilities.get((a, c), None)
            v_bc = surface_utilities.get((b, c), None)
            if v_ac is not None and v_bc is not None:
                deltas_i[c] = compute_action_preference_delta(v_ac, v_bc)
        if deltas_i:
            n_valid += 1
            result = detect_rank_reversal(deltas_i, epsilon)
            if result["reversal"]:
                n_reversals += 1
                reversal_details.append({"region": "i", "actions": (a, b),
                                         "result": result})
    for c, d in action_pairs:
        deltas_j = {}
        for a in actions:
            v_ac = surface_utilities.get((a, c), None)
            v_ad = surface_utilities.get((a, d), None)
            if v_ac is not None and v_ad is not None:
                deltas_j[a] = compute_action_preference_delta(v_ac, v_ad)
        if deltas_j:
            n_valid += 1
            result = detect_rank_reversal(deltas_j, epsilon)
            if result["reversal"]:
                n_reversals += 1
                reversal_details.append({"region": "j", "actions": (c, d),
                                         "result": result})
    rate = n_reversals / n_valid if n_valid > 0 else None
    return {"reversal_rate": rate, "n_reversals": n_reversals,
            "n_valid_opportunities": n_valid, "details": reversal_details}


def compute_best_action_by_context(surface_utilities: Dict[Tuple[str, str], float],
                                   focal_actions: List[str],
                                   context_actions: List[str],
                                   focal_region: str = "i",
                                   epsilon: float = EPSILON_RANK) -> Dict:
    best_actions = {}
    for ctx in context_actions:
        vals = {}
        for act in focal_actions:
            if focal_region == "i":
                key = (act, ctx)
            else:
                key = (ctx, act)
            if key in surface_utilities:
                vals[act] = surface_utilities[key]
        if not vals:
            best_actions[ctx] = {"action": None, "tied": True}
            continue
        max_v = max(vals.values())
        best = [a for a, v in vals.items() if abs(v - max_v) <= epsilon]
        if len(best) == 1:
            best_actions[ctx] = {"action": best[0], "tied": False}
        else:
            best_actions[ctx] = {"action": best, "tied": True}
    unique_non_tied = set()
    for ctx, info in best_actions.items():
        if not info["tied"]:
            unique_non_tied.add(info["action"])
    context_dependent = len(unique_non_tied) > 1
    return {"best_actions": best_actions,
            "context_dependent": context_dependent,
            "n_unique_non_tied": len(unique_non_tied)}


def compute_cross_difference(v_ac: float, v_bc: float,
                             v_ad: float, v_bd: float) -> float:
    return v_ac - v_bc - v_ad + v_bd


def compute_normalized_interaction(abs_interaction: float,
                                   surface_utilities: Dict,
                                   epsilon: float = EPSILON_RANK) -> float:
    nonzero_vs = [abs(v) for v in surface_utilities.values()
                  if abs(v) > epsilon]
    if not nonzero_vs:
        return 0.0
    import numpy as np
    median_abs = float(np.median(nonzero_vs))
    return abs_interaction / (median_abs + epsilon)


def compute_bac_rate(surface_utilities: Dict[Tuple[str, str], float],
                     focal_actions: List[str],
                     context_actions: List[str],
                     epsilon: float = EPSILON_RANK) -> Dict:
    result_i = compute_best_action_by_context(
        surface_utilities, focal_actions, context_actions, "i", epsilon)
    result_j = compute_best_action_by_context(
        surface_utilities, focal_actions, context_actions, "j", epsilon)
    n_dependent = 0
    n_total = 0
    if result_i["n_unique_non_tied"] > 0:
        n_total += 1
        if result_i["context_dependent"]:
            n_dependent += 1
    if result_j["n_unique_non_tied"] > 0:
        n_total += 1
        if result_j["context_dependent"]:
            n_dependent += 1
    rate = n_dependent / n_total if n_total > 0 else None
    return {"bac_rate": rate, "n_dependent": n_dependent,
            "n_total": n_total, "i": result_i, "j": result_j}


def aggregate_source_arms(rr_c1: Optional[float], rr_c2: Optional[float],
                          bac_c1: Optional[float], bac_c2: Optional[float]
                          ) -> Dict:
    if rr_c1 is None and rr_c2 is None:
        return {"unit_rr": None, "unit_bac": None, "available": False}
    if rr_c1 is None or rr_c2 is None:
        return {"unit_rr": None, "unit_bac": None, "available": False,
                "reason": "one_source_missing"}
    unit_rr = 0.5 * rr_c1 + 0.5 * rr_c2
    unit_bac = 0.5 * bac_c1 + 0.5 * bac_c2 if bac_c1 is not None and bac_c2 is not None else None
    return {"unit_rr": unit_rr, "unit_bac": unit_bac, "available": True,
            "rr_c1": rr_c1, "rr_c2": rr_c2, "bac_c1": bac_c1, "bac_c2": bac_c2}


def compute_scenario_means(units: List[Dict],
                           scenarios: List[str]) -> Dict:
    result = {}
    for scn in scenarios:
        scn_units = [u for u in units if u.get("scenario") == scn]
        rr_vals = [u["unit_rr"] for u in scn_units
                   if u.get("unit_rr") is not None]
        bac_vals = [u["unit_bac"] for u in scn_units
                    if u.get("unit_bac") is not None]
        import numpy as np
        result[scn] = {
            "RR_mean": float(np.mean(rr_vals)) if rr_vals else None,
            "BAC_mean": float(np.mean(bac_vals)) if bac_vals else None,
            "n_valid_rr": len(rr_vals),
            "n_valid_bac": len(bac_vals),
        }
    return result


def compute_g_rev(rr_stat: float, rr_per: float, rr_mix: float) -> float:
    return rr_stat - 0.5 * (rr_per + rr_mix)


def compute_g_bac(bac_stat: float, bac_per: float, bac_mix: float) -> float:
    return bac_stat - 0.5 * (bac_per + bac_mix)


def compute_g_interaction(im_stat: float, im_per: float, im_mix: float) -> float:
    return im_stat - 0.5 * (im_per + im_mix)
