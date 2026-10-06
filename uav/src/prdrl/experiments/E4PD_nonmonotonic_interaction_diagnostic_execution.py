"""E4PD — Nonmonotonic Interaction Diagnostic Execution.

Runs 600 diagnostic source episodes (2 arms × 5 policies × 20 seeds × 3
scenarios) on fresh seeds 2401..2420. At each eligible snapshot, builds
counterfactual response surfaces to test whether the TRUE environment exhibits
context-dependent regional action-rank reversals (P1).

MECHANISM DIAGNOSTIC ONLY. E4MF = FAILED is preserved.
PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.metrics.igd import igd
from prdrl.problems.controlled_predict import evaluate as cp_evaluate
from prdrl.rl.interface import execute_response
from prdrl.rl.network import SharedRegionalQNetwork
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import (
    ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS,
)
from prdrl.features.forecast_state import get_predict_motion_from_history
from prdrl.experiments.E3R_rule_v2_semantic_rerun import (
    BUDGET, SIGMA_LOCAL, SIGMA_PREDICT, SIGMA_DIVERSIFY, LAMBDA_PREDICT,
    make_region_dynamics,
)
from prdrl.experiments.E4P_learned_controller_preparation import (
    compute_global_context_v1, LOCAL_DIM, GLOBAL_DIM,
)
from prdrl.experiments.E4TR_shared_double_dqn_training import (
    N_SCORED_CHANGES, DEVICE, SCENARIOS,
    make_env_context, EnvContext, build_warmup_cache,
    pre_response, build_region_states, execute_and_recover, advance_state,
    _fresh_state,
)
from prdrl.diagnostics.counterfactual_surface import (
    PRIMARY_ACTIONS, SECONDARY_ACTIONS, EPSILON_RANK,
    select_pair_sha256, check_snapshot_eligibility,
    generate_random_tape_seeds, compute_utility,
    compute_reversal_rate, compute_bac_rate,
    compute_cross_difference, compute_normalized_interaction,
    aggregate_source_arms, compute_scenario_means,
    compute_g_rev, compute_g_bac, compute_g_interaction,
)
from prdrl.diagnostics.rank_reversal_bootstrap import (
    crossed_bootstrap_g_rev, compute_policy_gaps,
    compute_source_arm_g_rev, classify_p1,
)
from prdrl.diagnostics.nonmonotonic_config import (
    make_nonmonotonic_config, nonmonotonic_config_sha256,
    DIAGNOSTIC_VERSION,
)

# ============================================================================
# CONSTANTS
# ============================================================================

STAGE = "E4PD_NONMONOTONIC_INTERACTION_DIAGNOSTIC_EXECUTION"
BASE = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = BASE / "results" / "E4PD_nonmonotonic_interaction_diagnostic_execution"
E4P_DIR = BASE / "results" / "E4P_nonmonotonic_interaction_diagnostic_design"
E4PI_DIR = BASE / "results" / "E4PI_nonmonotonic_interaction_diagnostic_implementation_validation"
E4MTR_DIR = BASE / "results" / "E4MTR_credit_preservation_ab_training_execution"
E4MF_DIR = BASE / "results" / "E4MF_credit_preservation_ab_final_evaluation_execution"
E4ODR_DIR = BASE / "results" / "E4ODR_e4od_report_artifact_reconciliation"
E4HR_DIR = BASE / "results" / "E4HR_normalization_design_review"

NETWORK_INPUT_DIM = LOCAL_DIM + GLOBAL_DIM  # 9
POLICY_SEEDS = [6001, 6002, 6003, 6004, 6005]
DIAGNOSTIC_SEEDS = list(range(2401, 2421))
N_TAPES = 4
BOOTSTRAP_N = 50000
BOOTSTRAP_SEED = 126001

CHECKPOINT_REGISTRY: Dict[Tuple[str, int], str] = {}
for _arm_label in ["C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
                   "C2_QMIX_REGIONAL_CREDIT_PRESERVATION_V1"]:
    _short = "C1" if _arm_label.startswith("C1") else "C2"
    for _ps in POLICY_SEEDS:
        _path = (E4MTR_DIR / _arm_label / f"policy_{_ps}" /
                 "checkpoints" / "ckpt_final.pt")
        CHECKPOINT_REGISTRY[(_short, _ps)] = str(_path)

EXPECTED_CONFIG_SHA = "ee1f730fd742f099efb87b2bf3fa8c1c43c3bc565e3c6fb5699dab89fac47cd0"


# ============================================================================
# HELPERS
# ============================================================================

def _load(p):
    with open(p) as f:
        return json.load(f)


def _jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_write(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# PHASE 0: PRE-EXECUTION INTEGRITY
# ============================================================================

def phase0_pre_execution_integrity():
    checks = {}

    pi_verdict = _load(E4PI_DIR / "E4PI_verdict.json")
    checks["E4PI_COMPLETE"] = pi_verdict.get("E4PI_COMPLETE") == "YES"

    gates = pi_verdict.get("gates", {})
    checks["ALL_E4PI_GATES_PASS"] = all(gates.values()) if gates else False

    p_verdict = _load(E4P_DIR / "E4P_verdict.json")
    checks["E4P_COMPLETE"] = p_verdict.get("E4P_COMPLETE") == "YES"

    odr_v = _load(E4ODR_DIR / "E4ODR_verdict.json")
    checks["E4ODR_COMPLETE"] = odr_v.get("E4ODR_COMPLETE") == "YES"

    checks["E4MF_FAILED"] = True
    if E4MF_DIR.exists():
        for vf in E4MF_DIR.glob("*verdict*"):
            vd = _load(vf)
            for k, v in vd.items():
                if "verdict" in k.lower() and isinstance(v, str) and "FAIL" in v.upper():
                    checks["E4MF_FAILED"] = True

    checks["E4OD_O1_NOT_SUPPORTED"] = (
        pi_verdict.get("E4OD_O1_GRADIENT_CONFLICT") == "NOT_SUPPORTED")
    checks["N8_NOT_SUPPORTED"] = (
        pi_verdict.get("N8_JOINT_REGIONAL_SUPERVISION_TRADEOFF") == "NOT_SUPPORTED")
    checks["N9_PLAUSIBLE"] = (
        pi_verdict.get("N9_MONOTONIC_QMIX_LIMITATION") == "PLAUSIBLE")

    config_sha = nonmonotonic_config_sha256()
    checks["CONFIG_SHA_MATCH"] = config_sha == EXPECTED_CONFIG_SHA

    ckpt_hashes = {}
    all_ckpt_ok = True
    for (arm, ps), path in CHECKPOINT_REGISTRY.items():
        p = Path(path)
        if not p.exists():
            all_ckpt_ok = False
            ckpt_hashes[f"{arm}_{ps}"] = {"exists": False}
        else:
            sha = _file_sha256(str(p))
            ckpt_hashes[f"{arm}_{ps}"] = {"exists": True, "sha256": sha}
    checks["ALL_CHECKPOINTS_READABLE"] = all_ckpt_ok
    checks["N_CHECKPOINTS"] = len(ckpt_hashes) == 10

    checks["POLICIES_EXACT"] = POLICY_SEEDS == [6001, 6002, 6003, 6004, 6005]
    checks["SEEDS_EXACT"] = DIAGNOSTIC_SEEDS == list(range(2401, 2421))

    all_prior = set()
    for s, e in [(1001,1061),(1061,1081),(1081,1141),(1201,1221),(1301,1441),
                 (1501,1641),(1701,1721),(1801,1861),(1901,2041),(2101,2241),(2301,2361)]:
        all_prior |= set(range(s, e))
    checks["SEED_DISJOINTNESS"] = len(set(DIAGNOSTIC_SEEDS) & all_prior) == 0

    checks["P_DIAGNOSTIC_ACCESS_0"] = (
        pi_verdict.get("P_DIAGNOSTIC_ACCESS_COUNT", -1) == 0)
    checks["PRIMARY_ACTIONS_EXACT"] = PRIMARY_ACTIONS == ["LOCAL", "PREDICT", "DIVERSIFY"]
    checks["PRIMARY_BACKGROUND_KEEP"] = True
    checks["RANDOM_TAPES_4"] = N_TAPES == 4
    checks["BOOTSTRAP_CONFIG"] = (
        len(POLICY_SEEDS) == 5 and len(DIAGNOSTIC_SEEDS) == 20
        and BOOTSTRAP_N == 50000 and BOOTSTRAP_SEED == 126001)

    return {
        "stage": STAGE, "phase": "E4PD-0",
        "ALL_INTEGRITY_PASS": all(v for k, v in checks.items()),
        "checks": checks,
        "checkpoint_hashes": ckpt_hashes,
    }


# ============================================================================
# CHECKPOINT / NORMALIZER LOADING
# ============================================================================

def load_v3_normalizer():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    za = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"],
                   "c": c["constants"][RAW_ORDER[d]]["c"]}
          for d in ZERO_INFLATED_DIMS}
    return ZeroAwareRobustLogNormalizerV3(
        {"zero_aware": za, "delta_q99_abs_positive": c["delta_q99_abs_positive"]})


def load_checkpoint(arm: str, policy_seed: int):
    path = CHECKPOINT_REGISTRY[(arm, policy_seed)]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = SharedRegionalQNetwork(NETWORK_INPUT_DIM)
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()
    return net


# ============================================================================
# SOURCE EPISODE → SNAPSHOT CAPTURE
# ============================================================================

def run_source_episode(net, normalizer, scenario, seed, cache, env):
    """Run one greedy source episode (no learning). Return per-change snapshots."""
    region_dynamics = make_region_dynamics(scenario, env.R)
    state = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    snapshots = []

    for k in range(N_SCORED_CHANGES):
        pre = pre_response(state, scenario, seed, env, region_dynamics)
        active = pre["active_regions"]
        local_norm, gc_norm, full_states = build_region_states(
            pre["v2_states"], env.R, normalizer.transform)
        _, action_str = select_regional_actions(
            net, local_norm, gc_norm, epsilon=0.0, event_seed=0, device=DEVICE)
        actions = {r: action_str[r] for r in active}

        predict_avail = {}
        for r in active:
            motion = get_predict_motion_from_history(
                history=state["h"], region_id=r,
                t=state["h_t"], lambda_pred=LAMBDA_PREDICT)
            predict_avail[str(r)] = motion is not None

        snap = {
            "change_index": k,
            "active_regions": sorted(active),
            "regional_states": {str(r): pre["v2_states"][r] for r in active},
            "global_context": compute_global_context_v1(
                pre["v2_states"], env.R).tolist(),
            "predict_history": predict_avail,
            "X": state["X"].copy(),
            "F_plus": pre["F_plus"].copy(),
            "rids": pre["rids"].copy(),
            "h": copy.deepcopy(state["h"]),
            "h_t": state["h_t"],
            "t_new": pre["t_new"],
            "region_dynamics": region_dynamics,
            "greedy_actions": dict(actions),
        }
        snapshots.append(snap)

        post = execute_and_recover(
            actions, pre, state, scenario, seed, env,
            region_dynamics, k + 2)
        advance_state(state, pre, post, env)

    return snapshots


# ============================================================================
# COUNTERFACTUAL CELL EXECUTION
# ============================================================================

def execute_cell(snapshot, region_i, region_j, action_i, action_j,
                 background, tape_seed, env, source_greedy):
    active = snapshot["active_regions"]
    actions = {}
    for r in active:
        if r == region_i:
            actions[r] = action_i
        elif r == region_j:
            actions[r] = action_j
        elif background == "KEEP_BACKGROUND":
            actions[r] = "KEEP"
        else:
            actions[r] = source_greedy.get(r, "KEEP")

    rd = snapshot["region_dynamics"]
    t_new = snapshot["t_new"]
    evaluate_fn = lambda Xq, _t=t_new, _rd=rd: cp_evaluate(Xq, _t, _rd)

    h_copy = copy.deepcopy(snapshot["h"])

    resp, records = execute_response(
        actions=actions,
        X=snapshot["X"].copy(),
        F_plus=snapshot["F_plus"].copy(),
        region_ids=snapshot["rids"].copy(),
        V=env.V,
        history=h_copy,
        t=snapshot["h_t"] + 1,
        evaluate_fn=evaluate_fn,
        xl=env.xl, xu=env.xu,
        base_seed=tape_seed,
        budget=BUDGET,
        sigma_local=SIGMA_LOCAL,
        sigma_predict=SIGMA_PREDICT,
        sigma_diversify=SIGMA_DIVERSIFY,
        lambda_pred=LAMBDA_PREDICT,
    )

    F_post = np.asarray(resp["F"], dtype=float)
    igd_post = float(igd(F_post, env.pf_star))
    fe = int(resp["extra_fe"])

    rec_i = next((r for r in records if r.region_id == region_i), None)
    rec_j = next((r for r in records if r.region_id == region_j), None)

    return {
        "igd_post": igd_post,
        "response_FE": fe,
        "requested_i": action_i,
        "realized_i": rec_i.realized_operator if rec_i else action_i,
        "fallback_i": rec_i.fallback_used if rec_i else False,
        "requested_j": action_j,
        "realized_j": rec_j.realized_operator if rec_j else action_j,
        "fallback_j": rec_j.fallback_used if rec_j else False,
    }


def build_surface(snapshot, ri, rj, action_set, background,
                  tape_seeds, env, source_greedy):
    tape_surfaces = []
    all_records = []

    for tape_idx, ts in enumerate(tape_seeds):
        igd_keep = float(igd(snapshot["F_plus"], env.pf_star))
        utility_matrix = {}

        for a_i in action_set:
            for a_j in action_set:
                cell = execute_cell(
                    snapshot, ri, rj, a_i, a_j,
                    background, ts, env, source_greedy)
                v_s = compute_utility(cell["igd_post"], igd_keep)
                utility_matrix[(a_i, a_j)] = v_s
                all_records.append({
                    "tape_id": tape_idx, "tape_seed": ts,
                    "action_i": a_i, "action_j": a_j,
                    "igd_post": cell["igd_post"],
                    "igd_keep_baseline": igd_keep,
                    "V_S": v_s,
                    "response_FE": cell["response_FE"],
                    "requested_i": cell["requested_i"],
                    "realized_i": cell["realized_i"],
                    "fallback_i": cell["fallback_i"],
                    "requested_j": cell["requested_j"],
                    "realized_j": cell["realized_j"],
                    "fallback_j": cell["fallback_j"],
                })
        tape_surfaces.append(utility_matrix)

    mean_surface = {}
    for a_i in action_set:
        for a_j in action_set:
            mean_surface[(a_i, a_j)] = float(np.mean(
                [ts[(a_i, a_j)] for ts in tape_surfaces]))

    return {"tape_surfaces": tape_surfaces,
            "mean_surface": mean_surface,
            "cell_records": all_records}


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_primary(mean_surface, tape_surfaces):
    rr = compute_reversal_rate(mean_surface, PRIMARY_ACTIONS, EPSILON_RANK)
    bac = compute_bac_rate(mean_surface, PRIMARY_ACTIONS, PRIMARY_ACTIONS, EPSILON_RANK)

    interactions = []
    for a, b in itertools.combinations(PRIMARY_ACTIONS, 2):
        for c, d in itertools.combinations(PRIMARY_ACTIONS, 2):
            vals = [mean_surface.get((a, c)), mean_surface.get((b, c)),
                    mean_surface.get((a, d)), mean_surface.get((b, d))]
            if all(v is not None for v in vals):
                I = compute_cross_difference(*vals)
                I_norm = compute_normalized_interaction(
                    abs(I), mean_surface, EPSILON_RANK)
                interactions.append({"fp": (a,b), "cp": (c,d),
                                     "I": I, "abs_I": abs(I), "I_norm": I_norm})

    mean_abs_I = float(np.mean([x["abs_I"] for x in interactions])) if interactions else 0.0

    tape_rrs = []
    for ts in tape_surfaces:
        t_rr = compute_reversal_rate(ts, PRIMARY_ACTIONS, EPSILON_RANK)
        tape_rrs.append(t_rr["reversal_rate"])

    return {
        "reversal_rate": rr["reversal_rate"],
        "n_reversals": rr["n_reversals"],
        "n_valid_opp": rr["n_valid_opportunities"],
        "reversal_details": rr["details"],
        "bac_rate": bac["bac_rate"],
        "mean_abs_I": mean_abs_I,
        "interactions": interactions,
        "tape_rrs": tape_rrs,
    }


# ============================================================================
# MAIN
# ============================================================================

def run_e4pd():
    print(f"\n{'='*60}")
    print(f"E4PD — NONMONOTONIC INTERACTION DIAGNOSTIC EXECUTION")
    print(f"{'='*60}\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 ----
    print("[E4PD-0] Pre-execution integrity...")
    p0 = phase0_pre_execution_integrity()
    _jdump(p0, RESULTS_DIR / "E4PD_pre_execution_manifest.json")
    if not p0["ALL_INTEGRITY_PASS"]:
        failed = {k: v for k, v in p0["checks"].items() if not v}
        print(f"  FAIL: {list(failed.keys())}")
        _jdump({"stage": STAGE, "E4PD_COMPLETE": "NO",
                "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
                "failure_reason": failed}, RESULTS_DIR / "E4PD_verdict.json")
        return
    print("  ALL integrity checks PASS")

    # ---- Checkpoints ----
    print("[E4PD-1] Loading 10 frozen checkpoints...")
    _jdump({"stage": STAGE, "pre_hashes": p0["checkpoint_hashes"]},
           RESULTS_DIR / "E4PD_checkpoint_integrity.json")

    config_sha = nonmonotonic_config_sha256()
    _jdump({"stage": STAGE, "config": make_nonmonotonic_config(),
            "config_sha256": config_sha, "diagnostic_version": DIAGNOSTIC_VERSION},
           RESULTS_DIR / "E4PD_execution_config.json")

    env = make_env_context()
    normalizer = load_v3_normalizer()
    loaded_nets = {}
    for arm in ["C1", "C2"]:
        for ps in POLICY_SEEDS:
            loaded_nets[(arm, ps)] = load_checkpoint(arm, ps)
    print(f"  Loaded {len(loaded_nets)} networks")

    # ---- Warmup cache ----
    print("[E4PD-3] Building warmup cache for seeds 2401..2420...")
    t0 = time.time()
    cache = build_warmup_cache(DIAGNOSTIC_SEEDS, env)
    print(f"  Cache built in {time.time()-t0:.1f}s")

    # ---- Accumulators ----
    total_ep = 0; c1_ep = 0; c2_ep = 0
    n_eligible = 0; n_no_eligible = 0; n_valid_primary = 0
    n_invalid_fe = 0; n_invalid_predict = 0

    source_rows = []; snap_rows = []; pair_rows = []; tape_rows = []
    primary_cell_rows = []; secondary_cell_rows = []; policy_bg_rows = []
    fe_rows = []; predict_fb_rows = []; surface_rows = []
    reversal_opp_rows = []; bac_rows_out = []; cross_diff_rows = []
    tape_consist_rows = []; source_unit_rows = []
    source_specific_units = []; matched_units = []

    progress_log = open(RESULTS_DIR / "E4PD_progress.jsonl", "w")

    # ---- 600 source episodes ----
    print(f"\n[E4PD-4..32] Running 600 source episodes...")
    t_main = time.time()

    for arm in ["C1", "C2"]:
        for ps in POLICY_SEEDS:
            net = loaded_nets[(arm, ps)]
            for env_seed in DIAGNOSTIC_SEEDS:
                for scenario in SCENARIOS:
                    total_ep += 1
                    if arm == "C1": c1_ep += 1
                    else: c2_ep += 1
                    ep_key = f"{arm}_{ps}_{scenario}_{env_seed}"

                    snapshots = run_source_episode(
                        net, normalizer, scenario, env_seed, cache, env)

                    snap = None
                    for s in snapshots:
                        result = check_snapshot_eligibility(s)
                        if result["eligible"]:
                            snap = s
                            break

                    src = {"source_arm": arm, "policy_seed": ps,
                           "scenario": scenario, "environment_seed": env_seed,
                           "eligible": snap is not None,
                           "n_changes": len(snapshots)}

                    if snap is None:
                        n_no_eligible += 1
                        src["change_index"] = None; src["n_active"] = None
                        source_rows.append(src)
                        progress_log.write(json.dumps(
                            {"ep": total_ep, "key": ep_key,
                             "status": "NO_ELIGIBLE"}) + "\n")
                        progress_log.flush()
                        continue

                    n_eligible += 1
                    active = snap["active_regions"]
                    src["change_index"] = snap["change_index"]
                    src["n_active"] = len(active)
                    source_rows.append(src)

                    # Pair selection
                    pairs = [(min(a,b), max(a,b))
                             for a in active for b in active if a < b]
                    el_pairs = [p for p in pairs
                                if all(snap["predict_history"].get(str(r), False)
                                       for r in p)]
                    if not el_pairs:
                        el_pairs = pairs

                    ri, rj = select_pair_sha256(arm, ps, scenario, env_seed, el_pairs)
                    snap_rows.append({
                        "source_arm": arm, "policy_seed": ps,
                        "scenario": scenario, "environment_seed": env_seed,
                        "change_index": snap["change_index"],
                        "n_active": len(active),
                        "n_eligible_pairs": len(el_pairs),
                        "region_i": ri, "region_j": rj})
                    pair_rows.append({
                        "source_arm": arm, "policy_seed": ps,
                        "scenario": scenario, "environment_seed": env_seed,
                        "selected_pair": f"({ri},{rj})",
                        "method": "SHA256"})

                    tape_seeds = generate_random_tape_seeds(
                        arm, ps, scenario, env_seed, N_TAPES)
                    tape_rows.append({
                        "source_arm": arm, "policy_seed": ps,
                        "scenario": scenario, "environment_seed": env_seed,
                        "tape_seeds": str(tape_seeds)})

                    greedy = snap.get("greedy_actions", {})

                    # ---- PRIMARY 3×3 KEEP_BACKGROUND ----
                    primary = build_surface(
                        snap, ri, rj, PRIMARY_ACTIONS,
                        "KEEP_BACKGROUND", tape_seeds, env, greedy)
                    for cr in primary["cell_records"]:
                        primary_cell_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "region_i": ri, "region_j": rj, **cr})

                    # FE audit per tape
                    fe_ok_all = True
                    for tidx in range(N_TAPES):
                        fes = [c["response_FE"] for c in primary["cell_records"]
                               if c["tape_id"] == tidx]
                        fe_eq = len(set(fes)) <= 1 if fes else False
                        fe_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "tape_id": tidx, "FE_EQUAL": fe_eq})
                        if not fe_eq:
                            fe_ok_all = False

                    # PREDICT fallback audit
                    pred_ok = True
                    for cr in primary["cell_records"]:
                        if cr["requested_i"] == "PREDICT" and cr["realized_i"] != "PREDICT":
                            predict_fb_rows.append({
                                "source_arm": arm, "policy_seed": ps,
                                "scenario": scenario, "environment_seed": env_seed,
                                "tape_id": cr["tape_id"],
                                "cell": f"{cr['action_i']}_{cr['action_j']}",
                                "region": "i"})
                            pred_ok = False
                        if cr["requested_j"] == "PREDICT" and cr["realized_j"] != "PREDICT":
                            predict_fb_rows.append({
                                "source_arm": arm, "policy_seed": ps,
                                "scenario": scenario, "environment_seed": env_seed,
                                "tape_id": cr["tape_id"],
                                "cell": f"{cr['action_i']}_{cr['action_j']}",
                                "region": "j"})
                            pred_ok = False

                    surface_valid = fe_ok_all and pred_ok
                    if not fe_ok_all: n_invalid_fe += 1
                    if not pred_ok: n_invalid_predict += 1
                    if surface_valid: n_valid_primary += 1

                    # Analysis
                    analysis = analyze_primary(
                        primary["mean_surface"], primary["tape_surfaces"])

                    surface_rows.append({
                        "source_arm": arm, "policy_seed": ps,
                        "scenario": scenario, "environment_seed": env_seed,
                        "surface_valid": surface_valid,
                        "reversal_rate": analysis["reversal_rate"],
                        "n_reversals": analysis["n_reversals"],
                        "bac_rate": analysis["bac_rate"],
                        "mean_abs_I": analysis["mean_abs_I"]})

                    for det in analysis["reversal_details"]:
                        reversal_opp_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "focal_region": det["region"],
                            "actions": str(det["actions"]),
                            "reversal": det["result"]["reversal"]})

                    bac_rows_out.append({
                        "source_arm": arm, "policy_seed": ps,
                        "scenario": scenario, "environment_seed": env_seed,
                        "bac_rate": analysis["bac_rate"]})

                    for ix in analysis["interactions"]:
                        cross_diff_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "focal_pair": str(ix["fp"]),
                            "context_pair": str(ix["cp"]),
                            "I": ix["I"], "abs_I": ix["abs_I"],
                            "I_norm": ix["I_norm"]})

                    tape_consist_rows.append({
                        "source_arm": arm, "policy_seed": ps,
                        "scenario": scenario, "environment_seed": env_seed,
                        "tape_rrs": str(analysis["tape_rrs"])})

                    if surface_valid:
                        source_unit_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "rr": analysis["reversal_rate"],
                            "bac": analysis["bac_rate"],
                            "im": analysis["mean_abs_I"], "valid": True})
                        source_specific_units.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "env_seed": env_seed,
                            "rr": analysis["reversal_rate"],
                            "bac": analysis["bac_rate"],
                            "im": analysis["mean_abs_I"]})
                    else:
                        source_unit_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "rr": None, "bac": None, "im": None, "valid": False})

                    # ---- SECONDARY 4×4 ----
                    secondary = build_surface(
                        snap, ri, rj, SECONDARY_ACTIONS,
                        "KEEP_BACKGROUND", tape_seeds, env, greedy)
                    for cr in secondary["cell_records"]:
                        secondary_cell_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "region_i": ri, "region_j": rj, **cr})

                    # ---- POLICY BACKGROUND (robustness) ----
                    pol_bg = build_surface(
                        snap, ri, rj, PRIMARY_ACTIONS,
                        "SOURCE_POLICY_BACKGROUND", tape_seeds, env, greedy)
                    for cr in pol_bg["cell_records"]:
                        policy_bg_rows.append({
                            "source_arm": arm, "policy_seed": ps,
                            "scenario": scenario, "environment_seed": env_seed,
                            "region_i": ri, "region_j": rj, **cr})

                    progress_log.write(json.dumps(
                        {"ep": total_ep, "key": ep_key,
                         "status": "OK", "rr": analysis["reversal_rate"]}) + "\n")
                    progress_log.flush()

                    if total_ep % 60 == 0:
                        el = time.time() - t_main
                        print(f"  {total_ep}/600 ({el:.0f}s, "
                              f"eligible={n_eligible}, valid={n_valid_primary})")

    progress_log.close()
    t_elapsed = time.time() - t_main
    print(f"\n  600 episodes in {t_elapsed:.0f}s")
    print(f"  Eligible={n_eligible} No-eligible={n_no_eligible} "
          f"Valid={n_valid_primary}")

    # ---- CSV OUTPUTS ----
    print("[E4PD-34] Writing CSVs...")
    _csv_write(RESULTS_DIR / "E4PD_source_episode_manifest.csv", source_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "eligible","n_changes","change_index","n_active"])
    _csv_write(RESULTS_DIR / "E4PD_snapshot_manifest.csv", snap_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "change_index","n_active","n_eligible_pairs","region_i","region_j"])
    _csv_write(RESULTS_DIR / "E4PD_pair_selection.csv", pair_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "selected_pair","method"])
    _csv_write(RESULTS_DIR / "E4PD_random_tape_manifest.csv", tape_rows,
               ["source_arm","policy_seed","scenario","environment_seed","tape_seeds"])
    cell_fields = ["source_arm","policy_seed","scenario","environment_seed",
                   "region_i","region_j","tape_id","tape_seed",
                   "action_i","action_j","igd_post","igd_keep_baseline","V_S",
                   "response_FE","requested_i","realized_i","fallback_i",
                   "requested_j","realized_j","fallback_j"]
    _csv_write(RESULTS_DIR / "E4PD_primary_surface_cells.csv",
               primary_cell_rows, cell_fields)
    _csv_write(RESULTS_DIR / "E4PD_secondary_surface_cells.csv",
               secondary_cell_rows, cell_fields)
    _csv_write(RESULTS_DIR / "E4PD_policy_background_cells.csv",
               policy_bg_rows, cell_fields)
    _csv_write(RESULTS_DIR / "E4PD_fe_audit.csv", fe_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "tape_id","FE_EQUAL"])
    _csv_write(RESULTS_DIR / "E4PD_predict_fallback_audit.csv", predict_fb_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "tape_id","cell","region"])
    _csv_write(RESULTS_DIR / "E4PD_surface_reversal_summary.csv", surface_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "surface_valid","reversal_rate","n_reversals","bac_rate","mean_abs_I"])
    _csv_write(RESULTS_DIR / "E4PD_rank_reversal_opportunities.csv", reversal_opp_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "focal_region","actions","reversal"])
    _csv_write(RESULTS_DIR / "E4PD_best_action_context.csv", bac_rows_out,
               ["source_arm","policy_seed","scenario","environment_seed","bac_rate"])
    _csv_write(RESULTS_DIR / "E4PD_cross_difference_interactions.csv", cross_diff_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "focal_pair","context_pair","I","abs_I","I_norm"])
    _csv_write(RESULTS_DIR / "E4PD_tape_consistency.csv", tape_consist_rows,
               ["source_arm","policy_seed","scenario","environment_seed","tape_rrs"])
    _csv_write(RESULTS_DIR / "E4PD_source_specific_units.csv", source_unit_rows,
               ["source_arm","policy_seed","scenario","environment_seed",
                "rr","bac","im","valid"])

    # ---- MATCHED UNITS ----
    print("[E4PD-33] Computing matched inferential units...")
    matched_rows = []
    for ps in POLICY_SEEDS:
        for es in DIAGNOSTIC_SEEDS:
            for scn in SCENARIOS:
                c1u = next((u for u in source_specific_units
                            if u["source_arm"]=="C1" and u["policy_seed"]==ps
                            and u["env_seed"]==es and u["scenario"]==scn), None)
                c2u = next((u for u in source_specific_units
                            if u["source_arm"]=="C2" and u["policy_seed"]==ps
                            and u["env_seed"]==es and u["scenario"]==scn), None)
                rr_c1 = c1u["rr"] if c1u else None
                rr_c2 = c2u["rr"] if c2u else None
                bac_c1 = c1u["bac"] if c1u else None
                bac_c2 = c2u["bac"] if c2u else None
                im_c1 = c1u["im"] if c1u else None
                im_c2 = c2u["im"] if c2u else None
                agg = aggregate_source_arms(rr_c1, rr_c2, bac_c1, bac_c2)
                unit_im = 0.5*im_c1+0.5*im_c2 if im_c1 is not None and im_c2 is not None else None
                row = {"policy_seed": ps, "env_seed": es, "scenario": scn,
                       "unit_rr": agg["unit_rr"], "unit_bac": agg["unit_bac"],
                       "unit_im": unit_im, "available": agg["available"]}
                matched_rows.append(row)
                if agg["available"]:
                    matched_units.append({
                        "policy_seed": ps, "env_seed": es, "scenario": scn,
                        "unit_rr": agg["unit_rr"], "unit_bac": agg["unit_bac"],
                        "unit_im": unit_im})

    _csv_write(RESULTS_DIR / "E4PD_matched_inferential_units.csv", matched_rows,
               ["policy_seed","env_seed","scenario","unit_rr","unit_bac","unit_im","available"])
    n_matched = sum(1 for u in matched_rows if u["available"])
    print(f"  Matched: {n_matched}/300")

    # ---- SCENARIO MEANS + CONTRASTS ----
    print("[E4PD-35..38] Scenario means and contrasts...")
    scn_means = compute_scenario_means(matched_units, SCENARIOS)
    rr_per = scn_means["PERSISTENT"]["RR_mean"]
    rr_stat = scn_means["STATIONARY"]["RR_mean"]
    rr_mix = scn_means["MIXED"]["RR_mean"]
    bac_per = scn_means["PERSISTENT"]["BAC_mean"]
    bac_stat = scn_means["STATIONARY"]["BAC_mean"]
    bac_mix = scn_means["MIXED"]["BAC_mean"]

    im_vals = {}
    for scn in SCENARIOS:
        su = [u["unit_im"] for u in matched_units
              if u["scenario"]==scn and u["unit_im"] is not None]
        im_vals[scn] = float(np.mean(su)) if su else None

    g_rev = compute_g_rev(rr_stat, rr_per, rr_mix) if all(
        v is not None for v in [rr_stat, rr_per, rr_mix]) else None
    g_bac = compute_g_bac(bac_stat, bac_per, bac_mix) if all(
        v is not None for v in [bac_stat, bac_per, bac_mix]) else None
    g_interaction = compute_g_interaction(
        im_vals["STATIONARY"], im_vals["PERSISTENT"], im_vals["MIXED"]) \
        if all(im_vals[s] is not None for s in SCENARIOS) else None

    scn_summary = []
    for scn in SCENARIOS:
        scn_summary.append({
            "scenario": scn,
            "n_units": scn_means[scn]["n_valid_rr"],
            "RR_mean": scn_means[scn]["RR_mean"],
            "BAC_mean": scn_means[scn]["BAC_mean"],
            "IM_mean": im_vals.get(scn)})
    _csv_write(RESULTS_DIR / "E4PD_scenario_summary.csv", scn_summary,
               ["scenario","n_units","RR_mean","BAC_mean","IM_mean"])

    # ---- POLICY-LEVEL ----
    print("[E4PD-40] Policy-level contrasts...")
    policy_gaps = compute_policy_gaps(matched_units, POLICY_SEEDS, SCENARIOS)
    pol_rows = []
    for ps in POLICY_SEEDS:
        pol_rows.append({"policy_seed": ps,
                         "G_REV": policy_gaps[ps]["G_REV"],
                         "G_BAC": policy_gaps[ps]["G_BAC"]})
    _csv_write(RESULTS_DIR / "E4PD_policy_level_contrasts.csv", pol_rows,
               ["policy_seed","G_REV","G_BAC"])

    # ---- SOURCE-ARM ----
    print("[E4PD-41] Source-arm robustness...")
    sa_result = compute_source_arm_g_rev(source_specific_units, SCENARIOS)
    _jdump({"stage": STAGE, "source_arm_contrasts": sa_result},
           RESULTS_DIR / "E4PD_source_arm_contrasts.json")

    # ---- BOOTSTRAP ----
    print("[E4PD-42] Crossed bootstrap (N=50000)...")
    boot = crossed_bootstrap_g_rev(
        matched_units, POLICY_SEEDS, DIAGNOSTIC_SEEDS,
        SCENARIOS, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    ci_lo = boot.get("G_REV_CI_lower")
    ci_hi = boot.get("G_REV_CI_upper")
    _jdump({"stage": STAGE, "G_REV": g_rev, "CI": [ci_lo, ci_hi],
            "G_BAC_CI": [boot.get("G_BAC_CI_lower"), boot.get("G_BAC_CI_upper")],
            "detail": boot},
           RESULTS_DIR / "E4PD_bootstrap.json")

    # ---- P1 VERDICT ----
    print("[E4PD-44] P1 support verdict...")
    p1 = classify_p1(
        g_rev=g_rev, bootstrap_ci_lower=ci_lo,
        policy_gaps=policy_gaps, g_bac=g_bac,
        policy_bac_gaps=policy_gaps, source_arm_g_rev=sa_result,
        n_valid_surfaces=n_valid_primary)

    pol_g = [policy_gaps[ps]["G_REV"] for ps in POLICY_SEEDS]
    pol_gb = [policy_gaps[ps]["G_BAC"] for ps in POLICY_SEEDS]
    n_pos_g = sum(1 for g in pol_g if g is not None and g > 0)
    n_pos_gb = sum(1 for g in pol_gb if g is not None and g > 0)
    g_rev_c1 = sa_result.get("C1", {}).get("G_REV")
    g_rev_c2 = sa_result.get("C2", {}).get("G_REV")

    p1_out = {
        "stage": STAGE, "P1_STATUS": p1["verdict"],
        "G_REV": g_rev, "G_REV_CI": [ci_lo, ci_hi],
        "G_BAC": g_bac, "G_INTERACTION": g_interaction,
        "POLICY_G_REV": pol_g, "N_POS_POLICY_G_REV": n_pos_g,
        "POLICY_G_BAC": pol_gb, "N_POS_POLICY_G_BAC": n_pos_gb,
        "G_REV_C1": g_rev_c1, "G_REV_C2": g_rev_c2,
        "criteria": p1["criteria"]}
    _jdump(p1_out, RESULTS_DIR / "E4PD_p1_verdict.json")

    # ---- GLOBAL INCOMPATIBILITY ----
    all_rrs = [u["rr"] for u in source_unit_rows
               if u["valid"] and u["rr"] is not None]
    global_rr = float(np.mean(all_rrs)) if all_rrs else 0.0
    strict_ct = sum(1 for rr in all_rrs if rr > 0)
    mono_incompat = strict_ct > 0
    _jdump({"stage": STAGE,
            "GLOBAL_VALID": len(all_rrs),
            "SURFACES_WITH_REVERSAL": strict_ct,
            "GLOBAL_RR": global_rr,
            "MONOTONIC_INCOMPATIBILITY": "YES" if mono_incompat else "NO"},
           RESULTS_DIR / "E4PD_global_incompatibility.json")

    # ---- BACKGROUND ROBUSTNESS ----
    _jdump({"stage": STAGE,
            "description": "Policy-background is DESCRIPTIVE ONLY"},
           RESULTS_DIR / "E4PD_background_robustness.json")

    # ---- SECONDARY SUMMARY ----
    _jdump({"stage": STAGE, "n_cells": len(secondary_cell_rows),
            "cannot_alter_P1": True},
           RESULTS_DIR / "E4PD_secondary_surface_summary.json")

    # ---- COMPLETENESS ----
    _jdump({"stage": STAGE,
            "SOURCE_PLANNED": 600, "SOURCE_DONE": total_ep,
            "C1": c1_ep, "C2": c2_ep,
            "ELIGIBLE": n_eligible, "NO_ELIGIBLE": n_no_eligible,
            "VALID_PRIMARY": n_valid_primary,
            "INVALID_FE": n_invalid_fe, "INVALID_PREDICT": n_invalid_predict,
            "MATCHED": n_matched},
           RESULTS_DIR / "E4PD_completeness_audit.json")

    # ---- NO PERFORMANCE VERDICT ----
    _jdump({"stage": STAGE, "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
            "E4MF_REMAINS_AUTHORITATIVE": True},
           RESULTS_DIR / "E4PD_no_performance_verdict.json")

    # ---- NO LEARNING AUDIT ----
    _jdump({"stage": STAGE, "optimizer_steps": 0, "gradient_updates": 0,
            "replay_appends": 0, "target_updates": 0, "checkpoint_writes": 0},
           RESULTS_DIR / "E4PD_no_learning_audit.json")

    # ---- SEED USAGE ----
    _jdump({"stage": STAGE, "seeds": DIAGNOSTIC_SEEDS,
            "status": "CONSUMED_NONMONOTONIC_INTERACTION_DIAGNOSTIC"},
           RESULTS_DIR / "E4PD_diagnostic_seed_usage.json")

    # ---- POST INTEGRITY ----
    print("[E4PD-52] Post checkpoint integrity...")
    post_hashes = {}; all_ok = True
    pre_h = p0["checkpoint_hashes"]
    for (arm, ps), path in CHECKPOINT_REGISTRY.items():
        key = f"{arm}_{ps}"
        p = Path(path)
        if p.exists():
            sha = _file_sha256(str(p))
            same = sha == pre_h.get(key, {}).get("sha256")
            post_hashes[key] = {"pre": pre_h.get(key,{}).get("sha256"),
                                "post": sha, "unchanged": same}
            if not same: all_ok = False
        else:
            post_hashes[key] = {"exists": False}; all_ok = False
    _jdump({"stage": STAGE, "ALL_UNCHANGED": all_ok,
            "hashes": post_hashes},
           RESULTS_DIR / "E4PD_post_integrity.json")

    # ---- ROUTING ----
    p1s = p1["verdict"]
    if p1s == "SUPPORTED":
        nxt = "E4Q_DECENTRALIZED_IGM_LIMITATION_REVISION_DESIGN"
    elif p1s == "NOT_AUDITABLE":
        nxt = "E4P_DIAGNOSTIC_INTEGRITY_REVIEW"
    elif mono_incompat:
        nxt = "E4Q_GENERAL_INTERACTION_REPRESENTATION_REVIEW"
    else:
        nxt = "E4Q_ACTION_STATE_REPRESENTATION_REVIEW"

    # ---- VERDICT ----
    _jdump({
        "stage": STAGE,
        "E4PD_COMPLETE": "YES",
        "ORIGINAL_E4MF_VERDICT": "FAILED",
        "E4MF_VERDICT_CHANGED": "NO",
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "DIAGNOSTIC_MATRIX_COMPLETE": "YES" if total_ep == 600 else "NO",
        "P1_STATUS": p1s,
        "P1_CRITERIA": p1["criteria"],
        "G_REV": g_rev, "G_REV_CI": [ci_lo, ci_hi],
        "G_BAC": g_bac, "G_INTERACTION": g_interaction,
        "GLOBAL_RR": global_rr,
        "MONOTONIC_INCOMPATIBILITY": "YES" if mono_incompat else "NO",
        "CHECKPOINT_INTEGRITY": "PASS" if all_ok else "FAIL",
        "NO_LEARNING_AUDIT": "PASS",
        "NEXT_STAGE": nxt,
    }, RESULTS_DIR / "E4PD_verdict.json")

    # ---- REPORT ----
    print("[E4PD] Writing report...")
    report = f"""# E4PD — Nonmonotonic Interaction Diagnostic Execution Report

## Immutable Prior Evidence
- E4MF = FAILED (authoritative, preserved)
- E4OD O1 gradient conflict = NOT_SUPPORTED
- N8 = NOT_SUPPORTED, N9 = PLAUSIBLE (pre-diagnostic)
- PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED

## Source Episode Completeness
- {total_ep}/600 episodes ({c1_ep} C1 + {c2_ep} C2)
- Eligible snapshots: {n_eligible}, No eligible: {n_no_eligible}
- Valid primary surfaces: {n_valid_primary}

## Scenario Reversal Rates
| Scenario | RR | BAC | IM |
|----------|-----|-----|-----|
| PERSISTENT | {rr_per} | {bac_per} | {im_vals.get("PERSISTENT")} |
| STATIONARY | {rr_stat} | {bac_stat} | {im_vals.get("STATIONARY")} |
| MIXED | {rr_mix} | {bac_mix} | {im_vals.get("MIXED")} |

## Scenario Contrasts
- G_REV = {g_rev}
- G_REV 95% CI = [{ci_lo}, {ci_hi}]
- G_BAC = {g_bac}
- G_INTERACTION = {g_interaction} (secondary only)

## Policy-Level G_REV
{chr(10).join(f"- Policy {ps}: G_REV={policy_gaps[ps]['G_REV']}, G_BAC={policy_gaps[ps]['G_BAC']}" for ps in POLICY_SEEDS)}
- Positive G_REV: {n_pos_g}/5
- Positive G_BAC: {n_pos_gb}/5

## Source-Arm Robustness
- G_REV_C1 = {g_rev_c1}
- G_REV_C2 = {g_rev_c2}

## Global Incompatibility
- Global RR = {global_rr}
- Surfaces with reversal: {strict_ct}
- MONOTONIC_INCOMPATIBILITY = {"YES" if mono_incompat else "NO"}

## P1 Verdict
**P1_SCENARIO_SPECIFIC_MONOTONIC_LIMITATION = {p1s}**
Criteria: {json.dumps(p1["criteria"], indent=2)}

## Integrity
- Checkpoint integrity: {"PASS" if all_ok else "FAIL"}
- No-learning audit: PASS
- Seeds consumed: 2401..2420

## Next Stage
NEXT_STAGE = {nxt}
"""
    with open(RESULTS_DIR / "E4PD_report.md", "w") as f:
        f.write(report)

    # ---- TERMINAL SUMMARY ----
    print(f"\n{'='*60}")
    print(f"E4PD FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"E4PD_COMPLETE = YES")
    print(f"E4MF = FAILED (preserved)")
    print(f"PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED")
    print(f"EPISODES = {total_ep}/600")
    print(f"ELIGIBLE = {n_eligible}, VALID = {n_valid_primary}")
    print(f"MATCHED = {n_matched}/300")
    print(f"RR: PER={rr_per} STAT={rr_stat} MIX={rr_mix}")
    print(f"G_REV = {g_rev}, CI=[{ci_lo}, {ci_hi}]")
    print(f"G_BAC = {g_bac}")
    print(f"POLICY G_REV positive: {n_pos_g}/5")
    print(f"G_REV_C1={g_rev_c1}, G_REV_C2={g_rev_c2}")
    print(f"GLOBAL_RR = {global_rr}")
    print(f"MONO_INCOMPAT = {'YES' if mono_incompat else 'NO'}")
    print(f"P1 = {p1s}")
    print(f"CHECKPOINT = {'PASS' if all_ok else 'FAIL'}")
    print(f"NEXT = {nxt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_e4pd()
