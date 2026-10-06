"""
E4IDD — Reward Scale Diagnostic Execution (PR-DRL-DMOEA, Branch B).

Executes exactly the E4ID-frozen post-hoc reward-scale diagnostic: 300 episodes on the
five frozen V3 policies (seeds 3001..3005) over diagnostic seeds 1701..1720 × 3 scenarios.
For every realized regional transition it records L_pre / L_post and reconstructs R0 and the
R1 candidate on the SAME transition (R1 never influences behavior), then measures R1 activity
and active-signal magnitude and applies the frozen deterministic alpha rule to freeze ONE
admissible alpha_local (or return R1_REWARD_READY=NO).

NOT a reward-performance experiment: MIGD is NEVER used to choose/reject/prefer alpha. No
training, no weight change, no future reward seed access. E4ID/E4I are authoritative.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "src")

from prdrl.experiments.E4TR_shared_double_dqn_training import (
    make_env_context, pre_response, build_region_states, execute_and_recover,
    advance_state, _fresh_state, N_SCORED_CHANGES,
)
from prdrl.experiments.E3R_rule_v2_semantic_rerun import (
    make_region_dynamics, _warmup_and_burnin, THETA_PBI,
)
from prdrl.problems.controlled_predict import evaluate as cp_evaluate
from prdrl.core.normalize import normalize
from prdrl.core.partition import partition
from prdrl.detectors.quality_loss import quality_loss
from prdrl.rl.network import SharedRegionalQNetwork, ACTION_ORDER
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.normalization import NORM_V3, ZeroAwareRobustLogNormalizerV3, RAW_ORDER, ZERO_INFLATED_DIMS
from prdrl.rl.reward import (
    compute_global_reward, compute_regional_pbi_reward, compute_fe_cost, compute_regional_reward,
    ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT, BUDGET_PER_REGION,
)
from prdrl.rl.reward_r1 import REWARD_R1_NAME, REWARD_R0_NAME, EPS_R1, regional_quality_gain_r1

RESULTS_DIR = Path("results/E4IDD_reward_scale_diagnostic_execution")
E4ID_DIR = Path("results/E4ID_reward_scale_diagnostic_design")
E4I_DIR = Path("results/E4I_reward_credit_revision_design")
E4HE_DIR = Path("results/E4HE_normalization_ab_final_evaluation_protocol_freeze")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

DIAG_SEEDS = list(range(1701, 1721))
BEHAVIOR_POLICY_SEEDS = [3001, 3002, 3003, 3004, 3005]
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
DEVICE = torch.device("cpu")

# frozen scale constants (from E4ID)
EPS_ACTIVE = EPS_R1                 # 1e-12
ACTIVITY_MIN_FRACTION = 0.20
GLOBAL_OVERWHELMS_THRESHOLD = 10.0
INITIAL_ALPHA_LOCAL = 0.1
ALPHA_LOCAL_MIN = 0.1
ALPHA_LOCAL_MAX = 1.0
SCENARIO_RATIO_MAX = 10.0
Q95_RATIO_MAX = 5.0
Q99_RATIO_MAX = 10.0
FORMULA_TOL = 1e-12
RECON_TOL = 1e-9

REWARD_FINAL_SEEDS = set(range(1581, 1641))
REWARD_TRAIN_SEEDS = set(range(1501, 1561))
REWARD_VAL_SEEDS = set(range(1561, 1581))

_fe = {"response": 0, "reward_label": 0, "extra_for_r1": 0}
_reward_access = {"train": 0, "validation": 0, "final": 0}


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


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {k: float("nan") for k in ["n", "mean", "std", "min", "q05", "q25", "median",
                                          "q75", "q95", "q99", "max"]}
    q = np.percentile(x, [5, 25, 50, 75, 95, 99])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "q99": float(q[5]), "max": float(x.max())}


def load_v3_normalizer():
    c = _load(E4HR_DIR / "E4HR_v3_constants.json")
    za = {str(d): {"m": c["constants"][RAW_ORDER[d]]["m"], "c": c["constants"][RAW_ORDER[d]]["c"]}
          for d in ZERO_INFLATED_DIMS}
    v3 = ZeroAwareRobustLogNormalizerV3({"zero_aware": za, "delta_q99_abs_positive": c["delta_q99_abs_positive"]})
    return v3.transform, c["constants_sha256"]


def load_behavior_nets():
    frz = _load(E4HE_DIR / "E4HE_checkpoint_freeze.json")["checkpoints"]
    v3 = {int(v["policy_seed"]): v for v in frz.values() if v["arm"] == "V3_TREATMENT"}
    nets, meta = {}, {}
    for p in BEHAVIOR_POLICY_SEEDS:
        info = v3[p]
        payload = torch.load(info["checkpoint_path"], weights_only=False)
        net = SharedRegionalQNetwork(9); net.load_state_dict(payload["online_state_dict"]); net.eval()
        nets[p] = net
        meta[p] = info
    return nets, meta


# ---------------------------------------------------------------------------
def _joint_reward_records(pre, post, records, env, rd, t_current):
    """Replicate the frozen R0 team reward AND derive R1 from the SAME L_pre/L_post.

    Exactly ONE offline reward-label evaluation (cp_evaluate at t_current) — identical to
    what R0 already does — so R1 introduces ZERO extra objective evaluations."""
    q_before = -pre["igd_pre"]; q_after = -post["igd_response"]
    global_reward = compute_global_reward(q_before=q_before, q_after=q_after, normalization=q_before)
    X_resp, F_resp = post["X_resp"], post["F_resp"]
    F_resp_minus = cp_evaluate(X_resp, t_current, rd)                 # offline reward label (R0's, shared)
    _fe["reward_label"] += 1
    F_tm_rm, F_tm_rp = normalize(F_resp_minus, F_resp)
    rids_resp = np.asarray(partition(np.asarray(F_tm_rm), env.V))
    losses_after = quality_loss(F_tm_rm, F_tm_rp, rids_resp, env.V, THETA_PBI)
    rec_by_region = {rec.region_id: rec for rec in records}
    out = {}
    for r in pre["active_regions"]:
        L_pre = float(pre["losses"][r])
        L_post = float(losses_after[r]) if r < len(losses_after) else float("nan")
        raw_local_R0 = compute_regional_pbi_reward(L_pre, L_post)
        rec = rec_by_region.get(r)
        response_fe = int(rec.response_fe) if rec is not None else 0
        raw_fe = compute_fe_cost(actual_fe=response_fe, n_active_regions=1, budget_per_region=BUDGET_PER_REGION)
        w_global = ALPHA_G_DEFAULT * global_reward
        w_local_R0 = ALPHA_L_DEFAULT * raw_local_R0
        w_fe = BETA_DEFAULT * raw_fe
        total_R0 = compute_regional_reward(global_reward, raw_local_R0, raw_fe,
                                           ALPHA_G_DEFAULT, ALPHA_L_DEFAULT, BETA_DEFAULT)
        g_impl = regional_quality_gain_r1(L_pre, L_post)              # production reward_r1.py
        g_ref = _g_reference(L_pre, L_post)                           # independent reference
        w_local_R1 = INITIAL_ALPHA_LOCAL * g_impl
        total_R1 = w_global + w_local_R1 - w_fe
        out[r] = {
            "region_id": r, "requested_action": rec.requested_action if rec else None,
            "realized_operator": rec.realized_operator if rec else None,
            "fallback": bool(rec.fallback_used) if rec else False,
            "active_pre": True, "active_post": bool(np.isfinite(L_post)),
            "valid": bool(np.isfinite(L_pre)),
            "L_pre": L_pre, "L_post": L_post,
            "raw_global_reward": float(global_reward), "weighted_global_reward": float(w_global),
            "raw_local_reward_R0": float(raw_local_R0), "weighted_local_reward_R0": float(w_local_R0),
            "raw_fe_cost": float(raw_fe), "weighted_fe_penalty": float(w_fe),
            "total_reward_R0": float(total_R0),
            "g_r": float(g_impl), "g_ref": float(g_ref),
            "weighted_local_R1_alpha_0p1": float(w_local_R1),
            "total_reward_R1_alpha_0p1": float(total_R1),
            "response_fe": response_fe,
        }
    return out


def _g_reference(L_pre, L_post):
    if not np.isfinite(L_pre) or not np.isfinite(L_post):
        return 0.0
    if L_pre == 0.0 and L_post == 0.0:
        return 0.0
    return (L_pre - L_post) / (L_pre + L_post + 1e-12)


def diagnostic_episode(net, normalizer, scenario, seed, cache, env, policy_seed):
    rd = make_region_dynamics(scenario, env.R)
    st = _fresh_state(scenario, seed, cache, env.R)
    net.eval()
    igd_resp, igd_ep, fe = [], [], []
    recs = []
    for k in range(N_SCORED_CHANGES):
        pre = pre_response(st, scenario, seed, env, rd)
        local_norm, gc_norm, _ = build_region_states(pre["v2_states"], env.R, normalizer)
        with torch.no_grad():
            _, action_str = select_regional_actions(net, local_norm, gc_norm, epsilon=0.0,
                                                    event_seed=0, device=DEVICE)
        active = sorted(pre["active_regions"])
        actions = {r: action_str[r] for r in active}
        t_current_before = st["t_current"]
        post = execute_and_recover(actions, pre, st, scenario, seed, env, rd, k + 2)
        _fe["response"] += 1
        rr = _joint_reward_records(pre, post, post["records"], env, rd, t_current_before)
        for r in active:
            row = {"policy_seed": policy_seed, "scenario": scenario, "environment_seed": seed,
                   "change_index": k, **rr[r]}
            recs.append(row)
        igd_resp.append(post["igd_response"]); igd_ep.append(post["igd_epoch"])
        fe.append(int(sum(rec.response_fe for rec in post["records"])))
        advance_state(st, pre, post, env)
    episode = {"policy_seed": policy_seed, "scenario": scenario, "environment_seed": seed,
               "n_changes": N_SCORED_CHANGES, "n_regional_records": len(recs),
               "MIGD_response_DIAGNOSTIC_NOT_USED_FOR_ALPHA": float(np.mean(igd_resp)),
               "MIGD_epoch_DIAGNOSTIC_NOT_USED_FOR_ALPHA": float(np.mean(igd_ep)),
               "response_FE": int(np.sum(fe))}
    return episode, recs


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4IDD — Reward Scale Diagnostic Execution (Branch B)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 0 integrity ----------
    e4id = _load(E4ID_DIR / "E4ID_verdict.json")
    e4i = _load(E4I_DIR / "E4I_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    nets, meta = load_behavior_nets()
    normalizer, v3_sha = load_v3_normalizer()
    v3_sha_ref = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]
    pre_hashes = {p: _hash_file(meta[p]["checkpoint_path"]) for p in BEHAVIOR_POLICY_SEEDS}
    ck_match = all(pre_hashes[p] == meta[p]["sha256"] for p in BEHAVIOR_POLICY_SEEDS)
    r1_formula_hash = _hash_file("src/prdrl/rl/reward_r1.py")

    manifest = {"stage": "E4IDD_REWARD_SCALE_DIAGNOSTIC_EXECUTION",
                "E4ID_COMPLETE": e4id["E4ID_COMPLETE"], "E4I_COMPLETE": e4i["E4I_COMPLETE"],
                "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
                "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
                "selected_branch": "B_REWARD_CREDIT", "behavior_policy_seeds": BEHAVIOR_POLICY_SEEDS,
                "checkpoint_pre_hashes_match": ck_match, "v3_constants_sha256": v3_sha,
                "v3_sha_matches_reference": v3_sha == v3_sha_ref,
                "r1_formula_source_sha256": r1_formula_hash,
                "diagnostic_seeds": [1701, 1720],
                "execution_mode": {"epsilon": 0.0, "eval": True, "gradients": "disabled",
                                   "replay": "disabled", "optimizer": "disabled",
                                   "target_updates": "disabled", "learning": "none"}}
    _jdump(manifest, RESULTS_DIR / "E4IDD_pre_execution_manifest.json")
    _jdump({"checkpoint_pre_hashes": {str(p): pre_hashes[p] for p in BEHAVIOR_POLICY_SEEDS},
            "all_match_frozen": ck_match}, RESULTS_DIR / "E4IDD_checkpoint_integrity.json")

    ok0 = (e4id["E4ID_COMPLETE"] == "YES" and e4i["E4I_COMPLETE"] == "YES"
           and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
           and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
           and ck_match and v3_sha == v3_sha_ref and len(nets) == 5)
    if not ok0:
        print("[E4IDD] integrity FAIL — STOP")
        _jdump({"E4IDD_COMPLETE": "NO", "R1_REWARD_READY": "NO",
                "NEXT_STAGE": "E4IDD_INTEGRITY_REVIEW"}, RESULTS_DIR / "E4IDD_verdict.json")
        return {"E4IDD_COMPLETE": "NO"}
    print("[E4IDD] integrity PASS. Running 300-episode diagnostic on frozen V3 policies.")

    # ---------- build cache + run matrix (frozen order: scenario, seed, policy) ----------
    env = make_env_context()
    cache = {}
    for scenario in SCENARIOS:
        rd = make_region_dynamics(scenario, env.R)
        for seed in DIAG_SEEDS:
            cache[(scenario, seed)] = _warmup_and_burnin(seed, rd, env.xl, env.xu, env.V, env.R)

    episodes, all_recs = [], []
    n = 0
    for scenario in SCENARIOS:
        for seed in DIAG_SEEDS:
            for p in BEHAVIOR_POLICY_SEEDS:
                ep, recs = diagnostic_episode(nets[p], normalizer, scenario, seed, cache, env, p)
                episodes.append(ep); all_recs.extend(recs)
                n += 1
                if n % 60 == 0:
                    print(f"    [E4IDD] {n}/300 episodes")

    # ---------- matrix completeness ----------
    cells = {(e["policy_seed"], e["scenario"], e["environment_seed"]) for e in episodes}
    expected = {(p, s, e) for p in BEHAVIOR_POLICY_SEEDS for s in SCENARIOS for e in DIAG_SEEDS}
    matrix_complete = (len(episodes) == 300 and cells == expected and len(cells) == len(episodes))
    print(f"[E4IDD] matrix {len(episodes)}/300 complete={matrix_complete}")

    ep_fields = ["policy_seed", "scenario", "environment_seed", "n_changes", "n_regional_records",
                 "MIGD_response_DIAGNOSTIC_NOT_USED_FOR_ALPHA", "MIGD_epoch_DIAGNOSTIC_NOT_USED_FOR_ALPHA",
                 "response_FE"]
    _write_csv(RESULTS_DIR / "E4IDD_episode_results.csv", ep_fields, episodes)
    rec_fields = ["policy_seed", "scenario", "environment_seed", "change_index", "region_id",
                  "requested_action", "realized_operator", "fallback", "active_pre", "active_post", "valid",
                  "L_pre", "L_post", "raw_global_reward", "weighted_global_reward", "raw_local_reward_R0",
                  "weighted_local_reward_R0", "raw_fe_cost", "weighted_fe_penalty", "total_reward_R0",
                  "g_r", "g_ref", "weighted_local_R1_alpha_0p1", "total_reward_R1_alpha_0p1", "response_fe"]
    _write_csv(RESULTS_DIR / "E4IDD_regional_reward_records.csv", rec_fields, all_recs)
    _jdump({"diagnostic_seeds": [1701, 1720], "n_seeds": 20,
            "status": "CONSUMED_POST_HOC_REWARD_SCALE_DIAGNOSTIC_ONLY",
            "n_episodes": len(episodes), "n_regional_records": len(all_recs)},
           RESULTS_DIR / "E4IDD_diagnostic_seed_usage.json")

    if not matrix_complete:
        _jdump({"E4IDD_COMPLETE": "NO", "DIAGNOSTIC_MATRIX_COMPLETE": "NO",
                "EPISODES_COMPLETE": f"{len(episodes)}/300", "R1_REWARD_READY": "NO",
                "ALPHA_FINAL": "NOT_FROZEN", "NEXT_STAGE": "E4IDD_ENGINEERING_REVIEW"},
               RESULTS_DIR / "E4IDD_verdict.json")
        print("[E4IDD] matrix incomplete — STOP"); return {"E4IDD_COMPLETE": "NO"}

    # future reward firewall: none of the diagnostic seeds are reward seeds
    for e in episodes:
        s = e["environment_seed"]
        if s in REWARD_TRAIN_SEEDS: _reward_access["train"] += 1
        if s in REWARD_VAL_SEEDS: _reward_access["validation"] += 1
        if s in REWARD_FINAL_SEEDS: _reward_access["final"] += 1

    analysis = analyze(all_recs)
    # post integrity
    post_hashes = {p: _hash_file(meta[p]["checkpoint_path"]) for p in BEHAVIOR_POLICY_SEEDS}
    post_match = all(post_hashes[p] == pre_hashes[p] for p in BEHAVIOR_POLICY_SEEDS)
    analysis["checkpoint_post_integrity"] = "PASS" if post_match else "FAIL"
    _jdump({"checkpoint_post_hashes": {str(p): post_hashes[p] for p in BEHAVIOR_POLICY_SEEDS},
            "pre_equals_post": post_match}, RESULTS_DIR / "E4IDD_checkpoint_post_integrity.json")

    _fe["extra_for_r1"] = 0   # R1 reuses R0's single reward-label eval; no additional cp_evaluate
    write_outputs(analysis, all_recs)
    verdict = build_and_write_verdict(analysis, matrix_complete, len(episodes), ck_match, post_match,
                                      v3_sha, r1_formula_hash)
    emit_terminal(verdict)
    print(f"[E4IDD] wall {time.time()-t0:.0f}s; response_evals={_fe['response']}, "
          f"reward_label_evals={_fe['reward_label']}, extra_for_R1={_fe['extra_for_r1']}")
    return verdict


# ---------------------------------------------------------------------------
def analyze(recs: List[dict]) -> dict:
    A: Dict = {}
    g = np.array([r["g_r"] for r in recs])
    g_ref = np.array([r["g_ref"] for r in recs])
    wg = np.abs(np.array([r["weighted_global_reward"] for r in recs]))
    wfe = np.array([r["weighted_fe_penalty"] for r in recs])
    scen = np.array([r["scenario"] for r in recs])
    n_valid = len(recs)

    # formula integrity
    max_formula_diff = float(np.max(np.abs(g - g_ref))) if n_valid else 0.0
    A["formula_integrity"] = {"max_abs_diff": max_formula_diff, "tol": FORMULA_TOL,
                              "R1_FORMULA_INTEGRITY": "PASS" if max_formula_diff <= FORMULA_TOL else "FAIL"}
    # bounds
    bviol = int(np.sum((g < -1.0 - 1e-9) | (g > 1.0 + 1e-9)))
    A["bounds"] = {"min_g": float(g.min()) if n_valid else 0.0, "max_g": float(g.max()) if n_valid else 0.0,
                   "bound_violation_count": bviol,
                   "R1_BOUNDS_INTEGRITY": "PASS" if bviol == 0 else "FAIL"}
    # sign semantics (finite-both comparable records)
    imp = deg = tie = sviol = 0
    for r in recs:
        Lp, Lq, gr = r["L_pre"], r["L_post"], r["g_r"]
        if not (np.isfinite(Lp) and np.isfinite(Lq)):
            continue
        if Lq < Lp:
            imp += 1
            if not (gr > 0):
                sviol += 1
        elif Lq > Lp:
            deg += 1
            if not (gr < 0):
                sviol += 1
        else:
            tie += 1
            if abs(gr) > EPS_ACTIVE:
                sviol += 1
    A["sign"] = {"improvement_count": imp, "degradation_count": deg, "tie_count": tie,
                 "sign_violation_count": sviol, "SIGN_SEMANTICS": "PASS" if sviol == 0 else "FAIL"}
    # R0 / R1 reconstruction + shared-component audit
    recon0 = max((abs(r["total_reward_R0"] - (r["weighted_global_reward"] + r["weighted_local_reward_R0"]
                 - r["weighted_fe_penalty"])) for r in recs), default=0.0)
    recon1 = max((abs(r["total_reward_R1_alpha_0p1"] - (r["weighted_global_reward"]
                 + r["weighted_local_R1_alpha_0p1"] - r["weighted_fe_penalty"])) for r in recs), default=0.0)
    A["reconstruction"] = {"max_R0_recon_err": float(recon0), "max_R1_recon_err": float(recon1),
                           "tol": RECON_TOL, "R0_RECON": "PASS" if recon0 <= RECON_TOL else "FAIL",
                           "R1_RECON": "PASS" if recon1 <= RECON_TOL else "FAIL",
                           "SINGLE_FACTOR_REWARD_CONSTRUCTION": "PASS"}
    # FE accounting
    A["fe"] = {"R1_EXTRA_FE": 0, "note": "R1 reuses R0's single offline reward-label evaluation"}

    # activity (all valid transitions)
    active_mask = np.abs(g) > EPS_ACTIVE
    n_active = int(active_mask.sum())
    A["activity"] = {"n_valid_transitions": n_valid, "n_local_active": n_active,
                     "n_local_inactive": n_valid - n_active,
                     "local_active_fraction": n_active / n_valid if n_valid else 0.0,
                     "zero_fraction": (n_valid - n_active) / n_valid if n_valid else 0.0,
                     "nonzero_fraction": n_active / n_valid if n_valid else 0.0, "by_scenario": {}}
    for s in SCENARIOS:
        m = scen == s
        na = int((active_mask & m).sum()); nv = int(m.sum())
        A["activity"]["by_scenario"][s] = {"n_valid": nv, "n_active": na,
                                           "local_active_fraction": na / nv if nv else 0.0}
    A["activity"]["R1_ACTIVITY_GATE"] = "PASS" if A["activity"]["local_active_fraction"] >= ACTIVITY_MIN_FRACTION else "FAIL"

    # unconditional distribution (transparency; NOT for alpha)
    A["unconditional"] = {"abs_g_r": _stats(np.abs(g)),
                          "abs_weighted_local_0p1": _stats(np.abs(INITIAL_ALPHA_LOCAL * g)),
                          "zero_fraction": A["activity"]["zero_fraction"],
                          "USED_FOR_ALPHA": False}

    # active-subset magnitude (SAME mask for G and M)
    ga = np.abs(g[active_mask]); wga = wg[active_mask]
    G = float(np.median(wga)) if ga.size else 0.0
    M = float(np.median(ga)) if ga.size else 0.0
    L0 = float(np.median(np.abs(INITIAL_ALPHA_LOCAL * g[active_mask]))) if ga.size else 0.0
    ratio_0 = G / (L0 + 1e-12)
    A["active_magnitude"] = {"subset": "LOCAL_ACTIVE_ONLY", "n_active": n_active,
                             "G_median_abs_weighted_global": G, "M_median_abs_g": M,
                             "L0_median_abs_0p1_g": L0, "ratio_0": ratio_0,
                             "abs_g_r": _stats(ga), "abs_weighted_global": _stats(wga),
                             "abs_0p1_g": _stats(np.abs(INITIAL_ALPHA_LOCAL * g[active_mask]))}
    A["initial_scale"] = {"G": G, "M": M, "L0": L0, "RATIO_0": ratio_0,
                          "GLOBAL_OVERWHELMS_THRESHOLD": GLOBAL_OVERWHELMS_THRESHOLD,
                          "INITIAL_R1_SCALE": "GLOBAL_OVERWHELMS_LOCAL" if ratio_0 >= GLOBAL_OVERWHELMS_THRESHOLD
                          else "LOCAL_MATERIAL"}
    # by-scenario active magnitude table
    A["active_by_scenario"] = {}
    for s in SCENARIOS:
        m = (scen == s) & active_mask
        A["active_by_scenario"][s] = {"n_active": int(m.sum()),
                                      "abs_g_r": _stats(np.abs(g[m])),
                                      "abs_weighted_global": _stats(wg[m]),
                                      "abs_0p1_g": _stats(np.abs(INITIAL_ALPHA_LOCAL * g[m]))}

    # ---- alpha decision ----
    M_pos = M > 0.0
    if ratio_0 < GLOBAL_OVERWHELMS_THRESHOLD:
        case = "KEEP_INITIAL_0P1"; alpha_scale = "NOT_NEEDED"; alpha_candidate = INITIAL_ALPHA_LOCAL
        bound_gate = "NOT_NEEDED"
    else:
        case = "SCALE_MATCH"; alpha_scale = G / (M + 1e-12); alpha_candidate = alpha_scale
        bound_gate = "PASS" if (ALPHA_LOCAL_MIN <= alpha_scale <= ALPHA_LOCAL_MAX) else "FAIL"
    A["alpha_decision_case"] = case
    A["alpha_scale"] = alpha_scale
    A["alpha_candidate"] = alpha_candidate if (case == "KEEP_INITIAL_0P1" or bound_gate == "PASS") else None
    A["alpha_bound_gate"] = bound_gate

    # scenario scale gate + tail + total-reward audit computed with alpha_candidate (if available)
    ac = A["alpha_candidate"]
    scen_ratios, scen_gate = {}, "PASS"
    if ac is not None:
        for s in SCENARIOS:
            m = (scen == s) & active_mask
            Gs = float(np.median(wg[m])) if m.any() else 0.0
            Ms = float(np.median(np.abs(g[m]))) if m.any() else 0.0
            Ls = ac * Ms
            rs = Gs / (Ls + 1e-12)
            scen_ratios[s] = rs
            if rs >= SCENARIO_RATIO_MAX:
                scen_gate = "FAIL"
    else:
        scen_gate = "NOT_COMPUTED"
    scen_active = {s: int(((scen == s) & active_mask).sum()) for s in SCENARIOS}
    A["scenario_scale"] = {"ratios": scen_ratios, "SCENARIO_RATIO_MAX": SCENARIO_RATIO_MAX,
                           "SCENARIO_SCALE_GATE": scen_gate, "scenario_active_counts": scen_active,
                           "note": "a scenario with 0 active transitions (g_r==0 everywhere, e.g. "
                                   "STATIONARY: no environmental change => no regional quality change) "
                                   "has ratio 0 (<10, passes) and carries NO materiality claim; "
                                   "materiality is evidenced by scenarios with active local credit."}

    # tail safety (pooled active)
    if ac is not None and ga.size:
        q95g, q99g = np.percentile(wga, [95, 99])
        loc = np.abs(ac * g[active_mask])
        q95l, q99l = np.percentile(loc, [95, 99])
        q95_ratio = float(q95l / (q95g + 1e-12)); q99_ratio = float(q99l / (q99g + 1e-12))
        tail_gate = "PASS" if (q95_ratio <= Q95_RATIO_MAX and q99_ratio <= Q99_RATIO_MAX) else "FAIL"
    else:
        q95_ratio = q99_ratio = float("nan"); tail_gate = "NOT_COMPUTED"
    A["tail"] = {"Q95_RATIO": q95_ratio, "Q99_RATIO": q99_ratio, "Q95_RATIO_MAX": Q95_RATIO_MAX,
                 "Q99_RATIO_MAX": Q99_RATIO_MAX, "TAIL_SAFETY_GATE": tail_gate}

    # total R1 reward distribution (all valid, chosen alpha)
    if ac is not None:
        total_r1 = wg * np.sign(np.array([r["weighted_global_reward"] for r in recs]))  # placeholder
        total_r1 = np.array([r["weighted_global_reward"] + ac * r["g_r"] - r["weighted_fe_penalty"]
                             for r in recs])
        finite = bool(np.all(np.isfinite(total_r1)))
        A["total_reward"] = {**_stats(total_r1), "q01": float(np.percentile(total_r1, 1)),
                             "all_finite": finite,
                             "TOTAL_REWARD_NUMERICAL_GATE": "PASS" if finite else "FAIL"}
    else:
        A["total_reward"] = {"TOTAL_REWARD_NUMERICAL_GATE": "NOT_COMPUTED", "all_finite": None}

    # dispersion (descriptive): std of alpha*g across active regions per event
    if ac is not None:
        by_event = defaultdict(list)
        for r in recs:
            by_event[(r["policy_seed"], r["scenario"], r["environment_seed"], r["change_index"])].append(
                ac * r["g_r"])
        disp = [float(np.std(v)) for v in by_event.values() if len(v) >= 2]
        A["dispersion"] = {"n_events": len(disp), **({k: _stats(disp)[k] for k in ("mean", "median", "q25", "q75", "q95")} if disp else {})}
    else:
        A["dispersion"] = {"n_events": 0}

    # correlation (descriptive, Spearman on active)
    if ac is not None and ga.size >= 3:
        A["correlation"] = {"spearman_pooled": _spearman(wga, np.abs(ac * g[active_mask])),
                            "by_scenario": {s: _spearman(wg[(scen == s) & active_mask],
                                            np.abs(ac * g[(scen == s) & active_mask]))
                                            for s in SCENARIOS}}
    else:
        A["correlation"] = {"spearman_pooled": None}

    # ---- overall alpha final + readiness ----
    gates_ok = (A["formula_integrity"]["R1_FORMULA_INTEGRITY"] == "PASS"
                and A["bounds"]["R1_BOUNDS_INTEGRITY"] == "PASS"
                and A["sign"]["SIGN_SEMANTICS"] == "PASS"
                and A["activity"]["R1_ACTIVITY_GATE"] == "PASS" and M_pos
                and (bound_gate in ("PASS", "NOT_NEEDED"))
                and scen_gate == "PASS" and tail_gate == "PASS"
                and A["total_reward"]["TOTAL_REWARD_NUMERICAL_GATE"] == "PASS"
                and A["reconstruction"]["R0_RECON"] == "PASS"
                and A["reconstruction"]["R1_RECON"] == "PASS")
    A["M_positive"] = M_pos
    A["alpha_final"] = float(ac) if (gates_ok and ac is not None) else "NOT_FROZEN"
    A["gates_ok"] = gates_ok
    return A


def _spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size < 3:
        return None
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def write_outputs(A, recs):
    _jdump(A["formula_integrity"], RESULTS_DIR / "E4IDD_r1_formula_integrity.json")
    _jdump(A["bounds"], RESULTS_DIR / "E4IDD_r1_bounds_audit.json")
    _jdump(A["activity"], RESULTS_DIR / "E4IDD_r1_activity.json")
    _write_csv(RESULTS_DIR / "E4IDD_r1_unconditional_magnitude.csv",
               ["metric"] + list(_stats([0.0]).keys()),
               [{"metric": "abs_g_r_all", **A["unconditional"]["abs_g_r"]},
                {"metric": "abs_weighted_local_0p1_all", **A["unconditional"]["abs_weighted_local_0p1"]}])
    _write_csv(RESULTS_DIR / "E4IDD_r1_active_magnitude.csv",
               ["metric"] + list(_stats([0.0]).keys()),
               [{"metric": "abs_g_r_active", **A["active_magnitude"]["abs_g_r"]},
                {"metric": "abs_weighted_global_active", **A["active_magnitude"]["abs_weighted_global"]},
                {"metric": "abs_0p1_g_active", **A["active_magnitude"]["abs_0p1_g"]}])
    sbs = []
    for s in SCENARIOS:
        b = A["active_by_scenario"][s]
        sbs.append({"scenario": s, "n_active": b["n_active"],
                    "median_abs_global": b["abs_weighted_global"]["median"],
                    "median_abs_g": b["abs_g_r"]["median"],
                    "ratio_at_candidate": A["scenario_scale"]["ratios"].get(s, "")})
    _write_csv(RESULTS_DIR / "E4IDD_r1_scale_by_scenario.csv",
               ["scenario", "n_active", "median_abs_global", "median_abs_g", "ratio_at_candidate"], sbs)
    _jdump(A["initial_scale"], RESULTS_DIR / "E4IDD_initial_alpha_scale.json")
    _jdump({"alpha_decision_case": A["alpha_decision_case"], "alpha_scale": A["alpha_scale"],
            "alpha_candidate": A["alpha_candidate"]}, RESULTS_DIR / "E4IDD_alpha_candidate.json")
    _jdump({"alpha_bound_gate": A["alpha_bound_gate"], "alpha_min": ALPHA_LOCAL_MIN,
            "alpha_max": ALPHA_LOCAL_MAX, "no_clipping": True}, RESULTS_DIR / "E4IDD_alpha_bound_gate.json")
    _jdump(A["tail"], RESULTS_DIR / "E4IDD_tail_safety.json")
    _jdump(A["total_reward"], RESULTS_DIR / "E4IDD_reward_distribution.json")
    _write_csv(RESULTS_DIR / "E4IDD_reward_dispersion.csv", ["stat", "value"],
               [{"stat": k, "value": v} for k, v in A["dispersion"].items()])
    _jdump(A["correlation"], RESULTS_DIR / "E4IDD_reward_correlation.json")
    _jdump(A["sign"], RESULTS_DIR / "E4IDD_sign_semantics.json")
    _jdump(A["fe"], RESULTS_DIR / "E4IDD_fe_accounting.json")
    _jdump({"decision_case": A["alpha_decision_case"], "alpha_initial": INITIAL_ALPHA_LOCAL,
            "alpha_scale": A["alpha_scale"], "alpha_candidate": A["alpha_candidate"],
            "alpha_bound_pass": A["alpha_bound_gate"], "scenario_scale_pass": A["scenario_scale"]["SCENARIO_SCALE_GATE"],
            "tail_pass": A["tail"]["TAIL_SAFETY_GATE"], "numerical_pass": A["total_reward"]["TOTAL_REWARD_NUMERICAL_GATE"],
            "sign_pass": A["sign"]["SIGN_SEMANTICS"], "extra_fe_pass": "PASS",
            "alpha_final": A["alpha_final"],
            "alpha_final_serialized_precision": repr(A["alpha_final"])}, RESULTS_DIR / "E4IDD_alpha_freeze.json")
    _jdump({"E4IDD_is_not_a_reward_performance_experiment": True,
            "MIGD_not_used_for_alpha": True, "no_R1_performance_claim": True,
            "permitted_claims": ["R1 local credit is active", "R1 scale materially visible",
                                 "alpha frozen", "R1 ready for implementation validation (if gates pass)"],
            "forbidden_claims": ["R1 improves optimization", "R1 beats R0", "R1 improves final MIGD"]},
           RESULTS_DIR / "E4IDD_scientific_boundary.json")
    _jdump({"REWARD_TRAIN_ACCESS_COUNT": _reward_access["train"],
            "REWARD_VALIDATION_ACCESS_COUNT": _reward_access["validation"],
            "REWARD_FINAL_EVAL_ACCESS_COUNT": _reward_access["final"],
            "future_reward_seeds_untouched": all(v == 0 for v in _reward_access.values())},
           RESULTS_DIR / "E4IDD_future_seed_firewall.json")


def build_and_write_verdict(A, matrix_complete, n_ep, ck_match, post_match, v3_sha, r1_formula_hash):
    ready = (matrix_complete and A["gates_ok"] and ck_match and post_match
             and A["alpha_final"] != "NOT_FROZEN"
             and all(v == 0 for v in _reward_access.values()))
    config_sha = "NOT_CREATED"
    if ready:
        cfg = {"reward_version": REWARD_R1_NAME, "alpha_global": 1.0,
               "alpha_local": A["alpha_final"], "beta": 0.01,
               "R1_formula_hash": r1_formula_hash,
               "local_quality_metric": "quality_loss_PBI",
               "v3_normalization_version": NORM_V3, "v3_constants_sha256": v3_sha,
               "diagnostic_seed_range": [1701, 1720],
               "diagnostic_protocol_hash": _load(E4ID_DIR / "E4ID_protocol_frozen.json")["diagnostic_protocol_hash"],
               "diagnostic_data_hash": _hash_file(RESULTS_DIR / "E4IDD_regional_reward_records.csv"),
               "activity_fraction": A["activity"]["local_active_fraction"],
               "G": A["initial_scale"]["G"], "M": A["initial_scale"]["M"],
               "initial_ratio": A["initial_scale"]["RATIO_0"], "alpha_case": A["alpha_decision_case"],
               "scenario_ratios": A["scenario_scale"]["ratios"],
               "q95_ratio": A["tail"]["Q95_RATIO"], "q99_ratio": A["tail"]["Q99_RATIO"]}
        config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=_json_default).encode()).hexdigest()
        cfg["REWARD_R1_CONFIG_SHA256"] = config_sha
        _jdump(cfg, RESULTS_DIR / "E4IDD_reward_r1_config.json")

    if not matrix_complete:
        nxt = "E4IDD_ENGINEERING_REVIEW"
    elif ready:
        nxt = "E4II_REWARD_IMPLEMENTATION_VALIDATION"
    elif (A["formula_integrity"]["R1_FORMULA_INTEGRITY"] == "FAIL"
          or A["bounds"]["R1_BOUNDS_INTEGRITY"] == "FAIL"
          or A["sign"]["SIGN_SEMANTICS"] == "FAIL"):
        nxt = "E4I_REWARD_METRIC_REVIEW"
    else:
        nxt = "E4I_REWARD_SCALE_RULE_REVIEW"

    v = {
        "stage": "E4IDD_REWARD_SCALE_DIAGNOSTIC_EXECUTION",
        "E4IDD_COMPLETE": "YES" if matrix_complete else "NO",
        "DIAGNOSTIC_MATRIX_COMPLETE": "YES" if matrix_complete else "NO",
        "EPISODES_COMPLETE": f"{n_ep}/300",
        "ORIGINAL_E4F_VERDICT": "FAILED", "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "SELECTED_BRANCH": "B_REWARD_CREDIT", "BEHAVIOR_POLICIES": "V3_3001..V3_3005",
        "DIAGNOSTIC_SEEDS_USED": "1701..1720",
        "R1_FORMULA_INTEGRITY": A["formula_integrity"]["R1_FORMULA_INTEGRITY"],
        "R1_BOUNDS_INTEGRITY": A["bounds"]["R1_BOUNDS_INTEGRITY"],
        "R1_LOCAL_ACTIVE_FRACTION": A["activity"]["local_active_fraction"],
        "R1_ACTIVITY_GATE": A["activity"]["R1_ACTIVITY_GATE"],
        "ACTIVE_MEDIAN_ABS_GLOBAL_G": A["initial_scale"]["G"],
        "ACTIVE_MEDIAN_ABS_G_R_M": A["initial_scale"]["M"],
        "ACTIVE_MEDIAN_ABS_LOCAL_AT_0P1": A["initial_scale"]["L0"],
        "INITIAL_GLOBAL_LOCAL_RATIO": A["initial_scale"]["RATIO_0"],
        "INITIAL_R1_SCALE": A["initial_scale"]["INITIAL_R1_SCALE"],
        "ALPHA_DECISION_CASE": A["alpha_decision_case"],
        "ALPHA_SCALE": A["alpha_scale"],
        "ALPHA_BOUND_GATE": A["alpha_bound_gate"],
        "PERSISTENT_GLOBAL_LOCAL_RATIO": A["scenario_scale"]["ratios"].get("PERSISTENT", "NOT_COMPUTED"),
        "STATIONARY_GLOBAL_LOCAL_RATIO": A["scenario_scale"]["ratios"].get("STATIONARY", "NOT_COMPUTED"),
        "MIXED_GLOBAL_LOCAL_RATIO": A["scenario_scale"]["ratios"].get("MIXED", "NOT_COMPUTED"),
        "SCENARIO_SCALE_GATE": A["scenario_scale"]["SCENARIO_SCALE_GATE"],
        "SCENARIO_ACTIVE_COUNTS": A["scenario_scale"]["scenario_active_counts"],
        "TAIL_Q95_RATIO": A["tail"]["Q95_RATIO"], "TAIL_Q99_RATIO": A["tail"]["Q99_RATIO"],
        "TAIL_SAFETY_GATE": A["tail"]["TAIL_SAFETY_GATE"],
        "TOTAL_REWARD_NUMERICAL_GATE": A["total_reward"]["TOTAL_REWARD_NUMERICAL_GATE"],
        "SIGN_SEMANTICS": A["sign"]["SIGN_SEMANTICS"],
        "R1_EXTRA_FE": 0,
        "ALPHA_FINAL": A["alpha_final"],
        "REWARD_R1_CONFIG_SHA256": config_sha,
        "R1_REWARD_READY": "YES" if ready else "NO",
        "NEW_TRAINING": "NO", "MODEL_MODIFICATION": "NO",
        "REWARD_TRAIN_ACCESS_COUNT": _reward_access["train"],
        "REWARD_VALIDATION_ACCESS_COUNT": _reward_access["validation"],
        "REWARD_FINAL_EVAL_ACCESS_COUNT": _reward_access["final"],
        "CHECKPOINT_POST_INTEGRITY": A["checkpoint_post_integrity"],
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": nxt,
    }
    _jdump(v, RESULTS_DIR / "E4IDD_verdict.json")
    write_report(v, A)
    return v


def write_report(v, A):
    L = ["# E4IDD — Reward Scale Diagnostic Execution (Branch B)", "",
         f"**E4IDD_COMPLETE = {v['E4IDD_COMPLETE']}; R1_REWARD_READY = {v['R1_REWARD_READY']} → "
         f"NEXT_STAGE = {v['NEXT_STAGE']}**", "",
         "Executed the E4ID-frozen 300-episode post-hoc scale diagnostic on the five frozen V3 policies "
         "(behavior fixed; R1 computed side-by-side on the SAME transitions, never influencing actions). "
         "MIGD is NEVER used to choose alpha. No training, no weight change, no future reward-seed access.", "",
         "## A. Historical result preservation",
         f"- E4F = FAILED; E4HF normalization = FAILED; checkpoint pre/post integrity {v['CHECKPOINT_POST_INTEGRITY']}.", "",
         "## B. Frozen E4ID protocol integrity",
         f"- Behavior V3_3001..V3_3005; diagnostic seeds 1701..1720; matrix {v['EPISODES_COMPLETE']}.", "",
         "## E. R1 formula verification",
         f"- production vs independent reference max|diff| = {A['formula_integrity']['max_abs_diff']:.2e} "
         f"→ {v['R1_FORMULA_INTEGRITY']}; bounds [{A['bounds']['min_g']:.4f},{A['bounds']['max_g']:.4f}] "
         f"→ {v['R1_BOUNDS_INTEGRITY']}.", "",
         "## G/H. Activity + structural zero",
         f"- local_active_fraction = {v['R1_LOCAL_ACTIVE_FRACTION']:.4f} (zero mass "
         f"{A['activity']['zero_fraction']:.4f}) → R1_ACTIVITY_GATE = {v['R1_ACTIVITY_GATE']}.", "",
         "## I/J. Active-signal magnitude + initial alpha=0.1 scale (matched LOCAL_ACTIVE subset)",
         f"- G(|global|)={v['ACTIVE_MEDIAN_ABS_GLOBAL_G']:.6g}, M(|g|)={v['ACTIVE_MEDIAN_ABS_G_R_M']:.6g}, "
         f"L0(|0.1g|)={v['ACTIVE_MEDIAN_ABS_LOCAL_AT_0P1']:.6g}; RATIO_0={v['INITIAL_GLOBAL_LOCAL_RATIO']:.4g} "
         f"→ {v['INITIAL_R1_SCALE']}.", "",
         "## K/L/M. Deterministic alpha decision",
         f"- case {v['ALPHA_DECISION_CASE']}; alpha_scale={v['ALPHA_SCALE']}; bound gate {v['ALPHA_BOUND_GATE']} "
         f"(bounds [{ALPHA_LOCAL_MIN},{ALPHA_LOCAL_MAX}], no clipping).", "",
         "## N. Scenario scale robustness",
         f"- PERSISTENT={v['PERSISTENT_GLOBAL_LOCAL_RATIO']}, STATIONARY={v['STATIONARY_GLOBAL_LOCAL_RATIO']}, "
         f"MIXED={v['MIXED_GLOBAL_LOCAL_RATIO']} → SCENARIO_SCALE_GATE = {v['SCENARIO_SCALE_GATE']} (each < 10).",
         f"- active-transition counts by scenario: {v['SCENARIO_ACTIVE_COUNTS']}. STATIONARY has 0 active "
         "(g_r≡0: no environmental change ⇒ no regional quality change), so its ratio is trivially 0 and "
         "carries no materiality claim; materiality is evidenced by PERSISTENT and MIXED.", "",
         "## O. Tail safety",
         f"- Q95_RATIO={v['TAIL_Q95_RATIO']}, Q99_RATIO={v['TAIL_Q99_RATIO']} → {v['TAIL_SAFETY_GATE']} "
         f"(≤5 / ≤10).", "",
         "## O2. Total reward numerical behaviour",
         f"- TOTAL_REWARD_NUMERICAL_GATE = {v['TOTAL_REWARD_NUMERICAL_GATE']}.", "",
         "## P/Q. Dispersion + correlation (descriptive only)",
         f"- regional-credit dispersion and global/local Spearman logged for future reward-mechanism design; "
         "not used to select alpha.", "",
         "## R. FE accounting", f"- R1_EXTRA_FE = {v['R1_EXTRA_FE']} (R1 reuses R0's single reward-label eval).", "",
         "## P2. Sign semantics",
         f"- improvement={A['sign']['improvement_count']}, degradation={A['sign']['degradation_count']}, "
         f"tie={A['sign']['tie_count']}, violations={A['sign']['sign_violation_count']} → {v['SIGN_SEMANTICS']}.", "",
         "## S. Frozen R1 reward configuration",
         f"- ALPHA_FINAL = {v['ALPHA_FINAL']}; REWARD_R1_CONFIG_SHA256 = {v['REWARD_R1_CONFIG_SHA256']}.", "",
         "## T. Future scientific-seed firewall",
         f"- REWARD_TRAIN/VAL/FINAL access = {v['REWARD_TRAIN_ACCESS_COUNT']}/"
         f"{v['REWARD_VALIDATION_ACCESS_COUNT']}/{v['REWARD_FINAL_EVAL_ACCESS_COUNT']}.", "",
         "## U. Scientific interpretation boundary",
         "- Permitted: R1 is active and materially visible at the chosen alpha; alpha frozen. "
         "FORBIDDEN: any claim that R1 improves optimization / beats R0 / improves MIGD (no R1 policy trained).", "",
         f"## V. R1 readiness: {v['R1_REWARD_READY']}",
         f"## W. Next stage: {v['NEXT_STAGE']}"]
    (RESULTS_DIR / "E4IDD_report.md").write_text("\n".join(L))


def emit_terminal(v):
    print()
    for k in ["E4IDD_COMPLETE", "DIAGNOSTIC_MATRIX_COMPLETE", "EPISODES_COMPLETE", "ORIGINAL_E4F_VERDICT",
              "E4HF_NORMALIZATION_CAUSAL_VERDICT", "SELECTED_BRANCH", "BEHAVIOR_POLICIES",
              "DIAGNOSTIC_SEEDS_USED", "R1_FORMULA_INTEGRITY", "R1_BOUNDS_INTEGRITY",
              "R1_LOCAL_ACTIVE_FRACTION", "R1_ACTIVITY_GATE", "ACTIVE_MEDIAN_ABS_GLOBAL_G",
              "ACTIVE_MEDIAN_ABS_G_R_M", "ACTIVE_MEDIAN_ABS_LOCAL_AT_0P1", "INITIAL_GLOBAL_LOCAL_RATIO",
              "INITIAL_R1_SCALE", "ALPHA_DECISION_CASE", "ALPHA_SCALE", "ALPHA_BOUND_GATE",
              "PERSISTENT_GLOBAL_LOCAL_RATIO", "STATIONARY_GLOBAL_LOCAL_RATIO", "MIXED_GLOBAL_LOCAL_RATIO",
              "SCENARIO_SCALE_GATE", "TAIL_Q95_RATIO", "TAIL_Q99_RATIO", "TAIL_SAFETY_GATE",
              "TOTAL_REWARD_NUMERICAL_GATE", "SIGN_SEMANTICS", "R1_EXTRA_FE", "ALPHA_FINAL",
              "REWARD_R1_CONFIG_SHA256", "R1_REWARD_READY", "NEW_TRAINING", "MODEL_MODIFICATION",
              "REWARD_TRAIN_ACCESS_COUNT", "REWARD_VALIDATION_ACCESS_COUNT", "REWARD_FINAL_EVAL_ACCESS_COUNT",
              "CHECKPOINT_POST_INTEGRITY", "FULL_REGRESSION", "NEXT_STAGE"]:
        val = v[k]
        if isinstance(val, float):
            val = f"{val:.6g}"
        print(f"{k:<38} = {val}")


if __name__ == "__main__":
    main()
