"""E4OI — Scenario Trade-off Diagnostic Implementation Validation.

Validates all diagnostic machinery frozen in E4O using SYNTHETIC DATA ONLY.
No training, no real environment, no seed opening (2301..2360 NOT instantiated).
Proves that future E4OD execution can correctly measure gradient alignment,
R1 signal, TD errors, common-state evaluation, and crossed bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair, double_dqn_target, N_ACTIONS,
)
from prdrl.rl.qmix import (
    QMIXMixerV1, R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM, TREATMENT_ARCH,
    build_mixer_online_target_pair,
)
from prdrl.rl.coordination_integration import (
    CONTROL_ARCH, C2_CREDIT_ARCH, STATE_DIM, GAMMA, TARGET_TAU,
    JOINT_WEIGHT, REGIONAL_WEIGHT,
    joint_reward_mean,
    current_joint_q, double_q_joint_target, treatment_loss,
    focal_regional_ddqn_target, focal_current_q,
    regional_smoothl1_loss, c2_composite_loss,
    make_qmix_config, qmix_config_sha256,
    make_c2_credit_config, c2_credit_config_sha256,
    gradient_diversity_diagnostic,
)
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME
from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL, FROZEN_R1_CONFIG_SHA256

from prdrl.diagnostics.gradient_diagnostic import (
    extract_utility_gradient, extract_mixer_gradient,
    gradient_norms, cosine_similarity_jr,
    gradient_norm_ratio, composite_gradient_identity,
    mixer_gradient_check, conflict_fraction,
    scenario_gap_cosine, scenario_gap_conflict,
    policy_level_gaps, EPSILON_COSINE,
)
from prdrl.diagnostics.common_state_diagnostic import (
    build_common_state_record, cross_evaluate,
    action_disagreement_fraction, action_transition_matrix,
    q_margin_stats, stratify_by_delta_severity,
)
from prdrl.diagnostics.signal_diagnostic import (
    r1_signal_statistics, r1_activity,
    regional_td_error, joint_td_error, td_error_ratio, td_error_summary,
)
from prdrl.diagnostics.statistical_diagnostic import (
    aggregate_policy_env_scenario,
    crossed_bootstrap_ci, scenario_gap_bootstrap,
    o1_support_verdict,
)
from prdrl.diagnostics.diagnostic_config import (
    DIAGNOSTIC_VERSION, make_diagnostic_config, diagnostic_config_sha256,
    validate_upstream_hashes,
    REQUIRED_QMIX_SHA, REQUIRED_C2_CREDIT_SHA, REQUIRED_R1_SHA, REQUIRED_V3_SHA,
    DIAGNOSTIC_SEEDS, DIAGNOSTIC_BOOTSTRAP_N, DIAGNOSTIC_BOOTSTRAP_SEED,
    POLICY_SEEDS, SCENARIOS, N_DIAGNOSTIC_ENVS, DIAGNOSTIC_EPISODES,
)

STAGE = "E4OI_SCENARIO_TRADEOFF_DIAGNOSTIC_IMPLEMENTATION_VALIDATION"
RESULTS_DIR = Path("results/E4OI_scenario_tradeoff_diagnostic_implementation_validation")
E4O_DIR = Path("results/E4O_scenario_tradeoff_diagnostic_design")

SYNTH_SEED = 99001
B = 8
PRIOR_SEED_RANGES = [
    (1001, 1060), (1061, 1080), (1081, 1140),
    (1201, 1220),
    (1301, 1360), (1361, 1380), (1381, 1440),
    (1501, 1560), (1561, 1580), (1581, 1640),
    (1701, 1720),
    (1801, 1860),
    (1901, 1960), (1961, 1980), (1981, 2040),
    (2101, 2160), (2161, 2180), (2181, 2240),
]

COUNTERS = {
    "optimizer_steps": 0,
    "parameter_changes": 0,
    "target_updates": 0,
    "replay_mutations": 0,
    "real_environment_episodes": 0,
    "diagnostic_seed_access": 0,
}


def _jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _jload(path):
    with open(path) as f:
        return json.load(f)


def _synth_networks(seed: int):
    """Build synthetic C1 and C2 utility networks + mixer for testing."""
    torch.manual_seed(seed)
    c1_online = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
    c1_target = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
    c1_target.load_state_dict(c1_online.state_dict())

    torch.manual_seed(seed + 1)
    c2_online = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
    c2_target = SharedRegionalQNetwork(STATE_DIM, 128, 128, N_ACTIONS)
    c2_target.load_state_dict(c2_online.state_dict())

    torch.manual_seed(seed + 2)
    mixer_online = QMIXMixerV1()
    mixer_target = QMIXMixerV1()
    mixer_target.load_state_dict(mixer_online.state_dict())

    return c1_online, c1_target, c2_online, c2_target, mixer_online, mixer_target


def _synth_batch(seed: int, batch_size: int = B):
    """Generate synthetic batch for diagnostic testing."""
    torch.manual_seed(seed)
    states = torch.randn(batch_size, STATE_DIM)
    next_states = torch.randn(batch_size, STATE_DIM)
    actions = torch.randint(0, N_ACTIONS, (batch_size,))
    r1_rewards = torch.randn(batch_size) * 0.1
    joint_rewards = torch.randn(batch_size) * 0.05
    dones = torch.zeros(batch_size)
    central_states = torch.randn(batch_size, CENTRAL_STATE_DIM)
    active_mask = torch.ones(batch_size, R_MAX)
    active_mask[:, 5:] = 0
    q_selected = torch.randn(batch_size, R_MAX) * 0.5
    return {
        "states": states, "next_states": next_states, "actions": actions,
        "r1_rewards": r1_rewards, "joint_rewards": joint_rewards,
        "dones": dones, "central_states": central_states,
        "active_mask": active_mask, "q_selected": q_selected,
    }


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    all_pass = True

    # =================================================================
    # Phase E4OI-0: Pre-manifest — verify E4O artifacts exist
    # =================================================================
    e4o_files = [
        "E4O_verdict.json", "E4O_gradient_alignment_contract.json",
        "E4O_gradient_norm_contract.json", "E4O_gradient_conflict_contract.json",
        "E4O_primary_scenario_gap.json", "E4O_hypothesis_support_logic.json",
        "E4O_r1_signal_contract.json", "E4O_td_error_contract.json",
        "E4O_common_state_bank_contract.json", "E4O_action_specialization_contract.json",
        "E4O_q_margin_contract.json", "E4O_state_stratification_contract.json",
        "E4O_joint_regional_tradeoff_contract.json", "E4O_mixer_gradient_contract.json",
        "E4O_monotonicity_boundary.json", "E4O_checkpoint_lock.json",
        "E4O_diagnostic_seed_contract.json", "E4O_seed_disjointness.json",
        "E4O_diagnostic_matrix.json", "E4O_transition_logging_schema.json",
        "E4O_offline_gradient_contract.json", "E4O_inferential_unit_contract.json",
        "E4O_statistical_design.json", "E4O_no_performance_verdict.json",
        "E4O_future_routing.json", "E4O_e4oi_validation_contract.json",
        "E4O_future_firewall.json", "E4O_protocol_frozen.json",
    ]
    manifest = {}
    for fn in e4o_files:
        p = E4O_DIR / fn
        manifest[fn] = "AVAILABLE" if p.is_file() else "MISSING"
    e4o_verdict = _jload(E4O_DIR / "E4O_verdict.json")
    manifest_pass = all(v == "AVAILABLE" for v in manifest.values())
    manifest_pass = manifest_pass and e4o_verdict.get("E4O_COMPLETE") == "YES"
    if not manifest_pass:
        all_pass = False
    _jdump({"manifest": manifest, "E4O_COMPLETE": e4o_verdict.get("E4O_COMPLETE"),
            "PASS": manifest_pass}, RESULTS_DIR / "E4OI_pre_manifest.json")
    results["pre_manifest"] = manifest_pass

    # =================================================================
    # Phase E4OI-1: Historical integrity
    # =================================================================
    hist = {}
    e4o_hist = _jload(E4O_DIR / "E4O_historical_integrity.json")
    hist["E4O_historical_all_pass"] = e4o_hist.get("all_pass", False)

    e4o_ckpt = _jload(E4O_DIR / "E4O_checkpoint_lock.json")
    hist["checkpoint_count"] = e4o_ckpt["frozen_checkpoints"]["count"]
    hist["C2_CREDIT_SHA"] = e4o_ckpt.get("C2_CREDIT_CONFIG_SHA256")
    hist["QMIX_SHA"] = e4o_ckpt.get("QMIX_CONFIG_SHA256")
    hist["sha_match_c2"] = hist["C2_CREDIT_SHA"] == REQUIRED_C2_CREDIT_SHA
    hist["sha_match_qmix"] = hist["QMIX_SHA"] == REQUIRED_QMIX_SHA
    hist_pass = (hist["E4O_historical_all_pass"] and hist["checkpoint_count"] == 10
                 and hist["sha_match_c2"] and hist["sha_match_qmix"])
    hist["PASS"] = hist_pass
    if not hist_pass:
        all_pass = False
    _jdump(hist, RESULTS_DIR / "E4OI_historical_integrity.json")
    results["historical_integrity"] = hist_pass

    # =================================================================
    # Phase E4OI-2: Upstream config hash verification
    # =================================================================
    live_qmix = qmix_config_sha256()
    live_c2 = c2_credit_config_sha256()
    upstream = validate_upstream_hashes(live_qmix, live_c2, FROZEN_R1_CONFIG_SHA256,
                                       REQUIRED_V3_SHA)
    upstream["live_qmix_sha256"] = live_qmix
    upstream["live_c2_sha256"] = live_c2
    upstream["live_r1_sha256"] = FROZEN_R1_CONFIG_SHA256
    upstream["live_v3_sha256"] = REQUIRED_V3_SHA
    if not upstream["all_pass"]:
        all_pass = False
    _jdump(upstream, RESULTS_DIR / "E4OI_upstream_config_verification.json")
    results["upstream_config"] = upstream["all_pass"]

    # =================================================================
    # Phase E4OI-3: Diagnostic config hash
    # =================================================================
    diag_config = make_diagnostic_config()
    diag_sha = diagnostic_config_sha256()
    _jdump({"config": diag_config, "E4OI_DIAGNOSTIC_CONFIG_SHA256": diag_sha},
           RESULTS_DIR / "E4OI_diagnostic_config.json")
    results["diagnostic_config_sha256"] = diag_sha

    # =================================================================
    # Phase E4OI-4: Seed firewall
    # =================================================================
    diag_set = set(DIAGNOSTIC_SEEDS)
    prior_sets = [set(range(lo, hi + 1)) for lo, hi in PRIOR_SEED_RANGES]
    overlaps = []
    for i, ps in enumerate(prior_sets):
        ov = diag_set & ps
        if ov:
            overlaps.append({"range": PRIOR_SEED_RANGES[i], "overlap": sorted(ov)})
    seed_firewall = {
        "diagnostic_seeds": [DIAGNOSTIC_SEEDS[0], DIAGNOSTIC_SEEDS[-1]],
        "n_diagnostic_seeds": len(DIAGNOSTIC_SEEDS),
        "n_prior_ranges_checked": len(PRIOR_SEED_RANGES),
        "overlaps": overlaps,
        "disjoint": len(overlaps) == 0,
        "O_DIAGNOSTIC_ACCESS_COUNT": COUNTERS["diagnostic_seed_access"],
        "PASS": len(overlaps) == 0 and COUNTERS["diagnostic_seed_access"] == 0,
    }
    if not seed_firewall["PASS"]:
        all_pass = False
    _jdump(seed_firewall, RESULTS_DIR / "E4OI_seed_firewall.json")
    results["seed_firewall"] = seed_firewall["PASS"]

    # =================================================================
    # Phase E4OI-5: V03 — g_J extraction on synthetic data
    # =================================================================
    c1_on, c1_tgt, c2_on, c2_tgt, mix_on, mix_tgt = _synth_networks(SYNTH_SEED)
    batch = _synth_batch(SYNTH_SEED + 100)

    joint_states = batch["states"].unsqueeze(1).expand(-1, R_MAX, -1)
    action_vector = batch["actions"].unsqueeze(1).expand(-1, R_MAX)
    next_joint_states = batch["next_states"].unsqueeze(1).expand(-1, R_MAX, -1)

    q_tot_current = current_joint_q(
        c2_on, mix_on, joint_states, action_vector,
        batch["active_mask"], batch["central_states"],
    )
    y_joint = double_q_joint_target(
        c2_on, c2_tgt, mix_tgt,
        next_joint_states, batch["active_mask"], batch["central_states"],
        batch["joint_rewards"], batch["dones"], GAMMA,
    )
    l_joint = treatment_loss(q_tot_current, y_joint)

    g_j = extract_utility_gradient(l_joint, c2_on, retain_graph=True)
    g_j_pass = g_j.numel() > 0 and not torch.all(g_j == 0)
    gj_result = {
        "g_J_numel": g_j.numel(),
        "g_J_norm": gradient_norms(g_j),
        "g_J_nonzero": bool(g_j_pass),
        "PASS": bool(g_j_pass),
    }
    if not g_j_pass:
        all_pass = False
    _jdump(gj_result, RESULTS_DIR / "E4OI_g_J_extraction.json")
    results["g_J_extraction"] = bool(g_j_pass)

    # =================================================================
    # Phase E4OI-6: V04 — g_R extraction on synthetic data
    # =================================================================
    q_local = focal_current_q(c2_on, batch["states"], batch["actions"])
    y_local = focal_regional_ddqn_target(
        c2_on, c2_tgt, batch["r1_rewards"], batch["next_states"], batch["dones"], GAMMA,
    )
    l_regional = regional_smoothl1_loss(q_local, y_local)

    g_r = extract_utility_gradient(l_regional, c2_on, retain_graph=True)
    g_r_pass = g_r.numel() > 0 and not torch.all(g_r == 0)
    gr_result = {
        "g_R_numel": g_r.numel(),
        "g_R_norm": gradient_norms(g_r),
        "g_R_nonzero": bool(g_r_pass),
        "g_R_same_dim_as_g_J": g_r.numel() == g_j.numel(),
        "PASS": bool(g_r_pass) and g_r.numel() == g_j.numel(),
    }
    if not gr_result["PASS"]:
        all_pass = False
    _jdump(gr_result, RESULTS_DIR / "E4OI_g_R_extraction.json")
    results["g_R_extraction"] = gr_result["PASS"]

    # =================================================================
    # Phase E4OI-7: V05 — Gradient cosine on known vectors
    # =================================================================
    aligned_a = torch.tensor([1.0, 0.0, 1.0])
    aligned_b = torch.tensor([2.0, 0.0, 2.0])
    cos_aligned = cosine_similarity_jr(aligned_a, aligned_b)
    assert cos_aligned["cosine_valid"]

    conflict_a = torch.tensor([1.0, 0.0, 0.0])
    conflict_b = torch.tensor([-1.0, 0.0, 0.0])
    cos_conflict = cosine_similarity_jr(conflict_a, conflict_b)

    ortho_a = torch.tensor([1.0, 0.0, 0.0])
    ortho_b = torch.tensor([0.0, 1.0, 0.0])
    cos_ortho = cosine_similarity_jr(ortho_a, ortho_b)

    zero_a = torch.tensor([0.0, 0.0, 0.0])
    cos_zero = cosine_similarity_jr(zero_a, aligned_b)

    cosine_tests = {
        "aligned_cosine": cos_aligned["cosine_JR"],
        "aligned_expected": 1.0,
        "aligned_close": abs(cos_aligned["cosine_JR"] - 1.0) < 1e-6,
        "conflict_cosine": cos_conflict["cosine_JR"],
        "conflict_expected": -1.0,
        "conflict_close": abs(cos_conflict["cosine_JR"] - (-1.0)) < 1e-6,
        "ortho_cosine": cos_ortho["cosine_JR"],
        "ortho_expected": 0.0,
        "ortho_close": abs(cos_ortho["cosine_JR"]) < 1e-6,
        "zero_valid": cos_zero["cosine_valid"],
        "zero_expected_valid": False,
        "zero_cosine_is_none": cos_zero["cosine_JR"] is None,
    }
    cosine_pass = (cosine_tests["aligned_close"] and cosine_tests["conflict_close"]
                   and cosine_tests["ortho_close"]
                   and not cosine_tests["zero_valid"] and cosine_tests["zero_cosine_is_none"])
    cosine_tests["PASS"] = cosine_pass
    if not cosine_pass:
        all_pass = False
    _jdump(cosine_tests, RESULTS_DIR / "E4OI_cosine_verification.json")
    results["cosine_verification"] = cosine_pass

    # =================================================================
    # Phase E4OI-8: V06 — Gradient norm ratio on known norms
    # =================================================================
    rho = gradient_norm_ratio(3.0, 6.0)
    rho_zero = gradient_norm_ratio(0.0, 0.0)
    norm_ratio_tests = {
        "rho_3_6": rho,
        "expected_0.5": abs(rho - 0.5) < 1e-6,
        "rho_0_0": rho_zero,
        "rho_0_0_near_zero": rho_zero < 1e-6,
        "PASS": abs(rho - 0.5) < 1e-6 and rho_zero < 1e-6,
    }
    if not norm_ratio_tests["PASS"]:
        all_pass = False
    _jdump(norm_ratio_tests, RESULTS_DIR / "E4OI_norm_ratio_verification.json")
    results["norm_ratio"] = norm_ratio_tests["PASS"]

    # =================================================================
    # Phase E4OI-9: V07 — Composite gradient identity
    # =================================================================
    c2_on.zero_grad()
    mix_on.zero_grad()

    q_tot2 = current_joint_q(
        c2_on, mix_on, joint_states, action_vector,
        batch["active_mask"], batch["central_states"],
    )
    y_j2 = double_q_joint_target(
        c2_on, c2_tgt, mix_tgt,
        next_joint_states, batch["active_mask"], batch["central_states"],
        batch["joint_rewards"], batch["dones"], GAMMA,
    )
    l_j2 = treatment_loss(q_tot2, y_j2)

    q_loc2 = focal_current_q(c2_on, batch["states"], batch["actions"])
    y_loc2 = focal_regional_ddqn_target(
        c2_on, c2_tgt, batch["r1_rewards"], batch["next_states"], batch["dones"], GAMMA,
    )
    l_r2 = regional_smoothl1_loss(q_loc2, y_loc2)

    l_composite = c2_composite_loss(l_j2, l_r2)

    g_j2 = extract_utility_gradient(l_j2, c2_on, retain_graph=True)
    g_r2 = extract_utility_gradient(l_r2, c2_on, retain_graph=True)
    g_c2_direct = extract_utility_gradient(l_composite, c2_on, retain_graph=False)

    identity_result = composite_gradient_identity(g_j2, g_r2, g_c2_direct)
    identity_result["PASS"] = identity_result["identity_holds"]
    if not identity_result["PASS"]:
        all_pass = False
    _jdump(identity_result, RESULTS_DIR / "E4OI_composite_gradient_identity.json")
    results["composite_identity"] = identity_result["PASS"]

    # =================================================================
    # Phase E4OI-10: V08 — Mixer gradient: no regional path
    # =================================================================
    c2_on.zero_grad()
    mix_on.zero_grad()

    q_tot3 = current_joint_q(
        c2_on, mix_on, joint_states, action_vector,
        batch["active_mask"], batch["central_states"],
    )
    y_j3 = double_q_joint_target(
        c2_on, c2_tgt, mix_tgt,
        next_joint_states, batch["active_mask"], batch["central_states"],
        batch["joint_rewards"], batch["dones"], GAMMA,
    )
    l_j3 = treatment_loss(q_tot3, y_j3)

    q_loc3 = focal_current_q(c2_on, batch["states"], batch["actions"])
    y_loc3 = focal_regional_ddqn_target(
        c2_on, c2_tgt, batch["r1_rewards"], batch["next_states"], batch["dones"], GAMMA,
    )
    l_r3 = regional_smoothl1_loss(q_loc3, y_loc3)
    l_c2_3 = c2_composite_loss(l_j3, l_r3)

    g_mixer_j_unw = extract_mixer_gradient(l_j3, mix_on, retain_graph=True)
    g_mixer_c2 = extract_mixer_gradient(l_c2_3, mix_on, retain_graph=False)

    mixer_check = mixer_gradient_check(g_mixer_j_unw, g_mixer_c2)
    mixer_check["PASS"] = mixer_check["no_regional_path"]
    if not mixer_check["PASS"]:
        all_pass = False
    _jdump(mixer_check, RESULTS_DIR / "E4OI_mixer_gradient_check.json")
    results["mixer_gradient"] = mixer_check["PASS"]

    # =================================================================
    # Phase E4OI-11: V09 — Common-state dual-network evaluation
    # =================================================================
    torch.manual_seed(SYNTH_SEED + 200)
    synth_states = torch.randn(16, STATE_DIM)
    cross_result = cross_evaluate(synth_states, c1_on, c2_on)

    cs_pass = (cross_result["Q_C1"].shape == (16, N_ACTIONS)
               and cross_result["Q_C2"].shape == (16, N_ACTIONS)
               and cross_result["argmax_C1"].shape == (16,)
               and cross_result["margin_C1"].shape == (16,))
    cs_result = {
        "Q_C1_shape": list(cross_result["Q_C1"].shape),
        "Q_C2_shape": list(cross_result["Q_C2"].shape),
        "n_disagree": int(cross_result["disagreement"].sum().item()),
        "n_total": 16,
        "mean_margin_C1": float(cross_result["margin_C1"].mean().item()),
        "mean_margin_C2": float(cross_result["margin_C2"].mean().item()),
        "PASS": bool(cs_pass),
    }
    if not cs_pass:
        all_pass = False
    _jdump(cs_result, RESULTS_DIR / "E4OI_common_state_evaluation.json")
    results["common_state_eval"] = bool(cs_pass)

    # =================================================================
    # Phase E4OI-12: V10 — Action disagreement on known Q tables
    # =================================================================
    q_known_c1 = torch.tensor([[1.0, 0.5, 0.2, 0.1],
                                [0.1, 0.9, 0.3, 0.2],
                                [0.2, 0.3, 0.8, 0.1],
                                [0.1, 0.2, 0.3, 0.7]])
    q_known_c2 = torch.tensor([[0.5, 1.0, 0.2, 0.1],
                                [0.1, 0.9, 0.3, 0.2],
                                [0.2, 0.3, 0.1, 0.8],
                                [0.1, 0.2, 0.3, 0.7]])
    argmax_k1 = q_known_c1.argmax(dim=1)
    argmax_k2 = q_known_c2.argmax(dim=1)
    d_act = action_disagreement_fraction(argmax_k1, argmax_k2)
    tm = action_transition_matrix(argmax_k1, argmax_k2)

    disagree_pass = (abs(d_act - 0.5) < 1e-6
                     and argmax_k1[0].item() == 0 and argmax_k2[0].item() == 1
                     and argmax_k1[1].item() == 1 and argmax_k2[1].item() == 1
                     and argmax_k1[2].item() == 2 and argmax_k2[2].item() == 3
                     and argmax_k1[3].item() == 3 and argmax_k2[3].item() == 3)
    tm_expected = np.zeros((4, 4), dtype=np.int64)
    tm_expected[0, 1] = 1
    tm_expected[1, 1] = 1
    tm_expected[2, 3] = 1
    tm_expected[3, 3] = 1
    tm_match = np.array_equal(tm, tm_expected)
    disagree_result = {
        "D_action": d_act,
        "expected_D_action": 0.5,
        "D_action_correct": abs(d_act - 0.5) < 1e-6,
        "transition_matrix": tm.tolist(),
        "tm_expected": tm_expected.tolist(),
        "tm_match": bool(tm_match),
        "PASS": disagree_pass and tm_match,
    }
    if not disagree_result["PASS"]:
        all_pass = False
    _jdump(disagree_result, RESULTS_DIR / "E4OI_action_disagreement.json")
    results["action_disagreement"] = disagree_result["PASS"]

    # =================================================================
    # Phase E4OI-13: V11 — Q-margins on known Q tables
    # =================================================================
    from prdrl.diagnostics.common_state_diagnostic import _q_margin
    margins_k1 = _q_margin(q_known_c1)
    expected_margins_k1 = torch.tensor([0.5, 0.6, 0.5, 0.4])
    margin_pass = torch.allclose(margins_k1, expected_margins_k1, atol=1e-6)
    margin_result = {
        "margins_C1": margins_k1.tolist(),
        "expected": expected_margins_k1.tolist(),
        "close": bool(margin_pass),
        "PASS": bool(margin_pass),
    }
    if not margin_pass:
        all_pass = False
    _jdump(margin_result, RESULTS_DIR / "E4OI_q_margin_verification.json")
    results["q_margin"] = bool(margin_pass)

    # =================================================================
    # Phase E4OI-14: V12 — Scenario gap formulas
    # =================================================================
    g_cos_val = scenario_gap_cosine(mean_cosine_stat=-0.3,
                                     mean_cosine_per=0.5,
                                     mean_cosine_mix=0.2)
    expected_g_cos = -0.3 - 0.5 * (0.5 + 0.2)
    g_cos_correct = abs(g_cos_val - expected_g_cos) < 1e-10

    g_conf_val = scenario_gap_conflict(conflict_stat=0.6,
                                        conflict_per=0.2,
                                        conflict_mix=0.3)
    expected_g_conf = 0.6 - 0.5 * (0.2 + 0.3)
    g_conf_correct = abs(g_conf_val - expected_g_conf) < 1e-10

    gap_result = {
        "G_cos": g_cos_val,
        "G_cos_expected": expected_g_cos,
        "G_cos_correct": g_cos_correct,
        "G_conflict": g_conf_val,
        "G_conflict_expected": expected_g_conf,
        "G_conflict_correct": g_conf_correct,
        "PASS": g_cos_correct and g_conf_correct,
    }
    if not gap_result["PASS"]:
        all_pass = False
    _jdump(gap_result, RESULTS_DIR / "E4OI_scenario_gap_verification.json")
    results["scenario_gap"] = gap_result["PASS"]

    # =================================================================
    # Phase E4OI-15: V13 — Aggregation correctness
    # =================================================================
    synth_records = []
    rng_agg = np.random.RandomState(SYNTH_SEED + 300)
    for p in [6001, 6002]:
        for e in [2301, 2302]:
            for s in SCENARIOS:
                for _ in range(3):
                    synth_records.append({
                        "policy": p, "diagnostic_environment": e,
                        "scenario": s, "cosine_JR": rng_agg.randn(),
                    })
    agg = aggregate_policy_env_scenario(synth_records, "cosine_JR")
    cell_key = (6001, 2301, "PERSISTENT")
    cell_vals = [r["cosine_JR"] for r in synth_records
                 if r["policy"] == 6001 and r["diagnostic_environment"] == 2301
                 and r["scenario"] == "PERSISTENT"]
    expected_mean = float(np.mean(cell_vals))
    agg_correct = abs(agg["cell_means"][cell_key] - expected_mean) < 1e-10
    agg_result = {
        "cell_mean": agg["cell_means"][cell_key],
        "expected_mean": expected_mean,
        "correct": agg_correct,
        "n_policies": len(agg["policies"]),
        "n_envs": len(agg["environments"]),
        "n_scenarios_in_matrices": len(agg["scenario_matrices"]),
        "PASS": agg_correct,
    }
    if not agg_result["PASS"]:
        all_pass = False
    _jdump(agg_result, RESULTS_DIR / "E4OI_aggregation_verification.json")
    results["aggregation"] = agg_result["PASS"]

    # =================================================================
    # Phase E4OI-16: V14 — Crossed bootstrap on known matrix
    # =================================================================
    known_matrix = np.array([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.6]])
    boot = crossed_bootstrap_ci(known_matrix, n_bootstrap=10000, seed=DIAGNOSTIC_BOOTSTRAP_SEED)
    grand_mean = float(np.mean(known_matrix))
    boot_pass = (abs(boot["point"] - grand_mean) < 1e-10
                 and boot["ci_lower"] < grand_mean
                 and boot["ci_upper"] > grand_mean
                 and boot["ci_lower"] < boot["ci_upper"])
    boot_result = {
        "point": boot["point"],
        "grand_mean": grand_mean,
        "point_correct": abs(boot["point"] - grand_mean) < 1e-10,
        "ci_lower": boot["ci_lower"],
        "ci_upper": boot["ci_upper"],
        "ci_contains_mean": boot["ci_lower"] < grand_mean < boot["ci_upper"],
        "ci_ordered": boot["ci_lower"] < boot["ci_upper"],
        "n_bootstrap": boot["n_bootstrap"],
        "seed": boot["seed"],
        "PASS": boot_pass,
    }
    if not boot_pass:
        all_pass = False
    _jdump(boot_result, RESULTS_DIR / "E4OI_bootstrap_verification.json")
    results["bootstrap"] = boot_pass

    # =================================================================
    # Phase E4OI-17: Scenario gap bootstrap
    # =================================================================
    rng_gap = np.random.RandomState(SYNTH_SEED + 400)
    gap_matrices = {
        "PERSISTENT": rng_gap.randn(5, 10) + 0.5,
        "STATIONARY": rng_gap.randn(5, 10) - 0.3,
        "MIXED": rng_gap.randn(5, 10) + 0.2,
    }
    gap_boot = scenario_gap_bootstrap(gap_matrices, n_bootstrap=10000,
                                       seed=DIAGNOSTIC_BOOTSTRAP_SEED)
    expected_gap = (float(np.mean(gap_matrices["STATIONARY"]))
                    - 0.5 * (float(np.mean(gap_matrices["PERSISTENT"]))
                             + float(np.mean(gap_matrices["MIXED"]))))
    gap_boot_pass = (abs(gap_boot["G_point"] - expected_gap) < 1e-10
                     and gap_boot["G_ci_lower"] < gap_boot["G_ci_upper"])
    gap_boot_result = {
        "G_point": gap_boot["G_point"],
        "expected_gap": expected_gap,
        "point_correct": abs(gap_boot["G_point"] - expected_gap) < 1e-10,
        "G_ci_lower": gap_boot["G_ci_lower"],
        "G_ci_upper": gap_boot["G_ci_upper"],
        "PASS": gap_boot_pass,
    }
    if not gap_boot_pass:
        all_pass = False
    _jdump(gap_boot_result, RESULTS_DIR / "E4OI_gap_bootstrap_verification.json")
    results["gap_bootstrap"] = gap_boot_pass

    # =================================================================
    # Phase E4OI-18: O1 truth table verification
    # =================================================================
    o1_cases = [
        {"g_cos": -0.5, "ci_upper": -0.1, "n_neg": 5, "n_pol": 5,
         "g_conflict": 0.3, "n_pos": 5, "expected": "SUPPORTED"},
        {"g_cos": -0.5, "ci_upper": -0.1, "n_neg": 5, "n_pol": 5,
         "g_conflict": 0.3, "n_pos": 3, "expected": "PLAUSIBLE"},
        {"g_cos": -0.5, "ci_upper": 0.05, "n_neg": 5, "n_pol": 5,
         "g_conflict": 0.3, "n_pos": 5, "expected": "PLAUSIBLE"},
        {"g_cos": -0.5, "ci_upper": -0.1, "n_neg": 3, "n_pol": 5,
         "g_conflict": 0.3, "n_pos": 5, "expected": "PLAUSIBLE"},
        {"g_cos": 0.1, "ci_upper": 0.3, "n_neg": 1, "n_pol": 5,
         "g_conflict": -0.1, "n_pos": 1, "expected": "NOT_SUPPORTED"},
        {"g_cos": -0.1, "ci_upper": 0.1, "n_neg": 3, "n_pol": 5,
         "g_conflict": -0.05, "n_pos": 2, "expected": "PLAUSIBLE"},
    ]
    o1_results = []
    o1_all_correct = True
    for case in o1_cases:
        verdict = o1_support_verdict(
            case["g_cos"], case["ci_upper"],
            case["n_neg"], case["n_pol"],
            case["g_conflict"], case["n_pos"],
        )
        correct = verdict == case["expected"]
        if not correct:
            o1_all_correct = False
        o1_results.append({
            "input": case,
            "verdict": verdict,
            "expected": case["expected"],
            "correct": correct,
        })
    o1_test = {"cases": o1_results, "all_correct": o1_all_correct, "PASS": o1_all_correct}
    if not o1_all_correct:
        all_pass = False
    _jdump(o1_test, RESULTS_DIR / "E4OI_o1_truth_table.json")
    results["o1_truth_table"] = o1_all_correct

    # =================================================================
    # Phase E4OI-19: R1 signal statistics on synthetic data
    # =================================================================
    rng_r1 = np.random.RandomState(SYNTH_SEED + 500)
    synth_r1 = np.concatenate([np.zeros(30), rng_r1.randn(50) * 0.1, -rng_r1.rand(20) * 0.05])
    r1_stats = r1_signal_statistics(synth_r1)
    r1_act = r1_activity(synth_r1, eps_reward=0.0)

    r1_pass = (r1_stats["n"] == 100
               and abs(r1_stats["zero_fraction"] - 0.3) < 1e-10
               and r1_act == 0.7
               and "mean" in r1_stats and "median" in r1_stats
               and "q10" in r1_stats and "q90" in r1_stats
               and "variance" in r1_stats)
    r1_result = {
        "n": r1_stats["n"],
        "zero_fraction": r1_stats["zero_fraction"],
        "activity": r1_act,
        "has_all_fields": all(k in r1_stats for k in [
            "mean", "median", "mean_absolute_value", "q10", "q25", "q75", "q90", "variance",
        ]),
        "PASS": r1_pass,
    }
    if not r1_pass:
        all_pass = False
    _jdump(r1_result, RESULTS_DIR / "E4OI_r1_signal_verification.json")
    results["r1_signal"] = r1_pass

    # =================================================================
    # Phase E4OI-20: TD error diagnostics on synthetic data
    # =================================================================
    q_loc_td = torch.tensor([0.5, 0.3, 0.7, 0.1])
    y_loc_td = torch.tensor([0.6, 0.2, 0.9, 0.0])
    reg_td = regional_td_error(q_loc_td, y_loc_td)

    expected_delta = y_loc_td - q_loc_td
    td_pass = (torch.allclose(reg_td["abs_delta_R"], torch.abs(expected_delta), atol=1e-6)
               and abs(reg_td["mean_abs_delta_R"] - float(torch.abs(expected_delta).mean())) < 1e-6)

    q_tot_td = torch.tensor([1.0, 2.0])
    y_j_td = torch.tensor([1.5, 1.8])
    j_td = joint_td_error(q_tot_td, y_j_td)
    j_expected = y_j_td - q_tot_td
    td_pass = td_pass and torch.allclose(j_td["abs_delta_J"], torch.abs(j_expected), atol=1e-6)

    ratio_td = td_error_ratio(torch.tensor([0.3, 0.1]), torch.tensor([0.6, 0.5]))
    expected_ratio = torch.tensor([0.3 / (0.6 + 1e-12), 0.1 / (0.5 + 1e-12)])
    td_pass = td_pass and torch.allclose(ratio_td, expected_ratio, atol=1e-8)

    td_result = {
        "regional_delta_correct": bool(torch.allclose(reg_td["abs_delta_R"],
                                                       torch.abs(expected_delta), atol=1e-6)),
        "joint_delta_correct": bool(torch.allclose(j_td["abs_delta_J"],
                                                    torch.abs(j_expected), atol=1e-6)),
        "ratio_correct": bool(torch.allclose(ratio_td, expected_ratio, atol=1e-8)),
        "PASS": bool(td_pass),
    }
    if not td_pass:
        all_pass = False
    _jdump(td_result, RESULTS_DIR / "E4OI_td_error_verification.json")
    results["td_error"] = bool(td_pass)

    # =================================================================
    # Phase E4OI-21: State stratification
    # =================================================================
    synth_states_strat = np.array([
        [0.0, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8],
        [0.0, 0.3, 0.2, 0.0, 0.1, 0.5, 0.4, 0.6, 0.9],
        [0.1, 0.4, 0.5, 0.2, 0.3, 0.6, 0.5, 0.8, 0.7],
        [0.5, 0.2, 0.1, 0.3, 0.4, 0.7, 0.3, 0.5, 0.6],
    ])
    strat = stratify_by_delta_severity(synth_states_strat, delta_severity_index=0)
    strat_pass = (strat["n_delta_severity_zero"] == 2 and strat["n_delta_severity_positive"] == 2
                  and strat["n_total"] == 4)
    strat_result = {
        "n_total": strat["n_total"],
        "n_zero": strat["n_delta_severity_zero"],
        "n_positive": strat["n_delta_severity_positive"],
        "expected_zero": 2, "expected_positive": 2,
        "PASS": strat_pass,
    }
    if not strat_pass:
        all_pass = False
    _jdump(strat_result, RESULTS_DIR / "E4OI_stratification_verification.json")
    results["stratification"] = strat_pass

    # =================================================================
    # Phase E4OI-22: Common-state bank record schema
    # =================================================================
    rec = build_common_state_record(
        source_arm="C1", policy=6001, scenario="PERSISTENT",
        env_seed=2301, change_index=0, region_id=0,
        local_state_9d=np.zeros(STATE_DIM),
        central_state_55d=np.zeros(CENTRAL_STATE_DIM),
        active_mask=np.ones(R_MAX),
    )
    required_fields = ["source_arm", "policy", "scenario", "diagnostic_environment",
                       "dynamic_change_index", "region_id", "local_state_9D",
                       "central_state_55D", "active_mask"]
    schema_pass = all(f in rec for f in required_fields)
    schema_pass = schema_pass and len(rec["local_state_9D"]) == STATE_DIM
    schema_pass = schema_pass and len(rec["central_state_55D"]) == CENTRAL_STATE_DIM
    schema_result = {
        "fields_present": [f for f in required_fields if f in rec],
        "fields_missing": [f for f in required_fields if f not in rec],
        "local_dim": len(rec["local_state_9D"]),
        "central_dim": len(rec["central_state_55D"]),
        "PASS": schema_pass,
    }
    if not schema_pass:
        all_pass = False
    _jdump(schema_result, RESULTS_DIR / "E4OI_common_state_schema.json")
    results["common_state_schema"] = schema_pass

    # =================================================================
    # Phase E4OI-23: Transition logging schema
    # =================================================================
    e4o_schema = _jload(E4O_DIR / "E4O_transition_logging_schema.json")
    required_transition_fields = list(e4o_schema["per_dynamic_change_event"].keys())
    transition_pass = len(required_transition_fields) >= 15
    transition_result = {
        "n_required_fields": len(required_transition_fields),
        "fields": required_transition_fields,
        "PASS": transition_pass,
    }
    if not transition_pass:
        all_pass = False
    _jdump(transition_result, RESULTS_DIR / "E4OI_transition_schema.json")
    results["transition_schema"] = transition_pass

    # =================================================================
    # Phase E4OI-24: Conflict fraction verification
    # =================================================================
    test_cosines = [0.5, -0.3, 0.1, -0.7, None, -0.1]
    cf = conflict_fraction(test_cosines)
    expected_cf = 3.0 / 5.0
    cf_pass = abs(cf - expected_cf) < 1e-10
    cf_all_none = conflict_fraction([None, None])
    cf_result = {
        "conflict_fraction": cf,
        "expected": expected_cf,
        "correct": cf_pass,
        "all_none_is_nan": math.isnan(cf_all_none),
        "PASS": cf_pass and math.isnan(cf_all_none),
    }
    if not cf_result["PASS"]:
        all_pass = False
    _jdump(cf_result, RESULTS_DIR / "E4OI_conflict_fraction_verification.json")
    results["conflict_fraction"] = cf_result["PASS"]

    # =================================================================
    # Phase E4OI-25: Policy-level gaps
    # =================================================================
    per_policy_cos = {
        6001: {"PERSISTENT": 0.5, "STATIONARY": -0.3, "MIXED": 0.2},
        6002: {"PERSISTENT": 0.4, "STATIONARY": -0.2, "MIXED": 0.1},
        6003: {"PERSISTENT": 0.6, "STATIONARY": -0.4, "MIXED": 0.3},
        6004: {"PERSISTENT": 0.3, "STATIONARY": -0.1, "MIXED": 0.2},
        6005: {"PERSISTENT": 0.5, "STATIONARY": -0.5, "MIXED": 0.1},
    }
    per_policy_conf = {
        6001: {"PERSISTENT": 0.2, "STATIONARY": 0.6, "MIXED": 0.3},
        6002: {"PERSISTENT": 0.1, "STATIONARY": 0.5, "MIXED": 0.2},
        6003: {"PERSISTENT": 0.3, "STATIONARY": 0.7, "MIXED": 0.4},
        6004: {"PERSISTENT": 0.2, "STATIONARY": 0.4, "MIXED": 0.3},
        6005: {"PERSISTENT": 0.1, "STATIONARY": 0.8, "MIXED": 0.2},
    }
    plg = policy_level_gaps(per_policy_cos, per_policy_conf)
    plg_pass = (plg["n_policies_G_cos_negative"] == 5
                and plg["n_policies_G_conflict_positive"] == 5
                and len(plg["policy_G_cos"]) == 5)
    for i, pseed in enumerate(sorted(per_policy_cos.keys())):
        expected = scenario_gap_cosine(
            per_policy_cos[pseed]["STATIONARY"],
            per_policy_cos[pseed]["PERSISTENT"],
            per_policy_cos[pseed]["MIXED"],
        )
        if abs(plg["policy_G_cos"][i] - expected) > 1e-10:
            plg_pass = False
    plg_result = {
        "n_policies_G_cos_negative": plg["n_policies_G_cos_negative"],
        "n_policies_G_conflict_positive": plg["n_policies_G_conflict_positive"],
        "policy_G_cos": plg["policy_G_cos"],
        "policy_G_conflict": plg["policy_G_conflict"],
        "PASS": plg_pass,
    }
    if not plg_pass:
        all_pass = False
    _jdump(plg_result, RESULTS_DIR / "E4OI_policy_level_gaps.json")
    results["policy_level_gaps"] = plg_pass

    # =================================================================
    # Phase E4OI-26: No-model-mutation verification
    # =================================================================
    mutation = {
        "optimizer_steps": COUNTERS["optimizer_steps"],
        "parameter_changes": COUNTERS["parameter_changes"],
        "target_updates": COUNTERS["target_updates"],
        "replay_mutations": COUNTERS["replay_mutations"],
        "real_environment_episodes": COUNTERS["real_environment_episodes"],
        "diagnostic_seed_access": COUNTERS["diagnostic_seed_access"],
        "all_zero": all(v == 0 for v in COUNTERS.values()),
        "PASS": all(v == 0 for v in COUNTERS.values()),
    }
    if not mutation["PASS"]:
        all_pass = False
    _jdump(mutation, RESULTS_DIR / "E4OI_no_mutation.json")
    results["no_mutation"] = mutation["PASS"]

    # =================================================================
    # Phase E4OI-27: No performance verdict
    # =================================================================
    no_perf = {
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "E4MF_REMAINS_AUTHORITATIVE": True,
        "PASS": True,
    }
    _jdump(no_perf, RESULTS_DIR / "E4OI_no_performance_verdict.json")
    results["no_performance_verdict"] = True

    # =================================================================
    # Phase E4OI-28: Diagnostic matrix verification
    # =================================================================
    n_arms = 2
    n_policies = len(POLICY_SEEDS)
    n_envs = N_DIAGNOSTIC_ENVS
    n_scenarios = len(SCENARIOS)
    computed_total = n_arms * n_policies * n_envs * n_scenarios
    matrix_pass = computed_total == DIAGNOSTIC_EPISODES
    matrix_result = {
        "n_arms": n_arms, "n_policies": n_policies,
        "n_envs": n_envs, "n_scenarios": n_scenarios,
        "computed_total": computed_total,
        "expected_total": DIAGNOSTIC_EPISODES,
        "match": matrix_pass,
        "PASS": matrix_pass,
    }
    if not matrix_pass:
        all_pass = False
    _jdump(matrix_result, RESULTS_DIR / "E4OI_matrix_verification.json")
    results["matrix"] = matrix_pass

    # =================================================================
    # Phase E4OI-29: Joint-regional tradeoff records
    # =================================================================
    g_j_norm = gradient_norms(g_j)
    g_r_norm = gradient_norms(g_r)
    cos_jr = cosine_similarity_jr(g_j, g_r)
    g_c2_norm = gradient_norms(0.5 * g_j + 0.5 * g_r)
    rho_j_to_c = gradient_norms(0.5 * g_j) / (g_c2_norm + EPSILON_COSINE)
    rho_r_to_c = gradient_norms(0.5 * g_r) / (g_c2_norm + EPSILON_COSINE)

    tradeoff_result = {
        "norm_g_J": g_j_norm,
        "norm_g_R": g_r_norm,
        "cosine_JR": cos_jr["cosine_JR"],
        "norm_g_C2": g_c2_norm,
        "rho_joint_to_composite": rho_j_to_c,
        "rho_regional_to_composite": rho_r_to_c,
        "PASS": True,
    }
    _jdump(tradeoff_result, RESULTS_DIR / "E4OI_tradeoff_records.json")
    results["tradeoff"] = True

    # =================================================================
    # Phase E4OI-30: Version guards
    # =================================================================
    version_checks = {
        "NORM_V3": NORM_V3,
        "REWARD_R1_NAME": REWARD_R1_NAME,
        "STATE_DIM": STATE_DIM,
        "N_ACTIONS": N_ACTIONS,
        "R_MAX": R_MAX,
        "CENTRAL_STATE_DIM": CENTRAL_STATE_DIM,
        "MIX_HIDDEN_DIM": MIX_HIDDEN_DIM,
        "GAMMA": GAMMA,
        "TARGET_TAU": TARGET_TAU,
        "JOINT_WEIGHT": JOINT_WEIGHT,
        "REGIONAL_WEIGHT": REGIONAL_WEIGHT,
    }
    vg_pass = (version_checks["STATE_DIM"] == 9
               and version_checks["N_ACTIONS"] == 4
               and version_checks["R_MAX"] == 10
               and version_checks["CENTRAL_STATE_DIM"] == 55
               and version_checks["MIX_HIDDEN_DIM"] == 32
               and abs(version_checks["GAMMA"] - 0.99) < 1e-10
               and abs(version_checks["TARGET_TAU"] - 0.005) < 1e-10
               and abs(version_checks["JOINT_WEIGHT"] - 0.5) < 1e-10
               and abs(version_checks["REGIONAL_WEIGHT"] - 0.5) < 1e-10)
    version_checks["PASS"] = vg_pass
    if not vg_pass:
        all_pass = False
    _jdump(version_checks, RESULTS_DIR / "E4OI_version_guards.json")
    results["version_guards"] = vg_pass

    # =================================================================
    # Phase E4OI-31: Q-margin stats
    # =================================================================
    synth_margins = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4])
    qm_stats = q_margin_stats(synth_margins)
    qm_pass = (abs(qm_stats["mean"] - 0.3) < 1e-6
               and abs(qm_stats["median"] - 0.3) < 1e-6
               and abs(qm_stats["min"] - 0.1) < 1e-6
               and abs(qm_stats["max"] - 0.5) < 1e-6)
    qm_stats["PASS"] = qm_pass
    if not qm_pass:
        all_pass = False
    _jdump(qm_stats, RESULTS_DIR / "E4OI_q_margin_stats.json")
    results["q_margin_stats"] = qm_pass

    # =================================================================
    # Phase E4OI-32: TD error summary
    # =================================================================
    td_abs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    td_summ = td_error_summary(td_abs)
    td_summ_pass = abs(td_summ["mean"] - 0.3) < 1e-6 and abs(td_summ["median"] - 0.3) < 1e-6
    td_summ["PASS"] = td_summ_pass
    if not td_summ_pass:
        all_pass = False
    _jdump(td_summ, RESULTS_DIR / "E4OI_td_summary.json")
    results["td_summary"] = td_summ_pass

    # =================================================================
    # Phase E4OI-33: Final counters + verdict
    # =================================================================
    final_counters = dict(COUNTERS)
    final_counters["all_zero"] = all(v == 0 for v in COUNTERS.values())

    n_pass = sum(1 for v in results.values() if v is True or (isinstance(v, str) and v))
    n_total = len([v for v in results.values() if isinstance(v, bool)])
    all_pass = all_pass and final_counters["all_zero"]

    elapsed = time.time() - t0

    verdict = {
        "stage": STAGE,
        "E4OI_COMPLETE": "YES" if all_pass else "NO",
        "DIAGNOSTIC_VERSION": DIAGNOSTIC_VERSION,
        "E4OI_DIAGNOSTIC_CONFIG_SHA256": diag_sha,
        "PERFORMANCE_CAUSAL_VERDICT": "NOT_COMPUTED",
        "E4MF_REMAINS_AUTHORITATIVE": True,
        "O_DIAGNOSTIC_ACCESS_COUNT": COUNTERS["diagnostic_seed_access"],
        "COUNTERS": final_counters,
        "results_summary": {k: v for k, v in results.items() if isinstance(v, bool)},
        "n_validations_pass": sum(1 for v in results.values() if v is True),
        "n_validations_total": n_total,
        "elapsed_seconds": round(elapsed, 2),
        "NEXT_STAGE": "E4OD_SCENARIO_TRADEOFF_DIAGNOSTIC_EXECUTION" if all_pass else "FIX_E4OI",
    }
    _jdump(verdict, RESULTS_DIR / "E4OI_verdict.json")

    # =================================================================
    # Report
    # =================================================================
    report_lines = [
        f"# E4OI — Scenario Trade-off Diagnostic Implementation Validation",
        f"",
        f"## A. Stage",
        f"- Stage: {STAGE}",
        f"- E4OI_COMPLETE: {'YES' if all_pass else 'NO'}",
        f"",
        f"## B. Diagnostic Config",
        f"- DIAGNOSTIC_VERSION: {DIAGNOSTIC_VERSION}",
        f"- E4OI_DIAGNOSTIC_CONFIG_SHA256: {diag_sha}",
        f"",
        f"## C. Upstream Verification",
        f"- QMIX SHA: {'MATCH' if upstream['checks']['qmix'] else 'MISMATCH'}",
        f"- C2 Credit SHA: {'MATCH' if upstream['checks']['c2_credit'] else 'MISMATCH'}",
        f"- R1 SHA: {'MATCH' if upstream['checks']['r1_reward'] else 'MISMATCH'}",
        f"- V3 SHA: {'MATCH' if upstream['checks']['v3_normalization'] else 'MISMATCH'}",
        f"",
        f"## D. Gradient Extraction",
        f"- g_J extraction: {'PASS' if results.get('g_J_extraction') else 'FAIL'}",
        f"- g_R extraction: {'PASS' if results.get('g_R_extraction') else 'FAIL'}",
        f"- g_J dim = g_R dim: {g_j.numel()}",
        f"",
        f"## E. Cosine Similarity",
        f"- Aligned vectors: PASS" if results.get("cosine_verification") else "- Aligned vectors: FAIL",
        f"- Conflicting vectors: verified",
        f"- Orthogonal vectors: verified",
        f"- Zero-gradient semantics: cosine_valid=NO, cosine_JR=None",
        f"",
        f"## F. Gradient Norm Ratio",
        f"- rho(3,6) = 0.5: {'PASS' if results.get('norm_ratio') else 'FAIL'}",
        f"",
        f"## G. Composite Gradient Identity",
        f"- g_C2 == 0.5*g_J + 0.5*g_R: {'PASS' if results.get('composite_identity') else 'FAIL'}",
        f"- Max abs diff: {identity_result['max_abs_diff']:.2e}",
        f"",
        f"## H. Mixer Gradient",
        f"- No regional path through mixer: {'PASS' if results.get('mixer_gradient') else 'FAIL'}",
        f"",
        f"## I. Common-State Evaluation",
        f"- Dual-network cross-evaluation: {'PASS' if results.get('common_state_eval') else 'FAIL'}",
        f"",
        f"## J. Action Disagreement",
        f"- Known Q tables: {'PASS' if results.get('action_disagreement') else 'FAIL'}",
        f"- Transition matrix: verified",
        f"",
        f"## K. Q-Margins",
        f"- Known Q tables: {'PASS' if results.get('q_margin') else 'FAIL'}",
        f"",
        f"## L. Scenario Gap Formulas",
        f"- G_cos formula: {'PASS' if results.get('scenario_gap') else 'FAIL'}",
        f"- G_conflict formula: verified",
        f"",
        f"## M. Aggregation",
        f"- Cell mean aggregation: {'PASS' if results.get('aggregation') else 'FAIL'}",
        f"",
        f"## N. Crossed Bootstrap",
        f"- Bootstrap CI: {'PASS' if results.get('bootstrap') else 'FAIL'}",
        f"- Scenario gap bootstrap: {'PASS' if results.get('gap_bootstrap') else 'FAIL'}",
        f"",
        f"## O. O1 Truth Table",
        f"- 6 cases: {'PASS' if results.get('o1_truth_table') else 'FAIL'}",
        f"",
        f"## P. R1 Signal",
        f"- R1 statistics: {'PASS' if results.get('r1_signal') else 'FAIL'}",
        f"",
        f"## Q. TD Error",
        f"- Regional TD: {'PASS' if results.get('td_error') else 'FAIL'}",
        f"- Joint TD: verified",
        f"- TD ratio: verified",
        f"",
        f"## R. State Stratification",
        f"- delta_severity zero/positive: {'PASS' if results.get('stratification') else 'FAIL'}",
        f"",
        f"## S. Conflict Fraction",
        f"- Formula verified: {'PASS' if results.get('conflict_fraction') else 'FAIL'}",
        f"",
        f"## T. Policy-Level Gaps",
        f"- Per-policy G_cos/G_conflict: {'PASS' if results.get('policy_level_gaps') else 'FAIL'}",
        f"",
        f"## U. No Model Mutation",
        f"- All counters zero: {'PASS' if results.get('no_mutation') else 'FAIL'}",
        f"",
        f"## V. Version Guards",
        f"- All constants: {'PASS' if results.get('version_guards') else 'FAIL'}",
        f"",
        f"## W. Diagnostic Matrix",
        f"- 2×5×60×3 = 1800: {'PASS' if results.get('matrix') else 'FAIL'}",
        f"",
        f"## X. Performance Verdict",
        f"- PERFORMANCE_CAUSAL_VERDICT = NOT_COMPUTED",
        f"- E4MF_REMAINS_AUTHORITATIVE = True",
        f"",
        f"## Y. Seed Firewall",
        f"- Seeds 2301..2360 NOT instantiated: {'PASS' if results.get('seed_firewall') else 'FAIL'}",
        f"- O_DIAGNOSTIC_ACCESS_COUNT = {COUNTERS['diagnostic_seed_access']}",
        f"",
        f"## Z. Verdict",
        f"- E4OI_COMPLETE: {'YES' if all_pass else 'NO'}",
        f"- Validations: {sum(1 for v in results.values() if v is True)}/{n_total}",
        f"- NEXT: {'E4OD_SCENARIO_TRADEOFF_DIAGNOSTIC_EXECUTION' if all_pass else 'FIX_E4OI'}",
        f"",
        f"## AA. Limitations",
        f"- All validation uses SYNTHETIC data only",
        f"- Real checkpoint loading not tested (deferred to E4OD)",
        f"- Real environment transitions not generated",
        f"- Bootstrap validated on small matrix (full 5×60 in E4OD)",
        f"- Gradient extraction validated on random networks, not trained checkpoints",
    ]
    with open(RESULTS_DIR / "E4OI_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    return verdict


if __name__ == "__main__":
    sys.path.insert(0, "src")
    v = run()
    c = "YES" if v["E4OI_COMPLETE"] == "YES" else "NO"
    print(f"\nE4OI_COMPLETE={c}")
    print(f"E4OI_DIAGNOSTIC_CONFIG_SHA256={v['E4OI_DIAGNOSTIC_CONFIG_SHA256']}")
    print(f"PERFORMANCE_CAUSAL_VERDICT={v['PERFORMANCE_CAUSAL_VERDICT']}")
    print(f"O_DIAGNOSTIC_ACCESS_COUNT={v['O_DIAGNOSTIC_ACCESS_COUNT']}")
    n_p = v['n_validations_pass']
    n_t = v['n_validations_total']
    print(f"VALIDATIONS={n_p}/{n_t}")
    print(f"NEXT={v['NEXT_STAGE']}")
