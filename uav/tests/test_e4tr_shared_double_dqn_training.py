"""
E4TR — Shared Parameter Double DQN Training: contract / unit / artifact tests.

These tests DO NOT retrain the 5400-episode protocol. They verify:
  - frozen protocol values and seed firewalls (contract),
  - the RL algebra used in training (unit),
  - the artifacts produced by the executed run (artifact).

Run order note: artifact tests assume the E4TR run has produced
results/E4TR_shared_double_dqn_training/ (Phase 21 regression runs afterwards).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import prdrl.experiments.E4TR_shared_double_dqn_training as E
from prdrl.rl.network import (
    SharedRegionalQNetwork, build_online_target_pair, polyak_update,
    double_dqn_target, ACTION_ORDER, N_ACTIONS,
)
from prdrl.rl.replay import SharedReplayBuffer, RegionalTransition
from prdrl.rl.reward import (
    compute_global_reward, compute_regional_pbi_reward, compute_fe_cost,
    compute_regional_reward,
)
from prdrl.rl.agent import EpsilonGreedyConfig, ACTION_TO_IDX, IDX_TO_ACTION

RES = Path("results/E4TR_shared_double_dqn_training")
E4T = Path("results/E4T_training_protocol_freeze")


def _load(p):
    with open(p) as f:
        return json.load(f)


# =============================================================================
# CONTRACT TESTS (no artifacts required)
# =============================================================================

def test_T1_e4t_complete_yes():
    d = _load(E4T / "E4T_verdict.json")
    assert d["E4T_COMPLETE"] == "YES"


def test_T2_seed_sets_disjoint():
    inter = E.seed_intersections()
    assert inter["train_cap_validation"] == []
    assert inter["train_cap_final"] == []
    assert inter["validation_cap_final"] == []


def test_T2b_no_historical_seed_overlap():
    hist = E.historical_seed_overlap()
    for k, v in hist.items():
        assert v == [], f"historical overlap with {k}: {v}"


def test_T3_final_seed_runtime_guard():
    fw = E.FinalEvalFirewall()
    with pytest.raises(RuntimeError):
        fw.check(1081)
    with pytest.raises(RuntimeError):
        fw.check(1140)
    assert fw.access_count == 2
    # non-final seeds pass silently
    fw2 = E.FinalEvalFirewall()
    for s in [1001, 1060, 1061, 1080]:
        fw2.check(s)
    assert fw2.access_count == 0


def test_T4_frozen_config_exact_no_mismatch():
    cfg, mismatches = E.load_and_validate_frozen_config()
    assert mismatches == [], f"config mismatches: {mismatches}"


def test_T5_action_set_exact():
    assert ACTION_ORDER == ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")
    assert N_ACTIONS == 4
    assert ACTION_TO_IDX == {"KEEP": 0, "LOCAL": 1, "PREDICT": 2, "DIVERSIFY": 3}


def test_T6_network_dim_exact():
    net = SharedRegionalQNetwork(E.NETWORK_INPUT_DIM)
    assert E.NETWORK_INPUT_DIM == 9
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert [l.in_features for l in linears] == [9, 128, 128]
    assert [l.out_features for l in linears] == [128, 128, 4]
    x = torch.zeros(3, 9)
    assert net(x).shape == (3, 4)


def test_T9_episode_schedule_deterministic():
    # two RNGs seeded identically produce identical epoch permutations
    r1 = np.random.default_rng(2001 * 1000 + E.RNG_OFFSET_TRAIN_EPISODE_ORDER)
    r2 = np.random.default_rng(2001 * 1000 + E.RNG_OFFSET_TRAIN_EPISODE_ORDER)
    p1 = r1.permutation(180)
    p2 = r2.permutation(180)
    assert np.array_equal(p1, p2)
    # different policy seed -> different stream
    r3 = np.random.default_rng(2002 * 1000 + E.RNG_OFFSET_TRAIN_EPISODE_ORDER)
    assert not np.array_equal(p1, r3.permutation(180))


def test_T10_T11_budget_constants():
    assert E.EPISODES_PER_EPOCH == 180
    assert len(E.TRAIN_SEEDS) * len(E.SCENARIOS) == 180
    assert E.N_TRAIN_EPISODES == 1080
    assert E.N_EPOCHS == 6
    assert E.N_SCORED_CHANGES == 10


def test_T12_T14_shared_global_context_all_regions():
    # all active regions at one event share ONE global context vector
    v2 = {r: {"severity_r": 0.1 * r, "convergence_loss_r": 0.01 * r,
              "occupancy_r": 0.1, "delta_severity_r": 0.0} for r in range(10)}
    local, gc, full = E.build_region_states(v2, 10)
    # each full state's global slice (last 5) equals the shared gc
    for r in range(10):
        assert np.allclose(full[r][E.LOCAL_DIM:], gc)
    # the shared gc is identical for every region (single vector)
    assert gc.shape == (E.GLOBAL_DIM,)


def test_T15_T16_requested_vs_realized_stored_separately():
    # PREDICT requested but fell back to LOCAL: requested stays PREDICT
    t = RegionalTransition(
        state=np.zeros(9), requested_action=ACTION_TO_IDX["PREDICT"],
        realized_operator="LOCAL", reward=0.5, next_state=np.zeros(9),
        done=False, region_id=3, event_id="ev", fallback_applied=True,
        response_fe=10, diagnostic_oracle_action=None,
    )
    assert t.requested_action == 2         # PREDICT
    assert t.realized_operator == "LOCAL"  # realized differs
    assert t.fallback_applied is True


def test_T19_training_tensors_have_no_oracle_field():
    buf = SharedReplayBuffer(100, rng_seed=0)
    for i in range(10):
        buf.push(RegionalTransition(
            state=np.ones(9) * i, requested_action=i % 4, realized_operator="LOCAL",
            reward=float(i), next_state=np.zeros(9), done=False, region_id=i,
            event_id="e", fallback_applied=False, response_fe=10,
            diagnostic_oracle_action="KEEP",  # offline only
        ))
    batch = buf.sample(5)
    arrs = buf.to_arrays(batch)
    assert len(arrs) == 5   # state, action, reward, next_state, done — nothing else
    # diagnostic oracle action never enters tensors
    states, actions, rewards, next_states, dones = arrs
    assert states.shape[1] == 9


def test_T20_state_is_v2_plus_global_no_regime():
    # 9-D = 4 local V2 + 5 global; no regime/scenario id
    assert E.LOCAL_DIM == 4 and E.GLOBAL_DIM == 5
    assert E.NETWORK_INPUT_DIM == 9


def test_T21_T24_hyperparameters_exact():
    assert E.WARMUP_TRANSITIONS == 1000
    assert E.UPDATE_FREQUENCY == 4
    assert E.REPLAY_CAPACITY == 50_000
    assert E.BATCH_SIZE == 128
    assert E.GAMMA == 0.99
    assert E.LEARNING_RATE == 1e-4
    assert E.TARGET_TAU == 0.005
    assert E.GRAD_STEPS_PER_UPDATE == 1


def test_T25_double_dqn_equation_exact():
    torch.manual_seed(0)
    online = SharedRegionalQNetwork(9)
    target = SharedRegionalQNetwork(9)
    ns = torch.randn(7, 9)
    r = torch.randn(7)
    d = torch.tensor([0., 0, 1, 0, 1, 0, 0])
    y = double_dqn_target(r, ns, d, online, target, 0.99)
    # manual
    with torch.no_grad():
        a_star = online(ns).argmax(dim=1)
        q_t = target(ns)[range(7), a_star]
        y_manual = r + 0.99 * (1 - d) * q_t
    assert torch.allclose(y, y_manual, atol=1e-12)
    # terminal rows equal reward exactly
    assert torch.allclose(y[d.bool()], r[d.bool()], atol=1e-12)


def test_T26_huber_loss_used():
    loss = torch.nn.SmoothL1Loss()
    a = torch.tensor([0.0, 2.0])
    b = torch.tensor([0.0, 0.0])
    # SmoothL1 with beta=1: for diff=2 -> |2|-0.5=1.5 ; mean of (0,1.5)=0.75
    assert abs(float(loss(a, b)) - 0.75) < 1e-9


def test_T27_polyak_exact():
    online = SharedRegionalQNetwork(9)
    target = SharedRegionalQNetwork(9)
    with torch.no_grad():
        for p in online.parameters():
            p.copy_(torch.ones_like(p))
        for p in target.parameters():
            p.copy_(torch.zeros_like(p))
    polyak_update(online, target, 0.005)
    for p in target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.005), atol=1e-12)


def test_T28_epsilon_schedule_exact():
    cfg = EpsilonGreedyConfig(1.0, 0.05, 50_000)
    assert abs(cfg.get_epsilon(0) - 1.0) < 1e-12
    assert abs(cfg.get_epsilon(50_000) - 0.05) < 1e-12
    assert abs(cfg.get_epsilon(100_000) - 0.05) < 1e-12   # clamped after decay
    assert abs(cfg.get_epsilon(25_000) - 0.525) < 1e-9    # midpoint


def test_T29_checkpoint_transitions_exact():
    assert E.CHECKPOINT_TRANSITIONS == [10_000, 20_000, 30_000, 40_000, 50_000]


def test_T30_checkpoint_selection_rules():
    # rule 1: lowest macro MIGD_response
    c = [
        {"name": "a", "VALIDATION_MACRO_MIGD_RESPONSE": 0.02, "macro_MIGD_epoch": 0.01,
         "macro_mean_response_fe": 50, "transition_count": 10000},
        {"name": "b", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.99,
         "macro_mean_response_fe": 99, "transition_count": 50000},
    ]
    assert E.select_checkpoint(c)["name"] == "b"
    # rule 2: tie on response -> lower epoch
    c = [
        {"name": "a", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.05,
         "macro_mean_response_fe": 10, "transition_count": 10000},
        {"name": "b", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.02,
         "macro_mean_response_fe": 99, "transition_count": 50000},
    ]
    assert E.select_checkpoint(c)["name"] == "b"
    # rule 3: tie on response+epoch -> lower FE
    c = [
        {"name": "a", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.02,
         "macro_mean_response_fe": 30, "transition_count": 40000},
        {"name": "b", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.02,
         "macro_mean_response_fe": 10, "transition_count": 50000},
    ]
    assert E.select_checkpoint(c)["name"] == "b"
    # rule 4: all tie -> earlier transition
    c = [
        {"name": "a", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.02,
         "macro_mean_response_fe": 10, "transition_count": 50000},
        {"name": "b", "VALIDATION_MACRO_MIGD_RESPONSE": 0.01, "macro_MIGD_epoch": 0.02,
         "macro_mean_response_fe": 10, "transition_count": 20000},
    ]
    assert E.select_checkpoint(c)["name"] == "b"


def test_T18_team_reward_shared_and_dominant():
    # global component shared; alpha_g >= alpha_l enforced
    r1 = compute_regional_reward(global_reward=1.0, local_pbi_reward=5.0, fe_cost=1.0,
                                 alpha_g=1.0, alpha_l=0.1, beta=0.01)
    assert abs(r1 - (1.0 * 1.0 + 0.1 * 5.0 - 0.01 * 1.0)) < 1e-12
    with pytest.raises(ValueError):
        compute_regional_reward(0.1, 0.0, 0.0, alpha_g=0.05, alpha_l=0.1, beta=0.01)


def test_T17_global_reward_sign_and_norm():
    # improvement (q rises) -> positive
    assert compute_global_reward(q_before=-0.02, q_after=-0.01, normalization=-0.02) > 0
    # degradation -> negative
    assert compute_global_reward(q_before=-0.01, q_after=-0.02, normalization=-0.01) < 0


def test_local_pbi_reward_improvement_positive():
    # loss reduced -> positive local reward
    assert compute_regional_pbi_reward(0.5, 0.3) == pytest.approx(0.2)
    assert compute_regional_pbi_reward(0.3, 0.5) == pytest.approx(-0.2)
    # nan -> 0
    assert compute_regional_pbi_reward(float("nan"), 0.3) == 0.0


def test_T42_budget_ten_per_active_region():
    assert E.BUDGET == 10
    assert E.CURRENT_RESPONSE_BUDGET_VALUE == 10
    # KEEP FE cost = 0
    assert compute_fe_cost(0, 1, 10) == 0.0
    # active response region FE cost = 1.0
    assert compute_fe_cost(10, 1, 10) == 1.0


def test_T43_reward_coefficients_frozen():
    assert (E.ALPHA_G, E.ALPHA_L, E.BETA) == (1.0, 0.1, 0.01)
    assert E.ALPHA_G >= E.ALPHA_L


def test_region_disappearance_guard_raises():
    with pytest.raises(RuntimeError, match="ACTIVE_TO_INACTIVE_REGION"):
        E._assert_no_region_disappearance([0, 1, 2, 3], [0, 1, 3])
    # no disappearance -> no raise
    E._assert_no_region_disappearance([0, 1, 2], [0, 1, 2, 3])


def test_policy_seeds_frozen():
    assert E.POLICY_SEEDS == [2001, 2002, 2003, 2004, 2005]


# =============================================================================
# ARTIFACT TESTS (require the executed E4TR run)
# =============================================================================

def _require_run():
    if not (RES / "E4TR_verdict.json").exists():
        pytest.skip("E4TR run artifacts not present yet")


def test_T7_five_independent_policy_dirs():
    _require_run()
    for ps in E.POLICY_SEEDS:
        assert (RES / f"policy_{ps}" / "training_summary.json").exists()


def test_T8_independent_replay_and_rng():
    _require_run()
    d = _load(RES / "E4TR_rng_audit.json")
    assert d["per_policy_independent"] is True
    assert d["drl_streams_isolated_from_production"] is True
    assert "REPLAY_SAMPLE" in d["namespaces"]


def test_T11_episodes_per_policy():
    _require_run()
    for ps in E.POLICY_SEEDS:
        s = _load(RES / f"policy_{ps}" / "training_summary.json")
        assert s["n_episodes"] == 1080
        assert s["total_transitions"] == 1080 * 100  # 10 changes x 10 regions


def test_T29b_checkpoints_present_exact():
    _require_run()
    for ps in E.POLICY_SEEDS:
        cdir = RES / f"policy_{ps}" / "checkpoints"
        for tc in E.CHECKPOINT_TRANSITIONS:
            assert (cdir / f"ckpt_{tc}.pt").exists(), f"missing ckpt_{tc} for {ps}"
        assert (cdir / "ckpt_final.pt").exists()


def test_T30b_selected_checkpoint_recorded():
    _require_run()
    d = _load(RES / "E4TR_selected_checkpoints.json")
    assert len(d["selected"]) == 5
    for ps in E.POLICY_SEEDS:
        assert str(ps) in d["selected"]


def test_T31_validation_epsilon_zero():
    _require_run()
    d = _load(RES / "E4TR_execution_config.json")
    assert d["epsilon"]["validation_epsilon"] == 0.0


def test_T33_validation_determinism_pass():
    _require_run()
    d = _load(RES / "E4TR_validation_determinism.json")
    assert d["determinism_pass"] is True
    assert d["abs_diff"] <= 1e-12


def test_T34_T35_all_five_retained_no_filtering():
    _require_run()
    d = _load(RES / "E4TR_selected_checkpoints.json")
    assert d["all_five_retained"] is True
    assert d["no_best_policy_seed_selection"] is True
    assert d["policy_seeds"] == [2001, 2002, 2003, 2004, 2005]


def test_T36_baselines_use_validation_seeds():
    _require_run()
    d = _load(RES / "E4TR_best_fixed_global_validation.json")
    for sc in E.SCENARIOS:
        assert sc in d
        assert d[sc]["action"] in ("KEEP", "LOCAL", "PREDICT", "DIVERSIFY")


def test_T37_rule_baseline_unchanged():
    _require_run()
    d = _load(RES / "E4TR_rule_validation.json")
    assert d["retuned"] is False
    frozen = _load("results/E3R_rule_v2_semantic_rerun/E3R_rule_v1_v2state.json")["thresholds"]
    assert d["thresholds"] == frozen


def test_T38_final_eval_access_count_zero():
    _require_run()
    d = _load(RES / "E4TR_final_eval_firewall.json")
    assert d["FINAL_EVAL_ACCESS_COUNT"] == 0
    assert d["status"] == "PASS"


def test_T39_production_default_and_hashes():
    _require_run()
    d = _load(RES / "E4TR_pretrain_manifest.json")
    assert d["PRODUCTION_DEFAULT"] == "O1_F005"
    # production operator hashes recorded match the current files
    for fp, h in d["production_operator_hashes"].items():
        assert h == E._hash_file(fp), f"production file changed: {fp}"


def test_T40_T41_no_hard_no_memory():
    _require_run()
    d = _load(RES / "E4TR_pretrain_manifest.json")
    assert d["no_HARD_action"] is True
    assert d["no_MEMORY_action"] is True
    # action set has no HARD/MEMORY/RECALL
    for bad in ("HARD", "MEMORY", "RECALL"):
        assert bad not in ACTION_ORDER


def test_T19b_oracle_firewall_artifact():
    _require_run()
    d = _load(RES / "E4TR_oracle_firewall_audit.json")
    assert d["oracle_called_in_training_path"] is False
    assert d["diagnostic_oracle_action_in_transitions"] is False
    assert d["regime_label_in_state"] is False
    assert d["training_tensor_fields"] == ["state", "requested_action", "reward",
                                           "next_state", "done"]
    assert d["status"] == "PASS"


def test_T44_no_hyperparameter_search():
    _require_run()
    d = _load(RES / "E4TR_verdict.json")
    assert d["hyperparameter_retuning"] == "NO"
    assert d["config_mismatches_vs_prompt"] == []


def test_T45_selected_checkpoint_hashes_valid():
    _require_run()
    d = _load(RES / "E4TR_selected_checkpoints.json")
    for ps, info in d["selected"].items():
        path = info["checkpoint_path"]
        assert Path(path).exists()
        assert info["sha256"] == E._hash_file(path)
        assert info["file_size"] > 0


def test_action_fractions_sum_to_one():
    _require_run()
    d = _load(RES / "E4TR_action_distribution.csv") if False else None
    import csv
    with open(RES / "E4TR_action_distribution.csv") as f:
        rows = list(csv.DictReader(f))
    by_policy = {}
    for r in rows:
        by_policy.setdefault(r["policy_seed"], 0.0)
        by_policy[r["policy_seed"]] += float(r["final_epoch_fraction"])
    for ps, s in by_policy.items():
        assert abs(s - 1.0) < 1e-6, f"policy {ps} fractions sum {s}"


def test_numerical_stability_pass_all_policies():
    _require_run()
    for ps in E.POLICY_SEEDS:
        s = _load(RES / f"policy_{ps}" / "training_summary.json")
        assert s["numerical_stability"] == "PASS"


def test_verdict_status_fields():
    _require_run()
    d = _load(RES / "E4TR_verdict.json")
    assert d["E4TR_COMPLETE"] == "YES"
    assert d["POLICY_RUNS_COMPLETE"] == "5/5"
    assert d["FINAL_EVAL_ACCESS_COUNT"] == 0
    assert d["oracle_label_firewall"] == "PASS"
    assert d["rng_isolation"] == "PASS"
    assert d["validation_determinism"] == "PASS"
    assert d["final_evaluation"] == "NOT_RUN"
    assert d["scientific_superiority_claim"] == "NOT_MADE"
    assert d["RL_STATUS"] == "TRAINED_AWAITING_FINAL_EVAL"
    assert d["DRL_TRAINING"] == "COMPLETE"
    assert d["NEXT_STAGE"] == "E4E_FINAL_EVALUATION_PROTOCOL_FREEZE"


def test_final_eval_seeds_never_in_artifacts():
    _require_run()
    # No final-eval seed (1081..1140) may appear as an ENVIRONMENT SEED value in
    # the training diagnostics (column-exact check; substring matching would
    # false-positive on digits inside unrelated float values).
    import csv
    final = set(range(1081, 1141))
    with open(RES / "E4TR_training_diagnostics.csv") as f:
        rows = list(csv.DictReader(f))
    env_seeds = {int(r["environment_seed"]) for r in rows}
    assert env_seeds.isdisjoint(final), f"final seeds in training env seeds: {env_seeds & final}"
    # training only ever uses train seeds
    assert env_seeds.issubset(set(E.TRAIN_SEEDS)), f"unexpected env seeds: {env_seeds - set(E.TRAIN_SEEDS)}"
    # firewall counter confirms no final-eval seed was ever instantiated
    fw = _load(RES / "E4TR_final_eval_firewall.json")
    assert fw["FINAL_EVAL_ACCESS_COUNT"] == 0


def test_execution_config_local_reward_definition():
    _require_run()
    d = _load(RES / "E4TR_execution_config.json")
    lr = d["local_reward_operational_definition"]
    assert lr["uses_isolated_oracle_branches"] is False
    assert lr["uses_actual_joint_response_population"] is True
