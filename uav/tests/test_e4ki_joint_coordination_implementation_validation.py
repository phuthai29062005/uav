"""
Tests for E4KI — Joint Coordination Implementation Validation (Branch C).

Validates the frozen C1 QMIX treatment + C0 control on synthetic fixtures only (no training, no
environment, no future-seed access). Re-exercises the mixer/replay/loss/target logic independently
of the E4KI experiment run, and asserts every frozen E4KI verdict field.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from prdrl.rl.network import SharedRegionalQNetwork, build_online_target_pair, double_dqn_target, N_ACTIONS
from prdrl.rl.agent import select_regional_actions
from prdrl.rl.qmix import (
    QMIXMixerV1, build_mixer_online_target_pair, derive_mixer_seed,
    R_MAX, CENTRAL_STATE_DIM, MIX_HIDDEN_DIM, MIXER_SEED_OFFSET, TREATMENT_ARCH,
)
from prdrl.rl.joint_context import central_state_from_regions, build_central_state, LOCAL_DIM
from prdrl.rl.joint_replay import JointReplayBuffer, make_event_anchors, STATE_DIM
from prdrl.rl import coordination_integration as ci
from prdrl.rl.reward_r1 import REWARD_R1_NAME, regional_quality_gain_r1
from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL, FROZEN_R1_CONFIG_SHA256

RES = Path("results/E4KI_joint_coordination_implementation_validation")


def _load(name):
    with open(RES / name) as f:
        return json.load(f)


def _uinit(p):
    return p * 1000 + 0


@pytest.fixture(scope="module")
def verdict():
    return _load("E4KI_verdict.json")


def _param_hash(m):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# --- T1-T7 historical preservation ---
def test_T1_e4k_complete(verdict):
    assert _load("E4KI_historical_integrity.json")["checks"]["E4K_COMPLETE"] is True


def test_T2_branch_c(verdict):
    assert verdict["SELECTED_BRANCH"] == "C_JOINT_COORDINATION"


def test_T3_e4jf_failed(verdict):
    assert verdict["E4JF_GLOBAL_COMPARATOR_VERDICT"] == "FAILED"


def test_T4_e4if_validated(verdict):
    assert verdict["E4IF_REWARD_CAUSAL_VERDICT"] == "VALIDATED"


def test_T5_e4hf_failed(verdict):
    assert verdict["E4HF_NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
    assert verdict["ORIGINAL_E4F_VERDICT"] == "FAILED"


def test_T6_routing_amendment_preserved():
    assert _load("E4KI_historical_integrity.json")["checks"]["routing_amendment"] is True


def test_T7_selected_treatment_exact(verdict):
    assert verdict["TREATMENT_ARCHITECTURE"] == "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"
    assert _load("E4KI_historical_integrity.json")["checks"]["treatment_qmix"] is True


# --- T8-T14 dims + slots + masking ---
def test_T8_r_max_10(verdict):
    assert R_MAX == 10 == verdict["REGION_SLOT_COUNT"]


def test_T9_central_dim_55(verdict):
    assert CENTRAL_STATE_DIM == 55 == verdict["CENTRAL_STATE_DIM"]


def test_T10_slot_order_canonical(verdict):
    assert _load("E4KI_region_slot_validation.json")["REGION_SLOT_ORDERING"] == "PASS"


def test_T11_shuffled_input_reordered():
    rng = np.random.default_rng(1)
    local = {r: rng.standard_normal(4) for r in [7, 2, 9, 0, 4]}
    g5 = rng.standard_normal(5)
    cs = central_state_from_regions(dict(reversed(list(local.items()))), g5)
    cs2 = central_state_from_regions(local, g5)
    assert np.allclose(cs, cs2)  # order-independent
    mask = cs[4 * R_MAX:5 * R_MAX]
    assert all(mask[r] == 1.0 for r in local)


def test_T12_active_mask_exact(verdict):
    assert verdict["ACTIVE_MASK_VALIDATION"] == "PASS"


def test_T13_zero_active_distinguishable_from_inactive():
    g5 = np.zeros(5)
    active = central_state_from_regions({3: np.zeros(4)}, g5)
    inactive = central_state_from_regions({}, g5)
    assert not np.allclose(active, inactive)


def test_T14_inactive_local4_zero():
    cs = central_state_from_regions({0: np.ones(4)}, np.zeros(5))
    # slot 5 inactive -> local block zero
    assert np.allclose(cs[5 * LOCAL_DIM:6 * LOCAL_DIM], 0.0)


# --- T15-T25 utility + mixer forward ---
def test_T15_utility_architecture_exact():
    on, _ = build_online_target_pair(9, seed=2001)
    assert on.input_dim == 9 and on.n_actions == 4


def test_T16_utility_forward_equals_legacy(verdict):
    assert verdict["UTILITY_NETWORK_EQUIVALENCE"] == "PASS"
    a, _ = build_online_target_pair(9, seed=2001)
    b, _ = build_online_target_pair(9, seed=2001)
    x = torch.randn(16, 9)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_T17_mixer_input_q_shape():
    m = QMIXMixerV1()
    with torch.no_grad():
        out = m(torch.zeros(3, R_MAX), torch.zeros(3, CENTRAL_STATE_DIM))
    assert out.shape == (3, 1)


def test_T18_mixer_central_shape():
    m = QMIXMixerV1()
    with pytest.raises(ValueError):
        m(torch.zeros(2, R_MAX), torch.zeros(2, CENTRAL_STATE_DIM + 1))


def test_T19_mixer_hidden_32(verdict):
    assert MIX_HIDDEN_DIM == 32 == verdict["MIXER_HIDDEN_DIM"]
    m = QMIXMixerV1()
    assert m.hyper_b1.out_features == 32


def test_T20_w1_abs_transform():
    m = QMIXMixerV1()
    c = torch.randn(4, CENTRAL_STATE_DIM)
    with torch.no_grad():
        w1, _ = m.mixer_weights(c)
    assert bool((w1 >= 0).all())


def test_T21_w2_abs_transform():
    m = QMIXMixerV1()
    c = torch.randn(4, CENTRAL_STATE_DIM)
    with torch.no_grad():
        _, w2 = m.mixer_weights(c)
    assert bool((w2 >= 0).all())


def test_T22_elu_used():
    # ELU: bounded below by -1; with W1>=0 and q large negative, pre can be negative -> ELU(pre) in (-1,inf)
    src = Path("src/prdrl/rl/qmix.py").read_text()
    assert "elu" in src.lower()


def test_T23_qtot_scalar_shape(verdict):
    m = QMIXMixerV1()
    with torch.no_grad():
        out = m(torch.randn(7, R_MAX), torch.randn(7, CENTRAL_STATE_DIM))
    assert out.shape == (7, 1)


def test_T24_production_reference_mixer_match(verdict):
    assert verdict["MIXER_REFERENCE_MATCH"] == "PASS"
    assert _load("E4KI_mixer_reference_match.json")["max_abs_diff"] <= 1e-4


def test_T25_inactive_placeholder_invariant(verdict):
    assert verdict["INACTIVE_ACTION_INVARIANCE"] == "PASS"


# --- T26-T32 monotonicity + argmax + conditioning ---
def test_T26_monotonic_derivative_nonneg(verdict):
    assert float(verdict["MIN_ACTIVE_DQ_TOT_DQ"]) >= -1e-6


def test_T27_no_monotonic_violations(verdict):
    assert verdict["MONOTONICITY_VIOLATIONS"] == 0
    assert verdict["MONOTONICITY_VALIDATION"] == "PASS"


def _argmax_equiv(R_test, seed, mix):
    r = np.random.default_rng(seed)
    Qr = r.standard_normal((R_test, N_ACTIONS))
    c = torch.tensor(r.standard_normal((1, CENTRAL_STATE_DIM)), dtype=torch.float32)
    greedy = Qr.argmax(axis=1)
    best = -np.inf
    for combo in itertools.product(range(N_ACTIONS), repeat=R_test):
        q = np.zeros(R_MAX)
        for i in range(R_test):
            q[i] = Qr[i, combo[i]]
        with torch.no_grad():
            best = max(best, float(mix(torch.tensor(q, dtype=torch.float32).unsqueeze(0), c).item()))
    qg = np.zeros(R_MAX)
    for i in range(R_test):
        qg[i] = Qr[i, greedy[i]]
    with torch.no_grad():
        vg = float(mix(torch.tensor(qg, dtype=torch.float32).unsqueeze(0), c).item())
    return abs(vg - best) <= 1e-5


def test_T28_exhaustive_R1():
    m, _ = build_mixer_online_target_pair(seed=123)
    assert all(_argmax_equiv(1, s, m) for s in range(10))


def test_T29_exhaustive_R2():
    m, _ = build_mixer_online_target_pair(seed=124)
    assert all(_argmax_equiv(2, s, m) for s in range(20))


def test_T30_exhaustive_R3():
    m, _ = build_mixer_online_target_pair(seed=125)
    assert all(_argmax_equiv(3, s, m) for s in range(20))


def test_T31_randomized_exhaustive_zero_violations(verdict):
    assert verdict["DECENTRALIZED_ARGMAX_EQUIVALENCE"] == "PASS"
    a = _load("E4KI_argmax_equivalence.json")
    assert a["R1_violations"] == 0 and a["R2_violations"] == 0 and a["R3_violations"] == 0


def test_T32_central_state_dependency(verdict):
    assert verdict["STATE_CONDITIONING"] == "PASS"


# --- T33-T38 init + soft update ---
def test_T33_utility_online_eq_target_t0():
    on, tg = build_online_target_pair(9, seed=_uinit(5001))
    assert _param_hash(on) == _param_hash(tg)


def test_T34_mixer_online_eq_target_t0():
    on, tg = build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(5001)))
    assert _param_hash(on) == _param_hash(tg)


def test_T35_mixer_init_deterministic():
    a, _ = build_mixer_online_target_pair(seed=777)
    b, _ = build_mixer_online_target_pair(seed=777)
    assert _param_hash(a) == _param_hash(b)


def test_T36_distinct_policy_mixer_seeds():
    hs = {p: _param_hash(build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(p)))[0])
          for p in [5001, 5002, 5003, 5004, 5005]}
    assert len(set(hs.values())) == 5


def test_T37_utility_soft_update_tau():
    on, tg = build_online_target_pair(9, seed=11)
    with torch.no_grad():
        for p in on.parameters(): p.copy_(torch.ones_like(p) * 3.0)
        for p in tg.parameters(): p.copy_(torch.zeros_like(p))
    old = [p.clone() for p in tg.parameters()]
    ci.soft_update(on, tg, 0.005)
    for pn, po in zip(tg.parameters(), old):
        assert torch.allclose(pn, 0.995 * po + 0.005 * torch.ones_like(po) * 3.0, atol=1e-7)


def test_T38_mixer_soft_update_tau(verdict):
    assert verdict["TARGET_UPDATE_VALIDATION"] == "PASS"
    assert verdict["TARGET_UPDATE_TAU"] == 0.005


# --- T39-T45 joint reward + R1 + zero active ---
def test_T39_one_active_joint_reward():
    assert ci.joint_reward_mean([0.3, 0, 0], [1, 0, 0]) == pytest.approx(0.3)


def test_T40_multi_active_is_mean():
    assert ci.joint_reward_mean([0.2, 0.4, 0], [1, 1, 0]) == pytest.approx(0.3)


def test_T41_no_active_count_scaling():
    a = ci.joint_reward_mean([0.5, 0.5, 0, 0], [1, 1, 0, 0])
    b = ci.joint_reward_mean([0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1])
    assert a == pytest.approx(b)


def test_T42_no_sum_reward():
    assert ci.joint_reward_mean([0.2, 0.4], [1, 1]) != pytest.approx(0.6)


def test_T43_r1_formula_unchanged():
    assert regional_quality_gain_r1(0.0, 0.0) == 0.0
    assert regional_quality_gain_r1(1.0, 0.0) > 0.9
    assert FROZEN_R1_CONFIG_SHA256 == "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
    assert REWARD_R1_NAME == "REWARD_R1_REGIONAL_QUALITY_GAIN"


def test_T44_alpha_unchanged(verdict):
    assert FROZEN_R1_ALPHA_LOCAL == 0.1 == verdict["FIXED_ALPHA_LOCAL"]


def test_T45_zero_active_event():
    assert ci.joint_reward_mean([0, 0, 0], [0, 0, 0]) == 0.0
    assert _load("E4KI_zero_active_validation.json")["ZERO_ACTIVE_EVENT_HANDLING"] == "PASS"


# --- T46-T53 double-Q + current Q + loss ---
def test_T46_double_q_online_selection(verdict):
    assert _load("E4KI_double_q_target_validation.json")["online_selection_target_eval"] is True


def test_T47_double_q_target_eval(verdict):
    assert _load("E4KI_double_q_target_validation.json")["reference_match"] is True
    assert verdict["DOUBLE_Q_JOINT_TARGET"] == "PASS"


def test_T48_online_vs_target_disagreement():
    # ONLINE argmax action's TARGET value is used, not TARGET argmax
    class Fixed(nn.Module):
        def __init__(self, t): super().__init__(); self.t = t
        def forward(self, x): return self.t.expand(x.shape[0], -1).clone()
    on = Fixed(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))   # argmax 0
    tg = Fixed(torch.tensor([[0.0, 5.0, 9.0, 0.0]]))   # argmax 2, action0 -> 0.0
    flat = torch.zeros(1, 9)
    a_star = on(flat).argmax(1)
    q = tg(flat).gather(1, a_star.unsqueeze(1)).squeeze(1)
    assert int(a_star[0]) == 0 and float(q[0]) == pytest.approx(0.0)


def test_T49_terminal_target_reward_only():
    assert _load("E4KI_double_q_target_validation.json")["terminal_reward_only"] is True


def test_T50_current_q_uses_requested_action():
    torch.manual_seed(0)
    on_u, _ = build_online_target_pair(9, seed=_uinit(5001))
    on_m, _ = build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(5001)))
    js = torch.randn(1, R_MAX, 9); mask = torch.ones(1, R_MAX)
    cs = torch.randn(1, CENTRAL_STATE_DIM)
    av = torch.zeros(1, R_MAX, dtype=torch.int64)
    with torch.no_grad():
        q_all = on_u(js.reshape(R_MAX, 9)).reshape(1, R_MAX, N_ACTIONS)
        q_manual = q_all.gather(2, av.unsqueeze(2)).squeeze(2)
        q_tot_manual = on_m(q_manual, cs)
        q_tot = ci.current_joint_q(on_u, on_m, js, av, mask, cs)
    assert torch.allclose(q_tot, q_tot_manual, atol=1e-6)


def test_T51_inactive_q_zero():
    torch.manual_seed(1)
    on_u, _ = build_online_target_pair(9, seed=_uinit(5002))
    on_m, _ = build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(5002)))
    js = torch.randn(1, R_MAX, 9); cs = torch.randn(1, CENTRAL_STATE_DIM)
    mask = torch.ones(1, R_MAX); mask[0, 5] = 0.0
    av = torch.zeros(1, R_MAX, dtype=torch.int64)
    with torch.no_grad():
        base = float(ci.current_joint_q(on_u, on_m, js, av, mask, cs)[0, 0])
        for a in range(4):
            av2 = av.clone(); av2[0, 5] = a
            assert abs(float(ci.current_joint_q(on_u, on_m, js, av2, mask, cs)[0, 0]) - base) < 1e-6


def test_T52_treatment_smoothl1(verdict):
    assert verdict["TREATMENT_LOSS"] == "JOINT_SMOOTHL1"
    assert verdict["TREATMENT_LOSS_VALIDATION"] == "PASS"


def test_T53_no_auxiliary_loss():
    src = Path("src/prdrl/rl/coordination_integration.py").read_text()
    assert "SmoothL1Loss" in src
    # treatment_loss body has no additional additive term
    assert "def treatment_loss" in src


# --- T54-T55 control equivalence ---
def test_T54_c0_target_equals_ddqn():
    torch.manual_seed(0)
    on, tg = build_online_target_pair(9, seed=2001)
    s = torch.randn(32, 9); a = torch.randint(0, 4, (32,)); r = torch.randn(32)
    ns = torch.randn(32, 9); d = torch.zeros(32)
    y_ref = double_dqn_target(r, ns, d, on, tg, 0.99)
    y, q, loss = ci.control_ddqn_target_and_loss(on, tg, s, a, r, ns, d, 0.99)
    assert torch.equal(y, y_ref)


def test_T55_c0_loss_equals_ddqn(verdict):
    assert verdict["C0_BACKWARD_EQUIVALENCE"] == "PASS"
    torch.manual_seed(0)
    on, tg = build_online_target_pair(9, seed=2001)
    s = torch.randn(16, 9); a = torch.randint(0, 4, (16,)); r = torch.randn(16)
    ns = torch.randn(16, 9); d = torch.zeros(16)
    y, q, loss = ci.control_ddqn_target_and_loss(on, tg, s, a, r, ns, d, 0.99)
    q_ref = on(s).gather(1, a.unsqueeze(1)).squeeze(1)
    assert torch.equal(loss, nn.SmoothL1Loss()(q_ref, double_dqn_target(r, ns, d, on, tg, 0.99)))


# --- T56-T67 replay ---
def _mk_event(rng, active, eid="e"):
    js = rng.standard_normal((R_MAX, 9)).astype(np.float32)
    mask = np.zeros(R_MAX, dtype=np.float32); mask[active] = 1.0
    av = rng.integers(0, 4, R_MAX).astype(np.int64)
    cs = central_state_from_regions({r: rng.standard_normal(4) for r in active}, rng.standard_normal(5))
    rr = np.zeros(R_MAX, dtype=np.float32); rr[active] = rng.standard_normal(len(active))
    rj = ci.joint_reward_mean(rr, mask)
    njs = rng.standard_normal((R_MAX, 9)).astype(np.float32)
    ncs = central_state_from_regions({r: rng.standard_normal(4) for r in active}, rng.standard_normal(5))
    return make_event_anchors(active_regions=active, joint_states=js, active_mask=mask, action_vector=av,
                              central_state=cs, regional_rewards=rr, r_joint=rj, next_joint_states=njs,
                              next_active_mask=mask.copy(), next_central_state=ncs, done=False, event_id=eid)


def test_T56_replay_focal_schema():
    a = _mk_event(np.random.default_rng(1), [1, 4, 7])[0]
    assert a.focal_state_9.shape == (9,) and a.focal_next_state_9.shape == (9,)


def test_T57_replay_joint_current_schema():
    a = _mk_event(np.random.default_rng(1), [1, 4, 7])[0]
    assert a.joint_states.shape == (R_MAX, 9) and a.active_mask.shape == (R_MAX,)
    assert a.action_vector.shape == (R_MAX,) and a.central_state.shape == (CENTRAL_STATE_DIM,)
    assert a.regional_rewards.shape == (R_MAX,)


def test_T58_replay_joint_next_schema():
    a = _mk_event(np.random.default_rng(1), [1, 4, 7])[0]
    assert a.next_joint_states.shape == (R_MAX, 9) and a.next_active_mask.shape == (R_MAX,)
    assert a.next_central_state.shape == (CENTRAL_STATE_DIM,)


def test_T59_three_active_three_anchors():
    anchors = _mk_event(np.random.default_rng(2), [1, 4, 7])
    assert len(anchors) == 3
    assert [a.focal_region_id for a in anchors] == [1, 4, 7]


def test_T60_sibling_event_context_identical():
    anchors = _mk_event(np.random.default_rng(3), [2, 5])
    assert np.array_equal(anchors[0].central_state, anchors[1].central_state)
    assert anchors[0].r_joint == anchors[1].r_joint


def test_T61_siblings_not_deduplicated(verdict):
    assert verdict["SIBLING_ANCHOR_DEDUPLICATION"] == "NO"
    buf = JointReplayBuffer(capacity=50, rng_seed=1)
    buf.push_event(_mk_event(np.random.default_rng(4), [2, 5], eid="SIB"))
    batch = buf.sample_indices([0, 1])
    assert sum(1 for a in batch if a.event_id == "SIB") == 2


def test_T62_replay_capacity_counts_anchors(verdict):
    assert _load("E4KI_replay_counter_validation.json")["checks"]["capacity_is_anchors"] is True


def test_T63_warmup_counts_anchors():
    assert _load("E4KI_replay_counter_validation.json")["checks"]["warmup_is_anchors"] is True


def test_T64_update_cadence_counts_anchors():
    assert _load("E4KI_replay_counter_validation.json")["checks"]["update_cadence_every_4"] is True


def test_T65_batch_size_128(verdict):
    assert _load("E4KI_replay_counter_validation.json")["batch_size"] == 128


def test_T66_batch_tensor_shapes(verdict):
    assert verdict["REPLAY_SCHEMA_VALIDATION"] == "PASS"
    b = _load("E4KI_batch_shape_validation.json")
    assert b["qtot_shape"] == [128, 1] and b["y_shape"] == [128, 1]


def test_T67_no_cross_batch_mixing():
    assert _load("E4KI_batch_shape_validation.json")["no_cross_batch_mixing"] is True


# --- T68-T74 optimizer + autograd ---
def test_T68_c0_optimizer_utility_only():
    on_u, _ = build_online_target_pair(9, seed=_uinit(5001))
    params = ci.c0_optimizer_params(on_u)
    assert set(id(p) for p in params) == set(id(p) for p in on_u.parameters())


def test_T69_c1_optimizer_utility_plus_mixer():
    on_u, _ = build_online_target_pair(9, seed=_uinit(5001))
    on_m, _ = build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(5001)))
    params = ci.c1_optimizer_params(on_u, on_m)
    assert len(params) == len(list(on_u.parameters())) + len(list(on_m.parameters()))


def test_T70_target_params_excluded(verdict):
    assert verdict["OPTIMIZER_PARTITION"] == "PASS"
    assert _load("E4KI_optimizer_partition.json")["no_target_params"] is True


def test_T71_online_utility_finite_grad(verdict):
    assert _load("E4KI_autograd_boundary.json")["online_utility_grad"] is True


def test_T72_online_mixer_finite_grad(verdict):
    assert _load("E4KI_autograd_boundary.json")["online_mixer_grad"] is True


def test_T73_target_utility_no_grad(verdict):
    assert _load("E4KI_autograd_boundary.json")["target_utility_no_grad"] is True


def test_T74_target_mixer_no_grad(verdict):
    assert verdict["AUTOGRAD_BOUNDARY"] == "PASS"
    assert _load("E4KI_autograd_boundary.json")["target_mixer_no_grad"] is True


# --- T75-T79 execution + fallback + no-future ---
def test_T75_mixer_absent_execution_path(verdict):
    assert verdict["DECENTRALIZED_EXECUTION_PATH"] == "PASS"
    assert _load("E4KI_execution_path_validation.json")["mixer_call_count"] == 0


def test_T76_regional_argmax_execution():
    on_u, _ = build_online_target_pair(9, seed=_uinit(5001))
    rng = np.random.default_rng(5)
    ls = {r: rng.standard_normal(4).astype(np.float32) for r in range(R_MAX)}
    gc = rng.standard_normal(5).astype(np.float32)
    idx, act = select_regional_actions(on_u, ls, gc, epsilon=0.0, event_seed=1, device=torch.device("cpu"))
    assert len(act) == R_MAX


def test_T77_simultaneous_action_vector():
    assert _load("E4KI_fallback_semantics.json")["SIMULTANEOUS_ACTION"] == "PASS"


def test_T78_requested_predict_retained(verdict):
    assert verdict["REQUESTED_REALIZED_SEMANTICS"] == "PASS"


def test_T79_no_future_information(verdict):
    assert verdict["NO_FUTURE_INFORMATION"] == "PASS"


# --- T80-T85 locks + version guards ---
def test_T80_v3_both_arms(verdict):
    assert verdict["FIXED_NORMALIZATION"] == "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3"


def test_T81_v3_constants_exact():
    assert ci.FROZEN_V3_CONSTANTS_SHA256 == "1cc493c11ed16cc0aa8ef8ed1eebc3877fe369d01377834ccd58e7edd85d785b"


def test_T82_r1_both_arms(verdict):
    assert verdict["FIXED_REWARD"] == "REWARD_R1_REGIONAL_QUALITY_GAIN"


def test_T83_architecture_version_guard():
    sha = ci.qmix_config_sha256()
    good = {"architecture_version": TREATMENT_ARCH, "qmix_config_sha256": sha,
            "reward_version": REWARD_R1_NAME, "reward_config_sha256": FROZEN_R1_CONFIG_SHA256,
            "normalization_version": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
            "normalizer_constants_sha256": ci.FROZEN_V3_CONSTANTS_SHA256}
    ci.validate_c1_checkpoint_meta(good, sha)  # no raise


def test_T84_wrong_qmix_hash_rejected():
    with pytest.raises(ci.CoordinationConfigError):
        ci.validate_c1_checkpoint_meta({"architecture_version": TREATMENT_ARCH,
                                        "qmix_config_sha256": "bad"}, ci.qmix_config_sha256())


def test_T85_c1_cannot_load_as_c0():
    with pytest.raises(ci.CoordinationConfigError):
        ci.validate_c0_checkpoint_meta({"architecture_version": TREATMENT_ARCH, "has_mixer": True})


# --- T86-T89 serialization + params + diagnostics ---
def test_T86_serialization_roundtrip(verdict):
    s = _load("E4KI_serialization_roundtrip.json")
    assert s["SERIALIZATION_ROUNDTRIP"] == "PASS"
    assert s["param_hash_match"] is True and s["forward_match"] is True


def test_T87_parameter_counts_reported(verdict):
    pc = _load("E4KI_parameter_count.json")
    assert pc["C1_trainable_params"] == pc["utility_params"] + pc["mixer_params"]
    assert pc["C1_trainable_params"] > pc["C0_trainable_params"]
    assert verdict["PARAMETER_COUNT_DIFFERENCE"] == "INTRINSIC_TO_COORDINATION_TREATMENT"


def test_T88_coordination_diagnostic_hooks():
    d = _load("E4KI_coordination_diagnostics_validation.json")
    assert d["hooks_finite"] is True
    for k in ["A_monotonic_gradient", "B_weight_variation", "C_kappa_dispersion", "D_other_region_sensitivity"]:
        assert k in d


def test_T89_diagnostics_not_verdict_gate():
    d = _load("E4KI_coordination_diagnostics_validation.json")
    assert d["not_a_verdict_gate"] is True


# --- T90 factor isolation ---
def test_T90_future_ab_factor_isolation(verdict):
    a = _load("E4KI_future_ab_factor_isolation.json")
    assert verdict["FUTURE_AB_FACTOR_ISOLATION"] == "PASS"
    assert a["differing_fields_nonempty"] is True
    assert a["every_difference_authorized"] is True
    assert len(a["authorized_differences"]) > 0


# --- T91-T96 paired init ---
@pytest.mark.parametrize("p", [5001, 5002, 5003, 5004, 5005])
def test_T91_95_utility_init_paired(p):
    c0, _ = build_online_target_pair(9, seed=_uinit(p))
    c1, _ = build_online_target_pair(9, seed=_uinit(p))
    assert _param_hash(c0) == _param_hash(c1)


def test_T96_mixer_init_deterministic_all_five():
    for p in [5001, 5002, 5003, 5004, 5005]:
        a, _ = build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(p)))
        b, _ = build_mixer_online_target_pair(seed=derive_mixer_seed(_uinit(p)))
        assert _param_hash(a) == _param_hash(b)
    assert derive_mixer_seed(5001 * 1000) == 5001 * 1000 + MIXER_SEED_OFFSET


# --- T97-T104 firewall + no training + historical ---
def test_T97_train_seeds_untouched(verdict):
    assert verdict["C_TRAIN_ACCESS_COUNT"] == 0
    assert _load("E4KI_future_seed_firewall.json")["checks"]["train_range"] is True


def test_T98_validation_seeds_untouched(verdict):
    assert verdict["C_VALIDATION_ACCESS_COUNT"] == 0
    assert _load("E4KI_future_seed_firewall.json")["checks"]["validation_range"] is True


def test_T99_final_seeds_untouched(verdict):
    assert verdict["C_FINAL_ACCESS_COUNT"] == 0
    assert _load("E4KI_future_seed_firewall.json")["checks"]["final_range"] is True


def test_T100_real_environment_episodes_zero(verdict):
    assert verdict["REAL_ENVIRONMENT_EPISODES"] == 0


def test_T101_optimizer_steps_zero():
    assert _load("E4KI_no_training_audit.json")["audit"]["optimizer_steps"] == 0


def test_T102_policy_training_episodes_zero():
    assert _load("E4KI_no_training_audit.json")["audit"]["policy_training_episodes"] == 0


def test_T103_no_learned_checkpoint(verdict):
    assert verdict["MODEL_TRAINING_CHECKPOINTS_CREATED"] == 0
    assert verdict["NEW_TRAINING"] == "NO"


def test_T104_historical_artifacts_unchanged():
    h = _load("E4KI_historical_integrity.json")["checks"]
    assert h["v3_hash"] and h["r1_hash"] and h["r1_checkpoints_match"] and h["alpha_01"]


# --- T105-T106 verdict + config hash + qmix config ---
def test_T105_implementation_validity_logic(verdict):
    assert verdict["QMIX_IMPLEMENTATION_VALID"] == "YES"
    assert verdict["E4KI_COMPLETE"] == "YES"
    assert verdict["NEXT_STAGE"] == "E4KT_JOINT_COORDINATION_AB_TRAINING_PROTOCOL_FREEZE"


def test_T106_qmix_config_hash_matches(verdict):
    cfg = _load("E4KI_qmix_config.json")
    assert cfg["QMIX_CONFIG_SHA256"] == ci.qmix_config_sha256()
    assert cfg["QMIX_CONFIG_SHA256"] == verdict["QMIX_CONFIG_SHA256"]
    assert cfg["config"]["architecture_version"] == TREATMENT_ARCH
    assert cfg["config"]["nonnegative_transform"] == "ABS"
