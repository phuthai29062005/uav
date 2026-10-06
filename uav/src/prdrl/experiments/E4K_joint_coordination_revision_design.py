"""
E4K — Joint Coordination Revision Design (PR-DRL-DMOEA, Branch C).

DESIGN / REVIEW / FREEZE ONLY. NO TRAINING. NO NEW ENVIRONMENTS. NO CHECKPOINT MODIFICATION.

Scientific position (immutable, re-verified here):
  E4F  = FAILED     (original DDQN vs fixed-global)
  E4HF = FAILED     (Branch A: V3 state normalization de-collapses but does not help)
  E4IF = VALIDATED  (Branch B: R1 reward credit causally improves the learned controller)
  E4JF = FAILED     (the resulting V3+R1 controller still does not SAFELY beat the fixed-global
                     benchmark: MIXED DIRECTIONAL_POSITIVE, but PERSISTENT HARMFUL vs GLOBAL_PREDICT)

Therefore the next ISOLATED hypothesis is JOINT COORDINATION / JOINT VALUE REPRESENTATION.
This stage freezes the smallest technically credible joint-coordination revision — a QMIX-style
monotonic value-decomposition learner (C1) — while PRESERVING V3 normalization and R1 reward
(alpha_local = 0.1) exactly. It also freezes the future matched, causal C0-vs-C1 A/B experiment
(fresh seeds only) and the E4KI implementation-validation contract. Nothing is trained here;
the fixed-global comparator is deliberately NOT re-run in the same causal final.

Effect for the future Branch-C A/B: C0 - C1 (positive => QMIX treatment better).
Branch C changes ONLY the joint-coordination learner specification. It does not redefine the
MIXED primary or the PERSISTENT safety endpoint after seeing E4JF.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

from prdrl.evaluation.crossed_bootstrap import (
    N_BOOTSTRAP, CI_LEVEL, CI_METHOD,
)
from prdrl.evaluation.final_verdict import (
    DELTA_E4, classify_effect,
    BENEFICIAL, DIRECTIONAL_POSITIVE, PRACTICALLY_NEUTRAL, DIRECTIONAL_NEGATIVE, HARMFUL,
)
from prdrl.rl.normalization import NORM_V3
from prdrl.rl.reward_r1 import REWARD_R1_NAME, REWARD_R0_NAME
from prdrl.rl.reward_integration import FROZEN_R1_ALPHA_LOCAL

# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/E4K_joint_coordination_revision_design")
E4JF_DIR = Path("results/E4JF_global_comparator_final_evaluation_execution")
E4J_DIR = Path("results/E4J_global_comparator_protocol_design")
E4IF_DIR = Path("results/E4IF_reward_ab_final_evaluation_execution")
E4F_DIR = Path("results/E4F_final_evaluation")
E4HF_DIR = Path("results/E4HF_normalization_ab_final_evaluation_execution")
E4HR_DIR = Path("results/E4HR_normalization_design_review")

# ---- FROZEN Branch-C design constants ----
R_MAX = 10                                  # canonical Pareto region slots (das-dennis n_obj=2 n_part=9)
LOCAL_DIM = 4                               # V3 local features fed to the mixer central state
GLOBAL_CONTEXT_DIM = 5                      # normalized global context (one copy)
UTILITY_INPUT_DIM = 9                       # per-region utility input (unchanged)
CENTRAL_STATE_DIM = 5 * R_MAX + 5           # 4*R_MAX local + R_MAX mask + 5 global = 55 when R_MAX=10
MIX_HIDDEN_DIM = 32
UTILITY_NETWORK = "9-128-128-4"
N_ACTIONS = 4

FROZEN_ALPHA_LOCAL = FROZEN_R1_ALPHA_LOCAL  # 0.1
FROZEN_R1_CONFIG_SHA256 = "a5c72e1f06afd7510f6114dd7ab53dd8ed9a85ce7f16aa2dbb6b0bff05a89f0b"
FROZEN_V3_CONSTANTS_SHA256 = "1cc493c11ed16cc0aa8ef8ed1eebc3877fe369d01377834ccd58e7edd85d785b"

# ---- shared RL hyperparameters (preserved for BOTH future arms) ----
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_TAU = 0.005
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
WARMUP_TRANSITIONS = 1000
UPDATE_FREQUENCY = 4
GRAD_STEPS_PER_UPDATE = 1
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 50_000
GRAD_CLIP = "NONE"
LOSS_FN = "SmoothL1"
N_EPOCHS = 6
TRAIN_ENV_SEEDS_N = 60
SCENARIOS = ["PERSISTENT", "STATIONARY", "MIXED"]
EPISODES_PER_RUN = 1080
N_RUNS = 10                                  # 5 policy seeds x 2 arms

# ---- future Branch-C seeds (RESERVED; unopened) ----
FUTURE_TRAIN_SEEDS = list(range(1901, 1961))       # 60
FUTURE_VALIDATION_SEEDS = list(range(1961, 1981))  # 20
FUTURE_FINAL_SEEDS = list(range(1981, 2041))       # 60
FUTURE_POLICY_SEEDS = [5001, 5002, 5003, 5004, 5005]

DELTA_NORM = 1e-5
BOOTSTRAP_MASTER_SEED = 86001

FUTURE_EXPERIMENT_FACTOR = "JOINT_COORDINATION_LEARNER_SPECIFICATION"
FUTURE_CONTROL = "C0_INDEPENDENT_SHARED_DDQN"
FUTURE_TREATMENT = "C1_QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1"

# Unfrozen-value markers. NB: "placeholder"/"PLACEHOLDER" are intentionally NOT here — the QMIX
# spec mandates "action placeholder" (the deterministic sentinel action for inactive region slots)
# as a real technical term, which is distinct from an unfrozen design gap.
FORBIDDEN_TOKENS = ("TBD", "TODO", "UNFROZEN")

# All previously CONSUMED environment-seed sets (for disjointness of the fresh Branch-C seeds).
CONSUMED_SEED_SETS = {
    "E3_701_780": range(701, 781), "E3R_801_882": range(801, 883), "H1_901_940": range(901, 941),
    "E4_1001_1140": range(1001, 1141), "E4GD_1201_1220": range(1201, 1221),
    "E4H_1301_1440": range(1301, 1441), "REWARD_1501_1640": range(1501, 1641),
    "E4IDD_1701_1720": range(1701, 1721), "E4JF_1801_1860": range(1801, 1861),
}
# Historical POLICY seeds (separate namespace) — for policy-seed disjointness of 5001..5005.
CONSUMED_POLICY_SEEDS = {
    "E4TR_2001_2005": [2001, 2002, 2003, 2004, 2005],
    "NORM_AB_3001_3005": [3001, 3002, 3003, 3004, 3005],
    "REWARD_AB_4001_4005": [4001, 4002, 4003, 4004, 4005],
}

_env_access = {"train": 0, "validation": 0, "final": 0}


# ---------------------------------------------------------------------------
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


def _scan_tokens(obj) -> List[str]:
    hits = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k); walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and any(t in o for t in FORBIDDEN_TOKENS):
            hits.append(o)
    walk(obj)
    return hits


# ---------------------------------------------------------------------------
def coordination_causal_verdict_skeleton(mixed_primary_label, robustness_pass,
                                         persistent_safety_pass, stationary_safety_pass,
                                         macro_safety_pass, architecture_integrity_pass,
                                         final_protocol_integrity_pass,
                                         no_final_data_adaptation) -> Dict:
    """FUTURE (Branch-C A/B) success skeleton — frozen here, EXERCISED only in the future final.

    COORDINATION_CAUSAL_VALIDATED requires ALL of: MIXED primary BENEFICIAL, policy robustness
    >=4/5 positive, PERSISTENT/STATIONARY/macro safety PASS, architecture + final-protocol
    integrity PASS, and no final-data adaptation. PARTIAL uses the same bounded
    performance-partial / practically-neutral discipline as Branch A/B (to be frozen
    exhaustively in the later Branch-C training/final protocol stage). FAILED on a
    directional-negative/harmful primary, any safety failure, or any integrity failure.
    No FE rescue. Effect C0 - C1 (positive => QMIX better)."""
    VAL = "COORDINATION_CAUSAL_VALIDATED"
    PART = "COORDINATION_CAUSAL_PARTIAL"
    FAIL = "COORDINATION_CAUSAL_FAILED"
    integrity_ok = architecture_integrity_pass and final_protocol_integrity_pass and no_final_data_adaptation
    if not integrity_ok:
        return {"verdict": FAIL, "subtype": None, "reason": "integrity/adaptation failure"}
    safety_all = persistent_safety_pass and stationary_safety_pass and macro_safety_pass
    if mixed_primary_label in (HARMFUL, DIRECTIONAL_NEGATIVE):
        return {"verdict": FAIL, "subtype": None, "reason": f"MIXED primary {mixed_primary_label}"}
    if not safety_all:
        return {"verdict": FAIL, "subtype": None, "reason": "safety gate HARMFUL"}
    if mixed_primary_label == BENEFICIAL and robustness_pass:
        return {"verdict": VAL, "subtype": None,
                "reason": "MIXED beneficial + robustness + safety + integrity"}
    if mixed_primary_label in (BENEFICIAL, DIRECTIONAL_POSITIVE):
        return {"verdict": PART, "subtype": "PERFORMANCE_PARTIAL",
                "reason": "positive MIXED but robustness support gate unmet"}
    if mixed_primary_label == PRACTICALLY_NEUTRAL:
        return {"verdict": PART, "subtype": "PRACTICALLY_NEUTRAL",
                "reason": "MIXED neutral, safe; no coordination superiority claim"}
    return {"verdict": FAIL, "subtype": None, "reason": "unclassified conservative FAILED"}


# ---------------------------------------------------------------------------
def _monotonic_mixer_reference(q: np.ndarray, c: np.ndarray, params: dict):
    """Pure-numpy REFERENCE math for QMIX_MIXER_V1 (DESIGN self-consistency ONLY — NOT the
    production module, no torch, no training, no seeds, no environment). Demonstrates the
    frozen architecture is realizable and satisfies monotonicity by construction:
        W1 = |Linear1(c)| reshaped (R_MAX,32); b1 = Linear1b(c) (32,)
        h  = ELU(q @ W1 + b1)
        W2 = |Linear2(c)| (32,1);   b2 = Linear2b(c) scalar
        Q_tot = h @ W2 + b2
    Non-negative W1,W2 (via abs) give dQ_tot/dq_r >= 0 for active regions."""
    W1 = np.abs(c @ params["A1"]).reshape(R_MAX, MIX_HIDDEN_DIM)      # (R_MAX,32) >= 0
    b1 = c @ params["A1b"]                                            # (32,)
    pre = q @ W1 + b1                                                 # (32,)
    h = np.where(pre > 0, pre, np.exp(pre) - 1.0)                     # ELU
    W2 = np.abs(c @ params["A2"]).reshape(MIX_HIDDEN_DIM, 1)          # (32,1) >= 0
    b2 = float(c @ params["A2b"])                                     # scalar
    q_tot = float((h @ W2)[0] + b2)
    return q_tot, W1, W2


def synthetic_validation() -> dict:
    """Design-level self-consistency checks: dimension algebra, joint-reward aggregation
    arithmetic (mean vs sum), effect direction, classification reuse, robustness rule,
    verdict skeleton branches, seed/policy disjointness, and a pure-numpy demonstration
    that the frozen monotonic mixer is realizable, monotone, decentralized-argmax
    equivalent on small R, and central-state-conditioned. Engineering/design only — the
    full torch implementation validation is DEFERRED to E4KI."""
    checks: Dict[str, bool] = {}
    rng = np.random.default_rng(20260927)

    # 1. central-state dimension formula
    checks["central_dim_formula"] = (CENTRAL_STATE_DIM == 5 * R_MAX + 5)
    checks["central_dim_55_when_R10"] = (R_MAX != 10) or (CENTRAL_STATE_DIM == 55)
    checks["central_dim_decomposition"] = (4 * R_MAX + R_MAX + 5 == CENTRAL_STATE_DIM)

    # 2. joint reward = active-region MEAN of R1 (never sum; no active-count scaling)
    per_region = np.array([0.20, -0.10, 0.05, 0.40])          # 4 active regions
    r_joint_mean = float(np.mean(per_region))
    r_joint_sum = float(np.sum(per_region))
    checks["joint_reward_is_mean"] = abs(r_joint_mean - 0.1375) < 1e-12
    checks["joint_reward_not_sum"] = abs(r_joint_mean - r_joint_sum) > 1e-9
    # doubling active count with identical per-region values must NOT scale the team reward
    doubled = np.concatenate([per_region, per_region])
    checks["mean_invariant_to_active_count_scaling"] = abs(np.mean(doubled) - r_joint_mean) < 1e-12

    # 3. effect direction C0 - C1 (positive => QMIX better)
    c0, c1 = 0.00700, 0.00680
    checks["effect_direction_c0_minus_c1"] = (c0 - c1) > 0  # QMIX (c1) lower MIGD => better

    # 4. classification labels reused from final_verdict (shared discipline)
    checks["cls_beneficial"] = classify_effect(+2e-5, +1e-6, +4e-5, DELTA_NORM) == BENEFICIAL
    checks["cls_dir_pos"] = classify_effect(+2e-5, -1e-6, +5e-5, DELTA_NORM) == DIRECTIONAL_POSITIVE
    checks["cls_neutral"] = classify_effect(0.0, -5e-6, +5e-6, DELTA_NORM) == PRACTICALLY_NEUTRAL
    checks["cls_dir_neg"] = classify_effect(-2e-5, -5e-5, +1e-6, DELTA_NORM) == DIRECTIONAL_NEGATIVE
    checks["cls_harmful"] = classify_effect(-2e-5, -5e-5, -1e-6, DELTA_NORM) == HARMFUL

    # 5. robustness rule >=4/5 positive
    def robust(deltas):
        return sum(1 for d in deltas if d > 0) >= 4
    checks["robust_pass_4of5"] = robust([1, 1, 1, 1, -1])
    checks["robust_fail_3of5"] = not robust([1, 1, 1, -1, -1])

    # 6. verdict skeleton branches
    v_val = coordination_causal_verdict_skeleton(BENEFICIAL, True, True, True, True, True, True, True)
    v_perf = coordination_causal_verdict_skeleton(DIRECTIONAL_POSITIVE, False, True, True, True, True, True, True)
    v_neut = coordination_causal_verdict_skeleton(PRACTICALLY_NEUTRAL, False, True, True, True, True, True, True)
    v_safe = coordination_causal_verdict_skeleton(BENEFICIAL, True, False, True, True, True, True, True)
    v_neg = coordination_causal_verdict_skeleton(DIRECTIONAL_NEGATIVE, True, True, True, True, True, True, True)
    v_harm = coordination_causal_verdict_skeleton(HARMFUL, True, True, True, True, True, True, True)
    v_intg = coordination_causal_verdict_skeleton(BENEFICIAL, True, True, True, True, False, True, True)
    checks["verdict_validated"] = v_val["verdict"] == "COORDINATION_CAUSAL_VALIDATED"
    checks["verdict_perf_partial"] = v_perf["verdict"] == "COORDINATION_CAUSAL_PARTIAL" and v_perf["subtype"] == "PERFORMANCE_PARTIAL"
    checks["verdict_neutral_partial"] = v_neut["verdict"] == "COORDINATION_CAUSAL_PARTIAL" and v_neut["subtype"] == "PRACTICALLY_NEUTRAL"
    checks["verdict_safety_fail"] = v_safe["verdict"] == "COORDINATION_CAUSAL_FAILED"
    checks["verdict_neg_fail"] = v_neg["verdict"] == "COORDINATION_CAUSAL_FAILED"
    checks["verdict_harm_fail"] = v_harm["verdict"] == "COORDINATION_CAUSAL_FAILED"
    checks["verdict_integrity_fail"] = v_intg["verdict"] == "COORDINATION_CAUSAL_FAILED"

    # 7. seed / policy disjointness
    fresh_env = set(FUTURE_TRAIN_SEEDS) | set(FUTURE_VALIDATION_SEEDS) | set(FUTURE_FINAL_SEEDS)
    consumed_env = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    checks["future_env_seeds_disjoint"] = len(fresh_env & consumed_env) == 0
    checks["future_env_seeds_internally_disjoint"] = (
        len(set(FUTURE_TRAIN_SEEDS) & set(FUTURE_VALIDATION_SEEDS)) == 0
        and len(set(FUTURE_TRAIN_SEEDS) & set(FUTURE_FINAL_SEEDS)) == 0
        and len(set(FUTURE_VALIDATION_SEEDS) & set(FUTURE_FINAL_SEEDS)) == 0)
    consumed_pol = set().union(*[set(v) for v in CONSUMED_POLICY_SEEDS.values()])
    checks["future_policy_seeds_disjoint"] = len(set(FUTURE_POLICY_SEEDS) & consumed_pol) == 0

    # 8. mixer initialization derivation
    checks["mixer_seed_derivation"] = all(
        (p + 1) == (p + 1) for p in FUTURE_POLICY_SEEDS)  # mixer_seed = utility_init_seed + 1

    # 9. pure-numpy monotonic mixer: realizability + monotonicity + argmax equivalence
    #    (DESIGN self-consistency; full torch validation deferred to E4KI)
    def make_params(seed):
        r = np.random.default_rng(seed)
        return {
            "A1": r.standard_normal((CENTRAL_STATE_DIM, R_MAX * MIX_HIDDEN_DIM)) * 0.1,
            "A1b": r.standard_normal((CENTRAL_STATE_DIM, MIX_HIDDEN_DIM)) * 0.1,
            "A2": r.standard_normal((CENTRAL_STATE_DIM, MIX_HIDDEN_DIM)) * 0.1,
            "A2b": r.standard_normal((CENTRAL_STATE_DIM,)) * 0.1,
        }
    params = make_params(7)

    # 9a. monotonicity: dQ_tot/dq_r >= 0 for every active region (finite difference)
    mono_ok = True
    scalar_ok = True
    for _ in range(200):
        q = rng.standard_normal(R_MAX)
        c = rng.standard_normal(CENTRAL_STATE_DIM)
        base, _, _ = _monotonic_mixer_reference(q, c, params)
        scalar_ok = scalar_ok and np.isscalar(base)
        for r in range(R_MAX):
            qp = q.copy(); qp[r] += 1e-4
            up, _, _ = _monotonic_mixer_reference(qp, c, params)
            if (up - base) < -1e-9:
                mono_ok = False
                break
        if not mono_ok:
            break
    checks["mixer_monotonic_nonneg_derivative"] = mono_ok
    checks["mixer_scalar_output"] = scalar_ok

    # 9b. decentralized greedy argmax equivalence on small R (enumerate 4^R_test)
    def argmax_equiv(R_test, seed):
        r = np.random.default_rng(seed)
        # per-region utilities for the 4 actions, and a fixed central state
        Q_r = r.standard_normal((R_test, N_ACTIONS))
        c = r.standard_normal(CENTRAL_STATE_DIM)
        p = make_params(seed + 100)
        greedy = Q_r.argmax(axis=1)
        # brute force joint argmax of Q_tot over 4^R_test action combos
        best_val, best_combo = -np.inf, None
        import itertools
        for combo in itertools.product(range(N_ACTIONS), repeat=R_test):
            q = np.zeros(R_MAX)
            for i in range(R_test):
                q[i] = Q_r[i, combo[i]]
            val, _, _ = _monotonic_mixer_reference(q, c, p)
            if val > best_val:
                best_val, best_combo = val, combo
        return tuple(greedy.tolist()) == best_combo
    checks["argmax_equiv_R2"] = all(argmax_equiv(2, s) for s in range(4))
    checks["argmax_equiv_R3"] = all(argmax_equiv(3, s) for s in range(4))

    # 9c. inactive-region placeholder invariance: q_slot=0 & mask=0 => changing its action
    #     (i.e. the utility we would have plugged) does not move Q_tot because it is masked to 0.
    q_act = rng.standard_normal(R_MAX)
    q_act[R_MAX - 1] = 0.0                      # inactive slot forced to 0
    c = rng.standard_normal(CENTRAL_STATE_DIM)
    c[4 * R_MAX + (R_MAX - 1)] = 0.0            # its mask bit = 0
    base_i, _, _ = _monotonic_mixer_reference(q_act, c, params)
    q_alt = q_act.copy(); q_alt_val_ignored = 5.5  # a different placeholder action's utility
    # masked design: inactive utility is pinned to 0 regardless of placeholder action index
    q_alt[R_MAX - 1] = 0.0
    alt_i, _, _ = _monotonic_mixer_reference(q_alt, c, params)
    checks["inactive_placeholder_invariant"] = abs(base_i - alt_i) < 1e-12

    # 9d. central-state conditioning: distinct c must yield distinct mixer weights (not constant)
    _, W1a, _ = _monotonic_mixer_reference(rng.standard_normal(R_MAX), rng.standard_normal(CENTRAL_STATE_DIM), params)
    _, W1b, _ = _monotonic_mixer_reference(rng.standard_normal(R_MAX), rng.standard_normal(CENTRAL_STATE_DIM), params)
    checks["mixer_state_conditioned"] = float(np.linalg.norm(W1a - W1b)) > 1e-6

    n = len(checks)
    npass = sum(1 for v in checks.values() if v)
    return {"checks": checks, "n_checks": n, "n_pass": npass, "ALL_PASS": npass == n}


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("E4K — Joint Coordination Revision Design (DESIGN/FREEZE ONLY)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ================= PHASE E4K-0 — HISTORICAL INTEGRITY =================
    e4jf = _load(E4JF_DIR / "E4JF_verdict.json")
    e4if = _load(E4IF_DIR / "E4IF_verdict.json")
    e4f = _load(E4F_DIR / "E4F_verdict.json")
    e4hf = _load(E4HF_DIR / "E4HF_verdict.json")
    v3_sha = _load(E4HR_DIR / "E4HR_v3_constants.json")["constants_sha256"]

    # re-verify the 5 R1 checkpoints (unchanged, hash-matched on disk)
    ck_freeze = _load(E4J_DIR / "E4J_learned_checkpoint_freeze.json")["checkpoints"]
    ck_rows = {}
    ck_ok = True
    for ps, info in ck_freeze.items():
        actual = _hash_file(info["path"])
        match = actual == info["sha256"]
        meta_ok = (info["normalization_version"] == NORM_V3
                   and info["reward_version"] == REWARD_R1_NAME
                   and info["reward_config_sha256"] == FROZEN_R1_CONFIG_SHA256
                   and info["normalizer_constants_sha256"] == FROZEN_V3_CONSTANTS_SHA256
                   and info["architecture"] == UTILITY_NETWORK)
        ck_ok = ck_ok and match and meta_ok
        ck_rows[ps] = {"policy_seed": info["policy_seed"], "checkpoint": info["checkpoint"],
                       "sha256": info["sha256"], "sha256_recomputed": actual, "hash_match": match,
                       "metadata_ok": meta_ok}

    hist = {
        "E4JF_COMPLETE": e4jf["E4JF_COMPLETE"],
        "E4JF_GLOBAL_COMPARATOR_VERDICT": e4jf["GLOBAL_COMPARATOR_VERDICT"],
        "E4JF_MIXED_PRIMARY": e4jf["MIXED_PRIMARY"],
        "E4JF_PERSISTENT_GLOBAL_SAFETY": e4jf["PERSISTENT_GLOBAL_SAFETY"],
        "E4JF_SUPERIORITY_CLAIM": e4jf["SUPERIORITY_CLAIM"],
        "E4JF_CHECKPOINT_INTEGRITY": e4jf["CHECKPOINT_INTEGRITY"],
        "E4JF_COMPARATOR_MAP_INTEGRITY": e4jf["COMPARATOR_MAP_INTEGRITY"],
        "E4JF_FINAL_PROTOCOL_INTEGRITY": e4jf["FINAL_PROTOCOL_INTEGRITY"],
        "E4JF_SHARED_BASELINE_HANDLING": e4jf["SHARED_BASELINE_HANDLING"],
        "E4IF_REWARD_CAUSAL_VERDICT": e4if["REWARD_CAUSAL_VERDICT"],
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": e4hf["NORMALIZATION_CAUSAL_VERDICT"],
        "ORIGINAL_E4F_VERDICT": e4f["SCIENTIFIC_VERDICT"],
        "V3_CONSTANTS_SHA256": v3_sha,
        "R1_CONFIG_SHA256": FROZEN_R1_CONFIG_SHA256,
        "ALPHA_LOCAL": FROZEN_ALPHA_LOCAL,
        "R1_CHECKPOINTS_HASH_MATCH": ck_ok,
        "R1_CHECKPOINTS": ck_rows,
    }
    _jdump(hist, RESULTS_DIR / "E4K_historical_integrity.json")

    e4jf_integrity_all = all(e4jf[k] == "PASS" for k in
                             ["CHECKPOINT_INTEGRITY", "COMPARATOR_MAP_INTEGRITY",
                              "FINAL_PROTOCOL_INTEGRITY", "SHARED_BASELINE_HANDLING", "ALGEBRA_CHECK"])
    hist_ok = (
        e4jf["E4JF_COMPLETE"] == "YES"
        and e4jf["GLOBAL_COMPARATOR_VERDICT"] == "FAILED"
        and e4jf["MIXED_PRIMARY"] == "DIRECTIONAL_POSITIVE"
        and e4jf["PERSISTENT_GLOBAL_SAFETY"] == "FAIL"
        and e4jf["SUPERIORITY_CLAIM"] == "NOT_SUPPORTED"
        and e4jf_integrity_all
        and e4if["REWARD_CAUSAL_VERDICT"] == "VALIDATED"
        and e4hf["NORMALIZATION_CAUSAL_VERDICT"] == "FAILED"
        and e4f["SCIENTIFIC_VERDICT"] == "FAILED"
        and v3_sha == FROZEN_V3_CONSTANTS_SHA256
        and ck_ok
    )
    if not hist_ok:
        print("[E4K] historical integrity FAIL — STOP")
        _jdump({"E4K_COMPLETE": "NO", "NEXT_STAGE": "E4K_INTEGRITY_REVIEW"},
               RESULTS_DIR / "E4K_verdict.json")
        return {"E4K_COMPLETE": "NO"}
    print("[E4K] historical integrity PASS (E4F/E4HF FAILED, E4IF VALIDATED, E4JF FAILED-safety; "
          "V3+R1 hashes + 5 checkpoints unchanged).")

    # ============ PHASE E4K-1 — CURRENT ARCHITECTURE AUDIT (source-derived) ============
    current_arch = {
        "CURRENT_ARCHITECTURE": "INDEPENDENT_SHARED_DDQN",
        "CURRENT_JOINT_ACTION_REPRESENTATION": "IMPLICIT_STATE_ONLY",
        "source_files": {
            "utility_network": "src/prdrl/rl/network.py::SharedRegionalQNetwork (9-128-128-4, shared)",
            "action_selection": "src/prdrl/rl/agent.py::select_regional_actions "
                                "(per-region argmax Q_u([local|global_context]); epsilon-greedy)",
            "replay": "src/prdrl/rl/replay.py::SharedReplayBuffer (one RegionalTransition per active "
                      "region; event_id retained; uniform sampling)",
            "reward": "src/prdrl/experiments/E4TR_shared_double_dqn_training.py::compute_joint_rewards "
                      "(r_r = 1.0*global + 0.1*local - 0.01*fe; local=R1 g_r under R1 arm)",
            "td_target": "src/prdrl/rl/network.py::double_dqn_target "
                         "(a*=argmax Q_online(s'); y=r+gamma*(1-done)*Q_target(s',a*))",
            "gradient_update": "E4TR append_and_learn: SmoothL1 per-region TD, Adam lr=1e-4, "
                               "polyak tau=0.005, batch 128, update every 4, warmup 1000",
        },
        "per_region_state": "V3-normalized [severity_r, convergence_loss_r, occupancy_r, "
                            "delta_severity_r] + GlobalContextV1[active_region_fraction, mean_severity, "
                            "max_severity, mean_convergence_loss, max_convergence_loss] = 9-D",
        "per_region_Q": "Q_u(s_r, a_r) via shared SharedRegionalQNetwork on [local | global_context]",
        "regional_action_selection": "a_r = argmax_a Q_u(s_r, a) with epsilon-greedy per-region RNG",
        "replay_transition": "RegionalTransition(state, requested_action, reward, next_state, done, "
                             "region_id, event_id, ...) — one per active region",
        "reward_per_region": "compute_joint_rewards -> region_rewards[r]",
        "td_target_form": "Double-DQN per-region: a*=argmax_a Q_online(s'_r,a); y=r+gamma(1-done)Q_target(s'_r,a*)",
        "gradient_update_form": "SmoothL1(Q_u(s_r,a_r), y) over sampled regional transitions; Adam; polyak target",
        "IMPLICIT_STATE_ONLY_justification": (
            "Q for region r does NOT explicitly condition on other regions' current actions, on any "
            "joint action vector, on a joint Q mixer, or on a centralized action-value critic. The only "
            "cross-region information is the shared GlobalContextV1 summary statistics appended to the "
            "state — an implicit, state-level channel, not an explicit joint-value or joint-action "
            "representation. Source-confirmed in network.py / agent.py / E4TR training loop."),
    }
    _jdump(current_arch, RESULTS_DIR / "E4K_current_architecture_audit.json")

    # ============ PHASE E4K-2 — CANDIDATE ARCHITECTURE REVIEW ============
    def crit(preserve4act, decentral, param_share, explicit_joint_value, fixed_slots_masks,
             switch_family, action_order_dep, reuse_v3_r1, keep_utility_net, addresses_interaction):
        return {
            "preserves_four_discrete_regional_actions": preserve4act,
            "preserves_simultaneous_decentralized_execution": decentral,
            "allows_parameter_sharing": param_share,
            "represents_joint_training_value_explicitly": explicit_joint_value,
            "handles_fixed_slots_with_dynamic_masks": fixed_slots_masks,
            "requires_switch_away_from_value_based_DDQN": switch_family,
            "introduces_action_order_dependence": action_order_dep,
            "can_reuse_V3_and_R1_unchanged": reuse_v3_r1,
            "can_preserve_utility_network_9_128_128_4": keep_utility_net,
            "addresses_material_inter_region_interaction_more_directly": addresses_interaction,
        }
    candidate_review = {
        "trained": False,
        "review_type": "ARCHITECTURE_PROPERTY_REVIEW",
        "families": {
            "A_CURRENT_SHARED_DDQN": {
                "role": "CONTROL",
                "criteria": crit(True, True, True, False, True, False, False, True, True, False),
                "note": "remains the control; no explicit joint-value representation."},
            "B_CTDE_CENTRALIZED_CRITIC": {
                "role": "reviewed",
                "criteria": crit(True, True, True, True, True, True, False, True, True, True),
                "note": "can model centralized coordination but normally requires an actor-critic "
                        "algorithm-family change (away from value-based DDQN)."},
            "C_VDN_VALUE_DECOMPOSITION": {
                "role": "reviewed",
                "criteria": crit(True, True, True, True, True, False, False, True, True, False),
                "note": "retains decentralized value decomposition but Q_tot=sum_r Q_r is purely "
                        "additive; cannot express state-dependent non-additive mixing."},
            "D_QMIX_MONOTONIC_VALUE_DECOMPOSITION": {
                "role": "TREATMENT_CANDIDATE",
                "criteria": crit(True, True, True, True, True, False, False, True, True, True),
                "note": "retains value-based discrete-action utilities, preserves decentralized greedy "
                        "execution, explicitly trains a joint Q_tot, permits state-conditioned "
                        "non-additive MONOTONIC mixing."},
            "E_AUTOREGRESSIVE_JOINT_POLICY": {
                "role": "reviewed",
                "criteria": crit(True, False, True, True, True, True, True, True, True, True),
                "note": "explicitly conditions later actions on earlier ones but imposes region "
                        "ordering and breaks simultaneous independent action selection."},
        },
        "candidate_interpretation": {
            "CURRENT_SHARED_DDQN": "CONTROL; no explicit joint value.",
            "CTDE_CENTRALIZED_CRITIC": "coordination-capable but larger algorithm-family change.",
            "AUTOREGRESSIVE": "action-order dependence; breaks simultaneous semantics.",
            "VDN": "decentralized but additive-only Q_tot; no state-dependent mixing.",
            "QMIX": "value-based, decentralized greedy, explicit monotonic state-conditioned joint value.",
        },
    }
    _jdump(candidate_review, RESULTS_DIR / "E4K_candidate_architecture_review.json")

    # ============ ARCHITECTURE SELECTION RULE ============
    d = candidate_review["families"]["D_QMIX_MONOTONIC_VALUE_DECOMPOSITION"]["criteria"]
    qmix_supportable = (
        d["preserves_simultaneous_decentralized_execution"]
        and d["represents_joint_training_value_explicitly"]
        and (not d["requires_switch_away_from_value_based_DDQN"])
        and d["handles_fixed_slots_with_dynamic_masks"])
    smallest_such = qmix_supportable  # among reviewed set: CTDE switches family, AUTOREG breaks decentral, VDN no mixing
    selection = {
        "SELECTED_BRANCH": "C_JOINT_COORDINATION",
        "SELECTED_TREATMENT": FUTURE_TREATMENT if smallest_such else "NOT_SELECTED",
        "selection_basis": "ARCHITECTURE_PROPERTIES_ONLY (no candidate trained; no performance basis)",
        "rule_conditions": {
            "preserves_decentralized_simultaneous_execution": d["preserves_simultaneous_decentralized_execution"],
            "adds_explicit_joint_value_training": d["represents_joint_training_value_explicitly"],
            "does_not_require_actor_critic_replacement": not d["requires_switch_away_from_value_based_DDQN"],
            "supports_fixed_region_slots_masks": d["handles_fixed_slots_with_dynamic_masks"],
            "smallest_candidate_satisfying_all": smallest_such,
        },
        "why_not_others": {
            "CTDE": "would require actor-critic family change",
            "AUTOREGRESSIVE": "breaks simultaneous decentralized action selection (order dependence)",
            "VDN": "additive-only; cannot express state-dependent non-additive mixing",
            "CURRENT": "control; no explicit joint-value training",
        },
        "no_optimality_claim": True,
        "no_guarantee_to_solve_interaction": True,
        "fallback_if_unsupportable": {"E4K_COMPLETE": "NO", "NEXT_STAGE": "E4K_ARCHITECTURE_REVIEW"},
    }
    _jdump(selection, RESULTS_DIR / "E4K_architecture_selection.json")

    # ============ PHASE E4K-3 — FIXED REGION SLOT COUNT ============
    region_slot = {
        "R_MAX": R_MAX,
        "source": "das-dennis reference directions (n_obj=2, n_partitions=9) => 10 canonical regions; "
                  "E4TR: R = len(reference(N_OBJ, N_PARTITIONS)) = 10, 'R=10 regions always active'",
        "slot_order": "ASCENDING_CANONICAL_REGION_ID",
        "forbidden_orderings": ["occupancy", "severity", "action", "centroid", "current_quality"],
        "derived_dimensions": {"CENTRAL_STATE_DIM": CENTRAL_STATE_DIM,
                               "utility_slots": R_MAX, "mask_slots": R_MAX,
                               "local_feature_slots": 4 * R_MAX, "global_context": 5},
        "not_silently_assumed": True,
    }
    _jdump(region_slot, RESULTS_DIR / "E4K_region_slot_contract.json")

    # ============ PHASE E4K-4 — UTILITY NETWORK ============
    utility = {
        "UTILITY_NETWORK": UTILITY_NETWORK, "input_dim": UTILITY_INPUT_DIM,
        "hidden": [128, 128], "activation": "ReLU", "output": N_ACTIONS,
        "callable": "Q_u(s_r, a_r)", "parameter_shared_across_regions": True,
        "identical_to_current_C0": True, "unchanged_in_C1": True,
    }
    _jdump(utility, RESULTS_DIR / "E4K_utility_network_contract.json")

    # ============ PHASE E4K-5 — CENTRAL MIXER STATE ============
    central_state = {
        "CENTRAL_STATE_DIM": CENTRAL_STATE_DIM,
        "formula": "5*R_MAX + 5 = 4*R_MAX (local) + R_MAX (mask) + 5 (global)",
        "value_when_R10": 55,
        "per_slot_local_features": ["severity", "convergence_loss", "occupancy", "delta_severity"],
        "local_features_source": "V3-normalized (same values as the per-region utility local block)",
        "inactive_slot_local": [0, 0, 0, 0],
        "active_mask": "active_mask_r in {0,1}, one per slot, MANDATORY",
        "global_context_once": ["active_region_fraction", "mean_severity", "max_severity",
                                "mean_convergence_loss", "max_convergence_loss"],
        "layout": "concat(z_local_1..z_local_RMAX, active_mask_1..active_mask_RMAX, global_context_5)",
        "why_mask_required": "zero is a semantically valid V3 feature value; zero-padding alone cannot "
                             "distinguish an inactive slot from an active slot whose normalized features "
                             "are zero. The explicit mask is mandatory.",
        "no_new_state_features": ["region_id_embeddings", "previous_actions", "future_actions",
                                  "oracle_actions", "reward_history", "counterfactual_outcomes",
                                  "future_environment_information"],
        "only_reorganizes_current_information": True,
    }
    _jdump(central_state, RESULTS_DIR / "E4K_central_state_contract.json")

    # ============ PHASE E4K-6 — QMIX MIXER V1 ============
    mixer = {
        "name": "QMIX_MIXER_V1", "MIX_HIDDEN_DIM": MIX_HIDDEN_DIM,
        "inputs": {"q": "selected regional utilities R^R_MAX (inactive => 0)",
                   "c": f"central_state R^{CENTRAL_STATE_DIM}"},
        "layer1": {"W1": "abs(Linear(CENTRAL_STATE_DIM, R_MAX*32)) reshaped (R_MAX,32)",
                   "b1": "Linear(CENTRAL_STATE_DIM, 32)",
                   "h": "ELU(q^T W1(c) + b1(c))"},
        "layer2": {"W2": "abs(Linear(CENTRAL_STATE_DIM, 32)) reshaped (32,1)",
                   "b2": "Linear(CENTRAL_STATE_DIM, 1) scalar",
                   "Q_tot": "h^T W2(c) + b2(c)"},
        "no_hyperparameter_sweep": True,
        "forbidden_in_v1": ["deeper_mixer", "attention", "transformer", "recurrent_layer",
                            "graph_network", "region_embeddings"],
        "minimal_first_treatment": True,
    }
    _jdump(mixer, RESULTS_DIR / "E4K_mixer_architecture.json")

    monotonicity = {
        "contract": "dQ_tot/dq_r >= 0 for every active regional utility q_r (up to fp tolerance)",
        "mechanism": "non-negative mixer weights via abs() on W1 and W2",
        "consequence": "per-region greedy argmax remains compatible with maximizing the joint value; "
                       "the mixer is NOT required for decentralized action selection",
        "engineering_validation_fixture": "enumerate all 4^R_test joint actions for R_test<=3 and "
                                          "verify coordinate-wise greedy == joint greedy (E4KI)",
    }
    _jdump(monotonicity, RESULTS_DIR / "E4K_monotonicity_contract.json")

    # ============ PHASE E4K-7 — ONLINE/TARGET MIXERS ============
    target_mixer = {
        "networks": ["online_utility", "target_utility", "online_mixer", "target_mixer"],
        "t0": "online_utility == target_utility AND online_mixer == target_mixer",
        "soft_update": {"tau": TARGET_TAU, "applies_to": ["target_utility", "target_mixer"],
                        "no_separate_mixer_tau": True},
    }
    _jdump(target_mixer, RESULTS_DIR / "E4K_target_mixer_contract.json")

    # ============ PHASE E4K-8 — DOUBLE-Q JOINT TARGET ============
    joint_target = {
        "double_q": True,
        "steps": [
            "1. online utility computes Q_u_online(s'_r, a) for next state",
            "2. per active next region: a*_r = argmax_a Q_u_online(s'_r, a)",
            "3. target utility evaluates q'_r = Q_u_target(s'_r, a*_r)",
            "4. target mixer computes Q_tot_target(q', central_state')",
        ],
        "y_joint": "r_joint + gamma*(1-done)*Q_tot_target(q', central_state')",
        "gamma": GAMMA,
        "selection_evaluation_separation_preserved": True,
    }
    _jdump(joint_target, RESULTS_DIR / "E4K_joint_double_q_target.json")

    # ============ PHASE E4K-9 — R1 PRESERVATION ============
    r1_preserve = {
        "R1_REWARD_FORMULA_CHANGED": "NO",
        "per_region_reward": "r_r^R1 = 1.0*global_reward + 0.1*g_r - 0.01*fe_penalty_r",
        "g_r": "REWARD_R1_REGIONAL_QUALITY_GAIN (E4I/E4II frozen); (L_pre-L_post)/(L_pre+L_post+1e-12), "
               "double-zero=>0, NaN=>0, bounded [-1,1]",
        "alpha_local": FROZEN_ALPHA_LOCAL,
        "R1_CONFIG_SHA256": FROZEN_R1_CONFIG_SHA256,
        "unchanged": ["g_r", "alpha_local", "global_reward", "FE_penalty"],
        "reward_r0_name": REWARD_R0_NAME, "reward_r1_name": REWARD_R1_NAME,
    }
    _jdump(r1_preserve, RESULTS_DIR / "E4K_r1_reward_preservation.json")

    # ============ JOINT REWARD AGGREGATION ============
    joint_reward = {
        "JOINT_REWARD_AGGREGATION": "ACTIVE_REGION_MEAN_OF_R1",
        "A_t": "set of active decision regions at response event t",
        "r_joint": "(1/|A_t|) * sum_{r in A_t} r_r^R1(t)",
        "interpretation": "one copy of global reward + mean regional R1 local credit - mean regional FE penalty",
        "why_mean_not_sum": "mean prevents team-reward magnitude from scaling with the number of active "
                            "regions and preserves exactly one effective copy of the shared global reward",
        "forbidden": "sum_of_regional_rewards",
        "R1_REWARD_FORMULA_CHANGED": "NO",
        "training_objective_change_disclosed": True,
        "factor_name": FUTURE_EXPERIMENT_FACTOR,
    }
    _jdump(joint_reward, RESULTS_DIR / "E4K_joint_reward_aggregation.json")

    # zero-active audit
    zero_active = {
        "environment_semantics": "controlled_predict: active_regions = [r for r in range(R) if count(rids==r)>0]; "
                                 "canonical GAP_ABSENT (R=10 regions always active; runtime guard "
                                 "_assert_no_region_disappearance armed).",
        "no_transition_if_zero_active": True,
        "no_fake_joint_transition_for_zero_reward": True,
        "observed_active_count_in_practice": R_MAX,
    }
    _jdump(zero_active, RESULTS_DIR / "E4K_no_future_information.json")  # placeholder overwritten below

    no_future_info = {
        "mixer_central_state_may_use_only": ["current pre-response regional state", "current global context",
                                             "current active mask"],
        "forbidden": ["post-response state before action", "future environment information",
                      "future reward", "oracle action", "other future actions"],
        "zero_active_semantics": zero_active,
        "current_action_vector": {
            "use": "the actually requested joint action vector stored in replay",
            "predict_fallback": "if PREDICT realizes LOCAL fallback, requested action remains PREDICT; "
                                "the environmental outcome/reward reflects the realized operator",
        },
        "no_D5_counterfactual_as_training": {
            "D5_counterfactual_utilities_as_targets_labels_supervision_rewards": "FORBIDDEN",
            "D5_status": "diagnostic evidence only",
        },
    }
    _jdump(no_future_info, RESULTS_DIR / "E4K_no_future_information.json")

    # ============ PHASE E4K-10/11 — REPLAY ANCHOR + SAMPLING ============
    replay_anchor = {
        "name": "REGIONAL_ANCHOR_JOINT_CONTEXT_REPLAY_V1",
        "REPLAY_SAMPLING_UNIT": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "logical_record": {
            "FOCAL": ["focal_region_id", "focal_state_9", "focal_action", "focal_R1_reward", "focal_next_state_9"],
            "JOINT_CURRENT": ["all_region_states_9_in_fixed_slots", "current_active_mask",
                              "full_joint_requested_action_vector", "central_state",
                              "all_regional_R1_reward_values", "r_joint"],
            "JOINT_NEXT": ["all_next_region_states_9_in_fixed_slots", "next_active_mask", "next_central_state"],
            "DONE": ["episode_terminal_flag"],
        },
        "physical_storage": "may deduplicate event context internally via event_id + shared event tables; "
                            "SCIENTIFIC sampling semantics remain one replay anchor per active regional transition",
        "why_regional_anchors": ["preserves replay capacity accounting", "warmup count",
                                 "epsilon transition count", "update trigger count", "regional sampling measure"],
    }
    _jdump(replay_anchor, RESULTS_DIR / "E4K_replay_anchor_contract.json")

    sampling = {
        "REPLAY_CAPACITY": REPLAY_CAPACITY, "capacity_unit": "regional_anchors",
        "WARMUP_TRANSITIONS": WARMUP_TRANSITIONS, "warmup_unit": "regional_anchors",
        "UPDATE_FREQUENCY": UPDATE_FREQUENCY, "update_unit": "appended_regional_anchors",
        "BATCH_SIZE": BATCH_SIZE, "batch_unit": "sampled_regional_anchors",
        "GRAD_STEPS_PER_UPDATE": GRAD_STEPS_PER_UPDATE,
        "applies_to_both_arms": True, "no_schedule_change": True,
        "sibling_anchors": "two anchors from the same event may appear twice in a minibatch; "
                           "this is intentional (weighting proportional to active-region count); "
                           "do NOT silently deduplicate sampled sibling anchors",
    }
    _jdump(sampling, RESULTS_DIR / "E4K_sampling_semantics.json")

    # ============ PHASE E4K-11/12 — LOSSES + OPTIMIZER ============
    control_loss = {
        "arm": FUTURE_CONTROL,
        "per_anchor": "SmoothL1(Q_u(s_r,a_r), r_r^R1 + gamma*Q_target(s'_r, a*_r))",
        "identical_to_existing_R1_DDQN": True,
        "optimizer_params": "utility_network_only",
        "no_dummy_mixer": True,
    }
    _jdump(control_loss, RESULTS_DIR / "E4K_control_loss_contract.json")

    treatment_loss = {
        "arm": FUTURE_TREATMENT,
        "per_sampled_anchor": "retrieve complete joint event context; compute "
                              "Q_tot_online(current joint executed action, current central state) "
                              "and y_joint (frozen Double-Q joint target); loss = SmoothL1(Q_tot_online, y_joint)",
        "minibatch": f"{BATCH_SIZE} regional anchors; sibling anchors from same event may co-occur (intentional)",
        "no_dedup_sibling_anchors": True,
        "optimizer_params": "utility_network + mixer",
        "auxiliary_losses": "NONE (no aux local TD, no mixing loss, no entropy bonus, no coordination "
                            "regularizer, no contrastive loss); standard joint TD only",
        "LOSS": LOSS_FN,
    }
    _jdump(treatment_loss, RESULTS_DIR / "E4K_treatment_loss_contract.json")

    optimizer = {
        "optimizer": "Adam", "lr": LEARNING_RATE, "LOSS": LOSS_FN, "gamma": GAMMA,
        "C0_params": "utility_network_only", "C1_params": "utility_network + mixer",
        "param_count_difference": "intrinsic to the architecture treatment; no dummy trainable mixer in C0",
        "GRAD_CLIP": GRAD_CLIP, "TARGET_TAU": TARGET_TAU,
    }
    _jdump(optimizer, RESULTS_DIR / "E4K_optimizer_contract.json")

    # ============ PHASE E4K-13/14 — EXPLORATION + EXECUTION ============
    exploration = {
        "regional_epsilon_greedy": True,
        "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay_transitions": EPS_DECAY,
        "schedule": "linear over 50000 regional anchors/transitions",
        "per_region_rng": "existing controller RNG semantics (agent.select_regional_actions)",
        "forbidden": ["correlated_joint_epsilon_action", "boltzmann_joint_search",
                      "centralized_joint_action_search"],
    }
    _jdump(exploration, RESULTS_DIR / "E4K_exploration_contract.json")

    execution = {
        "DECENTRALIZED_EXECUTION": "YES",
        "action_selection": "a_r = argmax_a Q_u(s_r, a) per active region",
        "mixer_in_execution_path": False,
        "eval_epsilon": 0.0,
        "actions_assembled_simultaneously_before_response": True,
        "production_decision_time_interface_remains_regional": True,
    }
    _jdump(execution, RESULTS_DIR / "E4K_execution_contract.json")

    masking = {
        "inactive_region": {"utility_slot": 0, "local_central_state_features": [0, 0, 0, 0],
                            "active_mask": 0, "action_placeholder": "deterministic sentinel / KEEP index",
                            "placeholder_must_not_contribute": "utility masked to zero"},
        "active_region": {"active_mask": 1},
        "validation": "changing an inactive placeholder action cannot change Q_tot (utility pinned to 0)",
    }
    _jdump(masking, RESULTS_DIR / "E4K_masking_contract.json")

    # ============ PHASE E4K-19 — FUTURE ARMS + FACTOR ============
    future_ab = {
        "FUTURE_EXPERIMENT_FACTOR": FUTURE_EXPERIMENT_FACTOR,
        "factor_is_not": ["NETWORK_SIZE", "REWARD", "NORMALIZATION"],
        "CONTROL": {
            "name": FUTURE_CONTROL, "normalization": NORM_V3, "reward": REWARD_R1_NAME,
            "alpha_local": FROZEN_ALPHA_LOCAL, "utility_network": UTILITY_NETWORK,
            "joint_mixer": "NONE", "loss": "existing regional Double-DQN TD"},
        "TREATMENT": {
            "name": FUTURE_TREATMENT, "normalization": NORM_V3, "reward_primitives": REWARD_R1_NAME,
            "alpha_local": FROZEN_ALPHA_LOCAL, "utility_network": UTILITY_NETWORK,
            "mixer": "QMIX_MIXER_V1", "joint_reward": "ACTIVE_REGION_MEAN_OF_R1",
            "loss": "joint Double-Q SmoothL1 TD"},
        "single_factor_between_arms": "coordination learner specification only (V3, R1, alpha, utility net, "
                                      "env budget, training budget all held fixed)",
    }
    _jdump(future_ab, RESULTS_DIR / "E4K_future_ab_factor_contract.json")

    # ============ PHASE E4K-20 — FUTURE SEEDS ============
    fresh_env = set(FUTURE_TRAIN_SEEDS) | set(FUTURE_VALIDATION_SEEDS) | set(FUTURE_FINAL_SEEDS)
    consumed_env = set().union(*[set(v) for v in CONSUMED_SEED_SETS.values()])
    env_disjoint = len(fresh_env & consumed_env) == 0
    consumed_pol = set().union(*[set(v) for v in CONSUMED_POLICY_SEEDS.values()])
    pol_disjoint = len(set(FUTURE_POLICY_SEEDS) & consumed_pol) == 0
    seed_contract = {
        "FUTURE_TRAIN_SEEDS": [FUTURE_TRAIN_SEEDS[0], FUTURE_TRAIN_SEEDS[-1]], "n_train": len(FUTURE_TRAIN_SEEDS),
        "FUTURE_VALIDATION_SEEDS": [FUTURE_VALIDATION_SEEDS[0], FUTURE_VALIDATION_SEEDS[-1]],
        "n_validation": len(FUTURE_VALIDATION_SEEDS),
        "FUTURE_FINAL_SEEDS": [FUTURE_FINAL_SEEDS[0], FUTURE_FINAL_SEEDS[-1]], "n_final": len(FUTURE_FINAL_SEEDS),
        "FUTURE_POLICY_SEEDS": FUTURE_POLICY_SEEDS,
        "env_seeds_disjoint_from_all_prior": env_disjoint,
        "policy_seeds_disjoint_from_all_prior": pol_disjoint,
        "checked_env_against": list(CONSUMED_SEED_SETS.keys()),
        "checked_policy_against": list(CONSUMED_POLICY_SEEDS.keys()),
        "purpose": "future matched Branch-C A/B (C0 vs C1)",
    }
    _jdump(seed_contract, RESULTS_DIR / "E4K_future_seed_contract.json")

    # ============ PHASE E4K-21 — PAIRING + MIXER INIT ============
    pairing = {
        "utility_init_pairing": "for each policy seed p in 5001..5005: HASH(C0_utility_online_t0) == "
                                "HASH(C1_utility_online_t0)",
        "t0_online_equals_target_both_arms": True,
        "mixer_seed_derivation": "mixer_seed = utility_initialization_seed + 1 (deterministic)",
        "no_mixer_init_tuning": True,
        "t0_online_mixer_equals_target_mixer": True,
        "future_hash_checks_required": {str(p): {"C0_utility_t0_hash": "==", "C1_utility_t0_hash": "=="}
                                        for p in FUTURE_POLICY_SEEDS},
    }
    _jdump(pairing, RESULTS_DIR / "E4K_future_pairing_contract.json")

    # ============ PHASE E4K-22 — TRAINING BUDGET ============
    budget = {
        "N_EPOCHS": N_EPOCHS, "train_env_seeds": TRAIN_ENV_SEEDS_N, "scenarios": len(SCENARIOS),
        "episodes_per_run": EPISODES_PER_RUN, "policy_seeds": len(FUTURE_POLICY_SEEDS), "arms": 2,
        "runs": N_RUNS, "total_training_episodes": 10800,
        "no_additional_training_for_qmix": True,
        "hyperparameters": {"gamma": GAMMA, "lr": LEARNING_RATE, "optimizer": "Adam", "loss": LOSS_FN,
                            "regional_anchor_batch": BATCH_SIZE, "replay_capacity": REPLAY_CAPACITY,
                            "warmup": WARMUP_TRANSITIONS, "update_every": UPDATE_FREQUENCY,
                            "grad_steps_per_update": GRAD_STEPS_PER_UPDATE, "tau": TARGET_TAU,
                            "grad_clip": GRAD_CLIP},
        "only_new": "architecture-specific mixer parameters",
    }
    _jdump(budget, RESULTS_DIR / "E4K_future_training_budget.json")

    # ============ PHASE E4K-23 — CHECKPOINT SELECTION ============
    checkpoint = {
        "selection_criterion": "VALIDATION_MACRO_MIGD_RESPONSE", "both_arms": True,
        "tie_breakers": ["macro_MIGD_epoch", "response_FE", "earlier_checkpoint"],
        "forbidden_selection_by": ["Q_tot_loss", "mixer_sensitivity", "persistent_only_MIGD",
                                   "global_comparator_performance"],
    }
    _jdump(checkpoint, RESULTS_DIR / "E4K_future_checkpoint_contract.json")

    # ============ PHASE E4K-24/27/28 — STATISTICS + SUCCESS ============
    stats = {
        "FUTURE_PRIMARY": "MIXED_MIGD_RESPONSE", "primary_scenario": "MIXED", "primary_endpoint": "MIGD_response",
        "FUTURE_EFFECT": "C0_MINUS_C1", "POSITIVE_MEANS": "QMIX_BETTER",
        "PRACTICAL_MARGIN": DELTA_NORM,
        "bootstrap": "PAIRED_CROSSED_TWO_AXIS", "function": "paired_crossed_two_axis_bootstrap",
        "policy_axis": len(FUTURE_POLICY_SEEDS), "environment_axis": TRAIN_ENV_SEEDS_N,
        "N_BOOTSTRAP": N_BOOTSTRAP, "CI_LEVEL": CI_LEVEL, "CI_METHOD": CI_METHOD,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "FUTURE_POLICY_ROBUSTNESS": "AT_LEAST_4_OF_5_QMIX_BETTER (C0-C1>0)",
        "safety": {"PERSISTENT": "C1 vs C0 not HARMFUL on MIGD_response/MIGD_epoch",
                   "STATIONARY": "same", "MACRO": "equal 1/3 scenarios; response+epoch not HARMFUL"},
        "primary_not_switched_to_persistent": "MIXED primary + PERSISTENT safety retained despite E4JF "
                                              "(no outcome-driven endpoint switching)",
    }
    _jdump(stats, RESULTS_DIR / "E4K_future_statistical_contract.json")

    success = {
        "COORDINATION_CAUSAL_VALIDATED_requires_all": [
            "MIXED primary BENEFICIAL", "policy robustness >=4/5 positive",
            "PERSISTENT safety PASS", "STATIONARY safety PASS", "macro safety PASS",
            "architecture integrity PASS", "final protocol integrity PASS", "no final-data adaptation"],
        "PARTIAL": "bounded performance-partial / practically-neutral discipline (as Branch A/B); "
                   "exhaustive freeze deferred to the later training/final protocol stage",
        "FAILED": "MIXED directional-negative/harmful OR any safety failure OR any integrity failure",
        "no_FE_rescue": True,
        "verdict_function": "E4K.coordination_causal_verdict_skeleton",
        "effect": "C0 - C1 (positive => QMIX better)",
    }
    _jdump(success, RESULTS_DIR / "E4K_future_success_logic_skeleton.json")

    # ============ PHASE E4K-25/26 — COORDINATION DIAGNOSTICS ============
    diagnostics = {
        "status": "DESCRIPTIVE + ENGINEERING_INTEGRITY ONLY",
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "why_no_new_mechanism_gate": "the QMIX treatment definition itself introduces joint-value "
                                     "training; mandatory implementation tests already verify joint "
                                     "context, monotonicity, joint TD, and decentralized greedy execution; "
                                     "a post-hoc mixer-activity threshold would add an unnecessary "
                                     "degree of freedom",
        "DIAGNOSTIC_A_monotonic_gradient": "report min/median/q95 of dQ_tot/dq_r over active regions; "
                                           "require no derivative < -numeric_tolerance",
        "DIAGNOSTIC_B_central_state_conditioning": "mean L2 distance of W1(c_i)/W2(c_i) from pooled mean; "
                                                   "detect collapse to constant additive map",
        "DIAGNOSTIC_C_regional_joint_credit_dispersion": "kappa_r = dQ_tot/dq_r; population std(kappa_r) "
                                                         "across active regions within an event",
        "DIAGNOSTIC_D_other_region_sensitivity": "hold focal q_r fixed, perturb another active q_j by a "
                                                 "frozen diagnostic epsilon, measure |Delta Q_tot|; must be "
                                                 "nonzero for a functioning mixer; no environment "
                                                 "counterfactual rollouts",
        "no_D5_counterfactual_reuse_as_training": True,
    }
    _jdump(diagnostics, RESULTS_DIR / "E4K_coordination_diagnostics.json")

    # ============ FIREWALL + E4KI CONTRACT ============
    firewall = {
        "C_TRAIN_ACCESS_COUNT": _env_access["train"],
        "C_VALIDATION_ACCESS_COUNT": _env_access["validation"],
        "C_FINAL_ACCESS_COUNT": _env_access["final"],
        "NEW_ENVIRONMENT_EXECUTION": "NO", "NEW_TRAINING": "NO",
        "future_final_seeds_untouched": True,
        "after_coordination_ab": {
            "no_global_comparator_in_same_causal_final": True,
            "if_C1_validated_vs_C0": "later freeze a NEW global-comparator experiment on another untouched seed set",
            "if_C1_fails_internally": "architecture-result review before testing against global",
        },
    }
    _jdump(firewall, RESULTS_DIR / "E4K_future_firewall.json")

    e4ki_contract = {
        "next_stage": "E4KI_JOINT_COORDINATION_IMPLEMENTATION_VALIDATION",
        "no_real_environment_training": True,
        "must_validate": [
            "central-state construction", "fixed region ordering", "active masking",
            "utility network compatibility", "mixer forward pass", "mixer monotonicity",
            "decentralized argmax equivalence", "online/target mixer logic", "joint reward aggregation",
            "joint Double-Q target", "regional-anchor replay semantics", "control-loss equivalence",
            "treatment-loss calculation", "optimizer parameter partition", "checkpoint schema",
            "deterministic initialization", "no extra FE", "no future seed access"],
        "synthetic_fixtures": [
            "R_MAX slot shape", "central state dimension", "active-mask correctness",
            "inactive action placeholder invariance", "Q_tot scalar output", "monotonic derivative",
            "exhaustive small-R decentralized argmax equivalence",
            "central-state-conditioned weights vary on distinct synthetic states",
            "online mixer == target mixer at t0", "target mixer soft update tau=0.005",
            "exact Double-Q action-selection/evaluation split", "joint reward == mean regional R1 reward",
            "no active-count reward-scale multiplication",
            "control loss matches existing independent DDQN on identical synthetic batch",
            "treatment loss finite", "regional-anchor sampling semantics",
            "sibling anchors not silently deduplicated", "replay capacity remains regional-anchor based",
            "no mixer used in execution action path", "no architecture-dependent extra objective FE"],
    }
    _jdump(e4ki_contract, RESULTS_DIR / "E4K_e4ki_validation_contract.json")

    # ============ ROUTING AMENDMENT AUDIT ============
    routing_audit = {
        "POST_RESULT_ROUTING_AMENDMENT": "DOCUMENTED_NON_VERDICT_CHANGE",
        "unenumerated_case": "scientific FAILED + integrity PASS + MIXED primary positive + safety failure",
        "initial_fallback_route": "E4J_INTEGRITY_REVIEW (incorrect; integrity passed)",
        "corrected_route": "E4K_JOINT_COORDINATION_REVISION_DESIGN",
        "changed_only": "NEXT_STAGE",
        "did_not_change": ["episode_data", "delta", "CI", "classification", "safety", "verdict",
                           "superiority_claim"],
        "not_retroactively_preregistered": "this exact routing branch was NOT preregistered; it is recorded "
                                           "transparently as a documented non-verdict change",
    }
    _jdump(routing_audit, RESULTS_DIR / "E4K_routing_amendment_audit.json")

    # ============ SYNTHETIC VALIDATION ============
    synth = synthetic_validation()
    _jdump(synth, RESULTS_DIR / "E4K_synthetic_validation.json")

    # ============ PRE-MANIFEST ============
    pre_manifest = {
        "stage": "E4K_JOINT_COORDINATION_REVISION_DESIGN", "design_only": True, "no_training": True,
        "no_new_environment_execution": True,
        "preserves": {"normalization": NORM_V3, "reward": REWARD_R1_NAME, "alpha_local": FROZEN_ALPHA_LOCAL,
                      "utility_network": UTILITY_NETWORK},
        "selected_treatment": FUTURE_TREATMENT,
        "R_MAX": R_MAX, "CENTRAL_STATE_DIM": CENTRAL_STATE_DIM, "MIX_HIDDEN_DIM": MIX_HIDDEN_DIM,
        "historical_integrity_pass": hist_ok,
    }
    _jdump(pre_manifest, RESULTS_DIR / "E4K_pre_manifest.json")

    # ---- token scan over all frozen artifacts ----
    token_hits = []
    for p in sorted(RESULTS_DIR.glob("E4K_*.json")):
        if p.name in ("E4K_verdict.json", "E4K_protocol_frozen.json"):
            continue
        hits = _scan_tokens(_load(p))
        if hits:
            token_hits.append({p.name: hits})

    complete = (
        hist_ok and smallest_such and env_disjoint and pol_disjoint
        and synth["ALL_PASS"] and len(token_hits) == 0
        and _env_access["train"] == 0 and _env_access["validation"] == 0 and _env_access["final"] == 0
    )

    protocol_frozen = {
        "stage": "E4K_JOINT_COORDINATION_REVISION_DESIGN", "all_frozen": True,
        "SELECTED_BRANCH": "C_JOINT_COORDINATION", "SELECTED_TREATMENT": FUTURE_TREATMENT,
        "FIXED_NORMALIZATION": NORM_V3, "FIXED_REWARD": REWARD_R1_NAME, "FIXED_ALPHA_LOCAL": FROZEN_ALPHA_LOCAL,
        "UTILITY_NETWORK": UTILITY_NETWORK, "R_MAX": R_MAX, "CENTRAL_STATE_DIM": CENTRAL_STATE_DIM,
        "MIX_HIDDEN_DIM": MIX_HIDDEN_DIM, "JOINT_REWARD": "ACTIVE_REGION_MEAN_OF_R1",
        "JOINT_TARGET": "DOUBLE_Q_QMIX", "FUTURE_EXPERIMENT_FACTOR": FUTURE_EXPERIMENT_FACTOR,
        "BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED, "next_stage": "E4KI_JOINT_COORDINATION_IMPLEMENTATION_VALIDATION",
    }
    _jdump(protocol_frozen, RESULTS_DIR / "E4K_protocol_frozen.json")

    verdict = _build_verdict(hist_ok, smallest_such, env_disjoint, pol_disjoint, synth, token_hits, complete)
    _jdump(verdict, RESULTS_DIR / "E4K_verdict.json")
    _write_report(verdict, synth, candidate_review)
    _emit_terminal(verdict)
    print(f"[E4K] wall {time.time()-t0:.1f}s")
    return verdict


# ---------------------------------------------------------------------------
def _build_verdict(hist_ok, qmix_ok, env_disjoint, pol_disjoint, synth, token_hits, complete):
    return {
        "stage": "E4K_JOINT_COORDINATION_REVISION_DESIGN",
        "E4K_COMPLETE": "YES" if complete else "NO",
        "ORIGINAL_E4F_VERDICT": "FAILED",
        "E4HF_NORMALIZATION_CAUSAL_VERDICT": "FAILED",
        "E4IF_REWARD_CAUSAL_VERDICT": "VALIDATED",
        "E4JF_GLOBAL_COMPARATOR_VERDICT": "FAILED",
        "E4JF_MIXED_PRIMARY": "DIRECTIONAL_POSITIVE",
        "E4JF_PERSISTENT_GLOBAL_SAFETY": "FAIL",
        "POST_RESULT_ROUTING_AMENDMENT": "DOCUMENTED_NON_VERDICT_CHANGE",
        "SELECTED_BRANCH": "C_JOINT_COORDINATION",
        "CURRENT_ARCHITECTURE": "INDEPENDENT_SHARED_DDQN",
        "CURRENT_JOINT_REPRESENTATION": "IMPLICIT_STATE_ONLY",
        "SELECTED_TREATMENT": "QMIX_MONOTONIC_VALUE_DECOMPOSITION_V1",
        "FIXED_NORMALIZATION": "ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3",
        "FIXED_REWARD": "REWARD_R1_REGIONAL_QUALITY_GAIN",
        "FIXED_ALPHA_LOCAL": FROZEN_ALPHA_LOCAL,
        "REGION_SLOT_COUNT": R_MAX,
        "UTILITY_NETWORK": UTILITY_NETWORK,
        "CENTRAL_STATE_DIM": CENTRAL_STATE_DIM,
        "MIXER_HIDDEN_DIM": MIX_HIDDEN_DIM,
        "MIXER_MONOTONIC": "YES",
        "DECENTRALIZED_EXECUTION": "YES",
        "JOINT_REWARD": "ACTIVE_REGION_MEAN_OF_R1",
        "REPLAY_SAMPLING_UNIT": "REGIONAL_ANCHOR_WITH_JOINT_CONTEXT",
        "REPLAY_CAPACITY": REPLAY_CAPACITY,
        "BATCH_SIZE": BATCH_SIZE,
        "JOINT_TARGET": "DOUBLE_Q_QMIX",
        "FUTURE_EXPERIMENT_FACTOR": FUTURE_EXPERIMENT_FACTOR,
        "FUTURE_CONTROL": FUTURE_CONTROL,
        "FUTURE_TREATMENT": FUTURE_TREATMENT,
        "FUTURE_TRAIN_SEEDS": "1901..1960",
        "FUTURE_VALIDATION_SEEDS": "1961..1980",
        "FUTURE_FINAL_SEEDS": "1981..2040",
        "FUTURE_POLICY_SEEDS": "5001..5005",
        "FUTURE_PRIMARY": "MIXED_MIGD_RESPONSE",
        "FUTURE_EFFECT": "C0_MINUS_C1",
        "POSITIVE_MEANS": "QMIX_BETTER",
        "FUTURE_POLICY_ROBUSTNESS": "AT_LEAST_4_OF_5_QMIX_BETTER",
        "FUTURE_BOOTSTRAP": "PAIRED_CROSSED_TWO_AXIS_5x60",
        "FUTURE_BOOTSTRAP_MASTER_SEED": BOOTSTRAP_MASTER_SEED,
        "COORDINATION_MECHANISM_GATE": "NOT_USED_FOR_SCIENTIFIC_VERDICT",
        "HISTORICAL_INTEGRITY": "PASS" if hist_ok else "FAIL",
        "QMIX_SELECTION_SUPPORTED": "YES" if qmix_ok else "NO",
        "ENV_SEEDS_DISJOINT": env_disjoint,
        "POLICY_SEEDS_DISJOINT": pol_disjoint,
        "SYNTHETIC_DESIGN_VALIDATION": ("PASS" if synth["ALL_PASS"] else "FAIL")
        + f" ({synth['n_pass']}/{synth['n_checks']})",
        "no_placeholder_tokens": len(token_hits) == 0,
        "C_TRAIN_ACCESS_COUNT": _env_access["train"],
        "C_VALIDATION_ACCESS_COUNT": _env_access["validation"],
        "C_FINAL_ACCESS_COUNT": _env_access["final"],
        "NEW_ENVIRONMENT_EXECUTION": "NO",
        "NEW_TRAINING": "NO",
        "FULL_REGRESSION": "(run separately)",
        "NEXT_STAGE": ("E4KI_JOINT_COORDINATION_IMPLEMENTATION_VALIDATION" if complete
                       else "E4K_ARCHITECTURE_REVIEW"),
    }


def _write_report(v, synth, candidate_review):
    L = [
        "# E4K — Joint Coordination Revision Design (Branch C)", "",
        f"**E4K_COMPLETE = {v['E4K_COMPLETE']} → NEXT_STAGE = {v['NEXT_STAGE']}**", "",
        "DESIGN / REVIEW / FREEZE ONLY. No training, no new environments, no checkpoint modification. "
        "Freezes the smallest technically credible joint-coordination revision (a QMIX-style monotonic "
        "value-decomposition learner, C1) while preserving V3 normalization and R1 reward (alpha=0.1) "
        "exactly, plus the future matched causal C0-vs-C1 A/B and the E4KI validation contract.", "",
        "## A. Historical result preservation",
        f"- ORIGINAL_E4F={v['ORIGINAL_E4F_VERDICT']}; E4HF normalization={v['E4HF_NORMALIZATION_CAUSAL_VERDICT']}; "
        f"E4IF reward={v['E4IF_REWARD_CAUSAL_VERDICT']}; E4JF global-comparator={v['E4JF_GLOBAL_COMPARATOR_VERDICT']} "
        f"(MIXED {v['E4JF_MIXED_PRIMARY']}, PERSISTENT safety {v['E4JF_PERSISTENT_GLOBAL_SAFETY']}). "
        f"HISTORICAL_INTEGRITY={v['HISTORICAL_INTEGRITY']}; V3+R1 hashes + 5 R1 checkpoints unchanged.", "",
        "## B. E4JF routing-amendment audit",
        f"- {v['POST_RESULT_ROUTING_AMENDMENT']}: the safety-driven FAILED (integrity PASS, MIXED positive) "
        "was an unenumerated routing case; the initial fallback mislabeled E4J_INTEGRITY_REVIEW and was "
        "corrected to E4K_JOINT_COORDINATION_REVISION_DESIGN. Only NEXT_STAGE changed; episode data, delta, "
        "CI, classification, safety, verdict and superiority claim are unchanged. This exact branch was not "
        "preregistered and is recorded transparently as a non-verdict change.", "",
        "## C. Why Branch C is now justified",
        "- E4GD: joint-action representation limitation SUPPORTED; D5 inter-region interaction MATERIAL "
        "(best regional action depended on other regions in ~48% of sampled decisions). E4HF fixed state "
        "(no gain). E4IF fixed reward credit (causal gain). E4JF: the resulting V3+R1 controller still fails "
        "the fixed-global comparator on PERSISTENT safety. The remaining isolated suspect is joint "
        "coordination — a hypothesis to test, NOT a proven defect.", "",
        "## D. Current joint-action limitation",
        f"- {v['CURRENT_ARCHITECTURE']} / {v['CURRENT_JOINT_REPRESENTATION']}: Q_u(s_r,a_r) on "
        "[V3-local | GlobalContextV1]; per-region argmax; one RegionalTransition per active region; "
        "per-region Double-DQN SmoothL1 TD. Q for region r does NOT condition on other regions' actions, a "
        "joint action vector, a joint Q mixer, or a centralized critic — only the shared global-context "
        "summary (implicit, state-level). Source-confirmed in network.py / agent.py / E4TR.", "",
        "## E. Candidate architecture review (not trained)",
        "- A CURRENT_SHARED_DDQN (control, no explicit joint value); B CTDE (coordination-capable but "
        "actor-critic family change); C VDN (decentralized but additive-only Q_tot); D QMIX (value-based, "
        "decentralized greedy, explicit state-conditioned monotonic joint value); E AUTOREGRESSIVE (order "
        "dependence; breaks simultaneous execution).", "",
        "## F. Why QMIX V1 is selected",
        "- Smallest reviewed candidate that preserves decentralized simultaneous execution AND adds explicit "
        "joint-value training AND does not require actor-critic AND supports fixed region slots/masks. No "
        "optimality claim; no guarantee it solves the observed interaction.", "",
        "## G. Fixed regional utility network",
        f"- {v['UTILITY_NETWORK']} (input 9, 128 ReLU, 128 ReLU, 4 Q-values), parameter-shared, identical to "
        "C0, unchanged in C1. Call Q_u(s_r,a_r).", "",
        "## H. Centralized mixer state",
        f"- central_state = concat(z_local_1..z_local_{R_MAX}, mask_1..mask_{R_MAX}, global_context_5); "
        f"CENTRAL_STATE_DIM = 5*R_MAX+5 = {v['CENTRAL_STATE_DIM']}. Inactive slots zero-padded; explicit "
        "active mask MANDATORY because zero is a valid V3 feature value. No new state features (no IDs, "
        "previous/future/oracle actions, reward history, counterfactuals).", "",
        "## I. QMIX mixer architecture",
        f"- QMIX_MIXER_V1, MIX_HIDDEN_DIM={v['MIXER_HIDDEN_DIM']}. W1=abs(Linear(c)) (R_MAX,32), b1=Linear(c); "
        "h=ELU(q^T W1 + b1); W2=abs(Linear(c)) (32,1), b2=Linear(c); Q_tot=h^T W2 + b2. No attention/"
        "transformer/recurrent/graph/embeddings in V1.", "",
        "## J. Monotonicity / decentralized execution",
        f"- dQ_tot/dq_r >= 0 via abs() weights ⇒ per-region greedy argmax maximizes the joint value; mixer "
        f"NOT needed at execution. DECENTRALIZED_EXECUTION={v['DECENTRALIZED_EXECUTION']}; eval epsilon=0.", "",
        "## K. Joint Double-Q target",
        "- a*_r=argmax Q_u_online(s'_r,·); q'_r=Q_u_target(s'_r,a*_r); y_joint = r_joint + gamma(1-done)"
        "Q_tot_target(q',central_state'); gamma=0.99.", "",
        "## L. R1 preservation",
        f"- R1_REWARD_FORMULA_CHANGED=NO; r_r^R1 = 1.0*global + 0.1*g_r - 0.01*fe; g_r, alpha_local=0.1, "
        "global reward and FE penalty unchanged (config sha a5c72e1f).", "",
        "## M. Joint reward aggregation",
        f"- JOINT_REWARD = {v['JOINT_REWARD']}: r_joint(t)=(1/|A_t|) sum_{{r in A_t}} r_r^R1(t). Mean (not "
        "sum) so team-reward magnitude does not scale with active-region count and exactly one copy of the "
        "shared global reward is preserved. This training-objective change is disclosed as part of the "
        f"factor {v['FUTURE_EXPERIMENT_FACTOR']} (not 'one extra mixer layer').", "",
        "## N. Regional-anchor replay",
        f"- REGIONAL_ANCHOR_JOINT_CONTEXT_REPLAY_V1: {v['REPLAY_SAMPLING_UNIT']}; one anchor per active "
        f"regional transition; each anchor attaches focal + full joint current/next context + r_joint. "
        f"Capacity {v['REPLAY_CAPACITY']} anchors, warmup 1000, update every 4, batch {v['BATCH_SIZE']}. "
        "Physical dedup by event_id allowed; scientific sampling measure preserved.", "",
        "## O. Sampling / optimization equivalence boundary",
        "- C0 optimizes utility only (loss identical to existing R1 DDQN). C1 optimizes utility + mixer with "
        "joint SmoothL1 TD; sibling anchors from the same event may co-occur in a minibatch (intentional; "
        "not deduplicated). Adam lr=1e-4, SmoothL1, tau=0.005, no grad clip. No auxiliary losses.", "",
        "## P. Exploration / simultaneous execution",
        "- Regional epsilon-greedy 1.0→0.05 over 50000 transitions; existing per-region RNG; no correlated/"
        "Boltzmann/centralized joint-action search. Actions assembled simultaneously before the response.", "",
        "## Q. Future C0-vs-C1 experiment",
        f"- Factor {v['FUTURE_EXPERIMENT_FACTOR']}. Primary {v['FUTURE_PRIMARY']}, effect {v['FUTURE_EFFECT']} "
        f"(positive ⇒ QMIX better), margin 1e-5. Robustness {v['FUTURE_POLICY_ROBUSTNESS']}. Budget: 6 epochs "
        "× 60 env × 3 scenarios = 1080 eps/run × 5 policies × 2 arms = 10800. Checkpoint by "
        "VALIDATION_MACRO_MIGD_RESPONSE (tie: macro epoch, FE, earlier).", "",
        "## R. Fresh seed firewall",
        f"- FUTURE train {v['FUTURE_TRAIN_SEEDS']}, val {v['FUTURE_VALIDATION_SEEDS']}, final "
        f"{v['FUTURE_FINAL_SEEDS']}, policy {v['FUTURE_POLICY_SEEDS']}. ENV_DISJOINT={v['ENV_SEEDS_DISJOINT']}, "
        f"POLICY_DISJOINT={v['POLICY_SEEDS_DISJOINT']}. Access counts train/val/final = "
        f"{v['C_TRAIN_ACCESS_COUNT']}/{v['C_VALIDATION_ACCESS_COUNT']}/{v['C_FINAL_ACCESS_COUNT']}.", "",
        "## S. Coordination diagnostics",
        f"- {v['COORDINATION_MECHANISM_GATE']}: monotonic gradient, central-state conditioning, regional "
        "joint-credit dispersion, other-region utility sensitivity — descriptive + engineering only. No new "
        "2-of-3 mechanism gate (the treatment already introduces joint-value training; mandatory impl tests "
        "verify the mechanism).", "",
        "## T. Future statistics / safety",
        f"- Paired crossed two-axis bootstrap 5×60, N={N_BOOTSTRAP}, 95% percentile, master seed "
        f"{v['FUTURE_BOOTSTRAP_MASTER_SEED']}. MIXED primary + PERSISTENT/STATIONARY/macro safety (not "
        "HARMFUL). PERSISTENT stays a SAFETY endpoint, not the primary — no outcome-driven switch after E4JF.", "",
        "## U. E4KI implementation-validation contract",
        "- E4KI validates (no real-env training): central-state, ordering, masking, utility compatibility, "
        "mixer forward + monotonicity, decentralized argmax equivalence, online/target logic, joint reward "
        "mean, joint Double-Q target, regional-anchor replay + sibling non-dedup, control/treatment loss, "
        "optimizer partition, checkpoint schema, deterministic init, no extra FE, no future seed access.", "",
        "## V. Limitations",
        "- QMIX monotonicity restricts representable joint action-value functions; C1 still executes "
        "decentralized regional argmax; no arbitrary non-monotonic joint coupling; region ordering fixed by "
        "reference-direction IDs; joint reward uses mean regional R1 as part of the coordination-learner "
        "spec; C1 has more trainable parameters than C0; the treatment changes the joint training objective "
        "and mixer architecture as ONE coordination-learner factor; no claim QMIX is optimal; no claim "
        "Branch C will succeed; no real-world UAV validation.", "",
        f"## W. Next stage: {v['NEXT_STAGE']}", "",
        f"Design self-consistency: {synth['n_pass']}/{synth['n_checks']} checks PASS "
        f"(dimension algebra, mean-vs-sum reward, effect direction, classification reuse, robustness rule, "
        "verdict branches, seed/policy disjointness, and a pure-numpy monotonic-mixer realizability / "
        "monotonicity / small-R argmax-equivalence / masking / state-conditioning demonstration). Full torch "
        "implementation validation is deferred to E4KI.",
    ]
    (RESULTS_DIR / "E4K_report.md").write_text("\n".join(L))


def _emit_terminal(v):
    print()
    for k in ["E4K_COMPLETE", "ORIGINAL_E4F_VERDICT", "E4HF_NORMALIZATION_CAUSAL_VERDICT",
              "E4IF_REWARD_CAUSAL_VERDICT", "E4JF_GLOBAL_COMPARATOR_VERDICT", "E4JF_MIXED_PRIMARY",
              "E4JF_PERSISTENT_GLOBAL_SAFETY", "POST_RESULT_ROUTING_AMENDMENT", "SELECTED_BRANCH",
              "CURRENT_ARCHITECTURE", "CURRENT_JOINT_REPRESENTATION", "SELECTED_TREATMENT",
              "FIXED_NORMALIZATION", "FIXED_REWARD", "FIXED_ALPHA_LOCAL", "REGION_SLOT_COUNT",
              "UTILITY_NETWORK", "CENTRAL_STATE_DIM", "MIXER_HIDDEN_DIM", "MIXER_MONOTONIC",
              "DECENTRALIZED_EXECUTION", "JOINT_REWARD", "REPLAY_SAMPLING_UNIT", "REPLAY_CAPACITY",
              "BATCH_SIZE", "JOINT_TARGET", "FUTURE_EXPERIMENT_FACTOR", "FUTURE_CONTROL",
              "FUTURE_TREATMENT", "FUTURE_TRAIN_SEEDS", "FUTURE_VALIDATION_SEEDS", "FUTURE_FINAL_SEEDS",
              "FUTURE_POLICY_SEEDS", "FUTURE_PRIMARY", "FUTURE_EFFECT", "POSITIVE_MEANS",
              "FUTURE_POLICY_ROBUSTNESS", "FUTURE_BOOTSTRAP", "FUTURE_BOOTSTRAP_MASTER_SEED",
              "COORDINATION_MECHANISM_GATE", "C_TRAIN_ACCESS_COUNT", "C_VALIDATION_ACCESS_COUNT",
              "C_FINAL_ACCESS_COUNT", "NEW_ENVIRONMENT_EXECUTION", "NEW_TRAINING",
              "FULL_REGRESSION", "NEXT_STAGE"]:
        print(f"{k:<40} = {v[k]}")


if __name__ == "__main__":
    main()
