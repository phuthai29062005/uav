# FROZEN_CONTROLLER_AUDIT — E4V

All values verified from source code inspection (not documentation).
Source files: `src/prdrl/rl/network.py`, `src/prdrl/rl/coordination_graph.py`,
`src/prdrl/rl/agent.py`, `src/prdrl/rl/normalization.py`,
`src/prdrl/rl/reward_r1.py`, `src/prdrl/rl/interface.py`,
`src/prdrl/detectors/severity.py`, `src/prdrl/detectors/quality_loss.py`

## C1: Independent DQN (SharedRegionalQNetwork)

- **Architecture name**: `C1` (independent per-region; no coordination mechanism)
- **Network**: `SharedRegionalQNetwork` — shared parameters across all active regions
- **Input dim**: 9 (4 local + 5 global context)
- **Architecture**: 9 → 128 (ReLU) → 128 (ReLU) → 4
- **Trainable params**: 9×128 + 128 + 128×128 + 128 + 128×4 + 4 = 18,308
- **Output**: 4 Q-values for [KEEP, LOCAL, PREDICT, DIVERSIFY]
- **Inference**: Independent argmax per region (no coordination)
- **Training**: Double DQN with Polyak soft update

## C3: Pairwise Coordination Graph (C3_PAIRWISE_COORDINATION_GRAPH_V1)

- **Architecture name**: `C3_PAIRWISE_COORDINATION_GRAPH_V1`
- **Unary network**: SharedRegionalQNetwork 9 → 128 → 128 → 4 (18,308 params)
- **Pair network**: SharedPairNetwork 18 → 128 → 128 → 16 (21,008 params)
- **Total trainable params**: 39,316
- **Pair output reshape**: 16 → 4×4 matrix
- **Pair symmetrization**: M_ij = 0.5 * (H_ij + H_ji^T)
- **Pair centering**: zero row/column mean (double centering)
- **Graph topology**: complete undirected graph over active regions
- **Joint value**: Q_CG = (1/n)ΣU_i + (1/|E|)ΣP_ij
- **Inference**: MaxSum message passing, K=10 synchronous iterations
- **Message normalization**: subtract max
- **BETA_PAIR**: 1 (unary and pair equally weighted)
- **R_MAX**: 10 regions maximum

## C4: Gated Pairwise Coordination (C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1)

- **Architecture name**: `C4_CAUSAL_DYNAMICS_GATED_PAIRWISE_COORDINATION_V1`
- **Base controller**: C3 (identical except gate multiplication)
- **Unary network**: SharedRegionalQNetwork 9 → 128 → 128 → 4 (18,308 params)
- **Pair network**: SharedPairNetwork 18 → 128 → 128 → 16 (21,008 params)
- **Total trainable params**: 39,316 (same as C3)
- **Gate trainable params**: 0
- **Gate formula**: g_ij(t) = 0.5 * (|d_i(t)| + |d_j(t)|)
- **Gate signal**: d_r = abs(state_r[3]) = abs(V3-normalized delta_severity)
- **Gate signal dimension index**: 3 (GATE_SIGNAL_DIM)
- **Gate applied after**: pair symmetrization and double centering
- **Effective pair**: g_ij * P_ij
- **Joint value**: Q_C4 = (1/n)ΣU_i + (1/|E|)Σ(g_ij * P_ij)
- **Inference**: Gated MaxSum (pair potentials multiplied by g_ij)

## Shared Hyperparameters (all three controllers)

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam | coordination_graph.py make_c3_config/make_c4_config |
| Learning rate | 1e-4 | make_c3_config/make_c4_config |
| Gamma (discount) | 0.99 | make_c3_config/make_c4_config |
| Tau (Polyak) | 0.005 | make_c3_config/make_c4_config |
| Batch size | 128 | make_c3_config/make_c4_config |
| Replay capacity | 50,000 | make_c3_config/make_c4_config |
| Warmup | 1,000 | make_c3_config/make_c4_config |
| Update every | 4 | make_c3_config/make_c4_config |
| Gradient clipping | None | no_gradient_clipping: True |
| Loss | SmoothL1 (joint) | c3_loss/c4_loss |
| Exploration | ε-greedy, linear 1.0→0.05 over 50,000 transitions | agent.py EpsilonGreedyConfig |

## State Features (V2 schema, 9 dimensions)

| Dim | Feature | Mathematical definition | Oracle-free? |
|-----|---------|------------------------|-------------|
| 0 | severity_r | C[r] = 1 - exp(-S[r]/(κ*q_prev)), S[r]=median(‖F⁻-F⁺‖) | YES |
| 1 | convergence_loss_r | PBI-based quality loss: median(max(0, g⁺-g⁻)) per region | YES |
| 2 | occupancy_r | n_r / N_total | YES |
| 3 | delta_severity_r | C[r](t) - C[r](t-1) | YES |
| 4 | active_region_fraction | count(active) / R_MAX | YES |
| 5 | mean_severity | mean of C[r] across active regions | YES |
| 6 | max_severity | max of C[r] across active regions | YES |
| 7 | mean_convergence_loss | mean of quality_loss across active regions | YES |
| 8 | max_convergence_loss | max of quality_loss across active regions | YES |

**CRITICAL**: `convergence_loss_r` uses PBI decomposition against reference directions,
comparing F_minus (pre-change) to F_plus (post-change). This is computed from the
algorithm's own population and reference directions — it does NOT use true PF.

**CRITICAL**: `severity_r` uses ‖F_minus - F_plus‖ (objective-space distance between
pre-change and post-change evaluations). This does NOT use true PF.

**E4U LEAKAGE NOTE**: The E4U benchmark (benchmark_c4_v1) used IGD to true PF as
`convergence_loss` in the state construction. This constituted oracle leakage and
must NOT be replicated in E4V. The E4V adapter must use the frozen UAV state
construction (PBI-based quality loss) or a deployable equivalent.

## Normalization (V3: Zero-Aware Robust Log)

- **Version**: ZERO_AWARE_ROBUST_LOG_NORMALIZATION_V3
- **Nonneg zero-inflated dims** (0,1,5,6,7,8): x≤0→0; x>0→clip(log1p(x/m)/log1p(c/m), 0, 1)
- **Identity dims** (2,4): clip[0,1]
- **Delta dim** (3): clip(x/(a3+eps), -1, 1), sign/zero preserved
- **Constants**: fitted from TRAINING states only (not from test data)
- **Note**: V3 constants are domain-specific (fitted from UAV training). For benchmark
  adapter, constants must either be re-fitted from benchmark training data or the
  normalizer must be re-initialized appropriately.

## Reward (R1: Regional Quality Gain)

- **Version**: REWARD_R1_REGIONAL_QUALITY_GAIN
- **Formula**: r_r = α_g * r_global + α_local * g_r - β * FE_cost
- **g_r** = (L_pre - L_post) / (L_pre + L_post + eps) ∈ [-1, 1]
- **α_g** = 1.0, **α_local** = 0.1 (FROZEN), **β** = 0.01
- **L_pre/L_post**: regional quality loss (PBI-based, NOT true-PF-based)
- **No true PF in reward**: reward is entirely population-based

## Actions

| Index | Name | Response operator | Sigma | FE cost |
|-------|------|------------------|-------|---------|
| 0 | KEEP | No offspring | - | 0 |
| 1 | LOCAL | Gaussian perturbation | 0.02 | B=10 |
| 2 | PREDICT | Linear prediction + perturbation (fallback to LOCAL if no history) | 0.005 (λ=1.0) | B=10 |
| 3 | DIVERSIFY | Large Gaussian perturbation | 0.15 | B=10 |

## Initialization

- **Unary net init**: Xavier uniform with seeded Generator
- **Pair net init**: Xavier uniform with seeded Generator (pair_seed = utility_seed + 1)
- **Bias init**: zeros
- **Target net**: copy of online, requires_grad=False

## Seed Ranges (from frozen configs)

| Controller | Train | Validation | Final | Policy |
|-----------|-------|------------|-------|--------|
| C3 | 2501-2560 | 2561-2580 | 2581-2640 | 7001-7005 |
| C4 | 2701-2760 | 2761-2780 | 2781-2840 | 8001-8005 |
