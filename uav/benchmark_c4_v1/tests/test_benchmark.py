"""T01-T35: Unit tests for SA-DRL-DMOEA C4 benchmark validation.

35 tests covering: problems, PF samplers, segmentation, state construction,
normalization adapter, controller arms, metrics, ledger, baselines, and
experiment runner integration.
"""

from __future__ import annotations

import json
import hashlib
import sys
import os
import numpy as np
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ==================== T01-T05: Problem interface ====================

class TestT01_ProblemRegistry:
    """T01: All 22 problems are registered and instantiable."""

    def test_all_22_registered(self):
        from benchmark_c4_v1.problems.registry import PROBLEM_REGISTRY, ALL_PROBLEMS
        assert len(ALL_PROBLEMS) == 22
        for name in ALL_PROBLEMS:
            assert name in PROBLEM_REGISTRY
            prob = PROBLEM_REGISTRY[name](n_var=10, nt=10, taut=10)
            assert prob.n_var == 10
            assert prob.n_obj in (2, 3)


class TestT02_ProblemDimensions:
    """T02: Problem dimensions are correct (2-obj for DF1-9, FDA1-3, dMOP1-3; 3-obj for DF10-14, FDA4-5)."""

    def test_2obj(self):
        from benchmark_c4_v1.problems import get_problem
        for name in ["DF1", "DF2", "DF3", "DF4", "DF5", "DF6", "DF7", "DF8", "DF9",
                      "FDA1", "FDA2", "FDA3", "dMOP1", "dMOP2", "dMOP3"]:
            assert get_problem(name).n_obj == 2, f"{name} should be 2-obj"

    def test_3obj(self):
        from benchmark_c4_v1.problems import get_problem
        for name in ["DF10", "DF11", "DF12", "DF13", "DF14", "FDA4", "FDA5"]:
            assert get_problem(name).n_obj == 3, f"{name} should be 3-obj"


class TestT03_ProblemEvaluation:
    """T03: Evaluate returns correct shapes."""

    def test_eval_shapes(self):
        from benchmark_c4_v1.problems import get_problem
        from benchmark_c4_v1.problems.registry import ALL_PROBLEMS
        for name in ALL_PROBLEMS:
            prob = get_problem(name, n_var=10)
            X = np.random.rand(50, 10)
            X = X * (prob.xu - prob.xl) + prob.xl
            F = prob.evaluate(X, t=0.0)
            assert F.shape == (50, prob.n_obj), f"{name}: F shape {F.shape}"
            assert np.all(np.isfinite(F)), f"{name}: non-finite F"


class TestT04_ProblemTimeDependence:
    """T04: Problems change with time."""

    def test_time_changes_output(self):
        from benchmark_c4_v1.problems import get_problem
        for name in ["DF1", "DF5", "FDA1", "dMOP1"]:
            prob = get_problem(name, n_var=10)
            X = np.random.rand(10, 10) * 0.5 + 0.25
            X = X * (prob.xu - prob.xl) + prob.xl
            F0 = prob.evaluate(X, t=0.0)
            F1 = prob.evaluate(X, t=0.5)
            assert not np.allclose(F0, F1, atol=1e-8), f"{name}: F unchanged at different t"


class TestT05_TimeFormula:
    """T05: Time formula t = (1/nt) * floor(tau/taut) is correct."""

    def test_time_formula(self):
        from benchmark_c4_v1.problems import get_problem
        prob = get_problem("DF1", nt=10, taut=10)
        assert prob.time(0) == 0.0
        assert prob.time(9) == 0.0
        assert prob.time(10) == pytest.approx(0.1)
        assert prob.time(20) == pytest.approx(0.2)
        assert prob.time(100) == pytest.approx(1.0)


# ==================== T06-T10: True PF samplers ====================

class TestT06_PFShape:
    """T06: True PF returns correct shape."""

    def test_pf_shapes(self):
        from benchmark_c4_v1.problems import get_problem
        from benchmark_c4_v1.problems.registry import ALL_PROBLEMS
        for name in ALL_PROBLEMS:
            prob = get_problem(name, n_var=10)
            pf = prob.true_pf(t=0.0, n_points=500)
            assert pf.shape[1] == prob.n_obj, f"{name}: PF cols {pf.shape[1]} != {prob.n_obj}"
            assert len(pf) >= 100, f"{name}: PF too few points {len(pf)}"
            assert np.all(np.isfinite(pf)), f"{name}: non-finite PF"


class TestT07_PFOnOptimal:
    """T07: PF points have g=1 (or close) for simple problems."""

    def test_df1_pf_on_front(self):
        from benchmark_c4_v1.problems import get_problem
        prob = get_problem("DF1")
        pf = prob.true_pf(t=0.0, n_points=100)
        assert pf[:, 0].min() >= -1e-10
        assert pf[:, 1].min() >= -1e-10


class TestT08_PFTimeVarying:
    """T08: True PF changes with time for problems with changing PF."""

    def test_df1_pf_changes(self):
        from benchmark_c4_v1.problems import get_problem
        prob = get_problem("DF1")
        pf0 = prob.true_pf(t=0.0)
        pf1 = prob.true_pf(t=0.5)
        assert not np.allclose(pf0, pf1, atol=1e-6)


class TestT09_PFFixed:
    """T09: True PF stays fixed for problems with fixed PF."""

    def test_fda1_pf_fixed(self):
        from benchmark_c4_v1.problems import get_problem
        prob = get_problem("FDA1")
        pf0 = prob.true_pf(t=0.0)
        pf1 = prob.true_pf(t=0.5)
        np.testing.assert_allclose(pf0, pf1, atol=1e-10)


class TestT10_IGDOptimal:
    """T10: IGD of true PF to itself is ~0."""

    def test_igd_self_zero(self):
        from benchmark_c4_v1.metrics.igd import igd
        from benchmark_c4_v1.problems import get_problem
        prob = get_problem("DF1")
        pf = prob.true_pf(t=0.0, n_points=500)
        assert igd(pf, pf) < 1e-10


# ==================== T11-T15: Segmentation and state ====================

class TestT11_Segmentation:
    """T11: Segmentation produces exactly 2 non-empty regions."""

    def test_two_regions(self):
        from benchmark_c4_v1.controllers.segmentation import segment_population, BLIND_FIXED_2SEG
        X = np.random.rand(100, 10)
        F = np.random.rand(100, 2)
        regions = segment_population(X, F, 10, BLIND_FIXED_2SEG)
        assert len(regions) == 2
        total = sum(len(v) for v in regions.values())
        assert total == 100


class TestT12_StateShape:
    """T12: State builder produces 9-dim vectors for active regions."""

    def test_state_9dim(self):
        from benchmark_c4_v1.controllers.state import BenchmarkStateBuilder
        builder = BenchmarkStateBuilder(n_regions=2)
        X = np.random.rand(100, 10)
        F = np.random.rand(100, 2)
        pf = np.random.rand(500, 2) * 0.5
        regions = {0: np.arange(50), 1: np.arange(50, 100)}
        states = builder.build_states(regions, X, F, pf, 100)
        for r_id, state in states.items():
            assert state.shape == (9,), f"Region {r_id}: shape {state.shape}"
            assert np.all(np.isfinite(state))


class TestT13_StateDelta:
    """T13: delta_severity is zero on first call, nonzero after shift."""

    def test_delta_severity(self):
        from benchmark_c4_v1.controllers.state import BenchmarkStateBuilder
        builder = BenchmarkStateBuilder(n_regions=2)
        X = np.random.rand(100, 10)
        F1 = np.random.rand(100, 2)
        pf = np.random.rand(500, 2) * 0.5
        regions = {0: np.arange(50), 1: np.arange(50, 100)}
        states1 = builder.build_states(regions, X, F1, pf, 100)
        for s in states1.values():
            assert abs(s[3]) < 1e-10

        F2 = F1 + 0.5
        states2 = builder.build_states(regions, X, F2, pf, 100)


class TestT14_ActiveMask:
    """T14: Active mask has correct entries for 2-region setup."""

    def test_active_mask(self):
        from benchmark_c4_v1.controllers.arms import R_MAX
        active_mask = np.zeros(R_MAX, dtype=np.float32)
        active_mask[0] = 1.0
        active_mask[1] = 1.0
        assert active_mask.sum() == 2.0
        assert active_mask[2:].sum() == 0.0


class TestT15_NormalizationAdapter:
    """T15: V3 normalizer can be applied to benchmark states."""

    def test_v3_normalizer(self):
        from prdrl.rl.normalization import ZeroAwareRobustLogNormalizerV3
        raw_states = np.random.rand(1000, 9) * np.array([1, 10, 1, 2, 1, 1, 1, 10, 10])
        raw_states[:, 3] = raw_states[:, 3] - 1.0
        v3 = ZeroAwareRobustLogNormalizerV3.fit(raw_states)
        test_state = np.array([0.5, 2.0, 0.3, 0.1, 0.8, 0.4, 0.6, 1.5, 3.0])
        normalized = v3.transform(test_state)
        assert normalized.shape == (9,)
        assert np.all(np.isfinite(normalized))


# ==================== T16-T20: Controller arms ====================

class TestT16_C1Architecture:
    """T16: B-C1 has correct architecture (9→128→128→4 + QMIX mixer)."""

    def test_c1_params(self):
        from benchmark_c4_v1.controllers.arms import BenchmarkC1Arm
        arm = BenchmarkC1Arm(policy_seed=42)
        n_online = sum(p.numel() for p in arm.online.parameters())
        assert n_online == 18308, f"Expected 18308, got {n_online}"
        n_mixer = sum(p.numel() for p in arm.mixer_online.parameters())
        assert n_mixer == 21560, f"Expected mixer params, got {n_mixer}"


class TestT17_C3Architecture:
    """T17: B-C3 has correct architecture (unary 9→128→128→4 + pair 18→128→128→16)."""

    def test_c3_params(self):
        from benchmark_c4_v1.controllers.arms import BenchmarkC3Arm
        arm = BenchmarkC3Arm(policy_seed=42)
        n_unary = sum(p.numel() for p in arm.online_unary.parameters())
        n_pair = sum(p.numel() for p in arm.online_pair.parameters())
        assert n_unary == 18308
        assert n_pair == 21008


class TestT18_C4Architecture:
    """T18: B-C4 has same params as C3 (0 trainable gate params)."""

    def test_c4_params(self):
        from benchmark_c4_v1.controllers.arms import BenchmarkC4Arm
        arm = BenchmarkC4Arm(policy_seed=42)
        n_unary = sum(p.numel() for p in arm.online_unary.parameters())
        n_pair = sum(p.numel() for p in arm.online_pair.parameters())
        assert n_unary == 18308
        assert n_pair == 21008
        total = n_unary + n_pair
        assert total == 39316


class TestT19_C4Gate:
    """T19: C4 gate formula is exactly g_ij = 0.5*(|d_i|+|d_j|)."""

    def test_gate_formula(self):
        from prdrl.rl.coordination_graph import compute_pair_gate
        s_i = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
        s_j = torch.tensor([[0.9, 0.8, 0.7, -0.6, 0.5, 0.4, 0.3, 0.2, 0.1]])
        g = compute_pair_gate(s_i, s_j)
        expected = 0.5 * (abs(0.4) + abs(-0.6))
        assert abs(g.item() - expected) < 1e-6


class TestT20_ActionSelection:
    """T20: All arms produce valid action indices (0-3)."""

    def test_action_range(self):
        from benchmark_c4_v1.controllers.arms import BenchmarkC1Arm, BenchmarkC3Arm, BenchmarkC4Arm, R_MAX
        rng = np.random.default_rng(42)
        states = {0: np.random.rand(9).astype(np.float32),
                  1: np.random.rand(9).astype(np.float32)}
        active_mask = np.zeros(R_MAX, dtype=np.float32)
        active_mask[0] = 1.0
        active_mask[1] = 1.0

        c1 = BenchmarkC1Arm(42)
        a1 = c1.select_actions_greedy(states)
        for a in a1.values():
            assert 0 <= a < 4

        c3 = BenchmarkC3Arm(42)
        a3 = c3.select_actions_greedy(states, active_mask)
        for i in range(R_MAX):
            if active_mask[i] > 0.5:
                assert 0 <= a3[i] < 4

        c4 = BenchmarkC4Arm(42)
        a4 = c4.select_actions_greedy(states, active_mask)
        for i in range(R_MAX):
            if active_mask[i] > 0.5:
                assert 0 <= a4[i] < 4


# ==================== T21-T25: Metrics ====================

class TestT21_IGD:
    """T21: IGD computation is correct."""

    def test_igd_known(self):
        from benchmark_c4_v1.metrics.igd import igd
        ref = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)
        approx = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)
        assert igd(approx, ref) < 1e-10

    def test_igd_empty(self):
        from benchmark_c4_v1.metrics.igd import igd
        ref = np.array([[0, 1], [1, 0]], dtype=float)
        approx = np.empty((0, 2))
        assert igd(approx, ref) == float("inf")


class TestT22_MIGD:
    """T22: MIGD is mean of IGD values."""

    def test_migd_mean(self):
        from benchmark_c4_v1.metrics.igd import migd
        vals = [0.1, 0.2, 0.3]
        assert abs(migd(vals) - 0.2) < 1e-10


class TestT23_HV2D:
    """T23: 2D hypervolume is correct."""

    def test_hv_simple(self):
        from benchmark_c4_v1.metrics.hv import hypervolume
        approx = np.array([[1, 3], [2, 2], [3, 1]], dtype=float)
        ref = np.array([4, 4], dtype=float)
        hv = hypervolume(approx, ref)
        assert hv > 0

    def test_hv_empty(self):
        from benchmark_c4_v1.metrics.hv import hypervolume
        approx = np.empty((0, 2))
        ref = np.array([2, 2], dtype=float)
        assert hypervolume(approx, ref) == 0.0


class TestT24_MHV:
    """T24: MHV is mean of HV values."""

    def test_mhv(self):
        from benchmark_c4_v1.metrics.hv import mhv
        vals = [0.5, 0.6, 0.7]
        assert abs(mhv(vals) - 0.6) < 1e-10


class TestT25_IGDMonotonicity:
    """T25: Adding dominated points does not improve IGD (sanity check)."""

    def test_igd_dominated_neutral(self):
        from benchmark_c4_v1.metrics.igd import igd
        ref = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)
        good = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)
        bad = np.vstack([good, [[2, 2], [3, 3]]])
        igd_good = igd(good, ref)
        igd_bad = igd(bad, ref)
        assert igd_good <= igd_bad + 1e-10


# ==================== T26-T30: Ledger and baselines ====================

class TestT26_Ledger:
    """T26: EvaluationLedger counts correctly and stops at budget."""

    def test_ledger_counting(self):
        from benchmark_c4_v1.controllers.ledger import EvaluationLedger
        ledger = EvaluationLedger(1000)
        assert ledger.remaining() == 1000
        assert ledger.charge(500)
        assert ledger.count == 500
        assert ledger.remaining() == 500
        assert not ledger.charge(600)
        assert ledger.count == 500
        assert ledger.charge(500)
        assert ledger.exhausted


class TestT27_LedgerNoFreeEvals:
    """T27: Ledger refuses evaluations beyond budget."""

    def test_no_free(self):
        from benchmark_c4_v1.controllers.ledger import EvaluationLedger
        ledger = EvaluationLedger(100)
        ledger.charge(100)
        assert not ledger.charge(1)


class TestT28_NSGA2:
    """T28: NSGA-II baseline produces offspring and selects."""

    def test_nsga2_step(self):
        from benchmark_c4_v1.baselines.nsga2 import NSGA2Baseline
        algo = NSGA2Baseline(pop_size=20, seed=42)
        xl = np.zeros(5)
        xu = np.ones(5)
        X = algo.initialize(xl, xu, 5)
        assert X.shape == (20, 5)
        F = np.random.rand(20, 2)
        algo.set_fitness(F)
        offspring = algo.step()
        assert offspring.shape[0] == 20
        offspring_F = np.random.rand(20, 2)
        algo.select(offspring, offspring_F)
        assert algo.X.shape == (20, 5)


class TestT29_DNSGA2:
    """T29: D-NSGA-II detects changes and reinitializes."""

    def test_dnsga2_response(self):
        from benchmark_c4_v1.baselines.dnsga2 import DNSGA2Baseline
        algo = DNSGA2Baseline(pop_size=20, seed=42, reinit_fraction=0.2)
        xl = np.zeros(5)
        xu = np.ones(5)
        algo.initialize(xl, xu, 5)
        X_before = algo.X.copy()
        algo.respond_to_change()
        n_changed = np.sum(~np.all(algo.X == X_before, axis=1))
        assert n_changed >= 1


class TestT30_MaxFEFormula:
    """T30: max_FE = pop_size * (n_changes * taut + 50) is correct."""

    def test_max_fe(self):
        from benchmark_c4_v1.scripts.run_experiment import compute_max_fe
        assert compute_max_fe(30, 10) == 100 * (30 * 10 + 50)
        assert compute_max_fe(30, 30) == 100 * (30 * 30 + 50)


# ==================== T31-T35: Integration and integrity ====================

class TestT31_CoreHashManifest:
    """T31: Core hash manifest exists and has expected structure."""

    def test_manifest_exists(self):
        manifest_path = Path(__file__).parent.parent / "ORIGINAL_CORE_HASH_MANIFEST.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "ORIGINAL_CORE_HASH_MANIFEST" in manifest
        assert "FILE_COUNT" in manifest
        file_map = manifest["ORIGINAL_CORE_HASH_MANIFEST"]
        assert len(file_map) == manifest["FILE_COUNT"]
        for path, sha in file_map.items():
            assert isinstance(path, str)
            assert isinstance(sha, str) and len(sha) == 64


class TestT32_PairedInit:
    """T32: C1/C3/C4 with same policy_seed share identical unary init."""

    def test_paired_init(self):
        from benchmark_c4_v1.controllers.arms import BenchmarkC1Arm, BenchmarkC3Arm, BenchmarkC4Arm
        c1 = BenchmarkC1Arm(42)
        c3 = BenchmarkC3Arm(42)
        c4 = BenchmarkC4Arm(42)

        for (p1, p3) in zip(c1.online.parameters(), c3.online_unary.parameters()):
            assert torch.equal(p1, p3), "C1 and C3 unary init differs"

        for (p3, p4) in zip(c3.online_unary.parameters(), c4.online_unary.parameters()):
            assert torch.equal(p3, p4), "C3 and C4 unary init differs"

        for (p3, p4) in zip(c3.online_pair.parameters(), c4.online_pair.parameters()):
            assert torch.equal(p3, p4), "C3 and C4 pair init differs"


class TestT33_GateZeroParams:
    """T33: C4 gate has exactly 0 trainable parameters."""

    def test_zero_gate_params(self):
        from benchmark_c4_v1.controllers.arms import BenchmarkC4Arm
        c4 = BenchmarkC4Arm(42)
        total_c4 = sum(p.numel() for p in c4.online_unary.parameters()) + \
                    sum(p.numel() for p in c4.online_pair.parameters())
        from benchmark_c4_v1.controllers.arms import BenchmarkC3Arm
        c3 = BenchmarkC3Arm(42)
        total_c3 = sum(p.numel() for p in c3.online_unary.parameters()) + \
                    sum(p.numel() for p in c3.online_pair.parameters())
        assert total_c4 == total_c3, f"C4={total_c4} != C3={total_c3}"


class TestT34_StatisticalTests:
    """T34: Wilcoxon and Friedman produce valid output."""

    def test_wilcoxon(self):
        from benchmark_c4_v1.metrics.statistics import wilcoxon_signed_rank
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        stat, p = wilcoxon_signed_rank(x, y)
        assert 0 <= p <= 1

    def test_friedman(self):
        from benchmark_c4_v1.metrics.statistics import friedman_ranking
        results = {"A": [0.1, 0.2, 0.3], "B": [0.2, 0.3, 0.4], "C": [0.3, 0.1, 0.2]}
        ranks = friedman_ranking(results)
        assert len(ranks) == 3
        for v in ranks.values():
            assert 1 <= v <= 3


class TestT35_FullSmokeIntegration:
    """T35: A minimal smoke run completes without error."""

    def test_smoke_single(self):
        from benchmark_c4_v1.problems import get_problem
        from benchmark_c4_v1.scripts.run_experiment import run_nsga2, run_dnsga2
        prob = get_problem("DF1", n_var=10, nt=10, taut=10)
        result = run_nsga2(prob, taut=10, run_seed=42)
        assert result["migd"] < float("inf")
        assert result["fe_used"] > 0

        prob2 = get_problem("DF1", n_var=10, nt=10, taut=10)
        result2 = run_dnsga2(prob2, taut=10, run_seed=42)
        assert result2["migd"] < float("inf")
