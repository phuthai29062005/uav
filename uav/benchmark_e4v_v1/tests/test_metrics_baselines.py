"""Tests for metrics, baselines, and runner (T26-T48).

T26-T28: IGD metric correctness
T29-T30: MIGD metric
T31-T33: EvaluationLedger FE accounting
T34-T37: NSGA-II baseline
T38-T41: D-NSGA-II baseline
T42-T48: Runner integration
"""

import numpy as np
import pytest
from benchmark_e4v_v1.src.metrics.igd import igd, migd
from benchmark_e4v_v1.src.metrics.ledger import EvaluationLedger
from benchmark_e4v_v1.src.baselines.nsga2 import (
    NSGA2Baseline, _fast_non_dominated_sort, _crowding_distance,
)
from benchmark_e4v_v1.src.baselines.dnsga2 import DNSGA2Baseline
from benchmark_e4v_v1.src.problems.registry import get_problem
from benchmark_e4v_v1.src.problems.base import T0
from benchmark_e4v_v1.src.runner import run_baseline, run_single_experiment


# ── T26-T28: IGD ──────────────────────────────────────────────────

class TestIGD:

    def test_t26_igd_perfect_match(self):
        """IGD = 0 when approx exactly equals reference."""
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert igd(ref, ref) == pytest.approx(0.0)

    def test_t27_igd_known_value(self):
        """IGD against a known analytical case."""
        ref = np.array([[0.0, 0.0]])
        approx = np.array([[3.0, 4.0]])
        assert igd(approx, ref) == pytest.approx(5.0)

    def test_t27b_igd_multiple_reference_points(self):
        """IGD with multiple reference points."""
        ref = np.array([[0.0, 0.0], [1.0, 0.0]])
        approx = np.array([[0.0, 1.0]])
        # d([0,0],[0,1])=1.0, d([1,0],[0,1])=sqrt(2)
        expected = (1.0 + np.sqrt(2.0)) / 2
        assert igd(approx, ref) == pytest.approx(expected, abs=1e-10)

    def test_t28_igd_empty_approx(self):
        """IGD = inf when approximation set is empty."""
        ref = np.array([[0.0, 1.0]])
        assert igd(np.array([]).reshape(0, 2), ref) == float("inf")


# ── T29-T30: MIGD ─────────────────────────────────────────────────

class TestMIGD:

    def test_t29_migd_basic(self):
        """MIGD is mean of IGD values."""
        values = [0.1, 0.2, 0.3]
        assert migd(values) == pytest.approx(0.2)

    def test_t30_migd_empty(self):
        """MIGD = inf for empty list."""
        assert migd([]) == float("inf")


# ── T31-T33: EvaluationLedger ─────────────────────────────────────

class TestLedger:

    def test_t31_ledger_counts_all_evaluations(self):
        """Ledger counts every evaluation call."""
        prob = get_problem("DF1")
        ledger = EvaluationLedger(prob, budget=None)
        X = np.random.rand(10, 10)
        ledger.evaluate(X)
        assert ledger.count == 10
        ledger.evaluate(X[:5])
        assert ledger.count == 15

    def test_t32_ledger_budget_enforcement(self):
        """Ledger raises when budget exceeded."""
        prob = get_problem("DF1")
        ledger = EvaluationLedger(prob, budget=20)
        X = np.random.rand(10, 10)
        ledger.evaluate(X)
        assert ledger.count == 10
        assert ledger.remaining == 10
        with pytest.raises(RuntimeError, match="budget exceeded"):
            ledger.evaluate(np.random.rand(15, 10))

    def test_t33_ledger_evaluate_returns_correct_shape(self):
        """Ledger.evaluate returns correct objective values."""
        prob = get_problem("DF1")
        ledger = EvaluationLedger(prob)
        X = np.random.rand(5, 10)
        F = ledger.evaluate(X)
        assert F.shape == (5, 2)
        assert np.all(np.isfinite(F))


# ── T34-T37: NSGA-II ─────────────────────────────────────────────

class TestNSGA2:

    def test_t34_nds_known_fronts(self):
        """NDS correctly identifies dominated and non-dominated."""
        F = np.array([
            [1.0, 3.0],  # front 0 (non-dominated)
            [2.0, 2.0],  # front 0
            [3.0, 1.0],  # front 0
            [2.0, 3.0],  # front 1 (dominated by [1,3] and [2,2])
            [3.0, 3.0],  # front 2
        ])
        fronts, rank = _fast_non_dominated_sort(F)
        assert set(fronts[0]) == {0, 1, 2}
        assert rank[0] == 0
        assert rank[1] == 0
        assert rank[2] == 0
        assert rank[3] == 1
        assert rank[4] == 2

    def test_t35_crowding_distance_boundary(self):
        """Boundary solutions get infinite crowding distance."""
        F = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        cd = _crowding_distance(F, [0, 1, 2, 3])
        assert cd[0] == float("inf")
        assert cd[3] == float("inf")
        assert cd[1] > 0
        assert cd[2] > 0

    def test_t36_nsga2_population_size_preserved(self):
        """NSGA-II maintains constant population size."""
        alg = NSGA2Baseline(pop_size=50, seed=42)
        prob = get_problem("DF1")
        X = alg.initialize(prob.xl, prob.xu, prob.n_var)
        F = prob.evaluate(X)
        alg.set_fitness(F)
        for _ in range(10):
            off_X = alg.step()
            off_F = prob.evaluate(off_X)
            alg.select(off_X, off_F)
            assert len(alg.X) == 50
            assert len(alg.F) == 50

    def test_t37_nsga2_improves_on_static(self):
        """NSGA-II reduces IGD over generations on a static problem."""
        prob = get_problem("DF1", n_var=10)
        alg = NSGA2Baseline(pop_size=100, seed=42)
        X = alg.initialize(prob.xl, prob.xu, prob.n_var)
        F = prob.evaluate(X, t=0.0)
        alg.set_fitness(F)

        pf = prob.true_pf(t=0.0, n_points=500)
        igd_initial = igd(alg.F, pf)

        for _ in range(100):
            off_X = alg.step()
            off_F = prob.evaluate(off_X, t=0.0)
            alg.select(off_X, off_F)

        igd_final = igd(alg.F, pf)
        assert igd_final < igd_initial


# ── T38-T41: D-NSGA-II ───────────────────────────────────────────

class TestDNSGA2:

    def test_t38_dnsga2_detects_change(self):
        """D-NSGA-II detects environmental change."""
        prob = get_problem("DF1")
        alg = DNSGA2Baseline(pop_size=50, seed=42)
        X = alg.initialize(prob.xl, prob.xu, prob.n_var)
        F = prob.evaluate(X, t=0.0)
        alg.set_fitness(F)

        changed = alg.detect_change(lambda x: prob.evaluate(x, t=0.0))
        assert not changed

        changed = alg.detect_change(lambda x: prob.evaluate(x, t=0.5))
        assert changed

    def test_t39_dnsga2_reinit_fraction(self):
        """Respond to change reinitializes correct fraction."""
        alg = DNSGA2Baseline(pop_size=100, seed=42, reinit_fraction=0.2)
        prob = get_problem("DF1")
        X = alg.initialize(prob.xl, prob.xu, prob.n_var)
        F = prob.evaluate(X, t=0.0)
        alg.set_fitness(F)
        X_before = alg.X.copy()
        alg.respond_to_change()
        n_changed = np.sum(np.any(alg.X != X_before, axis=1))
        assert n_changed == 20

    def test_t40_dnsga2_is_nsga2_subclass(self):
        """D-NSGA-II inherits NSGA-II evolution operators."""
        alg = DNSGA2Baseline(pop_size=50, seed=42)
        assert isinstance(alg, NSGA2Baseline)

    def test_t41_dnsga2_population_size_stable(self):
        """D-NSGA-II maintains pop size through change detection."""
        prob = get_problem("DF1")
        alg = DNSGA2Baseline(pop_size=50, seed=42)
        X = alg.initialize(prob.xl, prob.xu, prob.n_var)
        F = prob.evaluate(X, t=0.0)
        alg.set_fitness(F)
        alg.detect_change(lambda x: prob.evaluate(x, t=0.5))
        alg.respond_to_change()
        assert len(alg.X) == 50


# ── T42-T48: Runner ──────────────────────────────────────────────

class TestRunner:

    def test_t42_runner_nsga2_30_measurements(self):
        """Runner produces exactly 30 IGD measurements."""
        result = run_single_experiment("DF1", "NSGA-II", seed=42, taut=10)
        assert result["n_measurements"] == 30
        assert len(result["igd_values"]) == 30

    def test_t43_runner_dnsga2_30_measurements(self):
        """Runner produces exactly 30 IGD measurements for D-NSGA-II."""
        result = run_single_experiment("DF1", "D-NSGA-II", seed=42, taut=10)
        assert result["n_measurements"] == 30

    def test_t44_runner_fe_count_nsga2(self):
        """NSGA-II uses exactly pop_size * total_gen FE."""
        result = run_single_experiment("DF1", "NSGA-II", seed=42, taut=10)
        total_gen = T0 + 30 * 10
        expected_fe = 100 * total_gen
        assert result["fe_count"] == expected_fe

    def test_t45_runner_fe_count_dnsga2_higher(self):
        """D-NSGA-II uses more FE than NSGA-II due to detection."""
        r_nsga = run_single_experiment("DF1", "NSGA-II", seed=42, taut=10)
        r_dnsga = run_single_experiment("DF1", "D-NSGA-II", seed=42, taut=10)
        assert r_dnsga["fe_count"] > r_nsga["fe_count"]

    def test_t46_runner_migd_finite(self):
        """MIGD is finite and positive for both algorithms."""
        for alg_name in ["NSGA-II", "D-NSGA-II"]:
            r = run_single_experiment("DF1", alg_name, seed=42, taut=10)
            assert 0 < r["migd"] < float("inf")

    def test_t47_runner_taut30_30_measurements(self):
        """Runner with taut=30 still produces 30 measurements."""
        result = run_single_experiment("DF1", "NSGA-II", seed=42, taut=30)
        assert result["n_measurements"] == 30

    def test_t48_runner_igd_values_positive(self):
        """All IGD values are positive (not zero or negative)."""
        result = run_single_experiment("DF1", "NSGA-II", seed=42, taut=10)
        for v in result["igd_values"]:
            assert v > 0
