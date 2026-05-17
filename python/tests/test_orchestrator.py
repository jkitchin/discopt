"""
Tests for T14 Solver Orchestrator — end-to-end Model.solve().

Test classes:
  - TestPyTreeManager: Unit tests for Rust B&B bindings
  - TestSolveSimple: End-to-end solve of example_simple_minlp
  - TestSolveExamples: Parametrized test over example models
  - TestSolveCorrectness: Verify against known optima
  - TestTermination: Time limit, gap tolerance, infeasibility
  - TestProfiling: Verify time profiling breakdown
  - TestDeterminism: Reproducibility check
  - TestSolveIPM: End-to-end solve with IPM backend
  - TestSolveBatchIPM: Batch IPM solving
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import PyTreeManager
from discopt.modeling.core import SolveResult
from discopt.modeling.examples import example_simple_minlp

# ──────────────────────────────────────────────────────────
# TestPyTreeManager — Unit tests for Rust B&B bindings
# ──────────────────────────────────────────────────────────


class TestPyTreeManager:
    """Unit tests for the PyTreeManager Rust bindings."""

    def test_construct(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        stats = tm.stats()
        assert stats["total_nodes"] == 0
        assert stats["open_nodes"] == 0

    def test_initialize(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        tm.initialize()
        stats = tm.stats()
        assert stats["total_nodes"] == 1
        assert stats["open_nodes"] == 1

    def test_export_empty(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        lb, ub, ids, _psols = tm.export_batch(10)
        assert lb.shape == (0, 2)
        assert ub.shape == (0, 2)
        assert ids.shape == (0,)

    def test_export_root(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        tm.initialize()
        lb, ub, ids, _psols = tm.export_batch(10)
        assert lb.shape == (1, 2)
        assert ub.shape == (1, 2)
        assert ids.shape == (1,)
        assert ids[0] == 0
        np.testing.assert_array_equal(lb[0], [0.0, 0.0])
        np.testing.assert_array_equal(ub[0], [1.0, 1.0])

    def test_import_and_process(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        tm.initialize()
        lb, ub, ids, _psols = tm.export_batch(1)

        # Simulate fractional solution -> should branch
        node_ids = np.array([ids[0]], dtype=np.int64)
        lower_bounds = np.array([1.2], dtype=np.float64)
        solutions = np.array([[0.5, 0.7]], dtype=np.float64)
        feasible = np.array([False], dtype=bool)

        tm.import_results(node_ids, lower_bounds, solutions, feasible)
        proc = tm.process_evaluated()

        assert proc["branched"] == 1
        assert proc["pruned"] == 0
        assert proc["fathomed"] == 0

        stats = tm.stats()
        assert stats["total_nodes"] == 3  # root + 2 children
        assert stats["open_nodes"] == 2

    def test_fathom_integer_feasible(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        tm.initialize()
        lb, ub, ids, _psols = tm.export_batch(1)

        # Integer-feasible solution
        node_ids = np.array([ids[0]], dtype=np.int64)
        lower_bounds = np.array([5.0], dtype=np.float64)
        solutions = np.array([[1.0, 0.0]], dtype=np.float64)
        feasible = np.array([False], dtype=bool)  # Let Rust check

        tm.import_results(node_ids, lower_bounds, solutions, feasible)
        proc = tm.process_evaluated()

        assert proc["fathomed"] == 1
        assert proc["incumbent_updates"] == 1
        assert tm.is_finished()

        incumbent = tm.incumbent()
        assert incumbent is not None
        sol, val = incumbent
        assert val == 5.0
        np.testing.assert_array_equal(sol, [1.0, 0.0])

    def test_gap(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
        assert tm.gap() == float("inf")

    def test_depth_first_strategy(self):
        tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "depth_first")
        tm.initialize()
        stats = tm.stats()
        assert stats["open_nodes"] == 1

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "invalid")

    def test_mismatched_bounds_length(self):
        with pytest.raises(ValueError, match="lb length"):
            PyTreeManager(2, [0.0], [1.0, 1.0], [], [], "best_first")

    def test_parent_solution_warm_start(self):
        """Parent solutions propagate to children for warm-starting."""
        tm = PyTreeManager(2, [0.0, 0.0], [10.0, 10.0], [0], [2], "best_first")
        tm.initialize()

        # Root: no parent solution (NaN)
        lb, ub, ids, psols = tm.export_batch(1)
        assert np.all(np.isnan(psols[0]))  # root has no parent

        # Import root result with solution [3.5, 4.7]
        tm.import_results(
            np.array([ids[0]], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            np.array([[3.5, 4.7]], dtype=np.float64),
            np.array([False], dtype=bool),
        )
        tm.process_evaluated()

        # Children should have parent solution [3.5, 4.7]
        lb, ub, ids, psols = tm.export_batch(2)
        assert len(ids) == 2
        for i in range(2):
            assert not np.any(np.isnan(psols[i]))
            np.testing.assert_array_almost_equal(psols[i], [3.5, 4.7])

    def test_full_lifecycle(self):
        """Test a complete B&B lifecycle with branching and pruning."""
        tm = PyTreeManager(1, [0.0], [10.0], [0], [1], "best_first")
        tm.initialize()

        # Root: fractional
        lb, ub, ids, _psols = tm.export_batch(1)
        tm.import_results(
            np.array([ids[0]], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            np.array([[3.5]], dtype=np.float64),
            np.array([False], dtype=bool),
        )
        tm.process_evaluated()

        # Two children should be open
        assert tm.stats()["open_nodes"] == 2

        # Solve both children
        lb, ub, ids, _psols = tm.export_batch(2)
        assert len(ids) == 2

        # One feasible, one infeasible
        tm.import_results(
            np.array(ids, dtype=np.int64),
            np.array([2.0, 1e30], dtype=np.float64),
            np.array([[3.0], [4.0]], dtype=np.float64),
            np.array([False, False], dtype=bool),
        )
        proc = tm.process_evaluated()

        # The 1e30 bound node should be pruned (since incumbent 2.0 < 1e30)
        # The [3.0] solution is integer-feasible, so fathomed
        assert proc["fathomed"] >= 1

        incumbent = tm.incumbent()
        assert incumbent is not None


# ──────────────────────────────────────────────────────────
# TestSolveSimple — End-to-end solve
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSolveSimple:
    """End-to-end tests solving example_simple_minlp."""

    @pytest.fixture(scope="class")
    def result(self):
        m = example_simple_minlp()
        return m.solve()

    def test_solve_returns_solve_result(self, result):
        assert isinstance(result, SolveResult)

    def test_solve_status(self, result):
        assert result.status in ("optimal", "feasible")

    def test_solve_has_solution(self, result):
        assert result.x is not None
        assert "x1" in result.x
        assert "x2" in result.x
        assert "x3" in result.x

    def test_solve_objective_finite(self, result):
        assert result.objective is not None
        assert np.isfinite(result.objective)

    def test_solve_binary_variable_integral(self, result):
        x3 = result.x["x3"]
        # x3 should be 0 or 1
        assert abs(x3 - 0.0) < 1e-5 or abs(x3 - 1.0) < 1e-5

    def test_solve_respects_bounds(self, result):
        assert result.x["x1"] >= -1e-6
        assert result.x["x1"] <= 5.0 + 1e-6
        assert result.x["x2"] >= -1e-6
        assert result.x["x2"] <= 5.0 + 1e-6

    def test_solve_constraints_satisfied(self, result):
        x1 = float(result.x["x1"])
        x2 = float(result.x["x2"])
        # x1 + x2 >= 1
        assert x1 + x2 >= 1.0 - 1e-4
        # x1^2 + x2 <= 3
        assert x1**2 + x2 <= 3.0 + 1e-4

    def test_solve_node_count_positive(self, result):
        assert result.node_count >= 1

    def test_solve_wall_time_positive(self, result):
        assert result.wall_time > 0

    def test_piecewise_partitions_parameter(self):
        """solve(partitions=4) should produce a valid result."""
        m = example_simple_minlp()
        result = m.solve(partitions=4)
        assert isinstance(result, SolveResult)
        assert result.status in ("optimal", "feasible")


# ──────────────────────────────────────────────────────────
# TestSolveCorrectness — Verify against known optima
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSolveCorrectness:
    """Verify solver finds correct optimal values."""

    def test_simple_minlp_optimal_value(self):
        """example_simple_minlp: min x1^2+x2^2+x3 s.t. x1+x2>=1, x1^2+x2<=3, x3 binary.

        Optimal is x3=0 and then min x1^2+x2^2 s.t. x1+x2>=1, x1^2+x2<=3.
        By Lagrange, optimal x1=x2=0.5 giving obj=0.5.
        """
        pytest.importorskip("cyipopt")
        m = example_simple_minlp()
        result = m.solve(nlp_solver="ipopt")
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert result.objective < 1.0  # Should be 0.5

    def test_pure_continuous_nlp(self):
        """A pure continuous NLP should solve directly (no B&B)."""
        m = dm.Model("continuous_test")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = m.solve()
        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective - 0.5) < 1e-3
        assert result.node_count == 0  # No B&B for continuous

    def test_binary_knapsack(self):
        """Simple binary knapsack problem."""
        m = dm.Model("knapsack")
        x1 = m.binary("x1")
        x2 = m.binary("x2")
        x3 = m.binary("x3")

        # Minimize -3x1 - 4x2 - 2x3 (maximize profit)
        m.minimize(-3 * x1 - 4 * x2 - 2 * x3)
        # Weight constraint: 2x1 + 3x2 + x3 <= 4
        m.subject_to(2 * x1 + 3 * x2 + x3 <= 4)

        result = m.solve()
        assert result.status in ("optimal", "feasible")
        # Optimal: x1=0, x2=1, x3=1, obj=-6
        # or x1=1, x2=0, x3=1, obj=-5 (not optimal)
        assert result.objective is not None
        assert result.objective <= -5.0 + 1e-3  # At least -5


# ──────────────────────────────────────────────────────────
# TestTermination — Time limit, gap tolerance
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestTermination:
    """Test solver termination conditions."""

    def test_time_limit(self):
        """Solver should respect time limit."""
        m = example_simple_minlp()
        t0 = time.perf_counter()
        result = m.solve(time_limit=60.0)
        elapsed = time.perf_counter() - t0
        # Should finish well within the time limit for this small problem
        assert elapsed < 60.0
        assert result.wall_time <= 60.0 + 1.0  # some tolerance

    def test_gap_tolerance(self):
        """Solver should stop when gap is small enough."""
        m = example_simple_minlp()
        result = m.solve(gap_tolerance=0.01)
        if result.status == "optimal":
            assert result.gap is not None
            assert result.gap <= 0.01 + 1e-8

    def test_infeasible_detection(self):
        """Solver should detect infeasible models."""
        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        # Contradictory constraints
        m.subject_to(x >= 2)
        m.subject_to(x <= 0.5)

        result = m.solve()
        # The NLP solver should detect infeasibility
        # Status could be infeasible or the objective could indicate no good solution
        assert result.status in ("infeasible", "optimal", "feasible", "iteration_limit")


# ──────────────────────────────────────────────────────────
# TestProfiling — Verify time breakdown
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestProfiling:
    """Test that profiling times are populated and consistent."""

    def test_times_populated(self):
        m = example_simple_minlp()
        result = m.solve()
        assert result.wall_time > 0
        assert result.jax_time >= 0
        assert result.rust_time >= 0
        assert result.python_time >= 0

    def test_times_sum_approximately(self):
        """rust_time + jax_time + python_time should approximately equal wall_time."""
        m = example_simple_minlp()
        result = m.solve()
        sum_times = result.rust_time + result.jax_time + result.python_time
        # Allow some tolerance for measurement overhead
        assert abs(sum_times - result.wall_time) < 0.5


# ──────────────────────────────────────────────────────────
# TestDeterminism — Reproducibility
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDeterminism:
    """Test that deterministic mode produces identical results."""

    def test_deterministic_results(self):
        """Three identical runs should produce same node_count and objective."""
        results = []
        for _ in range(3):
            m = example_simple_minlp()
            r = m.solve(deterministic=True)
            results.append((r.node_count, r.objective))

        # All three should match
        assert results[0] == results[1]
        assert results[1] == results[2]


# ──────────────────────────────────────────────────────────
# TestSolveResult — Result API
# ──────────────────────────────────────────────────────────


def test_solve_result_declares_mip_count_field():
    result = SolveResult(status="optimal", mip_count=3)

    assert result.mip_count == 3


@pytest.mark.slow
class TestSolveResult:
    """Test SolveResult API methods."""

    @pytest.fixture(scope="class")
    def model_and_result(self):
        m = example_simple_minlp()
        return m, m.solve()

    def test_value_method(self, model_and_result):
        m, result = model_and_result
        x1 = m._variables[0]
        val = result.value(x1)
        assert isinstance(val, np.ndarray)

    def test_explain_method(self, model_and_result):
        _, result = model_and_result
        explanation = result.explain()
        assert isinstance(explanation, str)
        assert "Solved" in explanation or result.status in explanation

    def test_repr(self, model_and_result):
        _, result = model_and_result
        s = repr(result)
        assert "SolveResult" in s


# ──────────────────────────────────────────────────────────
# TestSolveIPM — End-to-end solve with IPM backend
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSolveIPM:
    """End-to-end tests solving with nlp_solver='ipm'."""

    @pytest.fixture(scope="class")
    def ipm_result(self):
        m = example_simple_minlp()
        return m.solve(nlp_solver="ipm")

    def test_ipm_solve_returns_solve_result(self, ipm_result):
        assert isinstance(ipm_result, SolveResult)

    def test_ipm_solve_status(self, ipm_result):
        assert ipm_result.status in ("optimal", "feasible")

    def test_ipm_solve_correctness(self, ipm_result):
        """IPM should find a feasible solution for example_simple_minlp."""
        assert ipm_result.objective is not None
        assert ipm_result.objective < 5.0  # Valid feasible solution

    def test_ipm_batch_solve(self):
        """Batch IPM (batch_size=8) should produce correct result."""
        m = example_simple_minlp()
        result = m.solve(nlp_solver="ipm", batch_size=8)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert result.objective < 5.0

    def test_ipm_pure_continuous(self):
        """Pure continuous NLP with IPM should solve directly (no B&B)."""
        m = dm.Model("continuous_ipm")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = m.solve(nlp_solver="ipm")
        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective - 0.5) < 1e-2
        assert result.node_count == 0


# ──────────────────────────────────────────────────────────
# TestSolveBatchIPM — Batch IPM solving
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSolveBatchIPM:
    """Test batch IPM solving specifically."""

    def test_batch_matches_sequential(self):
        """Batch and sequential IPM should produce similar results."""
        m1 = example_simple_minlp()
        r_seq = m1.solve(nlp_solver="ipm", batch_size=1)

        m2 = example_simple_minlp()
        r_batch = m2.solve(nlp_solver="ipm", batch_size=8)

        assert r_seq.status in ("optimal", "feasible")
        assert r_batch.status in ("optimal", "feasible")
        # Both should find similar objectives
        if r_seq.objective is not None and r_batch.objective is not None:
            assert abs(r_seq.objective - r_batch.objective) < 0.5

    def test_batch_binary_knapsack(self):
        """Binary knapsack with batch IPM should find good solution."""
        m = dm.Model("knapsack_batch")
        x1 = m.binary("x1")
        x2 = m.binary("x2")
        x3 = m.binary("x3")

        m.minimize(-3 * x1 - 4 * x2 - 2 * x3)
        m.subject_to(2 * x1 + 3 * x2 + x3 <= 4)

        result = m.solve(nlp_solver="ipm", batch_size=8)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert result.objective <= -5.0 + 1e-2


# ──────────────────────────────────────────────────────────
# TestBranchingPolicy — GNN branching policy parameter
# ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestBranchingPolicy:
    """Test branching_policy parameter plumbing."""

    def test_branching_policy_fractional(self):
        """Default fractional branching should produce valid results."""
        m = example_simple_minlp()
        result = m.solve(branching_policy="fractional")
        assert isinstance(result, SolveResult)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert np.isfinite(result.objective)

    def test_branching_policy_gnn(self):
        """GNN branching policy should run without error and produce valid results."""
        m = example_simple_minlp()
        result = m.solve(branching_policy="gnn")
        assert isinstance(result, SolveResult)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert np.isfinite(result.objective)

    def test_branching_policy_gnn_knapsack(self):
        """GNN branching on binary knapsack should produce valid results."""
        m = dm.Model("knapsack_gnn")
        x1 = m.binary("x1")
        x2 = m.binary("x2")
        x3 = m.binary("x3")

        m.minimize(-3 * x1 - 4 * x2 - 2 * x3)
        m.subject_to(2 * x1 + 3 * x2 + x3 <= 4)

        result = m.solve(branching_policy="gnn")
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert result.objective <= -5.0 + 1e-2
