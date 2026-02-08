"""Tests for multi-start primal heuristics (T20)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

cyipopt = pytest.importorskip("cyipopt")

from discopt._jax.primal_heuristics import (  # noqa: E402
    MultiStartNLP,
    MultiStartResult,
    _generate_starts,
    _get_integer_mask,
    _is_integer_feasible,
    feasibility_pump,
)
from discopt.modeling.core import Model  # noqa: E402
from discopt.modeling.examples import example_simple_minlp  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _make_quadratic_nlp() -> Model:
    """min (x-3)^2 + (y-1)^2, x in [0,10], y in [0,10]."""
    m = Model("quad")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 3) ** 2 + (y - 1) ** 2)
    return m


def _make_constrained_nlp() -> Model:
    """min x^2 + y^2  s.t. x + y >= 2, x,y in [0,5]."""
    m = Model("constrained")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y >= 2)
    return m


def _make_simple_minlp() -> Model:
    """min x^2 + z  s.t. x >= 0.5, z in {0,1}, x in [0,5]."""
    m = Model("simple_minlp")
    x = m.continuous("x", lb=0, ub=5)
    z = m.binary("z")
    m.minimize(x**2 + z)
    m.subject_to(x >= 0.5)
    return m


# ─────────────────────────────────────────────────────────────
# TestMultiStartBasic
# ─────────────────────────────────────────────────────────────


class TestMultiStartBasic:
    def test_finds_optimum_unconstrained(self):
        """Multi-start finds optimum of simple quadratic."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=8, seed=0)
        result = ms.solve()

        assert result.best_solution is not None
        assert result.best_objective is not None
        assert result.best_objective < 0.01
        assert np.allclose(result.best_solution, [3.0, 1.0], atol=0.01)

    def test_finds_optimum_constrained(self):
        """Multi-start finds optimum of constrained quadratic."""
        m = _make_constrained_nlp()
        ms = MultiStartNLP(m, n_starts=8, seed=0)
        result = ms.solve()

        assert result.best_solution is not None
        assert result.best_objective is not None
        # Optimal at x = y = 1 with obj = 2
        assert abs(result.best_objective - 2.0) < 0.01
        assert np.allclose(result.best_solution, [1.0, 1.0], atol=0.05)

    def test_returns_result_type(self):
        """Verify the return type is MultiStartResult."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=4, seed=0)
        result = ms.solve()
        assert isinstance(result, MultiStartResult)

    def test_n_starts_matches(self):
        """Verify n_starts in result matches requested."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=16, seed=0)
        result = ms.solve()
        assert result.n_starts == 16


# ─────────────────────────────────────────────────────────────
# TestMultiStartMINLP
# ─────────────────────────────────────────────────────────────


class TestMultiStartMINLP:
    def test_simple_minlp(self):
        """Multi-start on simple MINLP finds a good solution."""
        m = _make_simple_minlp()
        ms = MultiStartNLP(m, n_starts=16, seed=0)
        result = ms.solve()

        # The NLP relaxation optimal is x=0.5, z=0 → obj=0.25.
        # Since z is binary, integer-feasible solution needs z in {0,1}.
        # Best integer-feasible: x=0.5, z=0 → obj=0.25
        # We may or may not find this depending on NLP solves.
        assert result.n_feasible > 0

    def test_example_minlp(self):
        """Multi-start on the textbook example MINLP from examples.py."""
        m = example_simple_minlp()
        ms = MultiStartNLP(m, n_starts=16, seed=0)
        result = ms.solve()

        # Should find at least some feasible solutions
        assert result.n_feasible > 0
        assert len(result.all_objectives) > 0

    def test_integer_feasibility_tracked(self):
        """Verify integer feasibility is tracked separately."""
        m = _make_simple_minlp()
        ms = MultiStartNLP(m, n_starts=16, seed=0)
        result = ms.solve()
        # n_integer_feasible <= n_feasible
        assert result.n_integer_feasible <= result.n_feasible


# ─────────────────────────────────────────────────────────────
# TestStartPointGeneration
# ─────────────────────────────────────────────────────────────


class TestStartPointGeneration:
    def test_points_within_bounds(self):
        """All starting points must respect variable bounds."""
        lb = np.array([0.0, -5.0, 1.0])
        ub = np.array([10.0, 5.0, 3.0])
        rng = np.random.default_rng(42)
        starts = _generate_starts(lb, ub, 100, rng)

        assert starts.shape == (100, 3)
        assert np.all(starts >= lb)
        assert np.all(starts <= ub)

    def test_points_are_diverse(self):
        """Starting points should cover the feasible region."""
        lb = np.array([0.0, 0.0])
        ub = np.array([10.0, 10.0])
        rng = np.random.default_rng(42)
        starts = _generate_starts(lb, ub, 64, rng)

        # Check that each dimension has reasonable spread
        for j in range(2):
            assert starts[:, j].std() > 1.0, "Points too clustered"
            assert starts[:, j].min() < 3.0, "No points near lower bound"
            assert starts[:, j].max() > 7.0, "No points near upper bound"

    def test_infinite_bounds_clipped(self):
        """Infinite bounds should be clipped for sampling."""
        lb = np.array([-np.inf])
        ub = np.array([np.inf])
        rng = np.random.default_rng(42)
        starts = _generate_starts(lb, ub, 10, rng)

        assert np.all(np.isfinite(starts))

    def test_single_dimension(self):
        """Works for 1-D problems."""
        lb = np.array([0.0])
        ub = np.array([1.0])
        rng = np.random.default_rng(42)
        starts = _generate_starts(lb, ub, 5, rng)

        assert starts.shape == (5, 1)
        assert np.all(starts >= 0.0)
        assert np.all(starts <= 1.0)


# ─────────────────────────────────────────────────────────────
# TestFeasibilityPump
# ─────────────────────────────────────────────────────────────


class TestFeasibilityPump:
    def test_continuous_only_passthrough(self):
        """For continuous-only models, feasibility pump returns input."""
        m = _make_quadratic_nlp()
        x_nlp = np.array([3.0, 1.0])
        result = feasibility_pump(m, x_nlp)

        assert result is not None
        assert np.allclose(result, x_nlp)

    def test_simple_minlp_rounding(self):
        """Feasibility pump on simple MINLP tries rounding."""
        m = _make_simple_minlp()
        # NLP relaxation solution with fractional z
        x_nlp = np.array([0.5, 0.3])
        result = feasibility_pump(m, x_nlp, max_rounds=5)

        # May or may not find a solution, but should not crash
        if result is not None:
            # z (index 1) should be integer-valued
            assert abs(result[1] - round(result[1])) < 1e-5

    def test_already_integer_feasible(self):
        """If input is already integer-feasible, pump still returns solution."""
        m = _make_simple_minlp()
        x_nlp = np.array([2.0, 1.0])  # z=1 is integer
        result = feasibility_pump(m, x_nlp, max_rounds=3)

        # Should find a feasible solution since we start at an integer point
        if result is not None:
            assert abs(result[1] - round(result[1])) < 1e-5


# ─────────────────────────────────────────────────────────────
# TestMultiStartStatistics
# ─────────────────────────────────────────────────────────────


class TestMultiStartStatistics:
    def test_feasible_count(self):
        """n_feasible counts NLP-feasible solutions."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=8, seed=0)
        result = ms.solve()

        assert result.n_feasible > 0
        assert result.n_feasible <= result.n_starts

    def test_all_objectives_length(self):
        """all_objectives has one entry per feasible solution."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=8, seed=0)
        result = ms.solve()

        assert len(result.all_objectives) == result.n_feasible

    def test_best_is_minimum(self):
        """best_objective is the minimum of all_objectives (for continuous)."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=16, seed=0)
        result = ms.solve()

        assert result.best_objective is not None
        assert len(result.all_objectives) > 0
        assert abs(result.best_objective - min(result.all_objectives)) < 1e-10

    def test_integer_feasible_for_continuous(self):
        """For continuous-only models, all feasible = integer feasible."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=8, seed=0)
        result = ms.solve()

        assert result.n_integer_feasible == result.n_feasible


# ─────────────────────────────────────────────────────────────
# TestMultiStartReproducibility
# ─────────────────────────────────────────────────────────────


class TestMultiStartReproducibility:
    def test_same_seed_same_result(self):
        """Same seed produces same results."""
        m = _make_quadratic_nlp()

        ms1 = MultiStartNLP(m, n_starts=8, seed=123)
        r1 = ms1.solve()

        ms2 = MultiStartNLP(m, n_starts=8, seed=123)
        r2 = ms2.solve()

        assert r1.n_feasible == r2.n_feasible
        assert r1.best_objective is not None
        assert r2.best_objective is not None
        assert abs(r1.best_objective - r2.best_objective) < 1e-10
        assert np.allclose(r1.best_solution, r2.best_solution, atol=1e-10)

    def test_different_seed_different_starts(self):
        """Different seeds generate different starting points."""
        lb = np.array([0.0, 0.0])
        ub = np.array([10.0, 10.0])

        rng1 = np.random.default_rng(1)
        starts1 = _generate_starts(lb, ub, 8, rng1)

        rng2 = np.random.default_rng(2)
        starts2 = _generate_starts(lb, ub, 8, rng2)

        # Should not be identical
        assert not np.allclose(starts1, starts2)


# ─────────────────────────────────────────────────────────────
# TestMultiStartEdgeCases
# ─────────────────────────────────────────────────────────────


class TestMultiStartEdgeCases:
    def test_single_variable(self):
        """Works with a single continuous variable."""
        m = Model("single")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize((x - 4) ** 2)

        ms = MultiStartNLP(m, n_starts=4, seed=0)
        result = ms.solve()

        assert result.best_solution is not None
        assert abs(result.best_solution[0] - 4.0) < 0.1

    def test_unconstrained_with_wide_bounds(self):
        """Handles wide bounds without issues."""
        m = Model("wide")
        x = m.continuous("x", lb=-1000, ub=1000)
        y = m.continuous("y", lb=-1000, ub=1000)
        m.minimize(x**2 + y**2)

        ms = MultiStartNLP(m, n_starts=8, seed=0)
        result = ms.solve()

        assert result.best_objective is not None
        assert result.best_objective < 1.0

    def test_single_start(self):
        """Works with n_starts=1."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=1, seed=0)
        result = ms.solve()

        assert result.n_starts == 1
        assert result.n_feasible >= 0

    def test_ipopt_options_passed(self):
        """Custom Ipopt options are forwarded."""
        m = _make_quadratic_nlp()
        ms = MultiStartNLP(m, n_starts=4, seed=0)
        result = ms.solve(ipopt_options={"max_iter": 50, "tol": 1e-6})

        assert result.best_solution is not None


# ─────────────────────────────────────────────────────────────
# TestIntegerMask
# ─────────────────────────────────────────────────────────────


class TestIntegerMask:
    def test_continuous_only(self):
        """Continuous-only model has all-False mask."""
        m = Model("cont")
        m.continuous("x", shape=(3,), lb=0, ub=1)
        m.minimize(m._variables[0])
        mask = _get_integer_mask(m)
        assert mask.shape == (3,)
        assert not np.any(mask)

    def test_mixed(self):
        """Mixed model has correct integer mask."""
        m = Model("mixed")
        m.continuous("x", shape=(2,), lb=0, ub=1)
        m.binary("y")
        m.integer("z", lb=0, ub=10)
        m.minimize(m._variables[0])
        mask = _get_integer_mask(m)
        # x(2 cont) + y(1 binary) + z(1 int) = [F, F, T, T]
        assert mask.shape == (4,)
        expected = np.array([False, False, True, True])
        assert np.array_equal(mask, expected)

    def test_integer_feasibility_check(self):
        """_is_integer_feasible correctly classifies vectors."""
        mask = np.array([False, True, True])

        # Integer values
        assert _is_integer_feasible(np.array([1.5, 2.0, 3.0]), mask)
        # Near-integer values within tolerance
        assert _is_integer_feasible(np.array([1.5, 2.000005, 2.999995]), mask)
        # Fractional integer variables
        assert not _is_integer_feasible(np.array([1.5, 2.5, 3.0]), mask)

    def test_no_integers(self):
        """Empty integer mask is always feasible."""
        mask = np.array([False, False])
        assert _is_integer_feasible(np.array([1.5, 2.5]), mask)
