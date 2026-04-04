"""Tests for nonlinear Branch & Bound (NLP-BB) solver mode."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_convex_minlp():
    """Convex MINLP: facility activation with quadratic operating costs.

    min  sum_i (c_i * y_i + x_i^2 + x_i)
    s.t. sum_i x_i >= D
         x_i <= M * y_i   for all i
         x_i >= 0, y_i in {0, 1}
    """
    k = 4
    fixed_costs = np.array([5.0, 8.0, 6.0, 7.0])
    M = 10.0
    D = 8.0

    m = dm.Model("convex_facility")
    y = m.binary("activate", shape=(k,))
    x = m.continuous("capacity", shape=(k,), lb=0, ub=M)

    # Build objective as a scalar sum to avoid vectorization issues
    obj = 0
    for i in range(k):
        obj = obj + fixed_costs[i] * y[i] + x[i] ** 2 + x[i]
    m.minimize(obj)

    # Demand: sum of capacities >= D
    total_cap = 0
    for i in range(k):
        total_cap = total_cap + x[i]
    m.subject_to(total_cap >= D)

    # Linking constraints
    for i in range(k):
        m.subject_to(x[i] <= M * y[i])

    return m


def _build_simple_convex_minlp():
    """Tiny convex MINLP for quick testing.

    min  x^2 + 3*y
    s.t. x + y >= 1
         x in [0, 5], y in {0, 1}

    Optimal: y=0 is infeasible (x >= 1, obj = 1 + 0 = 1);
             y=1 gives x + 1 >= 1 so x=0, obj = 0 + 3 = 3.
    Actually y=0 => x >= 1 => obj = 1; y=1 => x >= 0 => obj = 3.
    So optimal is y=0, x=1, obj=1.
    """
    m = dm.Model("simple_convex")
    x = m.continuous("x", lb=0, ub=5)
    y = m.binary("y")

    m.minimize(x**2 + 3 * y)
    m.subject_to(x + y >= 1)

    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNlpBbConvex:
    """NLP-BB on convex MINLPs should find optimal with certified gap."""

    def test_simple_convex_minlp(self):
        m = _build_simple_convex_minlp()
        result = m.solve(nlp_bb=True)

        assert result.status == "optimal"
        assert result.nlp_bb is True
        assert result.gap_certified is True
        assert result.objective is not None
        assert result.objective == pytest.approx(1.0, abs=1e-4)
        assert result.x["y"] == pytest.approx(0.0, abs=1e-5)
        assert result.x["x"] == pytest.approx(1.0, abs=1e-4)

    def test_facility_activation(self):
        m = _build_convex_minlp()
        result = m.solve(nlp_bb=True)

        assert result.status == "optimal"
        assert result.nlp_bb is True
        assert result.gap_certified is True
        assert result.gap is not None
        assert result.gap <= 1e-4

    def test_auto_selects_for_convex(self):
        """When nlp_bb=None (default), convex nonlinear MINLPs auto-select NLP-BB.

        Uses exp() to ensure the problem is classified as nonlinear (not MIQP),
        so it isn't dispatched to the specialized QP solver before reaching the
        NLP-BB auto-select logic.
        """
        m = dm.Model("convex_exp")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")

        m.minimize(dm.exp(x) + 3 * y)
        m.subject_to(x + y >= 1)

        result = m.solve()

        assert result.nlp_bb is True
        assert result.status == "optimal"

    def test_matches_spatial_bb(self):
        """NLP-BB and spatial B&B should find the same optimum."""
        m = _build_convex_minlp()

        result_nlpbb = m.solve(nlp_bb=True)
        result_spatial = m.solve(nlp_bb=False)

        assert result_nlpbb.status == "optimal"
        assert result_spatial.status == "optimal"
        assert result_nlpbb.objective == pytest.approx(result_spatial.objective, rel=1e-3)

    def test_batch_size_one(self):
        """NLP-BB should work with batch_size=1 (serial fallback)."""
        m = _build_simple_convex_minlp()
        result = m.solve(nlp_bb=True, batch_size=1)

        assert result.status == "optimal"
        assert result.nlp_bb is True
        assert result.objective == pytest.approx(1.0, abs=1e-4)


class TestNlpBbNonconvex:
    """NLP-BB on nonconvex MINLPs runs in heuristic mode."""

    def test_nonconvex_heuristic_mode(self, caplog):
        """Nonconvex + nlp_bb=True should warn and set gap_certified=False."""
        m = dm.Model("nonconvex")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.binary("y")

        m.minimize(dm.sin(x) + 2 * y)
        m.subject_to(x + y >= 0.5)

        import logging

        with caplog.at_level(logging.WARNING, logger="discopt"):
            result = m.solve(nlp_bb=True)

        assert result.nlp_bb is True
        assert result.gap_certified is False
        assert any("heuristic" in r.message for r in caplog.records)

    def test_nonconvex_default_uses_spatial(self):
        """Nonconvex + nlp_bb=None (default) should use spatial B&B."""
        m = dm.Model("nonconvex_default")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.binary("y")

        m.minimize(dm.sin(x) + 2 * y)
        m.subject_to(x + y >= 0.5)

        result = m.solve()

        # Should NOT auto-select NLP-BB for nonconvex
        assert result.nlp_bb is False


class TestNlpBbOverride:
    """Test manual override behavior of nlp_bb parameter."""

    def test_force_spatial_on_convex(self):
        """nlp_bb=False should force spatial B&B even for convex models."""
        m = _build_simple_convex_minlp()
        result = m.solve(nlp_bb=False)

        assert result.nlp_bb is False
        assert result.status == "optimal"
        assert result.objective == pytest.approx(1.0, abs=1e-4)

    def test_solve_result_fields(self):
        """SolveResult should have all expected NLP-BB fields."""
        m = _build_simple_convex_minlp()
        result = m.solve(nlp_bb=True)

        assert hasattr(result, "nlp_bb")
        assert hasattr(result, "gap_certified")
        assert result.wall_time > 0
        assert result.node_count >= 0
