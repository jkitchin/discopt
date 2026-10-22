"""Tests for the ``mccormick_bounds`` option of the B&B loop.

The ``"nlp"`` mode -- an NLP over the compositional McCormick objective
relaxation -- was retired in #1873 together with the relaxation compiler behind
it (docs/dev/nlp-mccormick-route-retirement.md); its unit tests went with it.
What remains pins the refusals and the modes that exist.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.modeling.core import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_nonconvex_model():
    """min x*y  s.t. x in [1,4], y in [1,4], x+y >= 3, x integer."""
    m = Model()
    x = m.integer("x", lb=1, ub=4)
    y = m.continuous("y", lb=1.0, ub=4.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 3.0)
    return m


def _simple_minimize_model():
    """min x^2 + y^2  s.t. x in [0,4], y in [0,4], x integer."""
    m = Model()
    x = m.integer("x", lb=0, ub=4)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(x**2 + y**2)
    return m


def _convex_quadratic_model():
    """min (x-1)^2 + (y-2)^2  s.t. x in [0,3], y in [0,3], x integer."""
    m = Model()
    x = m.integer("x", lb=0, ub=3)
    y = m.continuous("y", lb=0.0, ub=3.0)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    return m


# ===========================================================================
# C-18: the removed "midpoint" mode returned a non-bound
# ===========================================================================


class TestC18MidpointNotABound:
    """Correctness issue C-18.

    ``mccormick_bounds="midpoint"`` returned the convex underestimator's VALUE
    at the box midpoint, ``u(mid)``, and fed it into the node lower bound. But
    ``u(mid) <= f(mid)`` does NOT imply ``u(mid) <= min_box f``: the value can sit
    ABOVE the true box minimum, so it is not a valid lower bound and could fathom
    the node holding the true optimum -> false "optimal". The mode is removed and
    now rejected loudly.
    """

    def test_midpoint_mode_is_rejected_loudly(self):
        """Selecting the removed mode raises ValueError (never returns a non-bound)."""
        import pytest

        model = _simple_minimize_model()
        with pytest.raises(ValueError, match="C-18"):
            model.solve(mccormick_bounds="midpoint", max_nodes=1000)

    def test_unknown_mccormick_bounds_rejected(self):
        """A typo/unknown value is rejected rather than silently ignored."""
        import pytest

        model = _simple_minimize_model()
        with pytest.raises(ValueError, match="Unknown mccormick_bounds"):
            model.solve(mccormick_bounds="bogus", max_nodes=1000)


# ===========================================================================
# The retired "nlp" mode
# ===========================================================================


class TestNLPModeRetired:
    def test_nlp_mode_is_refused(self):
        with pytest.raises(ValueError, match="mccormick_bounds='nlp'.*has been removed"):
            _simple_minimize_model().solve(mccormick_bounds="nlp", max_nodes=1000)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Integration tests for McCormick bounds in solver."""

    def test_nonconvex_lp_route(self):
        """A nonconvex model with an explicit LP-form McCormick bound."""
        model = _simple_nonconvex_model()
        result = model.solve(mccormick_bounds="lp", max_nodes=500)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_auto_activates_for_dag_models(self):
        """'auto' mode should activate a valid bound path for DAG models."""
        model = _simple_minimize_model()
        result = model.solve(mccormick_bounds="auto", max_nodes=100)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_none_disables(self):
        """'none' mode should disable McCormick bounds."""
        model = _simple_minimize_model()
        result = model.solve(mccormick_bounds="none", max_nodes=100)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_global_optimality_with_bounds(self):
        """A convex model certifies through its NLP relaxation under ``"auto"``."""
        model = _convex_quadratic_model()
        result = model.solve(mccormick_bounds="auto", max_nodes=500)
        assert result.status in ("optimal", "feasible")
        if result.status == "optimal":
            # (x-1)^2 + (y-2)^2, x integer: optimal x=1, y=2, obj=0
            assert result.objective is not None
            np.testing.assert_allclose(result.objective, 0.0, atol=1e-3)
