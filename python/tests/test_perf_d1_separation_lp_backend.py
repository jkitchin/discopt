"""Phase-D (perf-d1): the edge-concave separation LP and the strong-branch LP
route to the in-house warm simplex, not a cold POUNCE IPM per call.

Regression guard for the backend-routing lever (bound-neutral: only the LP
*backend* changes; the separation math / branch rule are untouched). These tests
fail on the pre-perf-d1 code (which hard-coded ``lp_pounce.solve_lp`` in the
separator and let ``prefer_pounce`` reach the strong-branch solver on the default
``nlp_solver="pounce"`` path).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_separation_lp_solver_defaults_to_simplex(monkeypatch):
    """Default (flag unset / '1') -> the in-house simplex is the separation LP."""
    import discopt.solvers.lp_pounce as lp_pounce
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt._jax.edge_concave import _separation_lp_solver

    if not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")

    monkeypatch.delenv("DISCOPT_SEPARATION_LP_SIMPLEX", raising=False)
    assert _separation_lp_solver() is lp_simplex.solve_lp

    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "1")
    assert _separation_lp_solver() is lp_simplex.solve_lp

    # off-switch restores POUNCE
    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "0")
    assert _separation_lp_solver() is lp_pounce.solve_lp


def test_edge_concave_cut_sound_via_simplex_path(monkeypatch):
    """A cut derived through the simplex-backed separator is a valid under/over
    estimator over the whole box (removes no true point)."""
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt._jax.edge_concave import (
        EdgeConcaveQuadratic,
        separate_edge_concave_quadratic,
    )

    if not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")
    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "1")

    # q(x0,x1) = -x0^2 - x1^2 + x0*x1  (edge-concave: both square coeffs < 0)
    blk = EdgeConcaveQuadratic(
        var_idxs=(0, 1),
        sq={0: -1.0, 1: -1.0},
        bilin={(0, 1): 1.0},
        lin={},
        const=0.0,
        sense="under",
    )
    lb = np.array([-2.0, -2.0])
    ub = np.array([2.0, 2.0])
    xs = np.array([0.3, -0.4])
    # q_star well below the vertex-hull underestimator value at xs -> violated,
    # so the separator must emit a cut (which we then check is a valid estimator).
    q_star = -100.0

    cut = separate_edge_concave_quadratic(blk, lb, ub, xs, q_star)
    assert cut is not None, "expected a violated edge-concave cut"
    A, B = cut

    # validity: A·x + B <= q(x) at every sampled box point (underestimator)
    rng = np.random.default_rng(0)
    pts = lb + (ub - lb) * rng.random((500, 2))
    q = -(pts[:, 0] ** 2) - pts[:, 1] ** 2 + pts[:, 0] * pts[:, 1]
    est = pts @ A + B
    assert np.all(est <= q + 1e-6), "simplex-derived edge-concave cut removes a feasible point"


def test_strong_branch_lp_routes_to_simplex_on_default(monkeypatch):
    """On the default nlp_solver='pounce' path, strong branching's LP solver is
    the in-house simplex (flag ON), not POUNCE."""
    import discopt.solver as solvermod
    import discopt.solvers.lp_pounce as lp_pounce
    import discopt.solvers.lp_simplex as lp_simplex

    if not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")

    seen = {}

    class _StubEvaluator:
        n_constraints = 0

        def evaluate_gradient(self, x):
            return np.ones_like(np.asarray(x, dtype=float))

    def _capture_get_lp_solver(prefer_pounce=False):
        seen["prefer_pounce"] = prefer_pounce
        return lp_pounce.solve_lp if prefer_pounce else lp_simplex.solve_lp

    monkeypatch.setattr("discopt.solvers.lp_backend.get_lp_solver", _capture_get_lp_solver)

    sol = np.array([0.5, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    # flag ON (default): even with prefer_pounce=True from the caller, the routing
    # collapses prefer_pounce to False so the simplex is selected.
    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "1")
    solvermod._strong_branch_lp(
        _StubEvaluator(),
        sol,
        lb,
        ub,
        np.array([0, 1]),
        parent_lb=0.0,
        prefer_pounce=True,
    )
    assert seen["prefer_pounce"] is False, "flag ON should route strong-branch off POUNCE"

    # off-switch restores the caller-selected POUNCE preference
    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "0")
    solvermod._strong_branch_lp(
        _StubEvaluator(),
        sol,
        lb,
        ub,
        np.array([0, 1]),
        parent_lb=0.0,
        prefer_pounce=True,
    )
    assert seen["prefer_pounce"] is True, "flag OFF should honor the caller's POUNCE choice"
