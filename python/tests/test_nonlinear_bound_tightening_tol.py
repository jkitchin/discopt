"""Feasibility-tolerance behavior of nonlinear bound-tightening infeasibility guards.

Regression for issue #27a (Linux-only manifestation). The GDP hull perspective
reformulation is an O(eps) approximation at the integer faces, so constraint
bodies carry an eps-scale residual. Several bound-tightening rules previously
declared *hard* infeasibility from a residual as small as 1e-12 (a numeric-noise
floor), which on some platforms tipped a feasible problem into a spurious prune
and certified a wrong optimum. The guards now declare infeasibility only when the
constraint is violated by more than the feasibility tolerance; sub-tolerance
violations are deferred to the node NLP.

These tests exercise the guard math directly, so they are deterministic across
platforms (unlike the end-to-end solve, which only tripped on Linux numerics).
"""

from __future__ import annotations

from discopt._jax.nonlinear_bound_tightening import (
    _EMPTY_INTERVAL_FEAS_TOL,
    _tighten_univariate_quadratic_interval,
)

TOL = _EMPTY_INTERVAL_FEAS_TOL


class TestQuadraticIntervalTolerance:
    def test_subtolerance_vertex_violation_is_not_infeasible(self):
        # x**2 <= -r with 0 < r < TOL: the true min (0) exceeds rhs by r, a
        # sub-tolerance violation. Must NOT prune; returns the degenerate vertex.
        r = 0.5 * TOL
        interval = _tighten_univariate_quadratic_interval(1.0, 0.0, -r, -3.0, 3.0)
        assert interval is not None
        lo, hi = interval
        # Vertex is x = 0; clamped discriminant gives a (near-)degenerate box there.
        assert abs(lo) <= 1e-6
        assert abs(hi) <= 1e-6

    def test_supertolerance_vertex_violation_is_infeasible(self):
        # Violation well beyond tolerance is still a genuine infeasibility.
        interval = _tighten_univariate_quadratic_interval(1.0, 0.0, -1e-3, -3.0, 3.0)
        assert interval is None

    def test_feasible_quadratic_unchanged(self):
        # x**2 <= 4 over [-3, 3] tightens to [-2, 2].
        interval = _tighten_univariate_quadratic_interval(1.0, 0.0, 4.0, -3.0, 3.0)
        assert interval is not None
        lo, hi = interval
        assert lo == -2.0
        assert hi == 2.0

    def test_constant_constraint_subtolerance_is_feasible(self):
        # Degenerate "0 <= rhs" with rhs a hair negative (within tol): not pruned.
        interval = _tighten_univariate_quadratic_interval(0.0, 0.0, -0.5 * TOL, -3.0, 3.0)
        assert interval == (-3.0, 3.0)

    def test_constant_constraint_supertolerance_is_infeasible(self):
        interval = _tighten_univariate_quadratic_interval(0.0, 0.0, -1e-3, -3.0, 3.0)
        assert interval is None
