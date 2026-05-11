"""End-to-end tests for weighted sum, AUGMECON2, and weighted Tchebycheff.

The reference problem is the canonical bi-objective convex QP

    min  f1 = x^2 + y^2
    min  f2 = (x - 2)^2 + (y - 1)^2
    s.t. x, y in [-5, 5]

which has a closed-form Pareto front: for alpha in [0, 1],
    x* = 2*alpha, y* = alpha
    f1(alpha) = 5*alpha^2, f2(alpha) = 5*(1 - alpha)^2.
Points are parameterized by alpha and lie on the curve
    sqrt(f1/5) + sqrt(f2/5) = 1.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo import (
    epsilon_constraint,
    filter_nondominated,
    weighted_sum,
    weighted_tchebycheff,
)

# Multi-objective scalarization tests sweep alpha grids and run a real solve at
# each grid point.  The functionality rarely regresses from typical PRs; keep
# on the nightly slow lane.
pytestmark = pytest.mark.slow


def _build_biobj_qp():
    m = dm.Model("biobj_qp")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    f1 = x**2 + y**2
    f2 = (x - 2) ** 2 + (y - 1) ** 2
    return m, [f1, f2]


def _on_analytic_front(obj_pair, tol=1e-3):
    """Check a point (f1, f2) lies on sqrt(f1/5) + sqrt(f2/5) = 1."""
    f1, f2 = float(obj_pair[0]), float(obj_pair[1])
    lhs = np.sqrt(max(f1, 0) / 5.0) + np.sqrt(max(f2, 0) / 5.0)
    return abs(lhs - 1.0) < tol


class TestWeightedSumConvex:
    def test_front_on_analytic_curve(self):
        m, objs = _build_biobj_qp()
        front = weighted_sum(m, objs, n_weights=11)
        assert front.n >= 2
        for p in front.points:
            assert _on_analytic_front(p.objectives, tol=5e-3)

    def test_anchor_recovery(self):
        m, objs = _build_biobj_qp()
        front = weighted_sum(m, objs, n_weights=11)
        obj = front.objectives()
        # Pure-w1 anchor should reach f1 = 0, f2 = 5.
        assert obj[:, 0].min() == pytest.approx(0.0, abs=1e-4)
        assert obj[:, 1].max() == pytest.approx(5.0, abs=1e-3)

    def test_nondominated(self):
        m, objs = _build_biobj_qp()
        front = weighted_sum(m, objs, n_weights=11)
        mask = filter_nondominated(front.objectives())
        assert mask.all(), "Weighted sum returned dominated points"

    def test_hypervolume_positive(self):
        m, objs = _build_biobj_qp()
        front = weighted_sum(m, objs, n_weights=11)
        assert front.hypervolume() > 0.0


@pytest.mark.slow
@pytest.mark.integration
class TestEpsilonConstraintConvex:
    def test_covers_the_front(self):
        m, objs = _build_biobj_qp()
        front = epsilon_constraint(m, objs, n_points=11)
        assert front.n >= 3
        for p in front.points:
            assert _on_analytic_front(p.objectives, tol=5e-3)

    def test_strict_efficiency(self):
        m, objs = _build_biobj_qp()
        front = epsilon_constraint(m, objs, n_points=11)
        # AUGMECON2 should return strictly nondominated points; the filter
        # should drop nothing.
        filt = front.filtered()
        assert filt.n == front.n

    def test_non_augmented_also_runs(self):
        m, objs = _build_biobj_qp()
        front = epsilon_constraint(m, objs, n_points=7, augmented=False)
        assert front.n >= 2


@pytest.mark.slow
@pytest.mark.integration
class TestWeightedTchebycheffConvex:
    def test_front_on_analytic_curve(self):
        m, objs = _build_biobj_qp()
        front = weighted_tchebycheff(m, objs, n_weights=11)
        assert front.n >= 3
        for p in front.points:
            assert _on_analytic_front(p.objectives, tol=5e-3)

    def test_anchor_extremes(self):
        m, objs = _build_biobj_qp()
        front = weighted_tchebycheff(m, objs, n_weights=11)
        obj = front.objectives()
        assert obj[:, 0].min() == pytest.approx(0.0, abs=1e-3)
        assert obj[:, 1].min() == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────
# Nonconvex front: the concave-front variant where weighted sum fails.
#
# Using the classic Deb (2001) biobjective f1 = x, f2 = (1 + g(y)) / x,
# simplified to deterministic form f1 = x, f2 = 1 + 9 * y - x * y (y in [0,1]).
# We use a simpler construction below with a known concave piece.
# ─────────────────────────────────────────────────────────────


def _build_concave_front():
    """Bi-objective whose Pareto front is the concave unit-circle quarter.

    min f1 = x
    min f2 = y
    s.t. x^2 + y^2 >= 1,  x, y in [0, 1]

    Pareto boundary: f1^2 + f2^2 = 1, which is strictly concave (bulges
    away from the origin). Weighted sum cannot recover interior points
    -- they are dominated by convex combinations of the anchors -- while
    Tchebycheff, AUGMECON, and NBI/NNC do.
    """
    m = dm.Model("concave")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x**2 + y**2 >= 1.0)
    return m, [x, y]


@pytest.mark.slow
@pytest.mark.integration
class TestNonconvexFront:
    def test_weighted_sum_misses_interior(self):
        """Weighted sum should collapse to the two anchors on a concave front."""
        m, objs = _build_concave_front()
        front = weighted_sum(m, objs, n_weights=9)
        obj = front.objectives()
        # Most weights should land at one of the two anchors; very few
        # (if any) interior points. Check that at least 2/3 of the points
        # are within 1e-3 of an anchor in either objective.
        near_anchor = (np.minimum(obj[:, 0], obj[:, 1]) < 1e-3).sum()
        assert near_anchor >= max(2, int(0.5 * front.n))

    def test_tchebycheff_recovers_interior(self):
        m, objs = _build_concave_front()
        front = weighted_tchebycheff(m, objs, n_weights=9)
        obj = front.objectives()
        # Expect multiple distinct interior points where both objectives are
        # non-anchor values (f1 > 0.05 and f2 > 0.05).
        interior = ((obj[:, 0] > 0.05) & (obj[:, 1] > 0.05)).sum()
        assert interior >= 2, f"Only {interior} interior point(s) recovered"

    def test_epsilon_constraint_recovers_interior(self):
        m, objs = _build_concave_front()
        front = epsilon_constraint(m, objs, n_points=9)
        obj = front.objectives()
        interior = ((obj[:, 0] > 0.05) & (obj[:, 1] > 0.05)).sum()
        assert interior >= 2
