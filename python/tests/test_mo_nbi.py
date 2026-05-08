"""Tests for NBI and NNC geometric scalarizations."""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo import (
    filter_nondominated,
    normal_boundary_intersection,
    normalized_normal_constraint,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _build_biobj_qp():
    m = dm.Model("biobj_qp_nbi")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    f1 = x**2 + y**2
    f2 = (x - 2) ** 2 + (y - 1) ** 2
    return m, [f1, f2]


def _on_qp_front(obj_pair, tol=5e-3):
    f1, f2 = float(obj_pair[0]), float(obj_pair[1])
    return abs(np.sqrt(max(f1, 0) / 5.0) + np.sqrt(max(f2, 0) / 5.0) - 1.0) < tol


def _build_concave_front():
    m = dm.Model("concave_nbi")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x**2 + y**2 >= 1.0)
    return m, [x, y]


class TestNBIConvex:
    def test_front_on_analytic_curve(self):
        m, objs = _build_biobj_qp()
        front = normal_boundary_intersection(m, objs, n_points=9)
        assert front.n >= 3
        for p in front.points:
            assert _on_qp_front(p.objectives), f"NBI point off front: {p.objectives}"

    def test_anchors_recovered(self):
        m, objs = _build_biobj_qp()
        front = normal_boundary_intersection(m, objs, n_points=9)
        obj = front.objectives()
        assert obj[:, 0].min() == pytest.approx(0.0, abs=1e-4)
        assert obj[:, 1].min() == pytest.approx(0.0, abs=1e-4)

    def test_returned_points_nondominated(self):
        m, objs = _build_biobj_qp()
        front = normal_boundary_intersection(m, objs, n_points=9)
        mask = filter_nondominated(front.objectives())
        assert mask.all()

    def test_uniform_spacing(self):
        """NBI is designed to give near-uniform spacing for convex fronts."""
        m, objs = _build_biobj_qp()
        front = normal_boundary_intersection(m, objs, n_points=9)
        obj = front.objectives()
        order = np.argsort(obj[:, 0])
        sorted_obj = obj[order]
        diffs = np.linalg.norm(np.diff(sorted_obj, axis=0), axis=1)
        # Coefficient of variation below 15% for this convex problem.
        assert diffs.std() / diffs.mean() < 0.15


class TestNBIConcave:
    def test_recovers_interior(self):
        m, objs = _build_concave_front()
        front = normal_boundary_intersection(m, objs, n_points=9)
        obj = front.objectives()
        interior = ((obj[:, 0] > 0.05) & (obj[:, 1] > 0.05)).sum()
        assert interior >= 2, f"NBI recovered only {interior} interior points on concave front"


class TestNNCConvex:
    def test_front_on_analytic_curve(self):
        m, objs = _build_biobj_qp()
        front = normalized_normal_constraint(m, objs, n_points=9)
        assert front.n >= 3
        for p in front.points:
            assert _on_qp_front(p.objectives, tol=5e-3)

    def test_anchors_recovered(self):
        m, objs = _build_biobj_qp()
        front = normalized_normal_constraint(m, objs, n_points=9)
        obj = front.objectives()
        assert obj[:, 0].min() == pytest.approx(0.0, abs=1e-3)
        assert obj[:, 1].min() == pytest.approx(0.0, abs=1e-3)

    def test_returned_points_nondominated(self):
        m, objs = _build_biobj_qp()
        front = normalized_normal_constraint(m, objs, n_points=9)
        mask = filter_nondominated(front.objectives())
        assert mask.all()


class TestNNCConcave:
    def test_recovers_interior(self):
        m, objs = _build_concave_front()
        front = normalized_normal_constraint(m, objs, n_points=9)
        obj = front.objectives()
        interior = ((obj[:, 0] > 0.05) & (obj[:, 1] > 0.05)).sum()
        assert interior >= 2
