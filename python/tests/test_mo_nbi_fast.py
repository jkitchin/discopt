"""Fast NBI / NNC scalarization tests for the PR battery (#87).

A trimmed counterpart of ``test_mo_nbi.py`` (which is slow/integration
marked and excluded from the PR-time suites): one convex bi-objective QP
with a closed-form Pareto front, 5 NBI/NNC points, seconds total. Verifies
the geometric contract — anchor recovery, points on the analytic front,
non-dominance — not just execution.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo import (
    filter_nondominated,
    normal_boundary_intersection,
    normalized_normal_constraint,
)

pytestmark = pytest.mark.smoke


def _build_biobj_qp():
    m = dm.Model("biobj_qp_fast")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    f1 = x**2 + y**2
    f2 = (x - 2) ** 2 + (y - 1) ** 2
    return m, [f1, f2]


def _on_qp_front(obj_pair, tol=5e-3):
    # Pareto front of the two shifted paraboloids: sqrt(f1/5) + sqrt(f2/5) = 1.
    f1, f2 = float(obj_pair[0]), float(obj_pair[1])
    return abs(np.sqrt(max(f1, 0) / 5.0) + np.sqrt(max(f2, 0) / 5.0) - 1.0) < tol


def test_nbi_fast_anchor_recovery_and_nondominance():
    m, objs = _build_biobj_qp()
    front = normal_boundary_intersection(m, objs, n_points=5)
    assert front.n >= 3
    obj = front.objectives()
    # Anchors: each objective reaches its individual minimum (0).
    assert obj[:, 0].min() == pytest.approx(0.0, abs=1e-4)
    assert obj[:, 1].min() == pytest.approx(0.0, abs=1e-4)
    # None of the returned points dominates another.
    assert filter_nondominated(obj).all()


@pytest.mark.xfail(
    reason="#742: NBI interior subproblems return their unmoved CHIM starting "
    "points (dominated, off the analytic front). Probe flips when fixed.",
    strict=False,
)
def test_nbi_fast_interior_points_on_front():
    m, objs = _build_biobj_qp()
    front = normal_boundary_intersection(m, objs, n_points=5)
    for p in front.points:
        assert _on_qp_front(p.objectives), f"NBI point off front: {p.objectives}"


def test_nnc_fast_front_geometry():
    m, objs = _build_biobj_qp()
    front = normalized_normal_constraint(m, objs, n_points=5)
    assert front.n >= 3
    obj = front.objectives()
    assert obj[:, 0].min() == pytest.approx(0.0, abs=1e-4)
    assert obj[:, 1].min() == pytest.approx(0.0, abs=1e-4)
    for p in front.points:
        assert _on_qp_front(p.objectives), f"NNC point off front: {p.objectives}"
    assert filter_nondominated(obj).all()


def test_nbi_rejects_single_objective():
    m, objs = _build_biobj_qp()
    with pytest.raises((ValueError, TypeError)):
        normal_boundary_intersection(m, objs[:1], n_points=5)


def test_filter_nondominated_basic():
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
    mask = filter_nondominated(pts)
    assert mask.tolist() == [True, False, True]  # (2,2) dominated by (1,1)
