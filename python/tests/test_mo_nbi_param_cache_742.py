"""Regression tests for issue #742 — NBI interior subproblems returned
dominated (unmoved) CHIM points.

Root cause: the box-independent uniform-relaxation analysis cache
(``_uniform_relax_analysis``) is pinned on the model and keyed by a structural
token — ``(len(_variables), len(_constraints), id(_objective))`` — that does
**not** account for parameter *values*. NBI solves one max-``t`` subproblem per
CHIM weight, changing the CHIM-target :class:`Parameter` values between solves
while the model structure is fixed. The stale relaxation (embedding the previous
weight's target) produced an incorrect dual bound on the spatial B&B path that
fathomed the true Pareto point, so every interior subproblem terminated at the
dominated ``t = 0`` CHIM start. ``solve_model`` now resets the analysis cache at
the start of every solve, in lockstep with the convexity-classification reset.

These tests are intentionally *fast* and unmarked-``slow`` so they run in the
PR battery — the pre-existing on-front assertion in ``test_mo_nbi.py`` is
``slow``/``integration`` marked, so no CI lane exercised it and the regression
went unnoticed.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo import filter_nondominated, normal_boundary_intersection

pytestmark = [pytest.mark.pr_correctness]


def _on_qp_front(obj_pair, tol=5e-3):
    """Analytic front residual of the bi-objective QP used below.

    Minimizing ``f1 = x**2 + y**2`` and ``f2 = (x-2)**2 + (y-1)**2`` over the box
    has Pareto set on the segment between the two minimizers; in objective space
    the front satisfies ``sqrt(f1/5) + sqrt(f2/5) = 1``.
    """
    f1, f2 = float(obj_pair[0]), float(obj_pair[1])
    return abs(np.sqrt(max(f1, 0.0) / 5.0) + np.sqrt(max(f2, 0.0) / 5.0) - 1.0) < tol


def _build_biobj_qp():
    m = dm.Model("biobj_qp_742")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    f1 = x**2 + y**2
    f2 = (x - 2) ** 2 + (y - 1) ** 2
    return m, [f1, f2]


def test_nbi_interior_points_on_front():
    """Every NBI point — anchors *and* interior — lies on the analytic front.

    Before the fix, the interior points equalled their (dominated) CHIM starts,
    e.g. ``(3.75, 1.25)`` at ``f1 = 3.75`` where the true front has ``f2 ~= 0.09``.
    """
    m, objs = _build_biobj_qp()
    front = normal_boundary_intersection(m, objs, n_points=5, filter=False)
    assert front.n == 5
    off = [tuple(map(float, p.objectives)) for p in front.points if not _on_qp_front(p.objectives)]
    assert not off, f"NBI points off the analytic front (dominated CHIM starts): {off}"


def test_nbi_returned_points_are_nondominated():
    """The returned front must be mutually non-dominated (the dominated CHIM
    interior points were mutually non-dominated, so this alone did not catch the
    bug — kept as a companion invariant)."""
    m, objs = _build_biobj_qp()
    front = normal_boundary_intersection(m, objs, n_points=5)
    assert filter_nondominated(front.objectives()).all()


def test_resolve_after_parameter_change_is_sound():
    """Direct root-cause probe: the NBI max-``t`` subproblem solved twice on one
    model with only the CHIM-target parameters changed between solves.

    The second solve (interior weight ``w = [0.5, 0.5]``) must reach the Pareto
    point ``t = 0.25`` (objective ``(1.25, 1.25)``), not stall at the dominated
    ``t = 0`` CHIM point ``(2.5, 2.5)`` from the stale relaxation of the first
    (anchor) solve. This exercises the cache-reset path without the full sweep,
    so it is both fast and specific to the reintroduction of the bug.
    """
    m, objs = _build_biobj_qp()
    ideal = np.array([0.0, 0.0])
    span = np.array([5.0, 5.0])
    phi = np.array([[0.0, 1.0], [1.0, 0.0]])  # normalized anchors, min-form
    n_hat = -phi.sum(axis=0)  # Das-Dennis quasi-normal

    t = m.continuous("_mo_nbi_t", lb=-1e6, ub=1e6)
    b = [m.parameter("_mo_nbi_b_0", value=0.0), m.parameter("_mo_nbi_b_1", value=0.0)]
    for j in range(2):
        g = (objs[j] - float(ideal[j])) / float(span[j])
        m.subject_to(g - float(n_hat[j]) * t - b[j] == 0)
    m.minimize(-t)

    def solve_weight(w):
        target = phi.T @ (w / w.sum())
        b[0].value = np.asarray(float(target[0]))
        b[1].value = np.asarray(float(target[1]))
        return float(m.solve().x["_mo_nbi_t"])

    # First: an anchor weight (populates the cache); its optimum is t = 0.
    assert solve_weight(np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-4)
    # Second: an interior weight. With the stale cache this stalled at t = 0.
    assert solve_weight(np.array([0.5, 0.5])) == pytest.approx(0.25, abs=2e-3)
