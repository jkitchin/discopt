"""The feasible-distance cap must ignore columns that cannot move toward the row (#1284).

#1254 capped a row's violation at ``1e-4 * ||grad g||_inf``. Adding ``+ s`` with
``s in [0, 1]`` to the #1254 row makes ``||grad||_inf = 1`` through ``s`` alone,
although ``s = 0`` sits on the bound that blocks its only improving direction —
and the point ``y1 = y2 = -7, z = -20``, 0.87 away in ``y`` from satisfying the
row, was certified again. The same with ``s`` binary. True optimum
``log10(2e-7)`` (AM-GM with ``y1 + y2 >= -14``, at ``s = 0``).
"""

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.validation.feasibility import verify_point

TRUE_OPTIMUM = math.log10(2e-7)


def _model(kind):
    m = dm.Model("cap")
    y1 = m.continuous("y1", lb=-9.0, ub=-6.0)
    y2 = m.continuous("y2", lb=-9.0, ub=-6.0)
    z = m.continuous("z", lb=-20.0, ub=0.0)
    s = m.binary("s") if kind == "binary" else m.continuous("s", lb=0.0, ub=1.0)
    m.subject_to(y1 + y2 >= -14)
    m.subject_to(10.0**y1 + 10.0**y2 + s <= 10.0**z)
    m.minimize(z)
    return m


def test_improving_norm_drops_blocked_and_integer_columns():
    from discopt.validation.feasibility import improving_gradient_norms

    J = np.array([[2.3e-7, 2.3e-7, -2.3e-20, 1.0]])
    x = np.array([-7.0, -7.0, -20.0, 0.0])
    lb = np.array([-9.0, -9.0, -20.0, 0.0])
    ub = np.array([-6.0, -6.0, 0.0, 1.0])
    # body must decrease: s would have to go below its lower bound
    assert improving_gradient_norms(J, x, lb, ub, [1.0])[0] == pytest.approx(2.3e-7)
    # the other direction: s can move up, so it counts in full
    assert improving_gradient_norms(J, x, lb, ub, [-1.0])[0] == pytest.approx(1.0)
    # an integer column never counts
    mask = np.array([False, False, False, True])
    assert improving_gradient_norms(J, x, lb, ub, [-1.0], mask)[0] == pytest.approx(2.3e-7)
    # partial room scales the column's contribution
    x2 = x.copy()
    x2[3] = 0.5e-4
    assert improving_gradient_norms(J, x2, lb, ub, [1.0])[0] == pytest.approx(0.5)
    # a satisfied row keeps the plain sup-norm
    assert improving_gradient_norms(J, x, lb, ub, [0.0])[0] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_verify_point_rejects_the_far_point(kind):
    m = _model(kind)
    assert not verify_point(m, np.array([-7.0, -7.0, -20.0, 0.0])).ok


@pytest.mark.parametrize("kind", ["continuous", "binary"])
def test_solve_does_not_certify_below_the_optimum(kind):
    r = _model(kind).solve(time_limit=60)
    assert r.status in ("optimal", "feasible", "time_limit")
    if r.bound is not None and r.status == "optimal":
        assert r.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-4)
    if r.objective is not None:
        assert r.objective >= TRUE_OPTIMUM - 1e-4


def test_single_column_exact_repair_is_not_decided_by_roundoff():
    """portfol_roundlot's OA incumbent: ``x11 - 78000 x2 >= 0`` at ``x2 = 7.22e-11``
    (lower bound 0), ``x11 = 0`` integer. Moving ``x2`` onto its bound repairs the
    row exactly, so the point is 7e-11 from feasible and must stay inside the cap."""
    from discopt.validation.feasibility import (
        FEASIBLE_DISTANCE_TOL,
        improving_gradient_norms,
    )

    x2 = 7.220330978261474e-11
    J = np.array([[-78000.0, 1.0]])
    x = np.array([x2, 0.0])
    viol = 0.0 - (0.0 - 78000.0 * x2)
    g = improving_gradient_norms(
        J, x, np.zeros(2), np.full(2, np.inf), [-1.0], np.array([False, True])
    )
    assert viol <= FEASIBLE_DISTANCE_TOL * g[0]
