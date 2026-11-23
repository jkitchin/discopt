"""#2312: the 1e-12 floor of the feasible-distance cap is held to the distance rule.

``feasible_distance_cap`` holds a row's violation to ``floor + 1e-4 * ||grad||``
(#1254). Its absolute floor ``SMALL_ROW_ABS_FLOOR = 1e-12`` was not held to the
same rule: on a row whose largest partial derivative is ``G`` it admitted a point
``1e-12 / G`` away, which exceeds the declared 1e-4 once ``G < 1e-8``. And all
three gates (``verify_point``, ``solver._check_constraint_feasibility``,
``primal_heuristics._check_constraint_feasibility``) accepted any row violated by
at most 1e-12 before looking at a gradient.

Measured before the fix: #1254's model with ``y`` shifted down six decades was
certified ``optimal`` at -20 (optimum -12.699), and on ``x / y <= 0`` with
``y = 1e12`` the gate accepted ``x = 1``.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.validation.feasibility import (
    FEASIBLE_DISTANCE_TOL,
    SMALL_ROW_ABS_FLOOR,
    feasible_distance_cap,
    verify_point,
)

OPTIMUM = math.log10(2e-13)  # -12.69897...


def _shifted_epigraph() -> dm.Model:
    m = dm.Model("shifted_epigraph")
    y1 = m.continuous("y1", lb=-15.0, ub=-12.5)
    y2 = m.continuous("y2", lb=-15.0, ub=-12.5)
    z = m.continuous("z", lb=-20.0, ub=0.0)
    m.subject_to(y1 + y2 >= -26.0)
    m.subject_to(10.0**y1 + 10.0**y2 <= 10.0**z)
    m.minimize(z)
    return m


def _scaled_row(d: float) -> dm.Model:
    m = dm.Model("scaled_row")
    x = m.continuous("x", lb=-1, ub=1)
    y = m.continuous("y", lb=d, ub=2 * d)
    m.subject_to(x / y <= 0)
    m.minimize(-x)
    return m


@pytest.mark.unit
def test_floor_is_held_to_the_plain_norm():
    # A tiny plain norm shrinks the floor; an exactly flat row keeps it.
    cap = feasible_distance_cap(np.array([1e-13, 0.0, 1.0]))
    assert cap[0] == pytest.approx(2 * FEASIBLE_DISTANCE_TOL * 1e-13, rel=1e-12)
    assert cap[1] == SMALL_ROW_ABS_FLOOR
    assert cap[2] == pytest.approx(SMALL_ROW_ABS_FLOOR + FEASIBLE_DISTANCE_TOL, rel=1e-12)


@pytest.mark.unit
def test_pinned_column_keeps_its_floor_through_the_plain_norm():
    """#1392: an improving norm shrunk by a pinned column keeps the absolute floor
    when the row's plain sup-norm is ordinary."""
    cap = float(feasible_distance_cap(1e-10, None, 1.0))
    assert cap == pytest.approx(SMALL_ROW_ABS_FLOOR + FEASIBLE_DISTANCE_TOL * 1e-10, rel=1e-12)


@pytest.mark.unit
def test_verify_point_rejects_the_shifted_epigraph_point():
    m = _shifted_epigraph()
    assert not verify_point(m, np.array([-13.0, -13.0, -20.0])).ok
    # The true optimum is still accepted.
    assert verify_point(m, np.array([-13.0, -13.0, OPTIMUM + 1e-9])).ok


@pytest.mark.unit
@pytest.mark.parametrize("d", [1.0, 1e4, 1e8, 1e10, 1e12])
def test_scaled_row_acceptance_does_not_grow_with_the_denominator(d):
    m = _scaled_row(d)
    checked = 0
    # A point 1e-2 from ``x <= 0`` is never acceptable, at any d.
    assert not verify_point(m, np.array([1e-2, d])).ok
    checked += 1
    # A point well inside the declared distance stays acceptable.
    assert verify_point(m, np.array([1e-7, d])).ok
    checked += 1
    assert checked == 2


@pytest.mark.unit
@pytest.mark.parametrize("d", [1e8, 1e12])
def test_the_two_check_constraint_gates_agree(d):
    from discopt._relax import primal_heuristics as ph
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solver import _check_constraint_feasibility

    m = _scaled_row(d)
    ev = make_evaluator(m)
    bad = np.array([1e-2, d])
    assert not _check_constraint_feasibility(ev, bad, [-1e20], [0.0])
    cl, cu = np.array([-1e20]), np.array([0.0])
    assert not ph._check_constraint_feasibility(ev, bad, cl=cl, cu=cu)
    ok = np.array([1e-9, d])
    assert _check_constraint_feasibility(ev, ok, [-1e20], [0.0])
    assert ph._check_constraint_feasibility(ev, ok, cl=cl, cu=cu)


@pytest.mark.slow
@pytest.mark.correctness
def test_shifted_epigraph_is_not_certified_at_minus_twenty():
    r = _shifted_epigraph().solve(time_limit=30, threads=1)
    if r.gap_certified:
        assert r.objective == pytest.approx(OPTIMUM, abs=1e-4)
    if r.objective is not None:
        assert r.objective >= OPTIMUM - 1e-4
