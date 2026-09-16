"""Declared-box nonlinear bound tightening must not lose small terms to huge ones.

``SeparableQuadraticUpperBoundRule`` summed each row's minimum activity in floating
point and took leave-one-out sums as ``total - own``. With a column in the default
±9.999e19 box, ``-4.9995e20 + -1`` rounds to ``-4.9995e20``, the leave-one-out rest
of that column came out ``0`` instead of ``-1``, and the derived bound was tighter
than the row implies. Found by adversarial testing of the #1229 route: a 3-variable
feasible LP was returned ``infeasible`` with ``gap_certified=True`` on both LP
backends, because this pass runs before any routing.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.solver import _declared_box_tightening


def _two_fixed(x1_ub=None, rhs=-2.0):
    m = dm.Model("nbt_absorption")
    x0 = m.continuous("x0", lb=-3.0, ub=-3.0)
    x1 = m.continuous("x1") if x1_ub is None else m.continuous("x1", lb=0.0, ub=x1_ub)
    x2 = m.continuous("x2", lb=-1.0, ub=-1.0)
    m.subject_to(5 * x1 - x2 >= rhs)
    m.subject_to(-4 * x0 + 4 * x1 + 4 * x2 == 6)  # x1 = -0.5
    m.minimize(0 * x0 + 0 * x1)
    return m


def test_huge_box_column_keeps_the_feasible_point():
    lb, ub, stats = _declared_box_tightening(_two_fixed())
    assert not stats.infeasible, stats.infeasibility_reason
    assert lb[1] <= -0.5 <= ub[1]  # the only feasible x1 survives


@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_feasible_lp_is_not_certified_infeasible(monkeypatch, backend):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    res = _two_fixed().solve(time_limit=20)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(0.0, abs=1e-9)


def test_a_real_contradiction_is_still_proved():
    # 5 x1 + 1 >= 7 needs x1 >= 1.2, outside [0, 1]: widening must not hide this.
    _, _, stats = _declared_box_tightening(_two_fixed(x1_ub=1.0, rhs=7.0))
    assert stats.infeasible
