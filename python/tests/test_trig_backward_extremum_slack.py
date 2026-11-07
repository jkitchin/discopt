"""Rust FBBT ``sin``/``cos`` backward rule must not cut a strip next to an extremum.

The rule called a box monotone when an extremum sat within an absolute ``1e-12``
of an endpoint, then picked the piece from the box midpoint and inverted there,
which dropped the strip on the other side of the extremum. With
``TH0 = 1.570796326794`` (``pi/2 - 9e-13``), ``sin(TH0 + 1e-12*n) >= 0.5`` is
feasible at ``n = 0`` (``sin(TH0) = 1 - 4e-25``), yet root FBBT tightened ``n``
to ``[1, 1000]``: ``min n + y`` certified ``-1`` instead of ``-2``, and adding
``n <= 0`` made the model falsely ``infeasible``.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import pytest

TH0 = 1.570796326794  # pi/2 - 8.97e-13
TH_COS = 3.141592653589  # pi - 7.93e-13


def _model(atom, kind, offset=0.0):
    m = dm.Model()
    n = m.integer("n", lb=0, ub=3 if kind == "infeas" else 1000)
    y = m.continuous("y", lb=-2, ub=2)
    if atom == "sin":
        f = dm.sin(TH0 + offset + 1e-12 * n)
        m.subject_to(f >= (-1.0 if kind == "trivial" else 0.5))
    else:
        f = dm.cos(TH_COS + offset + 1e-12 * n)
        m.subject_to(f <= (1.0 if kind == "trivial" else -0.5))
    if kind == "infeas":
        m.subject_to(n <= 0)
    # Nonconvex decoy: routes the solve through spatial B&B, which runs FBBT.
    m.subject_to(y * y >= 1)
    m.minimize(n + y)
    return m


def test_reference_point_is_feasible():
    assert math.sin(TH0) >= 0.5 and math.cos(TH_COS) <= -0.5


@pytest.mark.parametrize("atom", ["sin", "cos"])
@pytest.mark.parametrize("kind", ["opt", "trivial", "infeas"])
def test_strip_next_to_extremum_is_kept(atom, kind):
    r = _model(atom, kind).solve(time_limit=60)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-2.0, abs=1e-6)
    assert float(r.x["n"]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("atom", ["sin", "cos"])
def test_control_extremum_well_outside(atom):
    # Control: shifted 1e-6 away, the box is plainly monotone; passes on main too.
    r = _model(atom, "opt", offset=-1e-6).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-2.0, abs=1e-6)
