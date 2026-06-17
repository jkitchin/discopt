"""Tests for the complementarity / MPEC handler (discopt.mpec).

Reference problem (a standard MPEC test):

    min (x-1)^2 + (y-1)^2   s.t.   0 <= x ⊥ y >= 0

The complementarity forces x·y = 0, so a global optimum sits at (1, 0) or
(0, 1), both with objective 1. We verify that:
  * the Scholtes homotopy drives x·y -> 0 and reaches objective 1,
  * the exact SOS1 reformulation reaches the same global optimum,
  * the two reformulations agree,
  * single-variable complementarity bound propagation is sound.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest

pytest.importorskip("pounce")

from discopt.mpec import (  # noqa: E402
    Complementarity,
    complementarity,
    reformulate_scholtes,
    solve_mpec,
    tighten_complementarity_bounds,
)

pytestmark = pytest.mark.requires_pounce


def _distance_model() -> tuple[dm.Model, dm.Variable, dm.Variable]:
    m = dm.Model("mpec_distance")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    return m, x, y


def test_complementarity_factory():
    m, x, y = _distance_model()
    p = complementarity(x, y, name="c0")
    assert isinstance(p, Complementarity)
    assert p.f is x and p.g is y and p.name == "c0"


def test_scholtes_reaches_global_optimum():
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y)], method="scholtes")
    assert res is not None and res.x is not None
    xv, yv = float(res.x[0]), float(res.x[1])
    # Objective ~ 1 and complementarity ~ satisfied.
    assert abs(float(res.objective) - 1.0) < 1e-3
    assert xv * yv < 1e-5
    assert min(abs(xv), abs(yv)) < 1e-3  # one side driven to zero


def test_sos1_reaches_global_optimum():
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y)], method="sos1")
    # Model.solve returns x as a name->value dict.
    assert res.objective is not None
    assert abs(float(res.objective) - 1.0) < 1e-3
    xv = float(np.asarray(res.x["x"]))
    yv = float(np.asarray(res.x["y"]))
    assert min(abs(xv), abs(yv)) < 1e-4  # exact complementarity


def test_scholtes_and_sos1_agree():
    m1, x1, y1 = _distance_model()
    r1 = solve_mpec(m1, [complementarity(x1, y1)], method="scholtes")
    m2, x2, y2 = _distance_model()
    r2 = solve_mpec(m2, [complementarity(x2, y2)], method="sos1")
    assert abs(float(r1.objective) - float(r2.objective)) < 1e-3


def test_reformulate_scholtes_adds_three_constraints_per_pair():
    m, x, y = _distance_model()
    before = len(m._constraints)
    reformulate_scholtes(m, [complementarity(x, y)], t=1e-3)
    assert len(m._constraints) - before == 3


def test_bound_propagation_fixes_zero_side():
    m = dm.Model("bp")
    a = m.continuous("a", lb=0.5, ub=3.0)  # strictly positive lower bound
    b = m.continuous("b", lb=0.0, ub=3.0)
    n_fixed = tighten_complementarity_bounds(m, [complementarity(a, b)])
    assert n_fixed == 1
    assert float(b.ub) == 0.0 and float(b.lb) == 0.0
    # a is untouched.
    assert float(a.lb) == 0.5


def test_bound_propagation_no_fix_when_both_free():
    m = dm.Model("bp2")
    a = m.continuous("a", lb=0.0, ub=3.0)
    b = m.continuous("b", lb=0.0, ub=3.0)
    assert tighten_complementarity_bounds(m, [complementarity(a, b)]) == 0


def test_unknown_method_raises():
    m, x, y = _distance_model()
    with pytest.raises(ValueError):
        solve_mpec(m, [complementarity(x, y)], method="bogus")
