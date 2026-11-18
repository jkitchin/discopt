"""#2252: one rule for the integer range of an INTEGER variable with a fractional bound.

Before the fix, ``m.integer("x", lb=1.000001, ub=3)`` was rounded differently per
route: ``min x`` (HiGHS MILP) returned x = 1, ``min x*x`` (LP/NLP-BB) returned
x = 1.0000010012543346, and ``min x - 1/x`` (native kernel) excluded x = 1 and
certified 1.5 at x = 2 with bound 1.499999999999 -- above the objective 0 of
x = 1, a point ``verify_point`` accepted.

The rule now lives where every box is installed (``Variable.lb``/``.ub``): a bound
within 1e-6 of an integer is that integer, otherwise ``lb`` rounds up and ``ub``
rounds down. Every route and ``verify_point`` read the same integral box, and the
OA-family exit publishes the integral point ``verify_point`` tested.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import (
    _INTEGER_BOUND_TOL,
    VarType,
    _is_integral_box,
    _readonly_bound,
    _snap_integer_bound,
)
from discopt.validation.feasibility import verify_point

# --- the rule itself ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "side", "expected"),
    [
        (1.000001, "lb", 1.0),  # within tol of 1 (the issue's bound)
        (1.000001, "ub", 1.0),
        (0.999999, "lb", 1.0),
        (0.999999, "ub", 1.0),
        (1.0000011, "lb", 2.0),  # beyond tol: lb rounds up
        (1.0000011, "ub", 1.0),  # ... and ub rounds down
        (1.5, "lb", 2.0),
        (1.5, "ub", 1.0),
        (-1.5, "lb", -1.0),
        (-1.5, "ub", -2.0),
        (2.9999999, "lb", 3.0),
        (2.9999999, "ub", 3.0),
        (2.0000005, "lb", 2.0),  # derived-bound noise must not cut off x = 2
        (7.0, "lb", 7.0),
        (-9.999e19, "lb", -9.999e19),
        (math.inf, "ub", math.inf),
        (-math.inf, "lb", -math.inf),
    ],
)
def test_snap_rule(value, side, expected):
    out = _snap_integer_bound(np.array(value, dtype=np.float64), side)
    assert float(out) == expected


@pytest.mark.parametrize("k", [-1000.0, -1.0, 0.0, 1.0, 1000.0])
def test_snap_boundary_is_symmetric(k):
    # ``k +/- 1e-6`` is "within 1e-6" whichever way binary64 rounds the decimal.
    for side in ("lb", "ub"):
        for v in (k + 1e-6, k - 1e-6):
            assert float(_snap_integer_bound(np.array(v), side)) == k, (k, v, side)


def test_snap_rule_ignores_binary64_subtraction_artifact():
    # ``ceil(lb - tol)`` would decide this case by how ``1.000001 - 1e-6`` rounds
    # (to 0.9999999999999999); the rule measures the distance to the integer.
    assert 1.000001 - _INTEGER_BOUND_TOL < 1.0
    assert float(_snap_integer_bound(np.array(1.000001), "lb")) == 1.0


def test_snap_requires_a_side():
    with pytest.raises(ValueError, match="side"):
        _readonly_bound(1.5, (), var_type=VarType.INTEGER)


def test_is_integral_box():
    assert _is_integral_box(np.array(3.0))
    assert _is_integral_box(np.array(math.inf))
    assert _is_integral_box(np.array([1.0, -math.inf, 4.0]))
    assert not _is_integral_box(np.array(1.5))
    assert not _is_integral_box(np.array([1.0, 2.5]))


# --- every door into a box applies it ----------------------------------------


def test_declaration_scalar_array_and_indexed():
    m = dm.Model()
    x = m.integer("x", lb=1.000001, ub=3.4)
    assert float(x.lb) == 1.0 and float(x.ub) == 3.0
    y = m.integer("y", shape=(3,), lb=[0.2, 1.000001, -0.5], ub=[2.7, 5.0, 0.5])
    np.testing.assert_array_equal(y.lb, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(y.ub, [2.0, 5.0, 0.0])
    s = dm.Set("s", ["a", "b"])
    z = m.integer("z", over=s, lb={"a": 0.5, "b": 1.0}, ub=4.5)
    np.testing.assert_array_equal(z.flat.lb, [1.0, 1.0])
    np.testing.assert_array_equal(z.flat.ub, [4.0, 4.0])


def test_setter_and_frozen_box_from_another_variable():
    m = dm.Model()
    x = m.integer("x", lb=0, ub=10)
    x.lb = 2.5
    x.ub = 7.0000004
    assert float(x.lb) == 3.0 and float(x.ub) == 7.0
    # A frozen box built for a CONTINUOUS variable is fractional; installing it on
    # an integer must not take the no-copy fast path past the snap.
    c = m.continuous("c", lb=1.25, ub=8.75)
    x.lb, x.ub = c.lb, c.ub
    assert float(x.lb) == 2.0 and float(x.ub) == 8.0
    assert float(c.lb) == 1.25  # the continuous box itself is untouched


def test_continuous_and_binary_unchanged():
    m = dm.Model()
    c = m.continuous("c", lb=1.000001, ub=2.5)
    assert float(c.lb) == 1.000001 and float(c.ub) == 2.5
    b = m.binary("b")
    assert float(b.lb) == 0.0 and float(b.ub) == 1.0


def test_fix_at_near_integral_value():
    m = dm.Model()
    x = m.integer("x", lb=0, ub=5)
    x.fix(2.0000001)
    assert float(x.lb) == 2.0 and float(x.ub) == 2.0
    x.unfix()
    assert float(x.lb) == 0.0 and float(x.ub) == 5.0


def test_serialization_round_trip():
    from discopt.serialize import dumps, loads

    m = dm.Model()
    x = m.integer("x", lb=1.000001, ub=3)
    m.minimize(x)
    x.fix(2.0)
    m2 = loads(dumps(m))
    v = m2._variables[0]
    assert float(v.lb) == 2.0
    v.unfix()
    assert float(v.lb) == 1.0 and float(v.ub) == 3.0


# --- the issue's reproducer: routes agree with each other and with verify_point --


def _solve(obj, lb):
    m = dm.Model()
    x = m.integer("x", lb=lb, ub=3)
    m.minimize(obj(x))
    r = m.solve(time_limit=30)
    return m, r


_OBJECTIVES = [
    ("x", lambda x: x, {1.0: 1.0, 2.0: 2.0}),
    ("x*x", lambda x: x * x, {1.0: 1.0, 2.0: 4.0}),
    ("x - 1/x", lambda x: x - 1 / x, {1.0: 0.0, 2.0: 1.5}),
]


@pytest.mark.parametrize(("lb", "x_star"), [(1.000001, 1.0), (1.0000001, 1.0), (1.5, 2.0)])
@pytest.mark.parametrize(("label", "obj", "value_at"), _OBJECTIVES, ids=[o[0] for o in _OBJECTIVES])
def test_routes_agree_with_verify_point(lb, x_star, label, obj, value_at):
    m, r = _solve(obj, lb)
    assert r.status == "optimal", (label, lb, r.status)
    x_val = float(r.x["x"])
    # An integer variable is published at an integral value on every route.
    assert x_val == x_star, (label, lb, x_val, r.algorithm_route)
    assert r.objective == pytest.approx(value_at[x_star], abs=1e-9)
    # The certificate never sits above a point verify_point accepts.
    check = verify_point(m, [1.0], with_objective=True)
    assert check.ok == (x_star == 1.0), (label, lb, check)
    if check.ok:
        assert r.bound <= check.objective + 1e-6, (label, lb, r.bound, check.objective)
    assert r.bound <= r.objective + 1e-6


# --- a box holding no integer --------------------------------------------------


@pytest.mark.parametrize(("label", "obj", "value_at"), _OBJECTIVES, ids=[o[0] for o in _OBJECTIVES])
def test_box_without_an_integer_is_infeasible_not_an_error(label, obj, value_at):
    # Stored as the crossed range [2, 1]; validate() must not mistake that for a
    # modeling error, and the solve proves it infeasible.
    m = dm.Model()
    x = m.integer("x", lb=1.2, ub=1.8)
    assert float(x.lb) == 2.0 and float(x.ub) == 1.0
    m.minimize(obj(x))
    r = m.solve(time_limit=20)
    assert r.status == "infeasible", (label, r.status)
    assert r.objective is None


def test_a_genuinely_crossed_integer_box_is_still_refused():
    m = dm.Model()
    m.integer("x", lb=5, ub=2)
    m.minimize(m._variables[0])
    with pytest.raises(ValueError, match="lb > ub"):
        m.validate()


def test_lp_spatial_engine_refuses_a_point_outside_a_crossed_box():
    # The engine's primal clips to the root box; on a crossed box the clip lands
    # outside it, and its verifier checked only rows: it certified ``optimal`` at
    # x0 = -2 for x0 in [-1.6, -1.07] (stored [-1, -2]).
    from discopt._relax.lp_spatial_bb import solve_lp_spatial_bb

    m = dm.Model()
    x0 = m.integer("x0", lb=-1.6, ub=-1.07)
    x1 = m.integer("x1", lb=-5.0, ub=0.0)
    m.subject_to(2 * x0 + x1 <= -6.6)
    m.minimize(4 * x1 + 1e6)
    res = solve_lp_spatial_bb(m, time_limit=10.0, gap_tolerance=1e-4)
    assert res is not None
    assert res.status == "infeasible" and res.x is None
