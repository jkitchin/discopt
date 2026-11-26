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


@pytest.mark.parametrize(
    ("lb", "ub"),
    [(5, 2), (3, 2), (1.8, 1.2), ([0, 3], [4, 2])],
    ids=["by3", "by1", "fractional_by1_after_snap", "array"],
)
def test_a_box_the_caller_crossed_is_still_refused(lb, ub):
    # ``lb=3, ub=2`` and ``lb=1.8, ub=1.2`` are stored crossed by exactly 1, like an
    # empty integer range; validate decides on the bounds as written (review of
    # #2255), so these stay modeling errors.
    m = dm.Model()
    x = m.integer("x", shape=np.shape(lb), lb=lb, ub=ub)
    m.minimize(dm.sum(x) if np.shape(lb) else x)
    with pytest.raises(ValueError, match="lb > ub"):
        m.validate()


def test_a_bound_assigned_later_that_crosses_is_refused():
    m = dm.Model()
    x = m.integer("x", lb=0, ub=5)
    m.minimize(x)
    x.lb = 6
    with pytest.raises(ValueError, match="lb > ub"):
        m.validate()


def test_empty_integer_box_round_trips_as_infeasible_not_an_error():
    import copy
    import pickle

    from discopt.serialize import dumps, loads

    m = dm.Model()
    x = m.integer("x", lb=1.2, ub=1.8)
    m.minimize(x)
    for m2 in (loads(dumps(m)), pickle.loads(pickle.dumps(m)), copy.deepcopy(m)):
        v = m2._variables[0]
        assert float(v.lb) == 2.0 and float(v.ub) == 1.0
        m2.validate()  # an empty integer range, not a modeling error
    assert loads(dumps(m)).solve(time_limit=20).status == "infeasible"


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


# --- the same rule on ROWS: a row over integers is held as verify_point holds it ----
#
# An integer point that misses a row by a sliver (2e-7 .. 3e-5) sits inside every
# engine's 1e-6 tolerance but outside ``verify_point``'s, which allows only float
# noise on a row no fractional move can repair. Each engine adopted that point,
# pruned the feasible next integer away, and returned ``error`` or a ``feasible``
# point the verifier refuses. Each case below names the engine it reached.


def _row_model(build, lb=0.0):
    m = dm.Model()
    x = m.integer("x", lb=lb, ub=10)
    extra = build(m, x)
    return m, extra


def _y_fixed_linear(obj):
    def build(m, x):
        y = m.integer("y", lb=0, ub=0)
        m.subject_to(2 * x + y >= 2.0000004)
        m.minimize(obj(x))
        return [0.0]

    return build


def _row(row, rhs, obj):
    def build(m, x):
        m.subject_to(row(x) >= rhs)
        m.minimize(obj(x))
        return []

    return build


_ROW_CASES = [
    # (id, build, lb, x*, objective at x*)
    ("spatial_ils_x2", _row(lambda x: x * x, 1.0000002, lambda x: x), 0.0, 2.0, 2.0),
    ("spatial_node_exp", _row(dm.exp, math.e * (1 + 1e-6), lambda x: x), 0.0, 2.0, 2.0),
    ("lift_unverified", _row(lambda x: x * x, 1.00003, lambda x: x - 1 / x), 1.0, 2.0, 1.5),
    ("oa_sqrt_x", _row(dm.sqrt, 1.0000002, lambda x: x), 0.0, 2.0, 2.0),
    ("oa_sqrt_xx", _row(dm.sqrt, 1.0000002, lambda x: x * x), 0.5, 2.0, 4.0),
    ("oa_log_xx", _row(dm.log, 2e-7, lambda x: x * x), 0.5, 2.0, 4.0),
    ("oa_linear_xx", _y_fixed_linear(lambda x: x * x), 0.5, 2.0, 4.0),
    ("highs_milp", _y_fixed_linear(lambda x: x), 0.0, 2.0, 2.0),
]


@pytest.mark.parametrize(
    ("build", "lb", "x_star", "f_star"),
    [c[1:] for c in _ROW_CASES],
    ids=[c[0] for c in _ROW_CASES],
)
def test_integer_point_off_a_row_by_a_sliver_is_not_the_answer(build, lb, x_star, f_star):
    m, extra = _row_model(build, lb)
    # The premise: x = 1 is refused by the arbiter, x* is accepted.
    assert not verify_point(m, [1.0] + extra).ok
    assert verify_point(m, [x_star] + extra).ok
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    xv = float(np.ravel(r.x["x"])[0])
    assert xv == x_star, (xv, r.algorithm_route)
    assert r.objective == pytest.approx(f_star, abs=1e-9)
    assert verify_point(m, [xv] + extra).ok
    assert r.bound <= r.objective + 1e-6


def test_round_single_integer_cut():
    from discopt.solvers.oa import _round_single_integer_cut

    integ = np.array([1, 0])
    # x >= 1.0000004 written as -0.5 x <= -0.5000002 -> x >= 2
    row, rhs = _round_single_integer_cut(np.array([-0.5, 0.0]), -0.5000002, integ)
    np.testing.assert_array_equal(row, [-1.0, 0.0])
    assert rhs == -2.0
    # x <= 2.7 -> x <= 2
    row, rhs = _round_single_integer_cut(np.array([2.0, 0.0]), 5.4, integ)
    np.testing.assert_array_equal(row, [1.0, 0.0])
    assert rhs == 2.0
    # resting on an integer up to float noise stays there
    _, rhs = _round_single_integer_cut(np.array([1.0, 0.0]), 2.0 - 1e-12, integ)
    assert rhs == 2.0
    # a continuous column, or two columns, is left alone
    row, rhs = _round_single_integer_cut(np.array([0.0, 1.0]), 2.5, integ)
    assert rhs == 2.5
    row, rhs = _round_single_integer_cut(np.array([1.0, 1.0]), 2.5, integ)
    assert rhs == 2.5


def test_round_single_integer_rows():
    from discopt.solvers.oa import _round_single_integer_rows, _SparseRow

    integ = np.array([1, 1, 0])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([10.0, 0.0, 5.0])  # column 1 fixed at 0
    rows = [
        _SparseRow.from_dense([2.0, 1.0, 0.0]),  # 2x + y >= 2.0000004 -> x >= 2
        _SparseRow.from_dense([1.0, 0.0, 0.0]),  # x <= 7.5 -> x <= 7
        _SparseRow.from_dense([1.0, 0.0, 1.0]),  # x + c <= 3: two free columns, untouched
        _SparseRow.from_dense([0.0, 0.0, 1.0]),  # c >= 0.5: continuous, untouched
    ]
    new_lb, new_ub = _round_single_integer_rows(
        lb, ub, rows, [2.0000004, 7.5, 3.0, 0.5], [">=", "<=", "<=", ">="], integ
    )
    np.testing.assert_array_equal(new_lb, [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(new_ub, [7.0, 0.0, 5.0])
    np.testing.assert_array_equal(lb, [0.0, 0.0, 0.0])  # inputs are not mutated


# --- review of #2257 -----------------------------------------------------------


def test_highs_verifier_resolve_never_adopts_a_tight_infeasible(monkeypatch):
    """The 1e-9 re-solve searches a SMALLER feasible set than the 1e-6 semantics
    ``verify_point`` uses, so its ``infeasible`` is no certificate for this model:
    the refused primary result must stand, labelled as declined."""
    import time

    import discopt.solver as S
    import discopt.solvers.lp_milp_highs as H

    m = dm.Model()
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=0)
    m.subject_to(2 * x + y >= 2.0000004)
    m.minimize(x)
    # Since the #2252 follow-up the HiGHS standard form rounds this single-free-
    # integer row to x >= 2, so the refused x = 1 no longer arises from it. Switch
    # that rounding off to reach the verifier re-solve this test guards.
    monkeypatch.setattr(S, "_round_single_integer_rows_sparse", lambda xl, xu, *a, **k: (xl, xu, 0))
    _, n_orig, sf = S._highs_std_form(m)
    primary = H.solve_milp_std(
        sf, time_limit=20.0, gap_tolerance=1e-4, max_nodes=10**6, n_struct=n_orig
    )
    assert primary.x is not None and float(primary.x[0]) == 1.0  # the refused point

    calls = []

    def tight_infeasible(*args, **kwargs):
        calls.append(kwargs.get("feasibility_tolerance"))
        return H.HighsOutcome("infeasible", gap_certified=True, highs_status="mocked")

    monkeypatch.setattr(H, "solve_milp_std", tight_infeasible)
    out = S._highs_milp_verifier_resolve(
        m, sf, primary, n_orig, 20.0, time.perf_counter(), 10**6, 1e-4, None
    )
    assert calls == [H.REFUTED_TIGHT_FEAS_TOL]  # the re-solve ran, at 1e-9
    assert out is primary and out.status != "infeasible"
    assert "milp/verifier_tight_resolve_declined" in out.labels


def test_raw_bounds_are_frozen_after_copy_and_pickle():
    import copy
    import pickle

    m = dm.Model()
    m.integer("y", lb=1.2, ub=1.8)
    for m2 in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        v = m2._variables[0]
        for raw in (v._raw_lb, v._raw_ub):
            assert raw is not None and not raw.flags.writeable
            with pytest.raises(ValueError):
                raw[...] = 0.0


def test_oa_rounding_skips_a_bound_too_large_to_round_exactly():
    # 1e-13 y >= 5 states y >= 5e13; with the 1e-9 relative noise allowance (5e4
    # there) a rounding would give 5e13 - 5e4, not the exact integer bound. Both OA
    # helpers leave such a row unrounded, as the pure-MILP helper does.
    from discopt.solvers.oa import (
        _round_single_integer_cut,
        _round_single_integer_rows,
        _SparseRow,
    )

    integ = np.array([1])
    lb, ub = _round_single_integer_rows(
        np.array([0.0]), np.array([1e14]), [_SparseRow.from_dense([1e-13])], [5.0], [">="], integ
    )
    assert lb.tolist() == [0.0] and ub.tolist() == [1e14]
    row, rhs = _round_single_integer_cut(np.array([-1e-13]), -5.0, integ)
    assert rhs == -5.0 and row.tolist() == [-1e-13]
    # A small bound still rounds exactly.
    row, rhs = _round_single_integer_cut(np.array([-1.0]), -2.0000004, integ)
    assert (row.tolist(), rhs) == ([-1.0], -3.0)


# ``result.objective`` is a claim that the returned ``x`` attains it (#1331), so on
# the integer-bilinear lift it is the UN-LIFTED model's objective at that point,
# exactly, on every model the lift is adopted on -- not the lifted MILP's linear
# expansion over bits and product columns, which each sit inside a 1e-6 row
# tolerance (``lift_unverified`` reported 1.4999999986 at x = 2, f = 1.5).
def _lift_family(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model()
    x = m.integer("x", lb=1, ub=int(rng.integers(4, 9)))
    y = m.integer("y", lb=1, ub=int(rng.integers(4, 9)))
    a, b, c = (float(v) for v in rng.uniform(0.5, 2.0, 3))
    m.subject_to(x * y >= float(rng.uniform(3, 10)))
    m.subject_to(a * x + b * y <= float(rng.uniform(8, 14)))
    if seed % 2:
        m.minimize(x * y * c - a / x + b * y * y - x)
    else:
        m.minimize(c * x * x - b / y + x * y)
    return m


def _lift_objective_cases():
    out = [(f"family{s}", lambda s=s: _lift_family(s)) for s in range(8)]
    case = next(c for c in _ROW_CASES if c[0] == "lift_unverified")
    out.append(("lift_unverified", lambda: _row_model(case[1], case[2])[0]))
    return out


@pytest.mark.parametrize("race", ["1", "0"], ids=["race", "legacy"])
@pytest.mark.parametrize(
    "build", [c[1] for c in _lift_objective_cases()], ids=[c[0] for c in _lift_objective_cases()]
)
def test_lift_reports_the_objective_its_point_attains(build, race, monkeypatch):
    import discopt.solver as S

    monkeypatch.setenv("DISCOPT_IPX_RACE", race)
    adopted = []
    orig = S._ipx_lift_point_refusal

    def spy(src, res):
        why = orig(src, res)
        adopted.append(why is None and getattr(res, "x", None) is not None)
        return why

    monkeypatch.setattr(S, "_ipx_lift_point_refusal", spy)
    m = build()
    r = m.solve(time_limit=30)
    # The probe fired: the lift ran and its point was the one judged.
    assert any(adopted), (adopted, r.algorithm_route)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    flat = [float(np.ravel(r.x[v.name])[0]) for v in m._variables]
    v = verify_point(m, flat, with_objective=True)
    assert v.ok, v.reason
    assert r.objective == v.objective, (r.objective, v.objective)
    assert r.bound <= r.objective + 1e-6


# #2352: ``result.objective`` is a claim that the returned ``x`` attains it (#1331).
# The spatial B&B exit integer-snaps the tree incumbent (C-3) and used to keep the
# NLP's value for the unsnapped point: on seed 2 below it returned (8, 1) with
# ``objective`` 3.75e-9 below f(8, 1), optimal and certified (seeds 2 and 4 of 8).
# The same consistency is asserted on the integer-bilinear lift by #2350's
# ``test_lift_reports_the_objective_its_point_attains``.
def _spatial_family(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model()
    x = m.integer("x", lb=1, ub=int(rng.integers(4, 9)))
    y = m.integer("y", lb=1, ub=int(rng.integers(4, 9)))
    a, b, c = (float(v) for v in rng.uniform(0.5, 2.0, 3))
    m.subject_to(x * y >= float(rng.uniform(3, 10)))
    m.subject_to(a * x + b * y <= float(rng.uniform(8, 14)))
    m.minimize(x * y * c - 1 / x + dm.exp(0.1 * y) - dm.log(x + y))
    return m


@pytest.mark.parametrize("seed", range(8))
def test_spatial_bb_reports_the_objective_its_point_attains(seed):
    m = _spatial_family(seed)
    r = m.solve(time_limit=30)
    # The probe reached the route under test.
    assert r.algorithm_route.startswith("spatial-bb"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    flat = [float(np.ravel(r.x[v.name])[0]) for v in m._variables]
    v = verify_point(m, flat, with_objective=True)
    assert v.ok, v.reason
    assert r.objective == v.objective, (r.objective, v.objective)
    assert r.bound <= r.objective + 1e-6
