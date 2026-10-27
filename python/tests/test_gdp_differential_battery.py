"""The GDP differential soundness battery (#1775 G0.3, spec §6.1).

Every shape below is lowered by every method in ``gdp_differential.METHODS`` and
checked in both directions (no feasible point cut, no infeasible point admitted)
by the harness in ``gdp_differential.py``; see that module for how the check
avoids trusting the solver under test.

The first group is the battery of ``docs/dev/gdp-module-review.md`` §4 (two-way,
multi-row, multi-var, equality, three-way, nonlinear-quadratic, unbounded). The
second group is the §6.1 extension. A shape a method *should* lower but refuses
today is ``xfail(strict=True)`` with the task or issue that will lift the refusal,
so the marker breaks the build the day the refusal goes away.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import pytest
from discopt.modeling.core import _DisjunctiveConstraint
from gdp_differential import METHODS, Shape, run_shape

# ---------------------------------------------------------------------------
# Review §4 shapes
# ---------------------------------------------------------------------------


def _two_way():
    m = dm.Model("two_way")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.minimize(x)
    return m


def _multi_row():
    m = dm.Model("multi_row")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 2, y >= 3], [x >= 5, y <= 4]], name="d")
    m.minimize(x + y)
    return m


def _multi_var():
    m = dm.Model("multi_var")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x + y <= 4], [x - y >= 3, x + 2 * y >= 12]], name="d")
    m.minimize(x + y)
    return m


def _equality():
    m = dm.Model("equality")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x == 3, y <= 5], [x >= 7]], name="d")
    m.minimize(x)
    return m


def _three_way():
    m = dm.Model("three_way")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 1], [x >= 3, x <= 4], [x >= 8]], name="d")
    m.minimize(x)
    return m


def _quadratic():
    m = dm.Model("quadratic")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 2, y >= 3], [x >= 5, x * x + y <= 40]], name="d")
    m.minimize(x + y)
    return m


def _unbounded():
    m = dm.Model("unbounded")
    x = m.continuous("x", lb=0)  # no finite upper bound: big-M has no valid M
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.minimize(x)
    return m


# ---------------------------------------------------------------------------
# §6.1 extension shapes
# ---------------------------------------------------------------------------


def _nested3():
    """Depth-3 nesting: [x<=3 ∧ ([y<=2 ∧ ([z<=1] ∨ [z>=4])] ∨ [y>=6])] ∨ [x>=7]."""
    m = dm.Model("nested3")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    inner = _DisjunctiveConstraint(disjuncts=[[z <= 1], [z >= 4]], name="inner")
    mid = _DisjunctiveConstraint(disjuncts=[[y <= 2, inner], [y >= 6]], name="mid")
    m.either_or([[x <= 3, mid], [x >= 7]], name="outer")
    m.minimize(x + y + z)
    return m


def _nested3_member(v, t):
    x, y, z = v["x"], v["y"], v["z"]
    inner = z <= 1 + t or z >= 4 - t
    mid = (y <= 2 + t and inner) or y >= 6 - t
    return (x <= 3 + t and mid) or x >= 7 - t


def _nested3_blocks():
    """``nested3`` as Disjunct blocks (#1775 G2.1: hull lowers block nesting too):
    A [x<=3, M1 [y<=2, I1 [z<=1] v I2 [z>=4]] v M2 [y>=6]] v B [x>=7]."""
    m = dm.Model("nested3_blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    i1, i2 = m.make_disjunct("I1"), m.make_disjunct("I2")
    i1.subject_to(z <= 1)
    i2.subject_to(z >= 4)
    m1, m2 = m.make_disjunct("M1"), m.make_disjunct("M2")
    m1.subject_to(y <= 2)
    m1.add_disjunction([i1, i2], name="inner")
    m2.subject_to(y >= 6)
    a, b = m.make_disjunct("A"), m.make_disjunct("B")
    a.subject_to(x <= 3)
    a.add_disjunction([m1, m2], name="mid")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + y + z)
    return m


def _nested3_blocks_member(v, t):
    """The indicators are columns, so the point carries its selection, which must
    be consistent (each level's indicators sum to its parent's)."""
    s = {k: round(v[f"{k}_active"]) for k in ("A", "B", "M1", "M2", "I1", "I2")}
    if s["A"] + s["B"] != 1 or s["M1"] + s["M2"] != s["A"] or s["I1"] + s["I2"] != s["M1"]:
        return False
    return (
        (not s["A"] or v["x"] <= 3 + t)
        and (not s["B"] or v["x"] >= 7 - t)
        and (not s["M1"] or v["y"] <= 2 + t)
        and (not s["M2"] or v["y"] >= 6 - t)
        and (not s["I1"] or v["z"] <= 1 + t)
        and (not s["I2"] or v["z"] >= 4 - t)
    )


_NESTED3_SELECTIONS = [
    {"A": 1, "B": 0, "M1": 1, "M2": 0, "I1": 1, "I2": 0},
    {"A": 1, "B": 0, "M1": 1, "M2": 0, "I1": 0, "I2": 1},
    {"A": 1, "B": 0, "M1": 0, "M2": 1, "I1": 0, "I2": 0},
    {"A": 0, "B": 1, "M1": 0, "M2": 0, "I1": 0, "I2": 0},
]


def _or_semantics():
    m = dm.Model("or_semantics")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 4], [x >= 2, x <= 6]], name="d", semantics="or")
    m.minimize(x)
    return m


def _or_indicators():
    """#1775 G1.4: OR with its indicators read -- both may be selected at once."""
    m = dm.Model("or_indicators")
    x = m.continuous("x", lb=0, ub=10)
    d = m.either_or([[x <= 4], [x >= 2, x <= 6]], name="d", semantics="or")
    d.indicators
    m.minimize(x)
    return m


def _or_indicators_member(v, t):
    a, b = round(v["d_0_active"]), round(v["d_1_active"])
    if a + b < 1:
        return False
    return (not a or v["x"] <= 4 + t) and (not b or 2 - t <= v["x"] <= 6 + t)


def _or_block():
    """#1775 G1.4: an OR ``add_disjunction`` block (``at_least(1)``)."""
    m = dm.Model("or_block")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("a")
    a.subject_to([x <= 4, y <= 5])
    b = m.make_disjunct("b")
    b.subject_to([x >= 2, x <= 6])
    m.add_disjunction([a, b], semantics="or")
    m.minimize(x + y)
    return m


def _or_block_member(v, t):
    a, b = round(v["a_active"]), round(v["b_active"])
    if a + b < 1:
        return False
    ok_a = v["x"] <= 4 + t and v["y"] <= 5 + t
    return (not a or ok_a) and (not b or 2 - t <= v["x"] <= 6 + t)


def _local_var():
    """``z`` appears only in the second disjunct; the first does not mention it."""
    m = dm.Model("local_var")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5, z >= x - 4, z <= 3]], name="d")
    m.minimize(x + z)
    return m


def _neg_univariate():
    """Univariate bound rows written with negative coefficients (spec §3.6)."""
    m = dm.Model("neg_univariate")
    x = m.continuous("x", lb=-5, ub=10)
    y = m.continuous("y", lb=-5, ub=10)
    m.either_or([[-2 * x <= -10, -y >= -1], [-x >= -1, -3 * y <= 6]], name="d")
    m.minimize(x + y)
    return m


def _separable_convex():
    m = dm.Model("separable_convex")
    x = m.continuous("x", lb=-1, ub=9)
    y = m.continuous("y", lb=-1, ub=9)
    m.either_or([[x * x + y * y <= 4], [(x - 6) * (x - 6) + (y - 6) * (y - 6) <= 4]], name="d")
    m.minimize(x + y)
    return m


def _non_separable():
    m = dm.Model("non_separable")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x * y >= 20], [x + y <= 2]], name="d")
    m.minimize(x + y)
    return m


def _domain_restricted():
    """``sqrt(x)`` sits in a disjunct that is inactive wherever ``x < 0``."""
    m = dm.Model("domain_restricted")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.sqrt(x) >= 2]], name="d")
    m.minimize(x)
    return m


def _domain_restricted_wrapped():
    """An undefined power folded to a finite enclosure: ``exp(-abs(x**0.5))`` in (0, 1].

    The row's own enclosure is finite, so only a per-node domain check sees that
    it is undefined for ``x < 0`` (#1779).
    """
    m = dm.Model("domain_restricted_wrapped")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.exp(-abs(x**0.5)) <= 0.5]], name="d")
    m.minimize(x)
    return m


def _block_disjunction():
    """``add_disjunction`` blocks: the indicators are user-visible model columns."""
    m = dm.Model("block_disjunction")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    d1 = m.make_disjunct("d1")
    d1.subject_to(x <= 2)
    d1.subject_to(y >= 3)
    d2 = m.make_disjunct("d2")
    d2.subject_to(x >= 5)
    m.add_disjunction([d1, d2], name="blk")
    m.minimize(x + y)
    return m


def _nested_blocks():
    """#1775 G1.2: a block disjunction nested in a Disjunct, plus a local variable.

    outer: A [x <= 3, inner: A1 [y >= 6] v A2 [y >= 8, x >= 2]] v B [x >= 7, z <= x - 5]
    with ``z`` local to B.
    """
    m = dm.Model("nested_blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a2 = m.make_disjunct("A2")
    a2.subject_to([y >= 8, x >= 2])
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    z = b.variable("z", lb=0, ub=10)
    b.subject_to([x >= 7, z <= x - 5])
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + y)
    return m


def _nested_blocks_member(v, t):
    a, b = round(v["A_active"]), round(v["B_active"])
    a1, a2 = round(v["A1_active"]), round(v["A2_active"])
    if a + b != 1 or a1 + a2 != a:
        return False
    ok = True
    if a:
        ok = ok and v["x"] <= 3 + t
    if a1:
        ok = ok and v["y"] >= 6 - t
    if a2:
        ok = ok and v["y"] >= 8 - t and v["x"] >= 2 - t
    if b:
        ok = ok and v["x"] >= 7 - t and v["z"] <= v["x"] - 5 + t
    return ok


#: Every valid selection of ``_nested_blocks`` (A with A1, A with A2, B) crossed
#: with a grid through its boundaries, so both directions are exercised on each.
_NESTED_GRID = [
    {"x": x, "y": y, "z": z, "A_active": a, "A1_active": a1, "A2_active": a2, "B_active": b}
    for a, a1, a2, b in ((1, 1, 0, 0), (1, 0, 1, 0), (0, 0, 0, 1))
    for x in (0.0, 2.0, 3.0, 3.5, 7.0, 8.0)
    for y in (0.0, 6.0, 7.0, 8.0)
    for z in (0.0, 3.0)
]


def _nested_in_block():
    """#1775 G1.2: a ``Model.disjunction`` nested in a Disjunct block."""
    m = dm.Model("nested_in_block")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a.subject_to(m.disjunction([[y >= 6], [y >= 8, x >= 2]], name="in"))
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + y)
    return m


def _nested_in_block_member(v, t):
    a, b = round(v["A_active"]), round(v["B_active"])
    if a + b != 1:
        return False
    if b:
        return v["x"] >= 7 - t
    inner = v["y"] >= 6 - t or (v["y"] >= 8 - t and v["x"] >= 2 - t)
    return v["x"] <= 3 + t and inner


def _block_member(v, t):
    a, b = round(v["d1_active"]), round(v["d2_active"])
    if a + b != 1:
        return False
    if a:
        return v["x"] <= 2 + t and v["y"] >= 3 - t
    return v["x"] >= 5 - t


def _logic_over_either_or():
    """``logical()`` over the indicators of an ``either_or`` (spec M4)."""
    m = dm.Model("logic_over_either_or")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    dx = m.either_or([[x <= 2], [x >= 5]], name="dx")
    dy = m.either_or([[y <= 2], [y >= 5]], name="dy")
    # x in its low mode forces y into its high mode.
    m.logical(dx.indicators[0].implies(dy.indicators[1]))
    m.minimize(x + y)
    return m


def _logic_member(v, t):
    """The indicators are model columns now: the point carries its selection."""
    sx = (round(v["dx_0_active"]), round(v["dx_1_active"]))
    sy = (round(v["dy_0_active"]), round(v["dy_1_active"]))
    if sum(sx) != 1 or sum(sy) != 1:
        return False
    x_ok = v["x"] <= 2 + t if sx[0] else v["x"] >= 5 - t
    y_ok = v["y"] <= 2 + t if sy[0] else v["y"] >= 5 - t
    return x_ok and y_ok and (not sx[0] or sy[1])


_UNBOUNDED_REFUSAL = (ValueError, r"(?i)bound")
_DOMAIN_REFUSAL = (ValueError, r"undefined on part of the variable box")

SHAPES: list[Shape] = [
    Shape("two_way", _two_way, lambda v, t: v["x"] <= 2 + t or v["x"] >= 5 - t),
    Shape(
        "multi_row",
        _multi_row,
        lambda v, t: (v["x"] <= 2 + t and v["y"] >= 3 - t) or (v["x"] >= 5 - t and v["y"] <= 4 + t),
    ),
    Shape(
        "multi_var",
        _multi_var,
        lambda v, t: (
            v["x"] + v["y"] <= 4 + t or (v["x"] - v["y"] >= 3 - t and v["x"] + 2 * v["y"] >= 12 - t)
        ),
    ),
    Shape(
        "equality",
        _equality,
        lambda v, t: (abs(v["x"] - 3) <= t and v["y"] <= 5 + t) or v["x"] >= 7 - t,
        points=[{"x": 3.0, "y": 1.0}, {"x": 3.0, "y": 5.0}, {"x": 3.0, "y": 6.0}],
    ),
    Shape(
        "three_way",
        _three_way,
        lambda v, t: v["x"] <= 1 + t or 3 - t <= v["x"] <= 4 + t or v["x"] >= 8 - t,
    ),
    Shape(
        "quadratic",
        _quadratic,
        lambda v, t: (
            (v["x"] <= 2 + t and v["y"] >= 3 - t)
            or (v["x"] >= 5 - t and v["x"] ** 2 + v["y"] <= 40 + t)
        ),
        points=[{"x": 5.5, "y": 9.75}, {"x": 6.0, "y": 4.0}, {"x": 6.0, "y": 5.0}],
        n_samples=30,
    ),
    Shape(
        "unbounded",
        _unbounded,
        lambda v, t: v["x"] <= 2 + t or v["x"] >= 5 - t,
        # big-M / mbigm / hull need a finite enclosure and refuse (spec §0.2 item
        # 5); simplex exists for exactly these rows (#1182) and must lower them,
        # and binary multiplication needs no bound on the body (#1775 G2.5).
        refuses={
            m: _UNBOUNDED_REFUSAL for m in METHODS if m not in ("simplex", "binary_multiplication")
        },
        box={"x": (0.0, 20.0)},
    ),
    Shape(
        "nested3",
        _nested3,
        _nested3_member,
        # Theorem 1 is stated over algebraic predicates; nesting is not one (#1182).
        refuses={"simplex": (Exception, r"nested disjunction is not an algebraic predicate")},
    ),
    Shape(
        "nested3_blocks",
        _nested3_blocks,
        _nested3_blocks_member,
        # Every consistent selection on a grid, so each disjunct's feasible region
        # is sampled (random indicators are rarely consistent).
        points=[
            {"x": x, "y": y, "z": z, **{f"{k}_active": a for k, a in sel.items()}}
            for sel in _NESTED3_SELECTIONS
            for x in (2.0, 5.0, 8.0)
            for y in (1.0, 4.0, 7.0)
            for z in (0.5, 2.5, 5.0)
        ],
    ),
    Shape(
        "or_semantics",
        _or_semantics,
        lambda v, t: v["x"] <= 6 + t,
        # #1775 G1.4: big-M lowers OR (auto routes it there; simplex is projection-
        # based); hull and mbigm refuse it, as Pyomo's do.
        refuses={m: (NotImplementedError, r"DisjunctionSemantics\.OR") for m in ("mbigm", "hull")},
    ),
    Shape(
        "or_indicators",
        _or_indicators,
        _or_indicators_member,
        points=[
            {"x": x, "d_0_active": a, "d_1_active": b}
            for x in (0.0, 2.0, 3.0, 4.0, 4.5, 6.0, 7.0)
            for a, b in ((1, 0), (0, 1), (1, 1), (0, 0))
        ],
        refuses={
            **{m: (NotImplementedError, r"DisjunctionSemantics\.OR") for m in ("mbigm", "hull")},
            "simplex": (ValueError, r"indicators were read"),
        },
    ),
    Shape(
        "or_block",
        _or_block,
        _or_block_member,
        points=[
            {"x": x, "y": y, "a_active": a, "b_active": b}
            for x in (0.0, 2.0, 3.0, 4.0, 4.5, 6.0, 7.0)
            for y in (0.0, 5.0)
            for a, b in ((1, 0), (0, 1), (1, 1), (0, 0))
        ],
    ),
    Shape(
        "local_var",
        _local_var,
        lambda v, t: (
            v["x"] <= 2 + t or (v["x"] >= 5 - t and v["z"] >= v["x"] - 4 - t and v["z"] <= 3 + t)
        ),
    ),
    Shape(
        "neg_univariate",
        _neg_univariate,
        lambda v, t: (
            (v["x"] >= 5 - t and v["y"] <= 1 + t) or (v["x"] <= 1 + t and v["y"] >= -2 - t)
        ),
    ),
    Shape(
        "separable_convex",
        _separable_convex,
        lambda v, t: (
            v["x"] ** 2 + v["y"] ** 2 <= 4 + t or (v["x"] - 6) ** 2 + (v["y"] - 6) ** 2 <= 4 + t
        ),
        points=[{"x": 0.0, "y": 0.0}, {"x": 6.0, "y": 7.0}, {"x": 3.0, "y": 3.0}],
        n_samples=30,
    ),
    Shape(
        "non_separable",
        _non_separable,
        lambda v, t: v["x"] * v["y"] >= 20 - t or v["x"] + v["y"] <= 2 + t,
        n_samples=30,
    ),
    Shape(
        "domain_restricted",
        _domain_restricted,
        lambda v, t: v["x"] <= -0.5 + t or (v["x"] >= 0 and math.sqrt(v["x"]) >= 2 - t),
        points=[{"x": -0.8}, {"x": -1.0}, {"x": 4.0}, {"x": 9.0}, {"x": 1.0}],
        # #1779: big-M, mbigm and simplex kept the sqrt row where its disjunct is
        # inactive, which imposed x >= 0 on the first disjunct (a false optimal).
        # Simplex and binary multiplication (#1775 G2.5), which keeps it too, refuse
        # it; big-M and mbigm split sqrt's operand into a column tied to x only
        # where the disjunct holds (_ArgumentDisaggregator). Hull, and auto (routed
        # to hull), lower it soundly.
        refuses={m: _DOMAIN_REFUSAL for m in ("simplex", "binary_multiplication")},
    ),
    Shape(
        "domain_restricted_wrapped",
        _domain_restricted_wrapped,
        lambda v, t: (
            v["x"] <= -0.5 + t or (v["x"] >= 0 and math.exp(-math.sqrt(v["x"])) <= 0.5 + t)
        ),
        points=[{"x": -0.8}, {"x": -1.0}, {"x": 1.0}, {"x": 9.0}],
        refuses={m: _DOMAIN_REFUSAL for m in ("simplex", "binary_multiplication")},
    ),
    Shape(
        "nested_blocks",
        _nested_blocks,
        _nested_blocks_member,
        n_samples=200,
        points=[
            {
                "x": 1.0,
                "y": 7.0,
                "z": 0.0,
                "A_active": 1,
                "A1_active": 1,
                "A2_active": 0,
                "B_active": 0,
            },
            {
                "x": 2.5,
                "y": 9.0,
                "z": 9.0,
                "A_active": 1,
                "A1_active": 0,
                "A2_active": 1,
                "B_active": 0,
            },
            # B selected: the inner block is off, so y is free.
            {
                "x": 8.0,
                "y": 0.0,
                "z": 2.0,
                "A_active": 0,
                "A1_active": 0,
                "A2_active": 0,
                "B_active": 1,
            },
            # B selected but an inner indicator on: violates sum(inner) == A.
            {
                "x": 8.0,
                "y": 7.0,
                "z": 2.0,
                "A_active": 0,
                "A1_active": 1,
                "A2_active": 0,
                "B_active": 1,
            },
            # A selected with no inner choice.
            {
                "x": 1.0,
                "y": 7.0,
                "z": 0.0,
                "A_active": 1,
                "A1_active": 0,
                "A2_active": 0,
                "B_active": 0,
            },
            *_NESTED_GRID,
        ],
    ),
    Shape(
        "nested_in_block",
        _nested_in_block,
        _nested_in_block_member,
        n_samples=120,
        points=[
            {"x": 1.0, "y": 7.0, "A_active": 1, "B_active": 0},
            {"x": 2.5, "y": 9.0, "A_active": 1, "B_active": 0},
            {"x": 8.0, "y": 0.0, "A_active": 0, "B_active": 1},
            {"x": 1.0, "y": 1.0, "A_active": 1, "B_active": 0},
        ],
    ),
    Shape(
        "block_disjunction",
        _block_disjunction,
        _block_member,
        points=[
            {"x": 1.0, "y": 5.0, "d1_active": 1.0, "d2_active": 0.0},
            {"x": 6.0, "y": 1.0, "d1_active": 0.0, "d2_active": 1.0},
            {"x": 6.0, "y": 1.0, "d1_active": 1.0, "d2_active": 0.0},
        ],
    ),
    Shape(
        "logic_over_either_or",
        _logic_over_either_or,
        _logic_member,
        points=[
            {
                "x": 1.0,
                "y": 6.0,
                "dx_0_active": 1,
                "dx_1_active": 0,
                "dy_0_active": 0,
                "dy_1_active": 1,
            },
            {
                "x": 6.0,
                "y": 1.0,
                "dx_0_active": 0,
                "dx_1_active": 1,
                "dy_0_active": 1,
                "dy_1_active": 0,
            },
            # x is in its low mode but y is not in its high one: the logic is violated.
            {
                "x": 1.0,
                "y": 1.0,
                "dx_0_active": 1,
                "dx_1_active": 0,
                "dy_0_active": 1,
                "dy_1_active": 0,
            },
            # The selection disagrees with x: dx_0 is selected but x is in disjunct 1.
            {
                "x": 6.0,
                "y": 6.0,
                "dx_0_active": 1,
                "dx_1_active": 0,
                "dy_0_active": 0,
                "dy_1_active": 1,
            },
        ],
        # Simplex has no selector binaries to bind indicators to (#1775 G1.1).
        refuses={"simplex": (ValueError, r"indicators were read")},
    ),
]

#: (shape, method) pairs that fail today but are planned to pass: the reason names
#: the task or open issue, and ``raises`` pins the failure, so neither an
#: unrelated harness error nor a silently-renamed API can keep the marker green.
#: Strict, so the marker fails the day the case starts passing.
XFAIL: dict[tuple[str, str], tuple[str, type[BaseException]]] = {}


def _params():
    for shape in SHAPES:
        for method in METHODS:
            xf = XFAIL.get((shape.name, method))
            marks = [pytest.mark.xfail(strict=True, reason=xf[0], raises=xf[1])] if xf else []
            yield pytest.param(shape, method, id=f"{shape.name}-{method}", marks=marks)


def test_battery_covers_every_method():
    from discopt._relax.gdp_reformulate import GDP_REFORMULATION_METHODS

    assert set(METHODS) == set(GDP_REFORMULATION_METHODS)


def test_xfail_table_names_real_cases():
    names = {s.name for s in SHAPES}
    assert all(s in names and m in METHODS for s, m in XFAIL)


@pytest.mark.parametrize("shape,method", list(_params()))
def test_battery(shape: Shape, method: str):
    rep = run_shape(shape, method)
    if rep.refused is not None:
        assert method in shape.refuses
        return
    # Prove the probe fired (CLAUDE.md §6): both directions were exercised.
    assert rep.feasible_checked > 0, rep
    assert rep.infeasible_checked > 0, rep
    assert rep.oracle_crosschecks == rep.comparisons, rep
    rep.assert_sound()


# ---------------------------------------------------------------------------
# The instrument detects what it claims to (CLAUDE.md §6): break a lowering on
# purpose and require the battery to say so, in each direction.
# ---------------------------------------------------------------------------


def test_harness_catches_a_dropped_hull_aggregation_row(monkeypatch):
    """The continuous (LP) path: without ``x = Σ v_k`` the hull no longer binds ``x``."""
    import discopt._relax.gdp_reformulate as gr

    real = gr._reformulate_disjunction_hull

    def drop_aggregation(dc, model, add_aux_binary, **kw):
        new_vars, new_cons = real(dc, model, add_aux_binary, **kw)
        # cons[0] is the selector row; cons[1] is the aggregation row of x.
        assert "_hull_" in str(new_cons[1].body) and new_cons[1].sense == "=="
        return new_vars, new_cons[:1] + new_cons[2:]

    monkeypatch.setattr(gr, "_reformulate_disjunction_hull", drop_aggregation)
    rep = run_shape(next(s for s in SHAPES if s.name == "two_way"), "hull")
    assert rep.admitted, rep
    with pytest.raises(AssertionError, match="infeasible point"):
        rep.assert_sound()


def test_harness_catches_a_too_small_big_m(monkeypatch):
    """A big-M a tenth of the valid one cuts feasible points of the active disjunct."""
    import discopt._relax.gdp_reformulate as gr

    real = gr._compute_big_m
    monkeypatch.setattr(gr, "_compute_big_m", lambda con, model, *a: 0.1 * real(con, model, *a))
    rep = run_shape(next(s for s in SHAPES if s.name == "two_way"), "big-m")
    assert rep.cut, rep
    with pytest.raises(AssertionError, match="feasible point"):
        rep.assert_sound()


def test_harness_catches_a_dropped_row(monkeypatch):
    """Dropping the second disjunct's rows admits points that satisfy neither disjunct."""
    import discopt._relax.gdp_reformulate as gr

    real = gr._reformulate_disjunction

    def drop_last_row(dc, model, add_aux_binary, **kw):
        new_vars, new_cons = real(dc, model, add_aux_binary, **kw)
        return new_vars, new_cons[:-1]

    monkeypatch.setattr(gr, "_reformulate_disjunction", drop_last_row)
    rep = run_shape(next(s for s in SHAPES if s.name == "two_way"), "big-m")
    assert rep.admitted, rep
    with pytest.raises(AssertionError, match="infeasible point"):
        rep.assert_sound()


def test_harness_refuses_a_predicate_that_disagrees_with_the_model():
    """A predicate for the wrong set is caught by the declared-model cross-check."""
    wrong = Shape("two_way_wrong", _two_way, lambda v, t: v["x"] <= 3 + t or v["x"] >= 5 - t)
    with pytest.raises(AssertionError, match="verify_point"):
        run_shape(wrong, "big-m")


@pytest.mark.parametrize("name", ["quadratic", "separable_convex", "non_separable"])
def test_nonlinear_hull_is_judged_without_a_solver(name):
    """The hull of a nonlinear disjunct goes through the affine-relaxation path."""
    rep = run_shape(next(s for s in SHAPES if s.name == name), "hull")
    assert rep.nonlinear_verdicts > 0, rep
    rep.assert_sound()
