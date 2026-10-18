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


def _or_semantics():
    m = dm.Model("or_semantics")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 4], [x >= 2, x <= 6]], name="d", semantics="or")
    m.minimize(x)
    return m


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
    m.either_or([[x <= 2], [x >= 5]], name="dx")
    m.either_or([[y <= 2], [y >= 5]], name="dy")
    dx, dy = m._constraints[-2], m._constraints[-1]
    # x in its low mode forces y into its high mode.
    m.logical(dx.indicators[0].implies(dy.indicators[1]))
    m.minimize(x + y)
    return m


_UNBOUNDED_REFUSAL = (ValueError, r"(?i)bound")

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
        # 5); simplex exists for exactly these rows (#1182) and must lower them.
        refuses={m: _UNBOUNDED_REFUSAL for m in METHODS if m != "simplex"},
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
        "or_semantics",
        _or_semantics,
        lambda v, t: v["x"] <= 6 + t,
        # G1.4 implements OR for big-M only; hull and mbigm refuse it, as Pyomo does,
        # and so does auto, which dispatches among them.
        refuses={
            m: (NotImplementedError, r"DisjunctionSemantics\.OR") for m in ("mbigm", "hull", "auto")
        },
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
        lambda v, t: (
            (v["x"] >= 5 - t or v["y"] >= 5 - t)
            and (v["x"] <= 2 + t or v["x"] >= 5 - t)
            and (v["y"] <= 2 + t or v["y"] >= 5 - t)
        ),
    ),
]

#: (shape, method) pairs that refuse today but are planned to lower; each names the
#: task or issue that lifts the refusal. Strict, so the marker fails when it lands.
XFAIL: dict[tuple[str, str], str] = {
    ("nested3", "hull"): "hull + nesting: #1124, #1775 G2.1",
    ("or_semantics", "big-m"): "OR (AT_LEAST_ONE) semantics under big-M: #1124, #1775 G1.4",
    **{
        ("domain_restricted", m): "sqrt row in an inactive disjunct cuts it; verifier NaN: #1779"
        for m in METHODS
    },
    **{
        ("logic_over_either_or", m): "logical() over either_or indicators: #1775 G1.1 (M4)"
        for m in METHODS
    },
}


def _params():
    for shape in SHAPES:
        for method in METHODS:
            reason = XFAIL.get((shape.name, method))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
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

    def drop_last_row(dc, model, add_aux_binary, lp_data=None):
        new_vars, new_cons = real(dc, model, add_aux_binary, lp_data=lp_data)
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
