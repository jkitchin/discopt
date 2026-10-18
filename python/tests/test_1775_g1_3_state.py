"""#1775 G1.3: fixing disjuncts -- the reduced subproblem and the current state.

Before G1.3 there was no way to fix a disjunct selection other than LOA's
internal big-M model with the integers fixed, which still evaluates every
inactive row (spec §5.2). ``discopt.gdp.state`` did not exist.

NC regime: nothing on the default solve path calls these functions; they are new
user-selected routes, registered as ``gdp.fix_disjuncts`` and
``gdp.transform_current_disjunctive_state``.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
from discopt.gdp import (
    gdp_tree,
    reduced_subproblem,
    reverse_current_disjunctive_state,
    transform_current_disjunctive_state,
)
from discopt.modeling.core import _DisjunctiveConstraint

pytestmark = pytest.mark.smoke


def _solve(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = m.solve(time_limit=60)
    assert r.status == "optimal", r.status
    return r.objective


def _either_or():
    """out: A [x <= 3, in: [y >= 6] v [y >= 8, x >= 2]] v B [x >= 7]; min x + y."""
    m = dm.Model("eo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    inner = m.disjunction([[y >= 6], [y >= 8, x >= 2]], name="in")
    m.either_or([[x <= 3, inner], [x >= 7]], name="out")
    m.minimize(x + y)
    return m


def _blocks():
    """The same set as Disjunct blocks (G1.2); min x + 3y."""
    m = dm.Model("blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a2 = m.make_disjunct("A2")
    a2.subject_to([y >= 8, x >= 2])
    a.add_disjunction([a1, a2])
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b])
    m.minimize(x + 3 * y)
    return m


def _selections(m):
    (out,) = gdp_tree(m).disjunctions
    a, b = out.disjuncts
    a1, a2 = a.children[0].disjuncts
    return {"A,A1": [a, a1], "A,A2": [a, a2], "B": [b]}


# Optima of each configuration, by hand:
#   either_or, min x + y:  A,A1 (0, 6) = 6;  A,A2 (2, 8) = 10;  B (7, 0) = 7
#   blocks,    min x + 3y: A,A1 18;          A,A2 26;           B 7
@pytest.mark.parametrize(
    "build, expected",
    [
        (_either_or, {"A,A1": 6.0, "A,A2": 10.0, "B": 7.0}),
        (_blocks, {"A,A1": 18.0, "A,A2": 26.0, "B": 7.0}),
    ],
)
def test_reduced_subproblem_solves_each_configuration(build, expected):
    m = build()
    rows_before = list(m._constraints)
    for key, sel in _selections(m).items():
        sub = reduced_subproblem(m, sel)
        assert not any(isinstance(c, _DisjunctiveConstraint) for c in sub._constraints)
        assert _solve(sub) == pytest.approx(expected[key], abs=1e-6), key
    # The input is untouched, and its own optimum is the best configuration.
    assert all(a is b for a, b in zip(m._constraints, rows_before))
    assert len(m._constraints) == len(rows_before)
    assert _solve(m) == pytest.approx(min(expected.values()), abs=1e-6)


def test_inactive_rows_are_dropped_not_relaxed():
    """Spec §5.2's reason to exist: the inactive disjunct is never evaluated.

    ``[x <= -0.5] v [sqrt(x) >= 2]`` on ``x in [-1, 10]``: the whole model refuses
    big-M (#1779), but the subproblem of the first disjunct has no sqrt row.
    """
    m = dm.Model("domain")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.sqrt(x) >= 2]], name="d")
    m.minimize(x)
    with pytest.raises(ValueError, match="undefined on part of the variable box"):
        m.solve(gdp_method="big-m")
    first, second = gdp_tree(m).disjunctions[0].disjuncts
    sub = reduced_subproblem(m, [first])
    assert [str(c) for c in sub._constraints] == ["(x - -0.5) <= 0.0"]
    assert _solve(sub) == pytest.approx(-1.0, abs=1e-6)
    assert _solve(reduced_subproblem(m, [second])) == pytest.approx(4.0, abs=1e-6)


def test_indicators_are_pinned_by_rows():
    m = _blocks()
    sel = _selections(m)["B"]
    sub = reduced_subproblem(m, sel)
    rows = [str(c) for c in sub._constraints]
    # The sum-to-parent and exactly(1) rows stay; every block row is gone except
    # B's; A, A1, A2 are pinned off and B on.
    assert rows == [
        "(Σ[2 terms] - A_active) == 0.0",
        "(Σ[2 terms] - 1) == 0.0",
        "(7 - x) <= 0.0",
        "(A_active - 0) == 0.0",
        "(A1_active - 0) == 0.0",
        "(A2_active - 0) == 0.0",
        "(B_active - 1) == 0.0",
    ]
    r = sub.solve(time_limit=60)
    assert round(float(r.x["B_active"])) == 1 and round(float(r.x["A1_active"])) == 0


def test_partial_state_promotes_the_undecided_child_and_reverses_exactly():
    m = _either_or()
    before = list(m._constraints)
    a = _selections(m)["A,A1"][0]
    token = transform_current_disjunctive_state(m, [a])
    # A's row is ordinary now; its nested disjunction is unconditional.
    kinds = [type(c).__name__ for c in m._constraints]
    assert kinds == ["Constraint", "_DisjunctiveConstraint"]
    assert _solve(m) == pytest.approx(6.0, abs=1e-6)
    token.reverse()
    assert len(m._constraints) == len(before)
    assert all(x is y for x, y in zip(m._constraints, before))
    assert _solve(m) == pytest.approx(6.0, abs=1e-6)


def test_reverse_refuses_after_further_changes():
    m = _either_or()
    token = transform_current_disjunctive_state(m, [_selections(m)["B"][0]])
    m.subject_to(m._variables[1] <= 9)
    with pytest.raises(ValueError, match="changed after"):
        token.reverse()


def test_current_state_reads_fixed_indicators():
    m = _blocks()
    a_ind = next(v for v in m._variables if v.name == "A_active")
    a1_ind = next(v for v in m._variables if v.name == "A1_active")
    with a_ind.fixed(1), a1_ind.fixed(1):
        token = transform_current_disjunctive_state(m)
        assert _solve(m) == pytest.approx(18.0, abs=1e-6)
        token.reverse()
    assert _solve(m) == pytest.approx(7.0, abs=1e-6)


def test_implication_can_be_selected():
    m = dm.Model("imp")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.if_then(z, [x >= 4])
    m.minimize(x)
    (imp,) = gdp_tree(m).disjunctions
    sub = reduced_subproblem(m, [imp.disjuncts[0]])
    assert _solve(sub) == pytest.approx(4.0, abs=1e-6)
    # Left unselected, an implication stays an indicator row.
    assert _solve(reduced_subproblem(m, [])) == pytest.approx(0.0, abs=1e-6)


def test_selection_errors():
    m = _either_or()
    (out,) = gdp_tree(m).disjunctions
    a, b = out.disjuncts
    a1 = a.children[0].disjuncts[0]
    with pytest.raises(ValueError, match="two disjuncts"):
        reduced_subproblem(m, [a, b])
    with pytest.raises(ValueError, match="select the enclosing disjunct"):
        reduced_subproblem(m, [b, a1])
    with pytest.raises(ValueError, match="needs one per active disjunction"):
        reduced_subproblem(m, [a])
    with pytest.raises(TypeError, match="DisjunctNode"):
        reduced_subproblem(m, [0])


def test_a_selection_resolves_on_a_copy():
    """Selections are addresses, so nodes from one model's tree work on its clone."""
    m = _blocks()
    sel = _selections(m)["A,A1"]
    assert _solve(reduced_subproblem(m.clone(), sel)) == pytest.approx(18.0, abs=1e-6)


def test_registered_transformations():
    assert {"gdp.fix_disjuncts", "gdp.transform_current_disjunctive_state"} <= set(dt.available())
    m = _blocks()
    sel = _selections(m)["B"]
    copy = dt.create_using("gdp.fix_disjuncts", m, selected=sel)
    assert _solve(copy) == pytest.approx(7.0, abs=1e-6)
    report = dt.check("gdp.fix_disjuncts", m, selected=sel)
    assert report.diff.removed_constraints and report.diff.added_constraints

    rows = list(m._constraints)
    dt.apply_to("gdp.transform_current_disjunctive_state", m, selected=_selections(m)["A,A1"])
    assert _solve(m) == pytest.approx(18.0, abs=1e-6)
    reverse_current_disjunctive_state(m)
    assert all(x is y for x, y in zip(m._constraints, rows)) and len(m._constraints) == len(rows)
    with pytest.raises(ValueError, match="no transform_current_disjunctive_state"):
        reverse_current_disjunctive_state(m)
