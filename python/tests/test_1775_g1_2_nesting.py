"""#1775 G1.2: nested disjunctions and local variables in ``Disjunct`` blocks.

Before G1.2 a ``Disjunct`` held algebraic rows only (spec M2/M3): there was no
``Disjunct.add_disjunction`` or ``Disjunct.variable``, and nesting existed only
inside ``either_or``. A nested block disjunction is active only when its enclosing
disjunct is selected -- its indicators sum to the parent's -- and a disjunction
from ``Model.disjunction`` passed to ``Disjunct.subject_to`` applies under the
same condition.

NC regime: a model without nesting is lowered exactly as before, pinned below
against a literal record of the pre-G1.2 rows and lowerings.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp import gdp_tree
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke

METHODS = ["big-m", "mbigm", "hull", "auto", "simplex"]


def _xy(name):
    m = dm.Model(name)
    return m, m.continuous("x", lb=0, ub=10), m.continuous("y", lb=0, ub=10)


def _nested_blocks():
    """outer: A [x <= 3, inner: A1 [y >= 6] v A2 [y >= 8, x >= 2]] v B [x >= 7]."""
    m, x, y = _xy("nested_blocks")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a2 = m.make_disjunct("A2")
    a2.subject_to([y >= 8, x >= 2])
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    return m, x, y


def _nested_disjunction():
    """The same set, with the inner disjunction from ``Model.disjunction``."""
    m, x, y = _xy("nested_dc")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a.subject_to(m.disjunction([[y >= 6], [y >= 8, x >= 2]], name="in"))
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    return m, x, y


def _only_nested():
    """A holds nothing but a nested block: A1 [y >= 6] v A2 [x >= 5, y >= 1]; B [x >= 7]."""
    m, x, y = _xy("only_nested")
    a = m.make_disjunct("A")
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a2 = m.make_disjunct("A2")
    a2.subject_to([x >= 5, y >= 1])
    a.add_disjunction([a1, a2])
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b])
    return m, x, y


# Optima by enumerating the selections by hand:
#   x + y:   A&A1 (0, 6) = 6,  A&A2 (2, 8) = 10, B (7, 0) = 7   -> 6
#   x - y:   A&A1 (0, 10) = -10, A&A2 (2, 10) = -8, B (7, 10) = -3 -> -10
#   x + 3y:  A&A1 18, A&A2 26, B 7 -> 7. Enforcing the inner block while B is
#            selected would force y >= 6 there: 25.
CASES = [
    (_nested_blocks, lambda x, y: x + y, 6.0),
    (_nested_blocks, lambda x, y: x - y, -10.0),
    (_nested_blocks, lambda x, y: x + 3 * y, 7.0),
    (_nested_disjunction, lambda x, y: x + y, 6.0),
    (_nested_disjunction, lambda x, y: x + 3 * y, 7.0),
    # x + 2y: A&A1 12, A&A2 (5, 1) = 7, B 7 -> 7;  2x + y: A1 6, A2 11, B 14 -> 6
    (_only_nested, lambda x, y: x + 2 * y, 7.0),
    (_only_nested, lambda x, y: 2 * x + y, 6.0),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("build, obj, truth", CASES)
def test_nested_solves_to_the_enumerated_optimum(build, obj, truth, method):
    """Fails before G1.2: ``Disjunct`` had no ``add_disjunction``, and a disjunction
    passed to ``subject_to`` reached ``if_then`` as if it were a row."""
    m, x, y = build()
    m.minimize(obj(x, y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # hull/simplex: nested rows on big-M
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_local_variable(method):
    """``z`` is declared on A: x >= z + 1 with z >= 3 there; B: x >= 6. min x is 4."""
    m, x, _ = _xy("local")
    a = m.make_disjunct("A")
    z = a.variable("z", lb=0, ub=10)
    a.subject_to([z >= 3, x >= z + 1])
    b = m.make_disjunct("B")
    b.subject_to(x >= 6)
    m.add_disjunction([a, b])
    m.minimize(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(4.0, abs=1e-6)
    assert a.local_vars == [z]


def test_local_variable_used_outside_its_disjunct_is_refused():
    m, x, _ = _xy("escape")
    a = m.make_disjunct("A")
    z = a.variable("z", lb=0, ub=10)
    a.subject_to(x >= z)
    b = m.make_disjunct("B")
    b.subject_to(x >= 6)
    m.add_disjunction([a, b])
    m.subject_to(z <= 4)  # outside A
    m.minimize(x)
    with pytest.raises(ValueError, match="declared local to disjunct 'A'"):
        m.solve()


def test_tree_attaches_nested_disjunctions_under_their_disjunct():
    m, _, _ = _nested_blocks()
    roots = gdp_tree(m).disjunctions
    (outer,) = roots
    a, b = outer.disjuncts
    (inner,) = a.children
    assert inner.parent is a and not b.children
    assert [d.indicator.variable.name for d in inner.disjuncts] == ["A1_active", "A2_active"]

    m, _, _ = _nested_disjunction()
    (outer,) = gdp_tree(m).disjunctions
    (inner,) = outer.disjuncts[0].children
    assert inner.kind == "either_or" and inner.name == "in"


def test_tree_finds_the_indicator_of_a_block_with_no_rows():
    """A holds only a nested block, so no row carries its indicator."""
    m, _, _ = _only_nested()
    (outer,) = gdp_tree(m).disjunctions
    assert [d.indicator.variable.name for d in outer.disjuncts] == ["A_active", "B_active"]
    assert len(outer.disjuncts[0].children) == 1


def test_declared_logic_reads_the_nesting():
    m, _, _ = _nested_blocks()
    m.minimize(m._variables[0])
    # columns: x, y, A, A1, A2, B
    col = {v.name: i for i, v in enumerate(m._variables)}

    def point(**kw):
        p = np.zeros(len(m._variables))
        for k, v in kw.items():
            p[col[k]] = v
        return p

    # B selected: the inner block is off, so y = 0 is fine.
    assert verify_point(m, point(x=8, y=0, B_active=1)).ok
    # A and A1 selected, rows hold.
    assert verify_point(m, point(x=1, y=7, A_active=1, A1_active=1)).ok
    # A selected with no inner choice: the sum row A1 + A2 == A fails.
    assert not verify_point(m, point(x=1, y=7, A_active=1)).ok
    # B selected but an inner indicator on: the sum row fails.
    assert not verify_point(m, point(x=8, y=7, B_active=1, A1_active=1)).ok
    # A and A1 selected but y < 6.
    assert not verify_point(m, point(x=1, y=2, A_active=1, A1_active=1)).ok

    m, _, _ = _nested_disjunction()
    m.minimize(m._variables[0])
    # columns: x, y, A, B. The nested disjunction applies only when A is selected.
    assert verify_point(m, np.array([8.0, 0.0, 0.0, 1.0])).ok
    assert not verify_point(m, np.array([1.0, 0.0, 1.0, 0.0])).ok
    assert verify_point(m, np.array([1.0, 7.0, 1.0, 0.0])).ok


def test_guards():
    m, x, y = _xy("guards")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    inner = m.disjunction([[y <= 1], [y >= 6]])
    a.subject_to(inner)
    with pytest.raises(ValueError, match="nested twice"):
        a.subject_to(inner)
    with pytest.raises(ValueError, match="cannot be nested"):
        a.add_disjunction([a])
    m.add_disjunction([a, b])
    # Anything added after add_disjunction lowered the block would be lost.
    with pytest.raises(ValueError, match="already passed to add_disjunction"):
        a.subject_to(y <= 5)
    with pytest.raises(ValueError, match="already passed to add_disjunction"):
        a.variable("late")
    # Its own indicators are not separate columns: nest Disjunct blocks for those.
    with pytest.raises(ValueError, match="nested in a Disjunct"):
        inner.indicators


def test_unattached_block_holding_only_a_nested_disjunction_is_reported():
    m, x, y = _xy("orphan")
    a = m.make_disjunct("A")
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a.add_disjunction([a1, m.make_disjunct("A2")])
    m.minimize(x)
    with pytest.raises(ValueError):
        m.solve()


def test_serialize_and_clone_keep_nesting(tmp_path):
    """Refused before schema 1.7 (plan WP-3b-2); now written, with the parent links
    and the block-disjunction record, so the reload solves to the same optimum."""
    from discopt.serialize import load, save

    m, x, y = _nested_blocks()
    m.minimize(x + 3 * y)
    save(m, tmp_path / "m.json")
    for copy in (load(tmp_path / "m.json"), m.clone()):
        r = copy.solve(time_limit=60)
        assert r.status == "optimal" and r.objective == pytest.approx(7.0, abs=1e-6)


def _flat():
    m, x, y = _xy("flat")
    a = m.make_disjunct("a")
    a.subject_to(x <= 2)
    a.subject_to(y >= 3)
    b = m.make_disjunct("b")
    b.subject_to(x >= 5)
    c = m.make_disjunct("c")
    m.add_disjunction([a, b, c], name="blk")
    d = m.make_disjunct("d")
    d.subject_to(y <= 1)
    e = m.make_disjunct("e")
    e.subject_to(y >= 8)
    m.add_disjunction([d, e])
    m.minimize(x + y)
    return m


#: ``_flat``'s declared rows and lowerings, recorded on ``origin/main`` at 9e6843ec
#: (before G1.2).
_PRE_G1_2: dict = {
    "big-m": [
        ["x", "y", "a_active", "b_active", "c_active", "d_active", "e_active"],
        [
            "((x - 2) - (8.08 * (1 - a_active))) <= 0.0",
            "((3 - y) - (3.03 * (1 - a_active))) <= 0.0",
            "((5 - x) - (5.05 * (1 - b_active))) <= 0.0",
            "(Σ[3 terms] - 1) == 0.0",
            "((y - 1) - (9.09 * (1 - d_active))) <= 0.0",
            "((8 - y) - (8.08 * (1 - e_active))) <= 0.0",
            "(Σ[2 terms] - 1) == 0.0",
        ],
    ],
    "declared": [
        "IND a_active 1 (x - 2) <= 0.0 ('add_disjunction:blk:0', 0, 3)",
        "IND a_active 1 (3 - y) <= 0.0 ('add_disjunction:blk:0', 0, 3)",
        "IND b_active 1 (5 - x) <= 0.0 ('add_disjunction:blk:0', 1, 3)",
        "(Σ[3 terms] - 1) == 0.0",
        "IND d_active 1 (y - 1) <= 0.0 ('add_disjunction::1', 0, 2)",
        "IND e_active 1 (8 - y) <= 0.0 ('add_disjunction::1', 1, 2)",
        "(Σ[2 terms] - 1) == 0.0",
    ],
    "hull": [
        [
            "x",
            "y",
            "a_active",
            "b_active",
            "c_active",
            "d_active",
            "e_active",
            "_hull_add_disjunction__1_v_y_0",
            "_hull_add_disjunction__1_v_y_1",
        ],
        [
            "((x - 2) - (8.08 * (1 - a_active))) <= 0.0",
            "((3 - y) - (3.03 * (1 - a_active))) <= 0.0",
            "((5 - x) - (5.05 * (1 - b_active))) <= 0.0",
            "(Σ[3 terms] - 1) == 0.0",
            "(y - (_hull_add_disjunction__1_v_y_0 + _hull_add_disjunction__1_v_y_1)) == 0.0",
            "(_hull_add_disjunction__1_v_y_0 - (1 * d_active)) <= 0.0",
            "(_hull_add_disjunction__1_v_y_0 - (0 * d_active)) >= 0.0",
            "(_hull_add_disjunction__1_v_y_1 - (10 * e_active)) <= 0.0",
            "(_hull_add_disjunction__1_v_y_1 - (8 * e_active)) >= 0.0",
            "((_hull_add_disjunction__1_v_y_0 - (1 * d_active)) - (0 * d_active)) <= 0.0",
            "(((8 * e_active) - _hull_add_disjunction__1_v_y_1) - (0 * e_active)) <= 0.0",
            "(Σ[2 terms] - 1) == 0.0",
        ],
    ],
    "mbigm": [
        ["x", "y", "a_active", "b_active", "c_active", "d_active", "e_active"],
        [
            "((x - 2) - (8.08 * (1 - a_active))) <= 0.0",
            "((3 - y) - (3.03 * (1 - a_active))) <= 0.0",
            "((5 - x) - (5.05 * (1 - b_active))) <= 0.0",
            "(Σ[3 terms] - 1) == 0.0",
            "((y - 1) - (9.09 * (1 - d_active))) <= 0.0",
            "((8 - y) - (8.08 * (1 - e_active))) <= 0.0",
            "(Σ[2 terms] - 1) == 0.0",
        ],
    ],
}


def test_a_model_without_nesting_is_unchanged():
    m = _flat()
    declared = [
        str(c)
        if not hasattr(c, "constraint")
        else f"IND {c.indicator.name} {c.active_value} {c.constraint} {c.disjunction}"
        for c in m._constraints
    ]
    assert declared == _PRE_G1_2["declared"]
    assert not hasattr(m, "_gdp_block_members")
    for method in ("big-m", "hull", "mbigm"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            low = reformulate_gdp(_flat(), method)
        names, rows = _PRE_G1_2[method]
        assert [v.name for v in low._variables] == names
        assert [str(c) for c in low._constraints] == rows
