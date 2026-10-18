"""#1775 G1.1: the GDP tree and ``either_or`` indicators.

Before G1.1 the selectors of an ``either_or`` were created inside the GDP pass, so
``logical()`` could not reference them (spec M4): there was no ``indicators``
attribute, and ``either_or`` returned ``None``. Each consumer also re-derived the
disjunctive structure from ``model._constraints`` itself.

Bound-neutrality (BN): a model that never reads ``indicators`` must lower exactly
as before. ``test_unread_indicators_lower_bit_identically`` pins the lowered
variables and rows against a literal record of the pre-G1.1 lowering.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_advisor import recommend_methods
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp import gdp_tree
from discopt.modeling.core import _DisjunctiveConstraint
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke


def _two(m=None):
    m = m or dm.Model("two")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    return m, x, y


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def test_either_or_returns_its_disjunction_with_indicators():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    assert isinstance(d, _DisjunctiveConstraint) and d is m._constraints[-1]
    n_vars, n_cons = len(m._variables), len(m._constraints)
    inds = d.indicators
    assert [b.variable.name for b in inds] == ["d_0_active", "d_1_active"]
    # Reading allocates the columns and their exactly(1) row, once.
    assert len(m._variables) == n_vars + 2 and len(m._constraints) == n_cons + 1
    assert [b.variable for b in d.indicators] == [b.variable for b in inds]
    assert len(m._variables) == n_vars + 2


def test_unread_indicators_allocate_nothing():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    assert d._indicator_vars is None
    gdp_tree(m)  # the tree is read-only
    assert d._indicator_vars is None and len(m._variables) == 2


def test_anonymous_disjunction_indicator_names_use_its_position():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]])
    assert [b.variable.name for b in d.indicators] == [
        "disjunction0_0_active",
        "disjunction0_1_active",
    ]


def test_nested_disjunction_has_no_indicators_yet():
    m, x, y = _two()
    inner = m.disjunction([[y <= 1], [y >= 4]], name="inner")
    m.either_or([[x <= 2, inner], [x >= 5]], name="outer")
    with pytest.raises(ValueError, match="G1.2"):
        inner.indicators


def test_indicator_name_collision_refuses():
    m, x, _ = _two()
    m.continuous("d_0_active", lb=0, ub=1)
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    with pytest.raises(ValueError, match="already used"):
        d.indicators
    assert d._indicator_vars is None


@pytest.mark.parametrize("method", ["big-m", "mbigm", "hull", "auto"])
def test_logic_over_either_or_indicators_solves(method):
    """Spec M4. ``min x + y`` with ``x<=2 => y>=5``: optimum 5, either way round.

    Fails before G1.1: ``either_or`` returned ``None`` and had no ``indicators``.
    """
    m, x, y = _two()
    dx = m.either_or([[x <= 2], [x >= 5]], name="dx")
    dy = m.either_or([[y <= 2], [y >= 5]], name="dy")
    m.logical(dx.indicators[0].implies(dy.indicators[1]))
    # Forbid the x-high branch, so the logic must be honoured: x <= 2, so y >= 5.
    m.logical(~dx.indicators[1])
    m.minimize(x + y)
    r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)
    assert float(r.x["y"]) >= 5.0 - 1e-6
    assert round(float(r.x["dx_0_active"])) == 1 and round(float(r.x["dy_1_active"])) == 1


def test_indicators_priced_in_the_objective():
    """The indicator is the selection: pricing it changes which disjunct is chosen."""
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    m.minimize(-x + 10 * d.indicators[1].variable)
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(-2.0, abs=1e-6)


def test_simplex_refuses_read_indicators():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    d.indicators
    with pytest.raises(ValueError, match="indicators were read"):
        reformulate_gdp(m, "simplex")


# ---------------------------------------------------------------------------
# Bound-neutrality of the lowering
# ---------------------------------------------------------------------------


def _lowered_record(method):
    m, x, y = _two()
    inner = m.disjunction([[y <= 1], [y >= 4]], name="inner")
    m.either_or([[x <= 2, inner], [x >= 5, x + y <= 12]], name="d")
    m.either_or([[x <= 3], [y >= 6], [x + y >= 15]])
    low = reformulate_gdp(m, method)
    return [v.name for v in low._variables], [str(c) for c in low._constraints]


#: The lowered rows of ``_lowered_record``'s model, recorded on ``origin/main`` at
#: 229e8d1 (before G1.1) with ``str(row)``.
_PRE_G1_1_ROWS: dict[str, list[str]] = {
    "auto": [
        "((_gdp_aux_disj_d_0_0 + _gdp_aux_disj_d_1_1) - 1) == 0.0",
        "((x - 2) - (8.08 * (1 - _gdp_aux_disj_d_0_0))) <= 0.0",
        "((_gdp_aux_nested_inner_0_2 + _gdp_aux_nested_inner_1_3) - _gdp_aux_disj_d_0_0) == 0.0",
        "((y - 1) - (9.09 * (1 - _gdp_aux_nested_inner_0_2))) <= 0.0",
        "((4 - y) - (4.04 * (1 - _gdp_aux_nested_inner_1_3))) <= 0.0",
        "((5 - x) - (5.05 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0",
        "(((x + y) - 12) - (8.08 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0",
        "(((_gdp_aux_hull_anon_0_4 + _gdp_aux_hull_anon_1_5) + _gdp_aux_hull_anon_2_6) - 1) == 0.0",
        "(x - ((_hull_anon_v_x_0 + _hull_anon_v_x_1) + _hull_anon_v_x_2)) == 0.0",
        "(y - ((_hull_anon_v_y_0 + _hull_anon_v_y_1) + _hull_anon_v_y_2)) == 0.0",
        "(_hull_anon_v_x_0 - (3 * _gdp_aux_hull_anon_0_4)) <= 0.0",
        "(_hull_anon_v_x_0 - (0 * _gdp_aux_hull_anon_0_4)) >= 0.0",
        "(_hull_anon_v_y_0 - (10 * _gdp_aux_hull_anon_0_4)) <= 0.0",
        "(_hull_anon_v_y_0 - (0 * _gdp_aux_hull_anon_0_4)) >= 0.0",
        "(_hull_anon_v_x_1 - (10 * _gdp_aux_hull_anon_1_5)) <= 0.0",
        "(_hull_anon_v_x_1 - (0 * _gdp_aux_hull_anon_1_5)) >= 0.0",
        "(_hull_anon_v_y_1 - (10 * _gdp_aux_hull_anon_1_5)) <= 0.0",
        "(_hull_anon_v_y_1 - (6 * _gdp_aux_hull_anon_1_5)) >= 0.0",
        "(_hull_anon_v_x_2 - (10 * _gdp_aux_hull_anon_2_6)) <= 0.0",
        "(_hull_anon_v_x_2 - (0 * _gdp_aux_hull_anon_2_6)) >= 0.0",
        "(_hull_anon_v_y_2 - (10 * _gdp_aux_hull_anon_2_6)) <= 0.0",
        "(_hull_anon_v_y_2 - (0 * _gdp_aux_hull_anon_2_6)) >= 0.0",
        "((_hull_anon_v_x_0 - (3 * _gdp_aux_hull_anon_0_4)) - (0 * _gdp_aux_hull_anon_0_4)) <= 0.0",
        "(((6 * _gdp_aux_hull_anon_1_5) - _hull_anon_v_y_1) - (0 * _gdp_aux_hull_anon_1_5)) <= 0.0",
        "(((15 * _gdp_aux_hull_anon_2_6) - (_hull_anon_v_x_2 + _hull_anon_v_y_2)) - (0 * "
        "_gdp_aux_hull_anon_2_6)) <= 0.0",
    ],
    "big-m": [
        "((_gdp_aux_disj_d_0_0 + _gdp_aux_disj_d_1_1) - 1) == 0.0",
        "((x - 2) - (8.08 * (1 - _gdp_aux_disj_d_0_0))) <= 0.0",
        "((_gdp_aux_nested_inner_0_2 + _gdp_aux_nested_inner_1_3) - _gdp_aux_disj_d_0_0) == 0.0",
        "((y - 1) - (9.09 * (1 - _gdp_aux_nested_inner_0_2))) <= 0.0",
        "((4 - y) - (4.04 * (1 - _gdp_aux_nested_inner_1_3))) <= 0.0",
        "((5 - x) - (5.05 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0",
        "(((x + y) - 12) - (8.08 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0",
        "(((_gdp_aux_disj_anon_0_4 + _gdp_aux_disj_anon_1_5) + _gdp_aux_disj_anon_2_6) - 1) == 0.0",
        "((x - 3) - (7.07 * (1 - _gdp_aux_disj_anon_0_4))) <= 0.0",
        "((6 - y) - (6.06 * (1 - _gdp_aux_disj_anon_1_5))) <= 0.0",
        "((15 - (x + y)) - (15.15 * (1 - _gdp_aux_disj_anon_2_6))) <= 0.0",
    ],
    "mbigm": [
        "((_gdp_aux_disj_d_0_0 + _gdp_aux_disj_d_1_1) - 1) == 0.0",
        "((x - 2) - (8.08 * (1 - _gdp_aux_disj_d_0_0))) <= 0.0",
        "((_gdp_aux_nested_inner_0_2 + _gdp_aux_nested_inner_1_3) - _gdp_aux_disj_d_0_0) == 0.0",
        "((y - 1) - (9.09 * (1 - _gdp_aux_nested_inner_0_2))) <= 0.0",
        "((4 - y) - (4.04 * (1 - _gdp_aux_nested_inner_1_3))) <= 0.0",
        "((5 - x) - (5.05 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0",
        "(((x + y) - 12) - (8.08 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0",
        "(((_gdp_aux_disj_anon_0_4 + _gdp_aux_disj_anon_1_5) + _gdp_aux_disj_anon_2_6) - 1) == 0.0",
        "((x - 3) - (7.07 * (1 - _gdp_aux_disj_anon_0_4))) <= 0.0",
        "((6 - y) - (6.06 * (1 - _gdp_aux_disj_anon_1_5))) <= 0.0",
        "((15 - (x + y)) - (15.15 * (1 - _gdp_aux_disj_anon_2_6))) <= 0.0",
    ],
}


@pytest.mark.parametrize("method", sorted(_PRE_G1_1_ROWS))
def test_unread_indicators_lower_bit_identically(method):
    """Selector names, order and every row are those of the pre-G1.1 lowering."""
    names, rows = _lowered_record(method)
    assert names[:9] == [
        "x",
        "y",
        "_gdp_aux_disj_d_0_0",
        "_gdp_aux_disj_d_1_1",
        "_gdp_aux_nested_inner_0_2",
        "_gdp_aux_nested_inner_1_3",
        f"_gdp_aux_{'hull' if method == 'auto' else 'disj'}_anon_0_4",
        f"_gdp_aux_{'hull' if method == 'auto' else 'disj'}_anon_1_5",
        f"_gdp_aux_{'hull' if method == 'auto' else 'disj'}_anon_2_6",
    ]
    assert rows == _PRE_G1_1_ROWS[method]


def test_read_indicators_replace_the_generated_selectors():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    cols = [b.variable for b in d.indicators]
    low = reformulate_gdp(m, "big-m")
    assert not any(v.name.startswith("_gdp_aux_") for v in low._variables)
    assert all(any(v is c for v in low._variables) for c in cols)
    rows = [str(c) for c in low._constraints]
    # The exactly(1) row is the model's own, not a second generated one.
    assert sum(r.endswith("== 0.0") for r in rows) == 1


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def test_tree_maps_all_three_apis():
    m, x, y = _two()
    inner = m.disjunction([[y <= 1], [y >= 4]], name="inner")
    d = m.either_or([[x <= 2, inner], [x >= 5]], name="d")
    a = m.make_disjunct("a")
    a.subject_to(x <= 1)
    a.subject_to(y <= 1)
    b = m.make_disjunct("b")
    b.subject_to(x >= 8)
    m.add_disjunction([a, b], name="blk")
    z = m.binary("z")
    m.if_then(z, [x <= 4, y <= 4], name="imp")
    m.logical(a.indicator.implies(~b.indicator))

    t = gdp_tree(m)
    kinds = [r.kind for r in t.disjunctions]
    assert kinds == ["either_or", "add_disjunction", "implication"]
    eo, blk, imp = t.disjunctions
    assert t.node_for(d) is eo
    assert [len(dj.rows) for dj in eo.disjuncts] == [1, 1]
    (child,) = eo.disjuncts[0].children
    assert child.name == "inner" and child.parent is eo.disjuncts[0]
    assert all(dj.indicator is None for dj in eo.disjuncts)
    assert [dj.indicator.variable.name for dj in blk.disjuncts] == ["a_active", "b_active"]
    assert [len(dj.rows) for dj in blk.disjuncts] == [2, 1]
    (only,) = imp.disjuncts
    assert only.indicator.variable is z and only.active_value == 1 and len(only.rows) == 2
    assert len(t.logic) == 1
    assert set(t.indicators) == {*blk.disjuncts, only}
    assert [n.name for n in t.walk()] == ["d", "inner", blk.name, imp.name]


def test_tree_sees_read_indicators():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    inds = d.indicators
    node = gdp_tree(m).node_for(d)
    assert [dj.indicator.variable for dj in node.disjuncts] == [b.variable for b in inds]


def test_advisor_reads_the_tree_with_unchanged_indices():
    m, x, y = _two()
    m.subject_to(x + y <= 15)
    m.either_or([[x <= 2], [x >= 5]], name="d0")
    a = m.make_disjunct("a")
    a.subject_to(y <= 1)
    b = m.make_disjunct("b")
    b.subject_to(y >= 8)
    m.add_disjunction([a, b], name="blk")
    m.either_or([[y <= 3], [y >= 6]], name="d1")
    advice = recommend_methods(m)
    idx = [i for i, c in enumerate(m._constraints) if isinstance(c, _DisjunctiveConstraint)]
    assert [a.disjunction_index for a in advice] == idx


# ---------------------------------------------------------------------------
# The verifier reads the selection
# ---------------------------------------------------------------------------


def test_declared_logic_holds_the_selected_disjunct():
    """With read indicators, the point's own selection must hold, not just any disjunct."""
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    d.indicators
    m.minimize(x)
    # columns: x, y, d_0_active, d_1_active
    assert verify_point(m, np.array([6.0, 0.0, 0.0, 1.0])).ok
    bad = verify_point(m, np.array([6.0, 0.0, 1.0, 0.0]))
    assert not bad.ok and "disjunction" in bad.reason


def test_serialization_keeps_the_indicator_link(tmp_path):
    from discopt.serialize import load, save

    m, x, y = _two()
    dx = m.either_or([[x <= 2], [x >= 5]], name="dx")
    dy = m.either_or([[y <= 2], [y >= 5]], name="dy")
    # The format carries no logic rows (pre-existing), so state it algebraically.
    m.subject_to(dx.indicators[0].variable <= dy.indicators[1].variable)
    m.subject_to(dx.indicators[1].variable == 0)
    m.minimize(x + y)
    path = tmp_path / "m.json"
    save(m, path)
    m2 = load(path)
    d2 = next(c for c in m2._constraints if isinstance(c, _DisjunctiveConstraint))
    assert d2._indicator_vars is not None
    assert [v.name for v in d2._indicator_vars] == ["dx_0_active", "dx_1_active"]
    assert d2._indicator_vars[0] is m2._variables[2]
    r = m2.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(5.0, abs=1e-6)


def test_clone_keeps_the_indicator_link():
    m, x, _ = _two()
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    d.indicators
    c = m.clone()
    d2 = next(k for k in c._constraints if isinstance(k, _DisjunctiveConstraint))
    assert d2._model is c
    assert all(any(v is w for w in c._variables) for v in d2._indicator_vars)
