"""#1775 G1.4: OR (at-least-one) disjunctions.

Before G1.4 ``DisjunctionSemantics.OR`` was declarable but every lowering refused
it, and ``add_disjunction`` refused it at model-build time. ``OR`` projects onto
the same ``x`` set as ``SELECT_ONE``; the two differ in the selector space, which
matters as soon as an indicator is read (priced, fixed or used in logic): under
OR several disjuncts may be selected together.

After: big-M (and ``auto``, which routes OR there) lowers OR with an
``at_least(1)`` selector row; a nested OR block gets ``sum >= parent`` plus
``y_k <= parent``; simplex lowers an OR ``either_or`` whose indicators were never
read (it is projection-based); hull and mbigm refuse it, as Pyomo's do, and
``add_disjunction`` blocks fall back to big-M indicator rows under them.

NC regime: SELECT_ONE models are lowered bit-identically (BN panel, 75/75).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_advisor import recommend_methods
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp import gdp_tree, reduced_subproblem
from discopt.modeling.core import DisjunctionSemantics

pytestmark = pytest.mark.smoke

_OR_REFUSAL = r"DisjunctionSemantics\.OR"


def _solve(m, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal", (method, r.status)
    return r.objective


def _or_max():
    """``[x <= 4] v [2 <= x <= 6]``, max x: 6 under either semantics."""
    m = dm.Model("or_max")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 4], [x >= 2, x <= 6]], name="d", semantics="or")
    m.maximize(x)
    return m


def _or_ind():
    """Selecting both pays: x=4 with both (4+3+3 = 10) beats x=6 alone (9)."""
    m = dm.Model("or_ind")
    x = m.continuous("x", lb=0, ub=10)
    d = m.either_or([[x <= 4], [x >= 2, x <= 6]], name="d", semantics="or")
    y = d.indicators
    m.maximize(x + 3 * y[0].variable + 3 * y[1].variable)
    return m


def _or_block():
    m = dm.Model("or_block")
    x = m.continuous("x", lb=0, ub=10)
    a = m.make_disjunct("a")
    a.subject_to(x <= 4)
    b = m.make_disjunct("b")
    b.subject_to([x >= 2, x <= 6])
    m.add_disjunction([a, b], semantics="or")
    m.maximize(x + 3 * a.indicator.variable + 3 * b.indicator.variable)
    return m


def _or_nested():
    """A [x <= 3, (A1 [y <= 5] OR A2 [y >= 4])] xor B [x >= 8].

    max y + 3 A1 + 3 A2 - x: A with A2 alone (y=10) gives 13; A with both
    (y <= 5) gives 11; B gives 2 and must pin A1 = A2 = 0 (else 8).
    """
    m = dm.Model("or_nested")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y <= 5)
    a2 = m.make_disjunct("A2")
    a2.subject_to(y >= 4)
    a.add_disjunction([a1, a2], semantics="or")
    b = m.make_disjunct("B")
    b.subject_to(x >= 8)
    m.add_disjunction([a, b])
    m.maximize(y + 3 * a1.indicator.variable + 3 * a2.indicator.variable - x)
    return m


# Under SELECT_ONE or_ind and or_block give 9 (x=6, one indicator).
@pytest.mark.parametrize("method", ["big-m", "auto", "simplex"])
def test_or_either_or_solves(method):
    assert _solve(_or_max(), method) == pytest.approx(6.0, abs=1e-6)


@pytest.mark.parametrize("method", ["big-m", "auto"])
def test_or_lets_both_indicators_be_selected(method):
    assert _solve(_or_ind(), method) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("method", ["big-m", "mbigm", "hull", "auto", "simplex"])
def test_or_block_solves_under_every_method(method):
    assert _solve(_or_block(), method) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("method", ["big-m", "mbigm", "hull", "auto"])
def test_nested_or_block_is_bounded_by_its_parent(method):
    assert _solve(_or_nested(), method) == pytest.approx(13.0, abs=1e-6)


@pytest.mark.parametrize("method", ["mbigm", "hull"])
def test_hull_and_mbigm_refuse_an_or_either_or(method):
    with pytest.raises(NotImplementedError, match=_OR_REFUSAL):
        _or_max().solve(gdp_method=method, time_limit=60)


def test_simplex_refuses_read_or_indicators_and_names_big_m_only():
    with pytest.raises(ValueError, match=r"indicators were read.*Use 'big-m'\.$"):
        _or_ind().solve(gdp_method="simplex", time_limit=60)


def test_big_m_selector_row_is_at_least_one():
    low = reformulate_gdp(_or_max(), "big-m")
    assert str(low._constraints[0]) == ("((_gdp_aux_disj_d_0_0 + _gdp_aux_disj_d_1_1) - 1) >= 0.0")


def test_tree_and_advisor_carry_or():
    (node,) = gdp_tree(_or_block()).disjunctions
    assert node.semantics is DisjunctionSemantics.OR
    (advice,) = recommend_methods(_or_max())
    assert advice.recommendation == "big-m"


def test_reduced_subproblem_accepts_several_selected_disjuncts_under_or():
    m = _or_max()
    d0, d1 = gdp_tree(m).disjunctions[0].disjuncts
    assert _solve(reduced_subproblem(m, [d0, d1]), "big-m") == pytest.approx(4.0, abs=1e-6)


def test_serialize_keeps_an_or_block():
    """Refused before schema 1.7 (plan WP-3b-2): written as select-one, an OR block
    would reload with a different selector space. The semantics now travel."""
    from discopt.serialize import dumps, loads

    m = _or_block()
    text = dumps(m, provenance=False)
    reloaded = loads(text)
    assert dumps(reloaded, provenance=False) == text
    (root,) = gdp_tree(reloaded).disjunctions
    assert root.semantics is DisjunctionSemantics.OR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1, r2 = m.solve(time_limit=60), reloaded.solve(time_limit=60)
    assert (r1.status, r1.objective, r1.node_count) == (r2.status, r2.objective, r2.node_count)


def test_rowless_or_block_keeps_its_semantics_in_the_tree():
    """An OR block whose disjuncts hold only nested disjunctions emits no tagged
    row; its tree node is rebuilt from the emission record (#1790), which must
    carry OR too -- else ``reduced_subproblem`` refuses selecting both."""
    m = dm.Model("rowless_or")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a1 = m.make_disjunct("A1")
    a1.subject_to(y <= 5)
    a2 = m.make_disjunct("A2")
    a2.subject_to(y >= 4)
    a.add_disjunction([a1, a2])
    b = m.make_disjunct("B")
    b1 = m.make_disjunct("B1")
    b1.subject_to(x >= 8)
    b.add_disjunction([b1])
    m.add_disjunction([a, b], semantics="or")
    m.maximize(y + 3 * a.indicator.variable + 3 * b.indicator.variable - x)
    (root,) = gdp_tree(m).disjunctions
    assert root.semantics is DisjunctionSemantics.OR
    # Both selected: A2 (y = 10) and B1 (x = 8) give 10 + 6 - 8 = 8.
    na, nb = root.disjuncts
    sel = [na, na.children[0].disjuncts[1], nb, nb.children[0].disjuncts[0]]
    assert _solve(reduced_subproblem(m, sel), "big-m") == pytest.approx(8.0, abs=1e-6)
    # A with A2 alone: y = 10, x = 0 gives 13.
    for method in ("big-m", "hull", "mbigm", "auto"):
        assert _solve(m, method) == pytest.approx(13.0, abs=1e-6), method


# ── #1795 review: a partial OR selection must not pin the rest to 0 ─────────


def _or_fixable():
    m = dm.Model("or_fix")
    x = m.continuous("x", lb=0, ub=10)
    a = m.make_disjunct("a")
    a.subject_to(x <= 4)
    b = m.make_disjunct("b")
    b.subject_to([x >= 2, x <= 6])
    m.add_disjunction([a, b], semantics="or")
    m.maximize(x + 3 * a.indicator.variable + 3 * b.indicator.variable)
    return m, a, b


@pytest.mark.smoke
def test_partial_or_transform_is_refused_not_narrowed():
    """Fixing ``a`` leaves ``b`` undecided under OR (optimum 10 with a=b=1); the
    transform used to pin ``b`` to 0 and certify 7."""
    from discopt.gdp import transform_current_disjunctive_state
    from discopt.transformations import model_fingerprint

    m, a, b = _or_fixable()
    a.indicator.variable.lb = 1.0
    a.indicator.variable.ub = 1.0
    before = model_fingerprint(m)
    with pytest.raises(ValueError, match="only some disjuncts of the OR disjunction"):
        transform_current_disjunctive_state(m)
    node_a = gdp_tree(m).disjunctions[0].disjuncts[0]
    with pytest.raises(ValueError, match="only some disjuncts of the OR disjunction"):
        transform_current_disjunctive_state(m, [node_a])
    assert model_fingerprint(m) == before
    assert _solve(m, "big-m") == pytest.approx(10.0)


@pytest.mark.smoke
def test_full_or_transform_and_reduced_subproblem_read_the_exact_set():
    from discopt.gdp import transform_current_disjunctive_state

    m, a, b = _or_fixable()
    root = gdp_tree(m).disjunctions[0]
    na, nb = root.disjuncts
    for sel, want in (([na], 7.0), ([nb], 9.0), ([na, nb], 10.0)):
        assert _solve(reduced_subproblem(m, sel), "big-m") == pytest.approx(want)
    token = transform_current_disjunctive_state(m, [na, nb])
    assert _solve(m, "big-m") == pytest.approx(10.0)
    token.reverse()
    assert _solve(m, "big-m") == pytest.approx(10.0)
