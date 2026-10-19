"""#1794: follow-ups to #1775 G1.3 (``discopt.gdp.state``) from the #1791 review.

Before the fix:

* B1 -- rows were dropped by the identity of the *wrapped* constraint, so a
  ``Constraint`` reused by another disjunction or an ``if_then`` lost that row
  too, and the reduced or transformed model certified ``optimal 1.0`` where the
  answer is 5.
* B2 -- ``reduced_subproblem`` rebuilt the model from ``_constraints`` alone and
  lost the fast-API rows (``optimal 1.0`` where the answer is 6), the fast-API
  objective, and the complementarity records.
* S1 -- a partial transform of nested structure left a model that ``validate``
  or ``gdp_tree`` refused, or (after #1790) a tree reporting the decided
  disjunction as still open.
* S2 -- a selection resolved by address alone, so a stale node or one from
  another model silently named a different disjunction.
* S3 -- ``_gdp_block_members`` was left describing rows the transform removed.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.gdp import gdp_tree, reduced_subproblem, transform_current_disjunctive_state
from discopt.gdp import state as st
from discopt.gdp.tree import Origin

pytestmark = pytest.mark.smoke


def _solve(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60)
    assert r.status == "optimal", r.status
    return r.objective


# ── B1: a shared Constraint object ─────────────────────────────────────────


def _shared_with_if_then():
    m = dm.Model("b1")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    c = x >= 5
    a = m.make_disjunct("A")
    a.subject_to(c)
    b = m.make_disjunct("B")
    b.subject_to(x >= 1)
    m.add_disjunction([a, b], name="D1")
    m.if_then(z, [c])
    m.subject_to(z >= 1)
    m.minimize(x)
    return m


def test_reduced_subproblem_keeps_an_if_then_row_sharing_a_disjunct_constraint():
    m = _shared_with_if_then()
    assert _solve(m) == pytest.approx(5.0, abs=1e-6)
    d1 = next(n for n in gdp_tree(m).disjunctions if n.kind == "add_disjunction")
    assert _solve(reduced_subproblem(m, [d1.disjuncts[1]])) == pytest.approx(5.0, abs=1e-6)


def test_selecting_the_implication_keeps_the_block_row_it_shares():
    m = _shared_with_if_then()
    imp = next(n for n in gdp_tree(m).disjunctions if n.kind == "implication")
    tok = transform_current_disjunctive_state(m, [imp.disjuncts[0]])
    assert _solve(m) == pytest.approx(5.0, abs=1e-6)
    tok.reverse()


def test_partial_transform_keeps_another_block_row_sharing_the_constraint():
    m = dm.Model("b1c")
    x = m.continuous("x", lb=0, ub=10)
    c = x >= 5
    a = m.make_disjunct("A")
    a.subject_to(c)
    b = m.make_disjunct("B")
    b.subject_to(x >= 1)
    m.add_disjunction([a, b], name="D1")
    p = m.make_disjunct("P")
    p.subject_to(c)
    q = m.make_disjunct("Q")
    q.subject_to(x >= 2)
    m.add_disjunction([p, q], name="D2")
    m.subject_to(p.indicator.variable >= 1)
    m.minimize(x)
    tok = transform_current_disjunctive_state(m, [gdp_tree(m).disjunctions[0].disjuncts[1]])
    assert _solve(m) == pytest.approx(5.0, abs=1e-6)
    tok.reverse()


# ── B2: what a fresh Model has to carry ────────────────────────────────────


def test_reduced_subproblem_keeps_fast_api_rows():
    m = dm.Model("b2")
    x = m.continuous("x", lb=0, ub=10, shape=(2,))
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([6.0]))
    m.either_or([[x[0] >= 1], [x[1] >= 1]], name="d")
    m.minimize(x[0] + x[1])
    sub = reduced_subproblem(m, [gdp_tree(m).disjunctions[0].disjuncts[0]])
    assert _solve(sub) == pytest.approx(6.0, abs=1e-6)


def test_reduced_subproblem_keeps_a_fast_api_objective():
    m = dm.Model("b2o")
    x = m.continuous("x", lb=0, ub=10, shape=(2,))
    m.either_or([[x[0] >= 3], [x[1] >= 4]], name="d")
    m.add_linear_objective(np.array([1.0, 1.0]), x)
    sub = reduced_subproblem(m, [gdp_tree(m).disjunctions[0].disjuncts[0]])
    assert _solve(sub) == pytest.approx(3.0, abs=1e-6)


def test_reduced_subproblem_carries_complementarities_and_blocks():
    m = dm.Model("b2c")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.complementarity(x, y)
    m.subject_to(x + y >= 2)
    m.either_or([[x >= 1], [x <= 9]], name="d")
    m.minimize(x + y)
    t = gdp_tree(m)
    sel = [n.disjuncts[0] for n in t.disjunctions]
    sub = reduced_subproblem(m, sel)
    assert len(sub._complementarities) == len(m._complementarities) == 1
    assert sub._complementarities[0] is m._complementarities[0]
    assert sub._gdp_factory_disjuncts == m._gdp_factory_disjuncts


def test_a_forgotten_disjunction_is_refused_not_dropped():
    m = dm.Model("float")
    x = m.continuous("x", lb=0, ub=10)
    m.disjunction([[x >= 4], [x >= 6]], name="lost")
    m.either_or([[x >= 1], [x >= 2]], name="d")
    m.minimize(x)
    with pytest.raises(ValueError, match="never added"):
        reduced_subproblem(m, [gdp_tree(m).disjunctions[0].disjuncts[0]])


# ── S1: partial transforms of nested structure ─────────────────────────────


def _either_or():
    m = dm.Model("eo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    inner = m.disjunction([[y >= 6], [y >= 8, x >= 2]], name="in")
    m.either_or([[x <= 3, inner], [x >= 7]], name="out")
    m.minimize(x + y)
    return m


@pytest.mark.parametrize("which, expected", [("B", 7.0), ("A,A2", 10.0)])
def test_partial_transform_consumes_a_nested_model_disjunction(which, expected):
    m = _either_or()
    (out,) = gdp_tree(m).disjunctions
    a, b = out.disjuncts
    sel = [b] if which == "B" else [a, a.children[0].disjuncts[1]]
    tok = transform_current_disjunctive_state(m, sel)
    assert _solve(m) == pytest.approx(expected, abs=1e-6)
    tok.reverse()
    assert m._gdp_factory_disjunctions and _solve(m) == pytest.approx(6.0, abs=1e-6)


def _blocks(nested: str):
    m = dm.Model("blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    if nested == "block":
        a1 = m.make_disjunct("A1")
        a1.subject_to(y >= 6)
        a2 = m.make_disjunct("A2")
        a2.subject_to([y >= 8, x >= 2])
        a.add_disjunction([a1, a2])
    else:
        a.subject_to(m.disjunction([[y >= 6], [y >= 8, x >= 2]], name="in"))
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b])
    m.minimize(x + 3 * y)
    return m


@pytest.mark.parametrize("nested", ["block", "dc"])
def test_deciding_a_block_parent_promotes_its_undecided_child(nested):
    m = _blocks(nested)
    rows = list(m._constraints)
    (root,) = gdp_tree(m).disjunctions
    inner = [d.name for d in root.disjuncts[0].children[0].disjuncts]
    tok = transform_current_disjunctive_state(m, [root.disjuncts[0]])
    # Was refused (parent owns no disjunction); with #1790's row-less nodes, the
    # decided A|B instead survived as a root and the undecided child vanished.
    (promoted,) = gdp_tree(m).disjunctions
    assert promoted.parent is None
    assert [d.name for d in promoted.disjuncts] == inner
    assert _solve(m) == pytest.approx(18.0, abs=1e-6)
    tok.reverse()
    assert all(p is q for p, q in zip(m._constraints, rows)) and len(m._constraints) == len(rows)
    assert _solve(m) == pytest.approx(7.0, abs=1e-6)


# ── S2: a selection must name this model's current structure ───────────────


def test_a_stale_selection_is_refused_after_rows_move():
    m = dm.Model("s2")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x >= 1], [x >= 2]], name="D0")
    m.either_or([[y >= 5], [y >= 9]], name="D1")
    m.minimize(x + y)
    d0_1 = gdp_tree(m).disjunctions[0].disjuncts[1]
    transform_current_disjunctive_state(m, [d0_1])
    with pytest.raises(ValueError, match="current GDP tree"):
        reduced_subproblem(m, [d0_1])


def test_a_selection_from_another_model_is_refused():
    m1 = dm.Model("a")
    x = m1.continuous("x", lb=0, ub=10)
    m1.either_or([[x >= 1], [x >= 2]], name="P")
    m2 = dm.Model("b")
    y = m2.continuous("y", lb=0, ub=10)
    m2.either_or([[y >= 5], [y >= 9]], name="Q")
    m2.minimize(y)
    with pytest.raises(ValueError, match="current GDP tree"):
        reduced_subproblem(m2, [gdp_tree(m1).disjunctions[0].disjuncts[1]])


# ── S3: the nesting record and unique root indices ─────────────────────────


def test_transform_keeps_the_block_member_record_in_step():
    m = _blocks("block")
    members = dict(m._gdp_block_members)
    (root,) = gdp_tree(m).disjunctions
    tok = transform_current_disjunctive_state(m, [root.disjuncts[1]])
    # B selected: the outer block and the block nested in A are both gone.
    assert m._gdp_block_members == {}
    tok.reverse()
    assert m._gdp_block_members == members


def test_resolve_tells_apart_two_roots_at_one_index():
    m = dm.Model("dup")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x >= 1], [x >= 2]], name="D0")
    m.either_or([[x >= 3], [x >= 4]], name="D1")
    tree = gdp_tree(m)
    r0, r1 = tree.disjunctions
    r1.origin = Origin(r0.origin.index, r1.origin.source)
    # Same index, different structure: each resolves to itself, never the other.
    assert st._resolve(tree, r1.disjuncts[1]) is r1.disjuncts[1]
    assert st._resolve(tree, r0.disjuncts[0]) is r0.disjuncts[0]


def test_resolve_refuses_two_identical_roots_at_one_index():
    m = dm.Model("dup2")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x >= 1], [x >= 2]], name="D")
    m.either_or([[x >= 1], [x >= 2]], name="D")
    tree = gdp_tree(m)
    r0, r1 = tree.disjunctions
    r1.origin = Origin(r0.origin.index, r1.origin.source)
    with pytest.raises(ValueError, match="identical disjunctions start at model row"):
        st._resolve(tree, r0.disjuncts[0])


# ── S4: the registry path's token and Model.save ───────────────────────────


def test_save_refuses_a_model_holding_an_unreversed_transform(tmp_path):
    import discopt.transformations as dt
    from discopt.gdp import reverse_current_disjunctive_state
    from discopt.serialize import SerializationError

    m = dm.Model("s4")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x >= 4], [x >= 7]], name="d")
    m.minimize(x)
    sel = [gdp_tree(m).disjunctions[0].disjuncts[1]]
    dt.apply_to("gdp.transform_current_disjunctive_state", m, selected=sel)
    with pytest.raises(SerializationError, match="_gdp_state_tokens"):
        m.save(tmp_path / "m.dopt")
    reverse_current_disjunctive_state(m)
    m.__dict__.pop("_gdp_state_tokens")
    m.save(tmp_path / "m.dopt")
