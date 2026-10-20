"""#1775 G2.1: ``gdp_method="hull"`` lowers nested disjunctions by a nested hull.

Before G2.1 the hull refused a disjunction nested in an ``either_or`` disjunct
(``NotImplementedError``), and lowered every disjunction nested in a ``Disjunct``
block -- a block from ``Disjunct.add_disjunction``, or a ``Model.disjunction``
passed to ``Disjunct.subject_to`` -- by big-M instead, with a ``UserWarning``.

After: the hull is built from the leaves up (Pyomo's ``gdp.hull``). The child is
hulled on its own selectors, and its rows become rows of the parent disjunct,
where they are hulled again; the child's selectors and disaggregated copies are
local to that disjunct. So every test below runs with ``UserWarning`` as an
error: a nested row that silently fell back to big-M fails it.

NC regime (spec §0.3): a model without nesting is lowered exactly as before,
which ``test_1775_g1_2_nesting.py::test_a_model_without_nesting_is_unchanged``
pins literally. Soundness is ``test_gdp_differential_battery.py``: ``nested3``
(``either_or``) and ``nested3_blocks`` (Disjunct blocks) under hull.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
import test_1775_g1_2_nesting as g12
import test_1790_g1_2_followups as f1790
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.modeling.core import _DisjunctiveConstraint

pytestmark = pytest.mark.smoke


def _strict_hull(m):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return reformulate_gdp(m, "hull")


def _strict_solve(m):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return m.solve(gdp_method="hull", time_limit=60)


def _nested3():
    m = dm.Model("nested3")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    inner = _DisjunctiveConstraint(disjuncts=[[z <= 1], [z >= 4]], name="inner")
    mid = _DisjunctiveConstraint(disjuncts=[[y <= 2, inner], [y >= 6]], name="mid")
    m.either_or([[x <= 3, mid], [x >= 7]], name="outer")
    return m, x, y, z


def _circles():
    """Nonlinear nesting: y <= 2 and (a circle at (1,1) or at (3,1)), or a circle
    at (2, 3.5); every circle has radius 0.5."""
    m = dm.Model("circles")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    inner = _DisjunctiveConstraint(
        disjuncts=[[(x - 1) ** 2 + (y - 1) ** 2 <= 0.25], [(x - 3) ** 2 + (y - 1) ** 2 <= 0.25]],
        name="inner",
    )
    m.either_or([[y <= 2, inner], [(x - 2) ** 2 + (y - 3.5) ** 2 <= 0.25]], name="outer")
    return m, x, y


# nested3, by enumerating the selections by hand (x, y, z each in [0, 10]):
#   x+y+z:  outer0&mid0&inner0 (0,0,0) = 0
#   -x-y-z: outer1 (10,10,10) = -30
#   x-y+z:  outer0&mid1 (0,10,0) = -10 ; outer1 (7,10,0) = -3 -> -10
#   -x+y-z: outer1 (10,0,10) = -20; outer0&mid0&inner1 (3,0,10) = -13 -> -20
#   x + (y-4)^2 + (z-2.5)^2: outer0&mid0 (0,2,1 or 4) = 4 + 2.25; outer0&mid1
#           (0,6,2.5) = 4 -> 4
NESTED3 = [
    (lambda x, y, z: x + y + z, 0.0),
    (lambda x, y, z: -x - y - z, -30.0),
    (lambda x, y, z: x - y + z, -10.0),
    (lambda x, y, z: -x + y - z, -20.0),
    (lambda x, y, z: x + (y - 4) ** 2 + (z - 2.5) ** 2, 4.0),
]

# circles: min x+y at the (1,1) circle, 2 - sqrt(0.5); min -y at the top circle,
# -4; min x-y at the top circle, 2 - 0.5 sqrt(2)... checked against big-M below.
CIRCLES = [
    lambda x, y: x + y,
    lambda x, y: -x - y,
    lambda x, y: -y,
    lambda x, y: x - y,
    lambda x, y: -x + 0.1 * y,
]


@pytest.mark.parametrize("obj, truth", NESTED3)
def test_either_or_nesting_is_hulled(obj, truth):
    """Fails before G2.1: ``NotImplementedError: the hull reformulation does not
    support nested disjunctions``."""
    m, x, y, z = _nested3()
    m.minimize(obj(x, y, z))
    r = _strict_solve(m)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("obj", CIRCLES)
def test_nonlinear_nesting_matches_big_m(obj):
    """The perspective of a child's perspective rows: same certified optimum as
    big-M, which has lowered nesting since G1.2."""
    m, x, y = _circles()
    m.minimize(obj(x, y))
    r = _strict_solve(m)
    m_b, x_b, y_b = _circles()
    m_b.minimize(obj(x_b, y_b))
    ref = m_b.solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal" and ref.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


@pytest.mark.parametrize(
    "build, obj, truth",
    g12.CASES + f1790.CASES,
    ids=[f"{b.__name__}-{i}" for i, (b, _, _) in enumerate(g12.CASES + f1790.CASES)],
)
def test_block_nesting_is_hulled(build, obj, truth):
    """Fails before G2.1: the nested block (or ``Model.disjunction`` in a block)
    went to big-M with ``UserWarning: gdp_method='hull' does not apply to ...``."""
    m, x, y = build()
    m.minimize(obj(x, y))
    r = _strict_solve(m)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


def test_the_lowering_holds_the_child_hull_and_no_big_m_row():
    m, x, y = g12._nested_blocks()
    m.minimize(x + y)
    low = _strict_hull(m)
    names = {v.name for v in low._variables}
    # The inner block's copies of y exist: it was hulled inside A ...
    assert any(n.startswith("_hull_add_disjunction_outer_") and "_inner_" in n for n in names)
    # ... and nothing was lowered by the nested big-M routine.
    assert not any(
        (getattr(c, "name", None) or "").startswith("_gdp_nested") for c in low._constraints
    )


def test_a_child_selector_is_bounded_by_its_parent():
    """The re-hulled select row: with A off, no inner disjunct may be on."""
    m, x, y = g12._nested_blocks()
    m.minimize(x + 3 * y)  # optimum 7 selects B; enforcing inner would cost 25
    r = _strict_solve(m)
    assert r.status == "optimal" and r.objective == pytest.approx(7.0, abs=1e-6)
    v = {var.name: float(r.value(var)) for var in m._variables if var.name.endswith("_active")}
    assert v["B_active"] == pytest.approx(1.0) and v["A1_active"] == pytest.approx(0.0)
    assert v["A2_active"] == pytest.approx(0.0)


def test_or_child_stays_on_big_m_and_says_so():
    """OR nested in a block is a big-M feature only (#1775 G1.4); the hull lowers
    its parent and leaves the child to big-M, with the warning, as before."""
    m, x, y = g12._xy("or_child")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1, a2 = m.make_disjunct("A1"), m.make_disjunct("A2")
    a1.subject_to(y >= 6)
    a2.subject_to(y >= 8)
    a.add_disjunction([a1, a2], name="inner", semantics="or")
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + y)
    with pytest.warns(UserWarning, match="does not apply"):
        r = m.solve(gdp_method="hull", time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(6.0, abs=1e-6)


def _sibling():
    """outer: A [x <= 3, inner: A1 [y >= 6] v A2 [y >= 8]] v B [x >= 7, x + y_A1 <= 9].

    B reads the indicator of a block nested in A. Enumerated by hand (x, y in
    [0, 10]; y_A1 = 0 whenever B holds): A&A1 x <= 3, y >= 6; A&A2 x <= 3, y >= 8;
    B x in [7, 9], y free.
    """
    m, x, y = g12._xy("sibling")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1, a2 = m.make_disjunct("A1"), m.make_disjunct("A2")
    a1.subject_to(y >= 6)
    a2.subject_to(y >= 8)
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to([x >= 7, x + a1.indicator.variable <= 9])
    m.add_disjunction([a, b], name="outer")
    return m, x, y


SIBLING = [
    (lambda x, y: x + y, 6.0),  # A&A1 (0, 6)
    (lambda x, y: -x - y, -19.0),  # B (9, 10)
    (lambda x, y: -x + 2 * y, -9.0),  # B (9, 0)
    (lambda x, y: x - 3 * y, -30.0),  # A (0, 10)
    (lambda x, y: (x - 2) ** 2 + (y - 3) ** 2, 9.0),  # A&A1 (2, 6)
]


@pytest.mark.parametrize("obj, truth", SIBLING)
def test_a_child_indicator_in_a_sibling_disjunct_is_hulled(obj, truth):
    """A nested block's indicator is local to its disjunct in the hull; read by a
    sibling disjunct too, it is shared there instead, and disaggregated (exact: it
    is 0 whenever the sibling holds). Fails before: ``ValueError: ... also appears
    in disjunct 1. Use gdp_method='big-m'`` (and before G2.1 the shape solved, on
    big-M with a ``UserWarning``)."""
    m, x, y = _sibling()
    m.minimize(obj(x, y))
    r = _strict_solve(m)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


def test_a_sibling_read_indicator_is_disaggregated():
    """The outer hull gives ``A1_active`` (read by B) a copy in each disjunct."""
    m, x, y = _sibling()
    low = _strict_hull(m)
    copies = {v.name for v in low._variables if "_v_" in v.name}
    assert any(n.endswith("_v_A1_active_0") for n in copies), sorted(copies)
    assert any(n.endswith("_v_A1_active_1") for n in copies), sorted(copies)


def test_a_grandchild_indicator_is_local_to_the_outer_disjunct():
    """The indicators of a block two levels down are local to A at the outer level,
    like the child's own: no copy anywhere. Fails before: they were disaggregated at
    the outer level, which only loosened the relaxation (B's copy was free in
    ``[0, y_B]``, decoupling the indicator from A's copy of it)."""
    m, x, y = g12._xy("grandchild")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1, a2 = m.make_disjunct("A1"), m.make_disjunct("A2")
    a2.subject_to(y <= 1)
    a1a, a1b = m.make_disjunct("A1a"), m.make_disjunct("A1b")
    a1a.subject_to(y >= 6)
    a1b.subject_to(y >= 8)
    a1.add_disjunction([a1a, a1b], name="gc")
    a.add_disjunction([a1, a2], name="ch")
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    low = _strict_hull(m)
    names = {v.name for v in low._variables}
    for ind in ("A1a_active", "A1b_active", "A1_active", "A2_active"):
        assert not any(f"_v_{ind}_" in n for n in names), (ind, sorted(names))


@pytest.mark.parametrize(
    "build", [g12._nested_blocks, g12._nested_disjunction, f1790._two_rowless_levels]
)
def test_transformations_check_passes_on_nested_models(build):
    """Spec §0.2 item 4: the lowering passes ``transformations.check`` in isolation
    (the input model is untouched; the copy is lowered)."""
    from discopt import transformations

    m, x, y = build()
    m.minimize(x + y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rep = transformations.check("gdp.hull", m)
    assert rep.result is not m
    assert any(v.name.startswith("_hull_") for v in rep.result._variables)
