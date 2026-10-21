"""#1775 G2.5: ``gdp_method="binary_multiplication"`` (Pyomo's ``gdp.binary_multiplication``).

Before: ``"binary_multiplication"`` was an unknown method; ``reformulate_gdp`` and
``Model.solve`` refused it, and no ``gdp.binary_multiplication`` transformation
existed.

After (NC: a new, user-selected lowering; no existing route changes): each disjunct
row ``body (sense) 0`` becomes ``body * y (sense) 0`` on its selector -- and an
indicator row with ``active_value == 0`` uses ``1 - y``. That is exact for a binary
``y`` and needs no bound on ``body``, so it lowers rows big-M refuses for an
unbounded enclosure. A body undefined on part of the box is refused, as under big-M,
because it stays in the model where its disjunct is inactive (#1779). Soundness on
every shape is checked by ``test_gdp_differential_battery`` (``METHODS`` includes it).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.modeling.core import _DisjunctiveConstraint

pytestmark = pytest.mark.smoke

BM = "binary_multiplication"


def _slabs():
    m = dm.Model("slabs")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 2, y >= 1, x + y <= 3], [x >= 5, -2 * x >= -16], [y <= 4]], name="d")
    return m, x, y


def _blocks():
    m = dm.Model("blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a, b, c = m.make_disjunct("A"), m.make_disjunct("B"), m.make_disjunct("C")
    a.subject_to([x <= 2, y >= 1, x + y <= 3])
    b.subject_to([x >= 5, -2 * x >= -16])
    c.subject_to(y <= 4)
    m.add_disjunction([a, b, c], name="d")
    return m, x, y


# Optima by enumerating the disjuncts by hand (box [0, 10]^2):
#   max x + y: A 3, B 18 (x <= 8), C 14            -> 18
#   min x - y: A -3 (0, 3), B -5 (5, 10), C -4     -> -5
#   max y:     A 3, B 10, C 4                       -> 10
#   max x:     A 2, B 8, C 10                       -> 10
CASES = [
    (lambda x, y: -(x + y), -18.0),
    (lambda x, y: x - y, -5.0),
    (lambda x, y: -y, -10.0),
    (lambda x, y: -x, -10.0),
]


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=60, **kw)


def test_rows_are_multiplied_by_their_selector():
    m, _, _ = _slabs()
    rows = [str(c) for c in reformulate_gdp(m, BM)._constraints]
    assert "((x - 2) * _gdp_aux_disj_d_0_0) <= 0.0" in rows
    assert "((-16 - (-2 * x)) * _gdp_aux_disj_d_1_1) <= 0.0" in rows
    assert "((y - 4) * _gdp_aux_disj_d_2_2) <= 0.0" in rows
    assert len(rows) == 7  # the select row plus one product per disjunct row


@pytest.mark.parametrize("build", [_slabs, _blocks])
@pytest.mark.parametrize("obj, truth", CASES)
def test_solve_reaches_the_optimum(build, obj, truth):
    m, x, y = build()
    m.minimize(obj(x, y))
    r = _solve(m, gdp_method=BM)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


def test_registered_transformation_matches():
    m, x, y = _slabs()
    m.minimize(x - y)
    t = dt.create_using("gdp.binary_multiplication", m)
    assert not any(isinstance(c, _DisjunctiveConstraint) for c in t._constraints)
    assert _solve(t).objective == pytest.approx(-5.0, abs=1e-6)


def test_an_unbounded_body_is_lowered_where_big_m_refuses():
    m = dm.Model("unbounded")
    x = m.continuous("x", lb=0)  # no upper bound: x <= 2 has no finite big-M
    m.either_or([[x <= 2], [x >= 5]])
    m.minimize((x - 4) ** 2)
    with pytest.raises(ValueError, match="cannot bound the body"):
        reformulate_gdp(m, "big-m")
    r = _solve(m, gdp_method=BM)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-6)  # x = 5


def test_nested_either_or():
    """outer [z <= 6 & inner(z <= 1 | z >= 4)] | [z >= 8]; min (z - 3)^2 -> z = 4."""
    m = dm.Model("nested")
    z = m.continuous("z", lb=0, ub=10)
    inner = _DisjunctiveConstraint(disjuncts=[[z <= 1], [z >= 4]], name="inner")
    m.either_or([[z <= 6, inner], [z >= 8]], name="outer")
    m.minimize((z - 3) ** 2)
    r = _solve(m, gdp_method=BM)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-6)


def test_nested_blocks_match_big_m():
    """A [x <= 3, A1 [y >= 6] | A2 [y >= 8, x >= 2]] | B [x >= 7]; min x + y -> 6."""
    m = dm.Model("nested_blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a, a1, a2, b = (m.make_disjunct(n) for n in ("A", "A1", "A2", "B"))
    a.subject_to(x <= 3)
    a1.subject_to(y >= 6)
    a2.subject_to([y >= 8, x >= 2])
    a.add_disjunction([a1, a2], name="inner")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + y)
    for method in (BM, "big-m"):
        r = _solve(m, gdp_method=method)
        assert r.status == "optimal", method
        assert r.objective == pytest.approx(6.0, abs=1e-6), method


def test_or_semantics_is_lowered():
    """OR: x <= 4 | (2 <= x <= 6); both may hold. max x -> 6."""
    m = dm.Model("or")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 4], [x >= 2, x <= 6]], name="d", semantics="or")
    m.maximize(x)
    r = _solve(m, gdp_method=BM)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(6.0, abs=1e-6)


def test_a_body_undefined_on_the_box_is_refused():
    m = dm.Model("domain")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.sqrt(x) >= 2]])
    with pytest.raises(ValueError, match="by binary multiplication: it is undefined"):
        reformulate_gdp(m, BM)


def test_indicator_active_on_zero_multiplies_by_one_minus_y():
    """``if_then(1 - b, ...)`` is active when ``b == 0`` (#1677): the row gets ``1 - b``."""
    m = dm.Model("ind")
    x = m.continuous("x", lb=0, ub=10)
    b = m.binary("b")
    m.if_then(b, [x >= 6])
    m.if_then(1 - b, [x <= 3])
    rows = [str(c) for c in reformulate_gdp(m, BM)._constraints]
    assert "((6 - x) * b) <= 0.0" in rows
    assert "((x - 3) * (1 - b)) <= 0.0" in rows
    m.minimize((x - 5) ** 2)
    r = _solve(m, gdp_method=BM)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-6)  # x = 6 with b = 1
