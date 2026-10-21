"""#1775 G2.8: ``gdp.partition_disjuncts`` and ``gdp.between_steps`` (P-split).

Before: neither transformation existed.

After (NC: new, user-selected transformations; no route changes): a disjunct row
``sum_p h_p(x_p) + c <= 0`` becomes ``sum_p a_p + c <= 0`` in the disjunct plus
global rows ``h_p(x_p) <= a_p``, each ``a_p`` bounded by the interval enclosure of
``h_p``. Exact for any ``h_p`` (an unselected disjunct takes ``a_p = hi_p``);
``between_steps`` lowers the split model by hull, whose relaxation then lies
between big-M's and the full hull's on convex rows (measured below).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp.partition import between_steps, partition_disjuncts
from discopt.modeling._relax_integrality import relax_integrality
from discopt.modeling.core import _DisjunctiveConstraint

pytestmark = pytest.mark.smoke


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=60, **kw)


def _balls(obj=lambda x: x[0] + 2 * x[1] - x[2] + 0.5 * x[3]):
    """Two unit balls in R^4, centred at +1 and -1: convex, separable rows."""
    m = dm.Model("balls")
    x = [m.continuous(f"x{i}", lb=-3, ub=3) for i in range(4)]
    m.either_or(
        [
            [sum((x[i] - 1) ** 2 for i in range(4)) <= 1],
            [sum((x[i] + 1) ** 2 for i in range(4)) <= 1],
        ],
        name="d",
    )
    m.minimize(obj(x))
    return m, x


def _aux(m):
    return [v for v in m._variables if v.name.startswith("_gdp_psplit_a")]


def test_rows_are_split_into_auxiliaries_with_interval_bounds():
    m, x = _balls()
    rep = partition_disjuncts(m, num_partitions=2)
    assert rep.rows_split == 2 and rep.aux_vars == 4
    aux = _aux(m)
    assert len(aux) == 4
    # Each part is two squares (x_i -/+ 1)^2 on [-3, 3]: enclosure [0, 32].
    assert all(float(a.lb) == 0.0 and float(a.ub) == 32.0 for a in aux)
    (dc,) = [c for c in m._constraints if isinstance(c, _DisjunctiveConstraint)]
    for d in dc.disjuncts:
        (row,) = d
        assert "_gdp_psplit_a" in str(row.body) and "x0" not in str(row.body)
    assert (
        sum("_gdp_psplit_a" in str(c) for c in m._constraints if not hasattr(c, "disjuncts")) == 4
    )


@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm"])
def test_the_split_model_keeps_the_optimum(method):
    ref = _solve(_balls()[0]).objective
    m, _ = _balls()
    partition_disjuncts(m)
    r = _solve(m, gdp_method=method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref, abs=1e-5)


def test_between_steps_keeps_the_optimum_and_its_relaxation_is_between():
    """Measured (#1775 G2.8): rBigM -12.68 < rBetween -11.53 < rHull -5.0 = optimum."""
    ref = _solve(_balls()[0]).objective
    r = _solve(between_steps(_balls()[0]))
    assert r.status == "optimal" and r.objective == pytest.approx(ref, abs=1e-5)

    def relaxed(mm):
        relax_integrality(mm)
        return _solve(mm).objective

    r_bigm = relaxed(reformulate_gdp(_balls()[0], "big-m"))
    r_between = relaxed(between_steps(_balls()[0]))
    # The full hull's relaxation equals the optimum here (-5.0, measured); it is not
    # re-solved, since discopt branches on its perspective rows (~60 s).
    assert r_bigm < r_between - 0.5
    assert r_between < ref - 0.5


@pytest.mark.parametrize("direction", [(1, 1, 1), (1, -1, 2), (-2, 1, -1), (0, 0, 1)])
def test_a_nonconvex_row_is_split_exactly(direction):
    """x0*x1 + x2**2 <= 1 is not convex; P-split is still exact (spec §6.2)."""

    def build():
        m = dm.Model("nc")
        x = [m.continuous(f"x{i}", lb=-2, ub=2) for i in range(3)]
        m.either_or([[x[0] * x[1] + x[2] ** 2 <= 1], [x[0] + x[1] >= 3]], name="d")
        m.minimize(sum(c * xi for c, xi in zip(direction, x)))
        return m

    ref = _solve(build())
    m = build()
    rep = partition_disjuncts(m)
    assert rep.rows_split == 2  # x0*x1 | x2**2, and x0 | x1 of the >= row
    r = _solve(m)
    assert r.status == ref.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


def test_ge_rows_are_negated_and_eq_and_single_part_rows_are_left():
    m = dm.Model("senses")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x**2 + y**2 >= 9, x + y == 5], [x**2 <= 1]], name="d")
    m.minimize(x + y)
    ref = _solve(m).objective
    rep = partition_disjuncts(m)
    assert rep.rows_split == 1
    assert any("sense '=='" in s for s in rep.skipped) and any("one part" in s for s in rep.skipped)
    assert _solve(m).objective == pytest.approx(ref, abs=1e-5)


def test_user_partitions_are_used_and_a_straddling_summand_is_refused():
    m, x = _balls()
    rep = partition_disjuncts(m, partitions=[[x[0]], [x[1]], [x[2], x[3]]])
    assert rep.aux_vars == 6  # three parts per row

    m2 = dm.Model("straddle")
    a = m2.continuous("a", lb=0, ub=1)
    b = m2.continuous("b", lb=0, ub=1)
    m2.either_or([[a * b + a <= 1], [a >= 0.5]])
    with pytest.raises(ValueError, match="uses variables of partitions"):
        partition_disjuncts(m2, partitions=[[a], [b]])
    with pytest.raises(ValueError, match="in no partition"):
        partition_disjuncts(m2, partitions=[[a]])


def test_refusals_unbounded_part_and_domain_gap():
    m = dm.Model("unb")
    x = m.continuous("x", lb=0)  # no upper bound
    y = m.continuous("y", lb=0, ub=1)
    m.either_or([[x**2 + y <= 1], [y >= 0.5]])
    with pytest.raises(ValueError, match="no finite upper bound"):
        partition_disjuncts(m)

    m2 = dm.Model("dom")
    x2 = m2.continuous("x", lb=-1, ub=4)
    y2 = m2.continuous("y", lb=0, ub=1)
    m2.either_or([[dm.sqrt(x2) + y2 <= 1], [y2 >= 0.5]])
    with pytest.raises(ValueError, match="undefined on part of the variable box"):
        partition_disjuncts(m2)


def test_a_row_with_a_parameter_is_left_unsplit():
    m = dm.Model("param")
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    p = m.parameter("p", value=1.0)
    m.either_or([[x**2 + p * y**2 <= 1], [x >= 1.5]])
    rep = partition_disjuncts(m)
    assert rep.rows_split == 0 and any("Parameter" in s for s in rep.skipped)


def test_indicator_and_nested_rows_are_split():
    m = dm.Model("blocks")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    a, b = m.make_disjunct("A"), m.make_disjunct("B")
    a.subject_to(x**2 + y**2 <= 1)
    b.subject_to([x >= 2])
    inner = m.disjunction([[x**2 + (y - 3) ** 2 <= 1], [x <= 0.5]], name="inner")
    b.subject_to(inner)
    m.add_disjunction([a, b], name="outer")
    m.minimize(-x - y)
    ref = _solve(m).objective
    rep = partition_disjuncts(m)
    assert rep.rows_split == 2  # the block row and the nested row
    assert _solve(m).objective == pytest.approx(ref, abs=1e-5)


def test_a_row_with_a_disjunct_local_variable_is_left_unsplit():
    """Its global rows would carry the local variable out of the block, which
    validate refuses (found by the battery probe on nested_blocks)."""
    m = dm.Model("local")
    x = m.continuous("x", lb=0, ub=3)
    a, b = m.make_disjunct("A"), m.make_disjunct("B")
    z = b.variable("z", lb=0, ub=3)
    a.subject_to(x <= 1)
    b.subject_to([x + z**2 <= 4, x >= 2])
    m.add_disjunction([a, b], name="d")
    m.minimize(-x)
    ref = _solve(m).objective
    rep = partition_disjuncts(m)
    assert any("disjunct-local" in s for s in rep.skipped)
    assert _solve(m).objective == pytest.approx(ref, abs=1e-5)


def test_a_user_big_m_on_a_split_row_is_dropped():
    m = dm.Model("bm")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    r = x**2 + y**2 <= 1
    d = m.either_or([[r, x <= 1], [x >= 2]], big_m={r: 20.0})
    partition_disjuncts(m)
    assert d.big_m is None


def test_registered_transformations():
    m, _ = _balls()
    p = dt.create_using("gdp.partition_disjuncts", m)
    assert len(_aux(p)) == 4 and not _aux(m)
    t = dt.create_using("gdp.between_steps", m)
    assert not _aux(m) and any(v.name.startswith("_gdp_psplit_a") for v in t._variables)
    ref = _solve(_balls()[0]).objective
    assert _solve(t).objective == pytest.approx(ref, abs=1e-5)
    m2, _ = _balls()
    dt.apply_to("gdp.between_steps", m2)  # adopts the result; its variables re-pointed
    assert all(v.model is m2 for v in m2._variables)
    assert _solve(m2).objective == pytest.approx(ref, abs=1e-5)


def test_num_partitions_below_two_is_refused():
    with pytest.raises(ValueError, match="num_partitions"):
        partition_disjuncts(_balls()[0], num_partitions=1)
