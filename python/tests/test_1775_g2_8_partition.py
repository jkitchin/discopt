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
from discopt.modeling.core import Constraint, _DisjunctiveConstraint

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
    # Each part is two squares (x_i -/+ 1)^2 on [-3, 3]: enclosure [0, 32], rounded
    # outward (evaluate_interval), so it contains the true range.
    for a in aux:
        assert float(a.lb) <= 0.0 and float(a.ub) >= 32.0
        assert float(a.lb) == pytest.approx(0.0, abs=1e-12)
        assert float(a.ub) == pytest.approx(32.0, rel=1e-12)
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


def test_dsl_ge_rows_eq_rows_and_single_part_rows():
    """The DSL stores ``x**2 + y**2 >= 9`` as ``9 - (x**2 + y**2) <= 0``, so this
    covers the summand signs only; the raw ``>=`` branch is tested below."""
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


def _raw(rows_a, rows_b, obj):
    m = dm.Model("raw")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.either_or([rows_a(x, y), rows_b(x, y)], name="d")
    m.minimize(obj(x, y))
    return m


# A raw ``Constraint(body, ">=", 0.0)`` keeps its sense, so it reaches the ``>=``
# negation, which DSL rows (always ``body <= 0``) never do. The oracle is worked
# out by hand, not by the unsplit solve (the base lowering mishandled raw rows,
# #1858). Each case kills the mutant that drops the negation (measured: -3.0).
_RAW_CASES = {
    # A: x^2+y^2 <= 1 (best -1 at (1, 0)); B: x+y-5 >= 0 in [0,3]^2 (best (3, 2)).
    "ge": (
        lambda x, y: [x**2 + y**2 - 1 <= 0],
        lambda x, y: [Constraint(body=x + y - 5, sense=">=", rhs=0.0)],
        lambda x, y: -x + 0.1 * y,
        -2.8,
    ),
    # The same feasible set with both rows raw: a raw ``<=`` and a raw ``>=``.
    "le_ge": (
        lambda x, y: [Constraint(body=x**2 + y**2 - 1, sense="<=", rhs=0.0)],
        lambda x, y: [Constraint(body=5 - x - y, sense="<=", rhs=0.0)],
        lambda x, y: -x + 0.1 * y,
        -2.8,
    ),
    "ge_both": (
        lambda x, y: [Constraint(body=1 - x**2 - y**2, sense=">=", rhs=0.0)],
        lambda x, y: [Constraint(body=x + y - 5, sense=">=", rhs=0.0)],
        lambda x, y: -x + 0.1 * y,
        -2.8,
    ),
}


@pytest.mark.parametrize("method", ["big-m", "hull"])
@pytest.mark.parametrize("case", sorted(_RAW_CASES))
def test_raw_ge_rows_are_split_exactly(case, method):
    rows_a, rows_b, obj, opt = _RAW_CASES[case]
    m = _raw(rows_a, rows_b, obj)
    rep = partition_disjuncts(m)
    assert rep.rows_split == 2
    r = _solve(m, gdp_method=method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=1e-5)


@pytest.mark.parametrize("sense", ["<=", ">="])
def test_a_raw_nonzero_rhs_row_is_refused_and_nothing_changes(sense):
    """A non-zero ``rhs`` is refused rather than folded into the split constant,
    as at every other door into a disjunct (#1858). Construction refuses it there,
    so the row is built with rhs=0 and given its rhs afterwards -- the one route
    by which it can reach partition_disjuncts. Splitting it would have turned it
    into a normalized row that Model.validate accepts."""
    m = dm.Model("raw_rhs")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    row = Constraint(body=x**2 + y**2, sense=sense, rhs=0.0)
    m.either_or([[x + y <= 1], [x**2 + y**2 - 4 <= 0, row]], name="d")
    m.minimize(-x - y)
    row.rhs = 4.0
    before = _state(m)
    with pytest.raises(ValueError, match=r"non-zero rhs \(4\.0\).*gdp\.partition_disjuncts"):
        partition_disjuncts(m)
    assert _state(m) == before


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


def _state(m):
    """Everything partition_disjuncts may touch, by identity and by text."""

    def disj(c):
        return (
            id(c.disjuncts),
            [(id(d), type(d), [id(r) for r in d], [str(r) for r in d]) for d in c.disjuncts],
            c.big_m,
        )

    return (
        [id(c) for c in m._constraints],
        [str(c) if not isinstance(c, _DisjunctiveConstraint) else disj(c) for c in m._constraints],
        [id(v) for v in m._variables],
        sorted(m._names),
        list(m._piecewise_domains),
        [(id(b._constraints), [id(r) for r in b._constraints]) for b in m._gdp_factory_disjuncts],
    )


def _two_disjunctions(second):
    """d1 splits over [[x], [y]]; d2 (``second``) refuses. Optimum -sqrt(2)."""
    m = dm.Model("atomic")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    z = m.continuous("z", lb=0, ub=3)
    m.either_or([[x**2 + y**2 <= 1], [x + y <= 0.5]], name="d1")
    m.either_or(second(x, y, z), name="d2")
    m.minimize(-x - y)
    return m, x, y


_REFUSING = {
    "straddle": (lambda x, y, z: [[x * y + x <= 1], [x <= 2]], "uses variables of partitions"),
    "no_partition": (lambda x, y, z: [[x + z <= 1], [x <= 2]], "in no partition"),
}


@pytest.mark.parametrize("how", ["direct", "apply_to"])
@pytest.mark.parametrize("why", sorted(_REFUSING))
def test_a_refusal_partway_through_leaves_the_model_unchanged(why, how):
    """Review of PR #1857 (p12): d1's rows used to be replaced before d2 refused,
    with the global rows still unadded, and the model then certified -5.0."""
    second, match = _REFUSING[why]
    m, x, y = _two_disjunctions(second)
    before = _state(m)
    rows_d1 = [list(d) for d in m._constraints[0].disjuncts]
    with pytest.raises(ValueError, match=match):
        if how == "direct":
            partition_disjuncts(m, partitions=[[x], [y]])
        else:
            dt.apply_to("gdp.partition_disjuncts", m, partitions=[[x], [y]])
    assert _state(m) == before
    assert [list(d) for d in m._constraints[0].disjuncts] == rows_d1
    r = _solve(m)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-(2.0**0.5), abs=1e-5)


def test_a_refusal_after_a_block_row_leaves_the_blocks_unchanged():
    m = dm.Model("blocks_atomic")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    a, b = m.make_disjunct("A"), m.make_disjunct("B")
    a.subject_to(x**2 + y**2 <= 1)
    b.subject_to([x + y <= 0.5])
    m.add_disjunction([a, b], name="outer")
    m.either_or([[x * y + x <= 1], [x <= 2]], name="d2")
    m.minimize(-x - y)
    before = _state(m)
    with pytest.raises(ValueError, match="uses variables of partitions"):
        partition_disjuncts(m, partitions=[[x], [y]])
    assert _state(m) == before
    assert _solve(m).objective == pytest.approx(-(2.0**0.5), abs=1e-5)


def test_a_tuple_disjunct_is_split_and_the_users_containers_are_not_assigned_into():
    """Review of PR #1857 (p11): ``d[j] = new`` raised TypeError on a tuple disjunct
    after the list disjunct before it had been rewritten (solve then gave -6.0)."""
    m = dm.Model("tuple")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    first = [x**2 + y**2 <= 1]
    second = (x**2 + y**2 >= 4, x + y <= 3.5)
    outer = [first, second]
    first_rows, second_rows = list(first), list(second)
    dc = m.either_or(outer, name="d")
    m.minimize(-x - y)
    rep = partition_disjuncts(m)
    assert rep.rows_split == 3
    # The user's lists were not written into; the disjunction holds new ones.
    assert outer == [first, second] and first == first_rows and list(second) == second_rows
    assert dc.disjuncts is not outer and isinstance(dc.disjuncts[1], tuple)
    assert all("_gdp_psplit_a" in str(r.body) for d in dc.disjuncts for r in d)
    r = _solve(m)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-3.5, abs=1e-5)


def test_partitions_of_vector_elements_and_duplicates_are_refused_with_guidance():
    m = dm.Model("vec")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.either_or([[x[0] ** 2 + x[1] ** 2 + y <= 1], [y >= 0.5]])
    m.minimize(-y)
    before = _state(m)
    with pytest.raises(ValueError, match="element of the vector variable 'x'"):
        partition_disjuncts(m, partitions=[[x[0]], [x[1]]])
    with pytest.raises(ValueError, match="duplicate variable 'y'"):
        partition_disjuncts(m, partitions=[[x, y], [y]])
    with pytest.raises(ValueError, match="not a list of variables"):
        partition_disjuncts(m, partitions=[y, x])
    assert _state(m) == before


def test_a_row_over_one_vector_variable_says_why_it_is_not_split():
    m = dm.Model("vec_skip")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.either_or([[x[0] ** 2 + x[1] ** 2 <= 1], [y >= 0.5]])
    m.minimize(-y)
    rep = partition_disjuncts(m)
    assert rep.rows_split == 0
    (reason,) = [s for s in rep.skipped if "disjunct 0" in s]
    assert "vector variable(s) ['x'] count as one variable" in reason


def _disc_or_line():
    m = dm.Model("stale")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.either_or([[x**2 + y**2 <= 40], [x + y <= 1]], name="d")
    m.minimize(-x - y)
    return m, x, y


@pytest.mark.parametrize("method", ["big-m", "hull"])
def test_widening_a_bound_after_partitioning_is_refused(method):
    """Review of PR #1857 (p5): partition on [0,3]^2, widen to [0,6]^2, and the
    auxiliaries' frozen bounds cut the new optimum -sqrt(80) (-6.0 certified)."""
    m, x, y = _disc_or_line()
    partition_disjuncts(m)
    x.ub = 6.0
    y.ub = 6.0
    with pytest.raises(ValueError, match="interval enclosure is .* outside the bounds"):
        _solve(m, gdp_method=method)
    # The guard reads the enclosure, not a fingerprint: tightening is fine.
    m2, x2, y2 = _disc_or_line()
    partition_disjuncts(m2)
    x2.ub = 2.0
    r = _solve(m2, gdp_method=method)
    assert r.status == "optimal" and r.objective == pytest.approx(-5.0, abs=1e-5)


def test_the_bound_guard_travels_with_copies_lowerings_and_documents():
    import copy
    import pickle

    from discopt import serialize

    m, _, _ = _disc_or_line()
    partition_disjuncts(m)
    assert len(m._piecewise_domains) == 4

    def widened_is_refused(mm):
        for v in mm._variables:
            if v.name in ("x", "y"):
                v.ub = 6.0
        with pytest.raises(ValueError, match="gdp.partition_disjuncts: auxiliary"):
            mm.validate()

    widened_is_refused(copy.deepcopy(m))
    widened_is_refused(pickle.loads(pickle.dumps(m)))
    widened_is_refused(serialize.loads(serialize.dumps(m)))
    widened_is_refused(between_steps(_disc_or_line()[0]))
    widened_is_refused(dt.create_using("gdp.partition_disjuncts", _disc_or_line()[0]))
    unbounded_below = dm.Model("ub")
    u = unbounded_below.continuous("u", lb=-1e30, ub=1)
    w = unbounded_below.continuous("w", lb=0, ub=1)
    unbounded_below.either_or([[-(u**2) - w <= 0], [w >= 0.5]])
    unbounded_below.minimize(w)
    partition_disjuncts(unbounded_below)
    lo = [lo for _, lo, _, _ in unbounded_below._piecewise_domains]
    assert float("-inf") in lo  # written as a token: a bare -inf is not JSON
    again = serialize.loads(serialize.dumps(unbounded_below))
    assert [r[1:3] for r in again._piecewise_domains] == [
        r[1:3] for r in unbounded_below._piecewise_domains
    ]
