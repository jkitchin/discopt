"""#1775 G2.9: ``gdp.basic_step`` and ``gdp.bilinear``.

Before: neither transformation existed.

After (NC: new, user-selected transformations; no route changes):

* ``apply_basic_step`` replaces SELECT_ONE disjunctions by their Cartesian
  product. Exact; its hull relaxation is at least as tight (measured below: 0 ->
  0.36 against an optimum of 1). Read indicators are tied to the new selectors.
* ``bilinear`` replaces ``b * x`` (b binary, x bounded) by ``v`` with
  ``[b == 1, v == x] v [b == 0, v == 0]``. Exact.

Both plan first and commit last: a refusal leaves the model unchanged.
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp.basic_step import apply_basic_step
from discopt.gdp.bilinear import bilinear
from discopt.modeling._relax_integrality import relax_integrality
from discopt.modeling.core import _DisjunctiveConstraint

pytestmark = pytest.mark.smoke


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=60, **kw)


def _shared(obj=lambda x, y: (x - 3) ** 2 + (y - 10) ** 2):
    """[x <= 2] v [x >= 8]  and  [x <= 5, y >= 6] v [x >= 5, y <= 4] on [0, 10]^2.

    Feasible combinations: x <= 2, y >= 6; x >= 8, y <= 4. Nearest to (3, 10) is
    (2, 10): optimum 1. (3, 10) itself lies in each disjunction's hull.
    """
    m = dm.Model("shared")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    d1 = m.either_or([[x <= 2], [x >= 8]], name="a")
    d2 = m.either_or([[x <= 5, y >= 6], [x >= 5, y <= 4]], name="b")
    m.minimize(obj(x, y))
    return m, d1, d2


def _relaxed_hull(m):
    low = reformulate_gdp(m, "hull")
    relax_integrality(low)
    return _solve(low).objective


# ── basic step ───────────────────────────────────────────────────────────────


def test_basic_step_builds_the_product():
    m, d1, d2 = _shared()
    new = apply_basic_step(m, [d1, d2])
    disjunctions = [c for c in m._constraints if isinstance(c, _DisjunctiveConstraint)]
    assert disjunctions == [new]
    assert [len(d) for d in new.disjuncts] == [3, 3, 3, 3]
    # combination (1, 0) is x >= 8 and x <= 5, y >= 6
    assert [str(r) for r in new.disjuncts[2]] == [
        str(r) for r in (*d1.disjuncts[1], *d2.disjuncts[0])
    ]


@pytest.mark.parametrize("method", ["big-m", "hull"])
@pytest.mark.parametrize(
    "obj",
    [lambda x, y: (x - 3) ** 2 + (y - 10) ** 2, lambda x, y: x - y, lambda x, y: -x - y],
)
def test_basic_step_keeps_the_optimum(obj, method):
    ref = _solve(_shared(obj)[0]).objective
    m, d1, d2 = _shared(obj)
    apply_basic_step(m, [d1, d2])
    r = _solve(m, gdp_method=method)
    assert r.status == "optimal" and r.objective == pytest.approx(ref, abs=1e-5)


def test_basic_step_tightens_the_hull_relaxation():
    """Measured: rHull 0.0 before, 0.36 after; the optimum is 1.0."""
    before = _relaxed_hull(_shared()[0])
    m, d1, d2 = _shared()
    apply_basic_step(m, [d1, d2])
    after = _relaxed_hull(m)
    assert before == pytest.approx(0.0, abs=1e-6)
    assert after == pytest.approx(0.36, abs=1e-4)
    assert after <= _solve(m).objective + 1e-6


def test_read_indicators_are_tied_to_the_product():
    """Logic on the original indicators still holds: forbid a-disjunct 0."""

    def build():
        m, d1, d2 = _shared()
        m.logical(~d1.indicators[0])  # x >= 8 forced, so x >= 8, y <= 4
        return m, d1, d2

    ref = _solve(build()[0])
    m, d1, d2 = build()
    apply_basic_step(m, [d1, d2])
    r = _solve(m)
    assert ref.objective == pytest.approx(25.0 + 36.0, abs=1e-5)  # (8, 4)
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


@pytest.mark.parametrize(
    "case, err, match",
    [
        ("one", ValueError, "at least two"),
        ("twice", ValueError, "listed twice"),
        ("foreign", ValueError, "not an either_or disjunction of this model"),
        ("or", NotImplementedError, "OR"),
        ("big_m", ValueError, "big_m"),
        ("too_big", ValueError, "max_disjuncts"),
    ],
)
def test_basic_step_refusals_leave_the_model_unchanged(case, err, match):
    m, d1, d2 = _shared()
    x = m._variables[0]
    kwargs = {}
    if case == "one":
        dcs = [d1]
    elif case == "twice":
        dcs = [d1, d1]
    elif case == "foreign":
        dcs = [d1, m.disjunction([[x <= 1], [x >= 2]])]
    elif case == "or":
        dcs = [d1, m.either_or([[x <= 1], [x >= 2]], semantics="or")]
    elif case == "big_m":
        dcs = [d1, m.either_or([[x <= 1], [x >= 2]], big_m=20.0)]
    else:
        dcs, kwargs = [d1, d2], {"max_disjuncts": 3}
    before = [id(c) for c in m._constraints]
    with pytest.raises(err, match=match):
        apply_basic_step(m, dcs, **kwargs)
    assert [id(c) for c in m._constraints] == before


def test_basic_step_registered():
    m, d1, d2 = _shared()
    t = dt.create_using("gdp.basic_step", m, disjunctions=[d1, d2])
    assert sum(isinstance(c, _DisjunctiveConstraint) for c in t._constraints) == 1
    assert sum(isinstance(c, _DisjunctiveConstraint) for c in m._constraints) == 2
    assert _solve(t).objective == pytest.approx(1.0, abs=1e-5)


# ── bilinear ─────────────────────────────────────────────────────────────────


def _bil():
    m = dm.Model("bil")
    z = m.binary("z")
    x = m.continuous("x", lb=-2, ub=5)
    y = m.continuous("y", lb=0, ub=3)
    m.subject_to(z * x + y >= 3)
    m.minimize(x + 2 * y + x * z)
    return m, z, x, y


def test_bilinear_replaces_each_pair_once():
    m, *_ = _bil()
    rep = bilinear(m)
    assert rep.products == 2 and rep.new_vars == 1  # z*x in the row and x*z in the objective
    (v,) = [w for w in m._variables if w.name.startswith("_gdp_bilinear_v")]
    assert (float(v.lb), float(v.ub)) == (-2.0, 5.0)
    assert "z" not in str(m._objective.expression)
    assert sum(isinstance(c, _DisjunctiveConstraint) for c in m._constraints) == 1


@pytest.mark.parametrize("method", ["big-m", "hull"])
def test_bilinear_keeps_the_optimum(method):
    ref = _solve(_bil()[0])
    m, *_ = _bil()
    bilinear(m)
    r = _solve(m, gdp_method=method)
    assert r.status == ref.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)


def test_bilinear_is_exact_pointwise():
    """For each z and a grid of x, the lowered model admits v = z * x and only it."""
    checks = 0
    for zv, xv in itertools.product((0.0, 1.0), np.linspace(-2, 5, 8)):
        m = dm.Model("pt")
        z = m.binary("z")
        x = m.continuous("x", lb=-2, ub=5)
        m.subject_to(z == zv)
        m.subject_to(x == xv)
        w = m.continuous("w", lb=-10, ub=10)
        m.subject_to(w == z * x)
        bilinear(m)
        for sense in (1, -1):
            m.minimize(sense * w)
            r = _solve(m)
            assert r.status == "optimal"
            assert float(r.x["w"]) == pytest.approx(zv * xv, abs=1e-6)
            checks += 1
    assert checks == 32


def test_bilinear_refuses_an_unbounded_variable_and_changes_nothing():
    m = dm.Model("unb")
    z = m.binary("z")
    x = m.continuous("x", lb=0)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(z * y <= 1)  # fine on its own, planned first
    m.subject_to(z * x <= 4)
    m.minimize(-x)
    before = ([id(v) for v in m._variables], [id(c) for c in m._constraints])
    with pytest.raises(ValueError, match="needs a finite box"):
        bilinear(m)
    assert ([id(v) for v in m._variables], [id(c) for c in m._constraints]) == before


def test_bilinear_registered():
    m, *_ = _bil()
    ref = _solve(_bil()[0]).objective
    t = dt.create_using("gdp.bilinear", m)
    assert not any(v.name.startswith("_gdp_bilinear_v") for v in m._variables)
    assert _solve(t).objective == pytest.approx(ref, abs=1e-5)
