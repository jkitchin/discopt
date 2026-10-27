"""#1779: a row undefined on part of the box, inside an inactive disjunct or indicator.

``min x, x in [-1, 10], [x <= -0.5] v [sqrt(x) >= 2]`` has optimum -1 (first
disjunct). Before the fix, big-m / mbigm / simplex / auto returned ``optimal`` at
4.0: the lowered row (``2 - sqrt(x) <= M (1 - y)``, or simplex's ``lam p(x)``)
stays in the model where its disjunct is inactive, so its domain ``x >= 0`` was
imposed on every disjunct. ``log(x) <= c`` and an undefined power folded to a
finite enclosure (``exp(-abs(x**0.5))``) did the same; ``if_then`` too. Hull
found -1 but ``Model.solve`` raised, because ``DeclaredLogic`` refused the whole
check on the NaN of the inactive disjunct.

After: those lowerings refuse loudly (a per-node domain check, not the row's
enclosure, which ``sqrt``'s clipping and outer operations hide); ``auto`` routes
such a disjunction to hull; and an undefined leaf fails only its own disjunct in
the verifier. Since then big-M (and mbigm) lower these rows exactly by giving the
restricted operand its own column, tied to it only where the disjunct holds
(``_ArgumentDisaggregator``); they still refuse an operand they cannot split, and
simplex still refuses.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_advisor import recommend_methods
from discopt._relax.gdp_reformulate import _domain_gap, reformulate_gdp
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.smoke

_REFUSAL = r"undefined on part of the variable box"


def _sqrt_disjunction():
    m = dm.Model("sqrt")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.sqrt(x) >= 2]], name="d")
    m.minimize(x)
    return m


def _log_disjunction():
    m = dm.Model("log")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.log(x) <= 1, x >= 2]], name="d")
    m.minimize(x)
    return m


def _wrapped_power_disjunction():
    m = dm.Model("wrapped")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.exp(-abs(x**0.5)) <= 0.5]], name="d")
    m.minimize(x)
    return m


def _unsplittable_disjunction():
    # sqrt's argument x**3 is not linear, so big-M cannot give it its own column
    # (_ArgumentDisaggregator): the row keeps no finite M and is refused.
    m = dm.Model("unsplittable")
    x = m.continuous("x", lb=-1, ub=10)
    m.either_or([[x <= -0.5], [dm.sqrt(x**3) >= 2]], name="d")
    m.minimize(x)
    return m


@pytest.mark.parametrize("build", [_sqrt_disjunction, _log_disjunction, _wrapped_power_disjunction])
@pytest.mark.parametrize("method", ["big-m", "mbigm"])
def test_big_m_splits_the_restricted_operand_and_finds_the_true_optimum(build, method):
    """Was ``optimal`` at 4.0 / 2.0 / 0.48 against a true optimum of -1, then (the
    #1779 fix) a refusal. The row is undefined unless its operand is in its domain,
    so that operand gets a column boxed to the domain and tied to it only where the
    disjunct holds: the first disjunct keeps ``x = -1``."""
    r = build().solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize("build", [_sqrt_disjunction, _log_disjunction, _wrapped_power_disjunction])
def test_simplex_keeping_the_body_out_of_the_selectors_reach_refuses(build):
    """Was: ``optimal`` at 4.0 / 2.0 / 0.48 against a true optimum of -1."""
    with pytest.raises(ValueError, match=_REFUSAL):
        build().solve(gdp_method="simplex", time_limit=60)


@pytest.mark.parametrize("method", ["big-m", "mbigm", "simplex"])
def test_an_operand_big_m_cannot_split_still_refuses(method):
    with pytest.raises(ValueError, match=_REFUSAL):
        _unsplittable_disjunction().solve(gdp_method=method, time_limit=60)


@pytest.mark.parametrize("method", ["hull", "auto"])
@pytest.mark.parametrize("build", [_sqrt_disjunction, _wrapped_power_disjunction])
def test_hull_and_auto_find_the_true_optimum(build, method):
    """Was: hull raised in ``DeclaredLogic``; auto returned a false optimum."""
    r = build().solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-6)


def _indicator_model():
    m = dm.Model("ind")
    x = m.continuous("x", lb=-1, ub=10)
    z = m.binary("z")
    m.if_then(z, [dm.sqrt(x) >= 2])
    m.subject_to(x >= 4 * z - 1)
    m.minimize(x)
    return m, x, z


def test_indicator_row_lowers_exactly():
    """``if_then`` has no hull route; it was ``optimal`` at 1e-11 against -1, then
    a refusal. Big-M now splits ``sqrt``'s operand: ``z = 0`` keeps ``x = -1``, and
    ``z = 1`` still enforces ``sqrt(x) >= 2``."""
    m, _, z = _indicator_model()
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(-1.0, abs=1e-6)
    m.subject_to(z == 1)
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(4.0, abs=1e-6)


def test_indicator_row_with_an_unsplittable_operand_refuses():
    m = dm.Model("ind")
    x = m.continuous("x", lb=-1, ub=10)
    z = m.binary("z")
    m.if_then(z, [dm.sqrt(x**3) >= 2])
    m.subject_to(x >= 4 * z - 1)
    m.minimize(x)
    with pytest.raises(ValueError, match=_REFUSAL):
        m.solve(time_limit=60)


def test_auto_routes_the_disjunction_to_hull():
    (advice,) = recommend_methods(_sqrt_disjunction())
    assert advice.recommendation == "hull" and "undefined" in advice.rationale


def test_defined_rows_still_lower_by_big_m():
    """No over-refusal: the same rows on a box inside their domain lower as before."""
    m = dm.Model("ok")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 0.5], [dm.sqrt(x) >= 2, dm.log(x + 1) <= 3]], name="d")
    m.minimize(x)
    low = reformulate_gdp(m, "big-m")
    assert len(low._constraints) == 4
    r = m.solve(gdp_method="big-m", time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(0.0, abs=1e-6)


def test_verifier_judges_an_undefined_inactive_disjunct_as_not_holding():
    """Was: ``verify_point`` refused (``non-finite residual``) and rejected x=-0.8."""
    m = _sqrt_disjunction()
    assert verify_point(m, np.array([-0.8])).ok
    assert verify_point(m, np.array([4.0])).ok
    bad = verify_point(m, np.array([-0.2]))  # neither disjunct; sqrt undefined there
    assert not bad.ok and "disjunction" in bad.reason


@pytest.mark.parametrize(
    "make, lo, hi, gap",
    [
        (lambda x: dm.sqrt(x), -1, 1, True),
        (lambda x: dm.sqrt(x), 0, 1, False),
        (lambda x: dm.log(x), -1, 1, True),
        (lambda x: dm.log(x), 0, 1, False),  # a point gap: left to the lowering
        (lambda x: dm.acos(x), -2, 0, True),
        (lambda x: dm.acos(x), -1, 1, False),
        (lambda x: dm.tan(x), 0, 2, False),  # poles are points
        (lambda x: 1 / x, -1, 1, False),  # so is a zero divisor
        (lambda x: x**0.5, -1, 1, True),
        (lambda x: x**-0.5, 0, 1, False),
        (lambda x: x**-2, -1, 1, False),
        (lambda x: x**2, -1, 1, False),
        (lambda x: dm.exp(-abs(x**0.5)), -1, 1, True),
        # Review of #1787: a self-product is a square, not an independent product.
        (lambda x: dm.sqrt(x * x + 1), -1, 10, False),
        (lambda x: dm.log(1 + x * x), -1, 10, False),
        (lambda x: dm.log1p(x * x), -1, 10, False),
        (lambda x: dm.sqrt((x - 2) * (x - 2)), -1, 10, False),
        (lambda x: dm.sqrt(x * x - 1), -1, 10, True),  # really undefined on (-1, 1)
    ],
)
def test_domain_gap_per_node(make, lo, hi, gap):
    m = dm.Model("g")
    x = m.continuous("x", lb=lo, ub=hi)
    assert (_domain_gap(make(x), m) is not None) is gap


def test_simplex_keeps_its_point_gap_capability():
    """#1182's reason to exist: ``1/x <= 1`` with ``x`` straddling 0, and an "off"
    disjunct sitting on the point where ``log`` is undefined. Both stay lowered."""
    m = dm.Model("recip")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 5.0) * (x - 5.0))
    m.either_or([[1.0 / x <= 1.0], [x >= 3.0]], name="recip")
    r = m.solve(gdp_method="simplex", time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(0.0, abs=1e-6)

    m = dm.Model("off_at_zero")
    c = m.continuous("c", lb=0, ub=10)
    m.either_or([[c <= 0], [dm.log(c) >= 1]], name="d")
    m.minimize(c)
    r = m.solve(gdp_method="simplex", time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("method", ["big-m", "mbigm", "simplex", "auto"])
def test_a_product_written_distance_row_is_not_refused(method):
    """Review of #1787: ``main`` solved this; the first #1779 check refused it,
    because interval multiplication saw ``x*x`` on ``[-1, 10]`` as ``[-10, 100]``."""
    m = dm.Model("dist")
    x = m.continuous("x", lb=-1, ub=10)
    y = m.continuous("y", lb=-3, ub=4)
    m.either_or([[x <= -0.5], [dm.sqrt((x - y) * (x - y) + 1) >= 2, x >= 0]], name="d")
    m.minimize(x + 0 * y)
    r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(-1.0, abs=1e-6)


def test_an_integer_valued_parameter_exponent_is_an_integer_power():
    """Review of #1787: ``x**p`` with ``p = 2`` was judged as a variable exponent."""
    m = dm.Model("par")
    x = m.continuous("x", lb=-1, ub=10)
    p = m.parameter("p", 2.0)
    assert _domain_gap(x**p, m) is None
    p.value = 0.5
    assert _domain_gap(x**p, m) is not None
