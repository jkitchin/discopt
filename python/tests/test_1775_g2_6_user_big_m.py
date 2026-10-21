"""#1775 G2.6: user big-M on ``either_or`` / ``disjunction``, with validation.

Before: there was no way to give a big-M; ``either_or`` refused a ``big_m``
keyword (``TypeError``) and every row got the computed M -- so a row whose body
has no finite interval enclosure could not be lowered by big-M at all.

After (NC: a new, user-selected input; models without ``big_m`` lower exactly as
before): ``big_m`` is a number or a ``{row: M, None: default}`` dict. Each value
is certified (interval bound, or a rigorous LP bound conditioned on each other
disjunct), refused with a witness point when the LP proves it too small, or used
uncertified -- and then ``Model.solve`` withholds the certificate, so a user M can
never produce a false one.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp.user_big_m import resolve_user_big_ms, uncertified_user_big_ms

pytestmark = pytest.mark.smoke


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return m.solve(time_limit=60, **kw)


def _rows(m, method="big-m"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return [str(c) for c in reformulate_gdp(m, method)._constraints]


# ── the API ──────────────────────────────────────────────────────────────────


def test_big_m_is_stored_per_row_with_a_default():
    m = dm.Model("api")
    x = m.continuous("x", lb=0, ub=100)
    c = x <= 2
    inner = m.disjunction([[x <= 1], [x >= 1.5]], name="inner")
    d = m.either_or([[c, inner], [x >= 5, x <= 12]], big_m={c: 11, None: 7}, name="d")
    assert d.big_m == ((11.0, None), (7.0, 7.0))  # a nested disjunction has no M
    d2 = m.either_or([[x <= 3], [x >= 4]], big_m=50, name="d2")
    assert d2.big_m == ((50.0,), (50.0,))
    assert m.either_or([[x <= 3], [x >= 4]]).big_m is None


@pytest.mark.parametrize(
    "bad, err, match",
    [
        (-1.0, ValueError, "finite and >= 0"),
        (float("inf"), ValueError, "finite and >= 0"),
        (True, TypeError, "must be a number"),
        ("10", TypeError, "must be a number"),
    ],
)
def test_bad_values_are_refused(bad, err, match):
    m = dm.Model("bad")
    x = m.continuous("x", lb=0, ub=10)
    with pytest.raises(err, match=match):
        m.either_or([[x <= 2], [x >= 5]], big_m=bad)


def test_a_key_that_is_not_a_row_is_refused():
    m = dm.Model("key")
    x = m.continuous("x", lb=0, ub=10)
    with pytest.raises(ValueError, match="not a row of this disjunction"):
        m.either_or([[x <= 2], [x >= 5]], big_m={x <= 2: 9.0})  # a different object


def test_the_lowering_uses_the_user_m():
    m = dm.Model("rows")
    x = m.continuous("x", lb=0, ub=100)
    c = x <= 2
    m.either_or([[c], [x >= 5, x <= 12]], big_m={c: 11, None: 7}, name="d")
    rows = _rows(m)
    assert "((x - 2) - (11 * (1 - _gdp_aux_disj_d_0_0))) <= 0.0" in rows
    assert "((5 - x) - (7 * (1 - _gdp_aux_disj_d_1_1))) <= 0.0" in rows


def test_rows_without_a_user_m_lower_as_before():
    def build(with_m):
        m = dm.Model("same")
        x = m.continuous("x", lb=0, ub=10)
        c = x <= 2
        m.either_or([[c], [x >= 5]], name="d", big_m={c: 9.0} if with_m else None)
        return m

    plain, partial = _rows(build(False)), _rows(build(True))
    assert plain[0] == partial[0] and plain[2] == partial[2]  # select row, x >= 5 row
    assert plain[1] != partial[1]


# ── certified: the solve keeps its certificate ───────────────────────────────


def test_m_above_the_interval_bound_is_certified():
    m = dm.Model("ival")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], big_m=20)
    m.maximize(x)
    r = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(10.0, abs=1e-6)


def test_m_below_the_interval_bound_is_certified_by_the_conditioned_lp():
    """x <= 2 has interval bound 98 on [0, 100]; with the other disjunct x <= 12
    selected its body is at most 10, so M = 11 is valid and certified."""
    m = dm.Model("cond")
    x = m.continuous("x", lb=0, ub=100)
    c = x <= 2
    d = m.either_or([[c], [x >= 5, x <= 12]], big_m={c: 11})
    verdict = resolve_user_big_ms(d, m)[id(c)]
    assert verdict.certified, verdict.reason
    m.maximize(x)
    r = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(12.0, abs=1e-6)


def test_an_unbounded_body_lowers_with_a_certified_user_m():
    """No upper bound on x: the computed big-M refuses; M = 25 is certified by the
    LP over the global row x <= 20, and the solve is certified."""
    m = dm.Model("unbounded")
    x = m.continuous("x", lb=0)
    m.subject_to(x <= 20)
    m.either_or([[x <= 2], [x >= 5]], name="plain")
    with pytest.raises(ValueError, match="cannot bound the body"):
        reformulate_gdp(m, "big-m")
    m2 = dm.Model("unbounded_m")
    x2 = m2.continuous("x", lb=0)
    m2.subject_to(x2 <= 20)
    m2.either_or([[x2 <= 2], [x2 >= 5]], big_m=25)
    m2.minimize((x2 - 4) ** 2)
    r = _solve(m2)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(1.0, abs=1e-6)


# ── refused: an LP witness proves the value too small ────────────────────────


def test_a_too_small_m_is_refused_with_a_witness():
    m = dm.Model("small")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 2
    m.either_or([[c], [x >= 5]], big_m={c: 3})
    m.maximize(x)
    with pytest.raises(ValueError, match=r"too small: the point \{x=10\}"):
        _solve(m)
    with pytest.raises(ValueError, match="too small"):
        dt.create_using("gdp.bigm", m)


def test_a_too_small_m_on_a_ge_row_is_refused():
    m = dm.Model("small_ge")
    x = m.continuous("x", lb=0, ub=10)
    c = x >= 5
    m.either_or([[x <= 2], [c]], big_m={c: 1})  # stored as 5 - x <= 0
    m.minimize(x)  # verify_point judges a witness against a model with an objective
    with pytest.raises(ValueError, match=r"the point \{x=0\}.*body is 5 > M"):
        reformulate_gdp(m, "big-m")


# ── uncertified: used as given, certificate withheld ─────────────────────────


def _nonlinear_uncertified():
    """x**2 <= 4 | x >= 5 under x <= 7: M = 50 is valid (x**2 - 4 <= 45) but the
    body is nonlinear and its interval bound on [0, 10] is 96."""
    m = dm.Model("nl")
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x <= 7)
    m.either_or([[x**2 <= 4], [x >= 5]], big_m=50)
    m.maximize(x)
    return m


def test_an_uncertified_m_withholds_the_certificate():
    m = _nonlinear_uncertified()
    reasons = uncertified_user_big_ms(m)
    assert len(reasons) == 1 and "not linear" in reasons[0]
    with pytest.warns(UserWarning, match="could not be certified"):
        r = m.solve(time_limit=60)
    assert r.status == "feasible"
    assert not r.gap_certified and r.bound is None and r.gap is None
    assert r.objective == pytest.approx(7.0, abs=1e-6)
    assert r.solver_stats["certificate/user_big_m_uncertified"] == 1.0


def test_an_uncertified_infeasibility_becomes_unknown():
    m = dm.Model("inf")
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x >= 8)
    m.either_or([[x**2 <= 4], [x <= 3]], big_m=50)
    m.minimize(x)
    r = _solve(m)
    assert r.status == "unknown" and not r.gap_certified


def test_a_too_small_m_ignored_by_hull_withholds_without_raising():
    m = dm.Model("hull_small")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 2
    m.either_or([[c], [x >= 5]], big_m={c: 3}, name="d")
    m.maximize(x)
    reasons = uncertified_user_big_ms(m)
    assert len(reasons) == 1 and "too small" in reasons[0]
    r = _solve(m, gdp_method="hull")
    assert r.status == "feasible" and not r.gap_certified
    assert r.objective == pytest.approx(10.0, abs=1e-6)


def test_an_ignored_m_warns_and_hull_keeps_its_certificate_when_certified():
    m = dm.Model("hull")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], big_m=20, name="d")
    m.maximize(x)
    with pytest.warns(UserWarning, match="big_m on disjunction 'd' is ignored"):
        r = m.solve(gdp_method="hull", time_limit=60)
    assert r.status == "optimal" and r.gap_certified


# ── nesting, pretransform, serialization ─────────────────────────────────────


def test_nested_disjunction_takes_its_user_m_unconditioned():
    """Inner rows are relaxed whenever their own selector is 0, also when the
    parent is not selected: only the global rows bound them."""
    m = dm.Model("nested")
    z = m.continuous("z", lb=0, ub=10)
    c = z <= 1
    inner = m.disjunction([[c], [z >= 4]], name="inner", big_m={c: 9.5})
    m.either_or([[z <= 6, inner], [z >= 8]], name="outer")
    rows = _rows(m)
    assert any("(9.5 * (1 - " in r for r in rows)
    m.minimize((z - 3) ** 2)
    r = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(1.0, abs=1e-6)
    # 8.5 is below 9 = max(z - 1) over the box, and z = 10 is feasible through the
    # outer z >= 8 arm, where the inner big-M row still binds: a witness, refused.
    m2 = dm.Model("nested_small")
    z2 = m2.continuous("z", lb=0, ub=10)
    c2 = z2 <= 1
    inner2 = m2.disjunction([[c2], [z2 >= 4]], name="inner", big_m={c2: 8.5})
    m2.either_or([[z2 <= 6, inner2], [z2 >= 8]], name="outer")
    m2.minimize(z2)
    with pytest.raises(ValueError, match="too small"):
        reformulate_gdp(m2, "big-m")


def test_bound_pretransform_keeps_the_user_ms_aligned():
    from discopt.gdp.bound_pretransform import bound_pretransform

    m = dm.Model("pt")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    xy = x + y <= 3
    d = m.either_or([[x <= 2, xy], [x >= 5]], name="d", big_m={xy: 17.5})
    bound_pretransform(m)
    assert [len(r) for r in d.disjuncts] == [1, 0]
    assert d.big_m == ((17.5,), ())
    assert any("(17.5 * (1 - " in r for r in _rows(m))


def test_serialization_refuses_a_user_big_m():
    from discopt.serialize import SerializationError, dumps

    m = dm.Model("ser")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], big_m=20, name="d")
    m.minimize(x)
    with pytest.raises(SerializationError, match="user big_m"):
        dumps(m)


# ── soundness on the differential battery ────────────────────────────────────


def test_certified_user_ms_cut_and_admit_nothing():
    """The §6.1 battery's checker on shapes with certified user Ms: no feasible
    point is cut and no infeasible one admitted; the probe must have fired."""
    from gdp_differential import Shape, run_shape

    def tight():
        m = dm.Model("tight")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=10)
        m.either_or([[x <= 2, y >= 3], [x >= 5, x <= 12, y <= 4]], big_m={None: 10.5})
        m.minimize(x)
        return m

    def unbounded():
        m = dm.Model("unb")
        x = m.continuous("x", lb=0)
        m.subject_to(x <= 20)
        m.either_or([[x <= 2], [x >= 5]], big_m=25)
        m.minimize(x)
        return m

    shapes = [
        Shape(
            "tight",
            tight,
            lambda v, t: (
                (v["x"] <= 2 + t and v["y"] >= 3 - t)
                or (5 - t <= v["x"] <= 12 + t and v["y"] <= 4 + t)
            ),
        ),
        Shape(
            "unbounded",
            unbounded,
            lambda v, t: v["x"] <= 20 + t and (v["x"] <= 2 + t or v["x"] >= 5 - t),
            box={"x": (0.0, 30.0)},
        ),
    ]
    feasible = infeasible = 0
    for shape in shapes:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # every M here is certified
            rep = run_shape(shape, "big-m")
        assert not rep.cut and not rep.admitted, (shape.name, rep.cut, rep.admitted)
        feasible += rep.feasible_checked
        infeasible += rep.infeasible_checked
    assert feasible > 20 and infeasible > 20
