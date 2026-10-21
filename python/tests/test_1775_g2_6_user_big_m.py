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
    verdict = resolve_user_big_ms(d, m)[(id(d), 0, 0)]
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


def _false_infeasibility():
    """M = 6 is too small for x <= 2: on the arm x >= 9, x**2 <= 90.25 allows x up
    to 9.5, where x - 2 = 7.5. The LP maximiser x = 10 violates x**2 <= 90.25, so
    there is no witness and the M is used uncertified. The lowered row gives
    x <= 8, so the lowered model is infeasible although x = 9 is optimal."""
    m = dm.Model("false_inf")
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x**2 <= 90.25)
    m.subject_to(x >= 5)
    r0 = x <= 2
    m.either_or([[r0], [x >= 9]], big_m={r0: 6.0})
    m.minimize(x)
    return m


def test_an_uncertified_infeasibility_becomes_unknown():
    """The guard is what turns the lowered model's infeasibility into unknown."""
    from unittest import mock

    with mock.patch.object(dm.Model, "_withhold_user_big_m_certificate", lambda *a, **k: None):
        r_off = _solve(_false_infeasibility())
    assert r_off.status == "infeasible" and r_off.gap_certified  # the false proof
    r = _solve(_false_infeasibility())
    assert r.status == "unknown" and not r.gap_certified
    assert r.bound is None
    assert r.solver_stats["certificate/user_big_m_uncertified"] == 1.0


def test_a_too_small_m_ignored_by_hull_keeps_the_hull_certificate():
    """Hull never reads the M, so it cannot weaken the hull result (#1851 review):
    the guard judges only the disjunctions a lowering of this solve read."""
    m = dm.Model("hull_small")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 2
    m.either_or([[c], [x >= 5]], big_m={c: 3}, name="d")
    m.maximize(x)
    reasons = uncertified_user_big_ms(m)  # standalone: every user M is judged
    assert len(reasons) == 1 and "too small" in reasons[0]
    r = _solve(m, gdp_method="hull")
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(10.0, abs=1e-6)
    with pytest.raises(ValueError, match="too small"):  # big-M does read it
        _solve(m)


def _domain_gap_model(big_m):
    m = dm.Model("gap")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=-1, ub=10)
    m.either_or([[dm.sqrt(y) <= 1, x <= 2], [x >= 5]], big_m=big_m, name="d")
    m.minimize((x - 3) ** 2 + y)
    return m


def test_a_domain_gap_does_not_break_a_hull_solve():
    """sqrt(y) on y in [-1, 10] has no big-M (domain gap); hull lowers it. The
    guard raised the domain-gap error after a working hull solve (#1851 review)."""
    plain = _solve(_domain_gap_model(None), gdp_method="hull")
    r = _solve(_domain_gap_model(50), gdp_method="hull")
    assert plain.status == "optimal" and plain.gap_certified
    assert r.status == plain.status and r.gap_certified == plain.gap_certified
    assert r.objective == pytest.approx(plain.objective, abs=1e-6)
    reasons = uncertified_user_big_ms(_domain_gap_model(50))  # reported, not raised
    assert len(reasons) == 1 and "undefined on part of the box" in reasons[0]
    with pytest.raises(ValueError, match="undefined on part"):
        _solve(_domain_gap_model(50))  # big-M cannot lower it, user M or not


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


def _nested_parent_unselected(m_c):
    """x <= 1 in the inner disjunction under the outer x <= 6 arm. The row is
    relaxed in A∩D (x in [4, 5]: x - 1 <= 4) *and* in the outer B arm, where the
    inner selectors are all 0 (x in [8, 10]: x - 1 <= 9)."""
    m = dm.Model("nested_parent")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 1
    inner = m.disjunction([[c], [x >= 4, x <= 5]], name="inner", big_m={c: m_c})
    m.either_or([[x <= 6, inner], [x >= 8]], name="outer")
    m.maximize(x)
    return m


def test_a_nested_m_valid_only_given_its_parent_is_not_certified():
    """M = 4.5 covers the inner sibling (4) but not the outer B arm (9): a nested
    disjunction must not be conditioned as if its parent were always selected.
    Conditioned that way, B is cut to x <= 5.5 and the solve certifies 5 against
    the true 10 (mutant M8 of the #1851 review)."""
    m = _nested_parent_unselected(4.5)
    with pytest.raises(ValueError, match=r"too small: the point \{x=10\}"):
        _solve(m)
    assert any("too small" in r for r in uncertified_user_big_ms(m))


def test_a_nested_m_is_conditioned_on_its_parent_path():
    """#1851 review point 7. The row x <= 1 is relaxed in A∩D (x - 1 <= 5 with
    A's x <= 6) and in B (x <= 9: x - 1 <= 8); the box alone gives 9. M = 8.5 is
    valid, and certified only by conditioning on both."""
    m = dm.Model("nested_path")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 1
    inner = m.disjunction([[c], [x >= 4]], name="inner", big_m={c: 8.5})
    m.either_or([[x <= 6, inner], [x >= 8, x <= 9]], name="outer")
    m.maximize(x)
    assert uncertified_user_big_ms(m) == []
    r = _solve(m)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(9.0, abs=1e-6)


def test_a_block_nested_m_is_judged_without_its_parent():
    """A disjunction in an ``add_disjunction`` block (``parent_indicator``): its
    parent is not visible, so only the global rows bound it. M = 4.5 covers the
    inner sibling but not the B block (x - 1 up to 8.5 at x = 9.5); x = 10 fails
    x**2 <= 90.25, so there is no witness and the M is used uncertified. The
    lowered model cuts B to x <= 5.5 and would certify 5 against the true 9.5:
    the guard must judge it with the block semantics and withhold."""
    m = dm.Model("block_nested")
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x**2 <= 90.25)
    c = x <= 1
    inner = m.disjunction([[c], [x >= 4, x <= 5]], name="inner", big_m={c: 4.5})
    a, b = m.make_disjunct("A"), m.make_disjunct("B")
    a.subject_to([x <= 6, inner])
    b.subject_to(x >= 8)
    m.add_disjunction([a, b], name="outer")
    m.maximize(x)
    assert inner.parent_indicator is not None
    r = _solve(m)
    assert r.status == "feasible" and not r.gap_certified and r.bound is None
    assert r.solver_stats["certificate/user_big_m_uncertified"] == 1.0


def test_one_row_in_two_places_keeps_each_m():
    """Ms are keyed by position, not by the row object (#1851 review nit): the
    same ``Constraint`` in the outer and the nested disjunction keeps both."""
    m = dm.Model("shared")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 2
    inner = m.disjunction([[c], [x >= 4]], name="inner", big_m={c: 20.0})
    m.either_or([[c, inner], [x >= 5]], name="outer", big_m={c: 9.0})
    rows = _rows(m)
    assert sum("(9 * (1 - " in r for r in rows) == 1
    assert sum("(20 * (1 - " in r for r in rows) == 1


# ── exact ties, LP failures ──────────────────────────────────────────────────


@pytest.mark.parametrize("sense", ["<=", ">="])
def test_an_m_equal_to_the_interval_bound_is_certified(sense):
    """The interval enclosure is the one the computed M rests on; an M equal to
    it is as valid (#1851 review point 4: a 1e-9 slack refused every tie)."""
    m = dm.Model("tie")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 2 if sense == "<=" else x >= 5
    other = x >= 5 if sense == "<=" else x <= 2
    d = m.either_or([[c], [other]], big_m={c: 8.0 if sense == "<=" else 5.0})
    verdict = resolve_user_big_ms(d, m)[(id(d), 0, 0)]
    assert verdict.certified, verdict.reason
    m.maximize(x) if sense == "<=" else m.minimize(x)
    r = _solve(m)
    assert r.status == "optimal" and r.gap_certified


def _conditioned():
    m = dm.Model("cond")
    x = m.continuous("x", lb=0, ub=100)
    c = x <= 2
    d = m.either_or([[c], [x >= 5, x <= 12]], big_m={c: 11})
    return m, d


def test_an_lp_that_does_not_end_optimal_leaves_the_m_uncertified(monkeypatch):
    import discopt.solvers.lp_backend as lpb
    from discopt.solvers import SolveStatus

    real = lpb.get_exact_lp_solver()

    def failing(**kw):
        res = real(**kw)
        res.status = SolveStatus.ITERATION_LIMIT
        return res

    monkeypatch.setattr(lpb, "get_exact_lp_solver", lambda: failing)
    m, d = _conditioned()
    verdict = resolve_user_big_ms(d, m)[(id(d), 0, 0)]
    assert not verdict.certified and "ended" in verdict.reason


def test_no_rigorous_lp_bound_leaves_the_m_uncertified(monkeypatch):
    import discopt._relax.gdp_reformulate as gr

    monkeypatch.setattr(gr, "_safe_lp_max", lambda *a, **k: None)
    m, d = _conditioned()
    verdict = resolve_user_big_ms(d, m)[(id(d), 0, 0)]
    assert not verdict.certified and "no rigorous LP bound" in verdict.reason


def test_an_infeasible_region_lp_says_why_it_is_not_certified():
    """#1851 review point 6: the other disjunct is empty, so the row is never
    relaxed, but a float LP's infeasible is not a proof; the reason says so."""
    m = dm.Model("empty_region")
    x = m.continuous("x", lb=0, ub=10)
    c = x <= 2
    d = m.either_or([[c], [x >= 5, x <= 4]], big_m={c: 1.0})
    verdict = resolve_user_big_ms(d, m)[(id(d), 0, 0)]
    assert not verdict.certified
    assert "reports infeasible" in verdict.reason and "not a proof" in verdict.reason


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
