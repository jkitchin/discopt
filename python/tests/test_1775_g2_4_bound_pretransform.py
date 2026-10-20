"""#1775 G2.4: ``gdp.bound_pretransformation`` (Pyomo's ``gdp.bound_pretransformation``).

Before: no such transformation; a disjunct row bounding one variable was lowered
like any other row (a big-M row, or a hull disaggregation).

After: a registered, user-selected transformation (NC) replaces the univariate rows
of each SELECT_ONE disjunction by ``sum_k L_k y_k <= x <= sum_k U_k y_k`` on its
indicators. Its automatic use inside the big-M and mbigm lowerings is BC, behind
``DISCOPT_GDP_BOUND_PRETRANSFORM`` (default OFF); with the flag OFF the lowering is
today's, which ``test_flag_off_lowering_is_unchanged`` pins.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.gdp.bound_pretransform import bound_pretransform

pytestmark = pytest.mark.smoke

FLAG = "DISCOPT_GDP_BOUND_PRETRANSFORM"


def _slabs():
    """x in [0,2] & y >= 1 | x >= 5 & -2x >= -16 (x <= 8) | y <= 4, box [0,10]^2."""
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
#   max x + y:   A: x+y <= 3 -> 3;  B: x <= 8, y <= 10 -> 18;  C: x <= 10, y <= 4 -> 14
#   min x - y:   A: x = 0, y = 3 -> -3;  B: x = 5, y = 10 -> -5;  C: x = 0, y = 4 -> -4
#   max y:       A: y <= 3 (x + y <= 3) -> 3;  B: 10;  C: 4
#   max x:       A 2, B 8, C 10 -> 10
CASES = [
    (lambda x, y: -(x + y), -18.0),
    (lambda x, y: x - y, -5.0),
    (lambda x, y: -y, -10.0),
    (lambda x, y: -x, -10.0),
]


def test_the_rows_are_replaced_sign_correctly():
    m, x, y = _slabs()
    rep = bound_pretransform(m)
    # x <= 2, y >= 1 (A); x >= 5, -2x >= -16 (B); y <= 4 (C) go; x + y <= 3 stays.
    assert rep.disjunctions == 1 and rep.rows_removed == 5 and rep.rows_added == 4
    (dc,) = [c for c in m._constraints if type(c).__name__ == "_DisjunctiveConstraint"]
    assert [len(d) for d in dc.disjuncts] == [1, 0, 0]
    text = "\n".join(str(c) for c in m._constraints)
    # -2x >= -16 is an UPPER bound x <= 8 (Pyomo's mbigm shortcut misreads the sign).
    assert "(8 * d_1_active)" in text
    assert "(x - (5 * d_1_active)) >= 0.0" in text


@pytest.mark.parametrize("build", [_slabs, _blocks])
@pytest.mark.parametrize("obj, truth", CASES)
@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm"])
def test_transformed_model_keeps_the_optimum(build, obj, truth, method):
    m, x, y = build()
    m.minimize(obj(x, y))
    t = dt.create_using("gdp.bound_pretransformation", m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = t.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("build", [_slabs, _blocks])
@pytest.mark.parametrize("obj, truth", CASES)
@pytest.mark.parametrize("method", ["big-m", "mbigm"])
def test_flag_on_keeps_the_optimum(build, obj, truth, method, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m, x, y = build()
    m.minimize(obj(x, y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("build", [_slabs, _blocks])
def test_flag_on_replaces_the_univariate_big_m_rows(build, monkeypatch):
    def rows(on):
        monkeypatch.setenv(FLAG, "1" if on else "0")
        m, _, _ = build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return [str(c) for c in reformulate_gdp(m, "big-m")._constraints]

    off, on = rows(False), rows(True)
    assert len(on) == len(off) - 5 + 4  # five big-M rows -> four bound rows


@pytest.mark.parametrize("build", [_slabs, _blocks])
def test_flag_off_lowering_is_unchanged(build, monkeypatch):
    called = []
    import discopt.gdp.bound_pretransform as bp

    real = bp.pretransform_disjunction
    monkeypatch.setattr(
        bp, "pretransform_disjunction", lambda *a, **k: called.append(1) or real(*a, **k)
    )
    monkeypatch.setenv(FLAG, "0")
    m, _, _ = build()
    off = [str(c) for c in reformulate_gdp(m, "big-m")._constraints]
    monkeypatch.delenv(FLAG)
    m, _, _ = build()
    unset = [str(c) for c in reformulate_gdp(m, "big-m")._constraints]
    assert off == unset and not called


def test_or_disjunction_is_refused():
    m = dm.Model("or")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], semantics="or")
    with pytest.raises(NotImplementedError, match="OR"):
        bound_pretransform(m)


def test_an_infinite_fallback_bound_keeps_the_rows():
    """x has no upper bound of its own: a disjunct without an x <= row would
    contribute +inf, so the upper side is not emitted and its row stays."""
    m = dm.Model("inf")
    x = m.continuous("x", lb=0)  # default upper bound: the 'no bound' sentinel
    m.either_or([[x <= 2], [x >= 5]], name="d")
    rep = bound_pretransform(m)
    assert rep.rows_added == 1  # only x >= 5 d_1 (lower side, fallback lb = 0)
    (dc,) = [c for c in m._constraints if type(c).__name__ == "_DisjunctiveConstraint"]
    assert [len(d) for d in dc.disjuncts] == [1, 0]


def test_nested_disjunctions_are_left_alone():
    from discopt.modeling.core import _DisjunctiveConstraint

    m = dm.Model("nested")
    z = m.continuous("z", lb=0, ub=10)
    inner = _DisjunctiveConstraint(disjuncts=[[z <= 1], [z >= 4]], name="inner")
    m.either_or([[z <= 6, inner], [z >= 8]], name="outer")
    rep = bound_pretransform(m)
    assert rep.disjunctions == 0 and rep.rows_added == 0
    assert any("nested" in s for s in rep.skipped)


def test_transformations_check_runs_in_isolation():
    m, x, y = _slabs()
    m.minimize(x + y)
    rep = dt.check("gdp.bound_pretransformation", m)
    assert rep.result is not m
