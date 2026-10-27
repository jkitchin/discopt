"""Big-M lowers a row whose operand is undefined where its disjunct is inactive.

GDPlib ``stranded_gas`` raised ``ValueError: GDP big-M reformulation cannot bound
the body of constraint 'mtype_exists[A500]_then_0' from below``: its learning
curve ``lf == 0.9 ** (log(S) / log 2)`` sits in a disjunct that also requires
``S >= 1``, while the other disjunct allows ``S = 0`` -- the pole of ``log``. No
finite M exists for the row as written, since M must hold where the disjunct is
inactive. ``_ArgumentDisaggregator`` gives the operand ``S`` its own column ``w``
boxed to its range over the *active* region and adds ``S - w == 0`` to the same
disjunct: exact when active, and bounded (finite M) when not.

The soundness cases are the ones where the inactive disjunct wins *at* the pole:
a lowering that leaked ``S >= 1`` (or ``log``'s domain) into every disjunct would
miss them.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import _DisjunctiveConstraint, reformulate_gdp

pytestmark = pytest.mark.smoke

_METHODS = ["big-m", "mbigm"]


def _counts(m):
    x = m.integer("x", shape=(3,), lb=0, ub=4)
    y = m.continuous("y", lb=-5, ub=5)
    return x, y, x[0] + x[1] + x[2]


def _pole(form, weight):
    """``on: s >= 1, y == log(s)`` v ``off: y == -1``; maximize ``y - weight * s``."""
    m = dm.Model(f"pole_{form}")
    _, y, s = _counts(m)
    if form == "block":
        on, off = m.make_disjunct("on"), m.make_disjunct("off")
        on.subject_to(s >= 1)
        on.subject_to(y == dm.log(s))
        off.subject_to(y == -1)
        m.add_disjunction([on, off], name="d")
    else:
        m.either_or([[s >= 1, y == dm.log(s)], [y == -1]], name="d")
    m.maximize(y - weight * s)
    return m


def _pole_optimum(weight):
    return max([np.log(k) - weight * k for k in range(1, 13)] + [-1.0])


@pytest.mark.parametrize("form", ["block", "either_or"])
@pytest.mark.parametrize("method", _METHODS)
def test_the_active_disjunct_wins_away_from_the_pole(form, method):
    """Was the ``ValueError`` above. ``log(3) - 0.9``, at ``s = 3``."""
    r = _pole(form, 0.3).solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(_pole_optimum(0.3), abs=1e-6)


@pytest.mark.parametrize("form", ["block", "either_or"])
@pytest.mark.parametrize("method", _METHODS)
def test_the_inactive_disjunct_wins_at_the_pole(form, method):
    """``off`` at ``s = 0`` gives -1; ``on`` is at best ``log(1) - 2 = -2``. A
    lowering that imposed ``s >= 1`` everywhere would report -2 or worse."""
    r = _pole(form, 2.0).solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-6)
    assert _pole_optimum(2.0) == -1.0


def _gap(form):
    """A region gap: ``sqrt(x - 1)`` is undefined on ``x < 1``, where ``b`` lives."""
    m = dm.Model(f"gap_{form}")
    x = m.continuous("x", lb=-5, ub=5)
    z = m.continuous("z", lb=0, ub=10)
    if form == "block":
        a, b = m.make_disjunct("a"), m.make_disjunct("b")
        a.subject_to(x >= 1)
        a.subject_to(z <= dm.sqrt(x - 1))
        b.subject_to(x <= -2)
        b.subject_to(z <= 0.5)
        m.add_disjunction([a, b], name="g")
    else:
        m.either_or([[x >= 1, z <= dm.sqrt(x - 1)], [x <= -2, z <= 0.5]], name="g")
    m.maximize(z - 0.1 * x * x)
    return m


@pytest.mark.parametrize("form", ["block", "either_or"])
@pytest.mark.parametrize("method", _METHODS)
def test_a_region_gap_lowers_exactly(form, method):
    xs = np.linspace(1, 5, 400001)
    want = max(float(np.max(np.sqrt(xs - 1) - 0.1 * xs**2)), 0.5 - 0.4)
    r = _gap(form).solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(want, abs=1e-6)


def _nested(form):
    """The pole row one level down: ``[(s in [1, 2], y == log s) v (s >= 3, y ==
    log s)] v [y == -1]``."""
    m = dm.Model(f"nested_{form}")
    _, y, s = _counts(m)
    if form == "block":
        i1, i2, off = m.make_disjunct("i1"), m.make_disjunct("i2"), m.make_disjunct("off")
        i1.subject_to(s >= 1)
        i1.subject_to(s <= 2)
        i1.subject_to(y == dm.log(s))
        i2.subject_to(s >= 3)
        i2.subject_to(y == dm.log(s))
        off.subject_to(y == -1)
        on = m.make_disjunct("on")
        on.add_disjunction([i1, i2], name="inner")
        m.add_disjunction([on, off], name="outer")
    else:
        inner = _DisjunctiveConstraint(
            disjuncts=[[s >= 1, s <= 2, y == dm.log(s)], [s >= 3, y == dm.log(s)]],
            name="inner",
        )
        m.either_or([[inner], [y == -1]], name="outer")
    m.maximize(y - 0.3 * s)
    return m


@pytest.mark.parametrize("form", ["block", "either_or"])
@pytest.mark.parametrize("method", _METHODS)
def test_a_nested_pole_row_lowers_exactly(form, method):
    r = _nested(form).solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(_pole_optimum(0.3), abs=1e-6)


def _globally_bounded():
    """No row of ``on`` bounds ``s`` itself: ``s >= t`` (model-wide) and ``t >= 1``
    (``on``) do together, so only the model-wide LP stage finds ``W``."""
    m = dm.Model("global_stage")
    _, y, s = _counts(m)
    t = m.continuous("t", lb=0, ub=12)
    m.subject_to(s - t >= 0)
    m.either_or([[t >= 1, y == dm.log(s)], [y == -1, t <= 0]], name="d")
    m.maximize(y - 2.0 * s)
    return m


@pytest.mark.parametrize("method", _METHODS)
def test_the_model_wide_stage_bounds_what_the_disjunct_alone_cannot(method):
    low = reformulate_gdp(_globally_bounded(), method)
    cols = [v for v in low._variables if v.name.startswith("_gdp_aux_arg")]
    assert len(cols) == 1
    assert float(cols[0].lb) == pytest.approx(1.0, abs=1e-6)
    r = _globally_bounded().solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-6)


def test_the_split_column_is_boxed_to_the_active_range():
    low = reformulate_gdp(_pole("either_or", 0.3), "big-m")
    cols = [v for v in low._variables if v.name.startswith("_gdp_aux_arg")]
    assert len(cols) == 1
    lo, hi = float(cols[0].lb), float(cols[0].ub)
    assert 1.0 - 1e-6 <= lo <= 1.0 and hi == pytest.approx(12.0, abs=1e-6)


@pytest.mark.parametrize("method", _METHODS)
def test_a_pole_no_active_row_excludes_still_refuses(method):
    """Without ``s >= 1`` the ``on`` disjunct itself reaches the pole: there is no
    range to box ``w`` to, and big-M refuses as before."""
    m = dm.Model("unguarded")
    _, y, s = _counts(m)
    m.either_or([[y == dm.log(s)], [y == -1]], name="d")
    m.maximize(y - 0.3 * s)
    with pytest.raises(ValueError, match="cannot bound the body"):
        reformulate_gdp(m, method)


@pytest.mark.slow
def test_stranded_gas_lowers_by_big_m():
    """The GDPlib instance itself: 18 rows split, each operand bounded by the
    disjunct-local LP stage (121 s -> ~2 s against the model-wide LP)."""
    pytest.importorskip("pyomo")
    sg = pytest.importorskip("gdplib.stranded_gas")
    from discopt.pyomo.gdp import from_pyomo_gdp

    low = reformulate_gdp(from_pyomo_gdp(sg.build_model()), "big-m")
    cols = [v for v in low._variables if v.name.startswith("_gdp_aux_arg")]
    assert len(cols) == 18
    assert all(float(v.lb) >= 1.0 - 1e-6 for v in cols)
