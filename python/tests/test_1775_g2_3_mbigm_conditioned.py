"""#1775 G2.3: multiple big-M conditioned on each disjunct (``DISCOPT_MBIGM_CONDITIONED``).

Before: ``gdp_method="mbigm"`` gives each disjunct row one M, the exact-LP bound
of its body over the global linear rows and the box, and lowers it to
``body <= M (1 - y_k)``.

With the flag ON, a row ``c`` of disjunct ``k`` of a SELECT_ONE disjunction
without nesting gets one ``M_{c,j}`` per other disjunct ``j`` -- the exact-LP
bound over the global rows **and disjunct j's linear rows** -- and becomes
``body <= sum_{j != k} M_{c,j} y_j``. Each ``M_{c,j}`` is at most today's M.

BC, default OFF: with the flag OFF (or unset) the lowering is unchanged, which
``test_flag_off_is_todays_lowering`` pins against the unconditioned route.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt._relax import gdp_reformulate as gr
from discopt._relax.gdp_reformulate import reformulate_gdp

pytestmark = pytest.mark.smoke

FLAG = "DISCOPT_MBIGM_CONDITIONED"


def _three_slabs():
    """x in [0,2] (y <= 1) or [4,6] (y <= 5) or [8,10] (y <= 9), y in [0, 20].

    Today's M for ``y - 1 <= 0`` is max(y) - 1 = 19 over the box. Conditioned on
    the second disjunct it is 5 - 1 = 4, on the third 9 - 1 = 8."""
    m = dm.Model("slabs")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=20)
    m.either_or(
        [[x <= 2, y <= 1], [x >= 4, x <= 6, y <= 5], [x >= 8, y <= 9]],
        name="slab",
    )
    return m, x, y


def _blocks():
    """The same slabs as Disjunct blocks (the add_disjunction route)."""
    m = dm.Model("slab_blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=20)
    a, b, c = m.make_disjunct("A"), m.make_disjunct("B"), m.make_disjunct("C")
    a.subject_to([x <= 2, y <= 1])
    b.subject_to([x >= 4, x <= 6, y <= 5])
    c.subject_to([x >= 8, y <= 9])
    m.add_disjunction([a, b, c], name="slab")
    return m, x, y


def _rows(m):
    return [str(c) for c in m._constraints]


def _lower(build, monkeypatch, on):
    monkeypatch.setenv(FLAG, "1" if on else "0")
    m, x, y = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return reformulate_gdp(m, "mbigm")


@pytest.mark.parametrize("build", [_three_slabs, _blocks])
def test_flag_off_is_todays_lowering(build, monkeypatch):
    """OFF and unset give the same rows (default OFF), and neither takes the new route."""
    called = []
    real = gr._reformulate_disjunction_mbigm_conditioned
    monkeypatch.setattr(
        gr,
        "_reformulate_disjunction_mbigm_conditioned",
        lambda *a, **k: called.append(1) or real(*a, **k),
    )
    off = _rows(_lower(build, monkeypatch, on=False))
    monkeypatch.delenv(FLAG, raising=False)
    m, _, _ = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unset = _rows(reformulate_gdp(m, "mbigm"))
    assert off == unset and not called


@pytest.mark.parametrize("build", [_three_slabs, _blocks])
def test_flag_on_conditions_each_m(build, monkeypatch):
    """Fails before G2.3: the route does not exist. The y-row of the first slab gets
    M = 4.04+ against the second slab and 8.08+ against the third, both below today's
    single M = 19.19."""
    low = _lower(build, monkeypatch, on=True)
    text = "\n".join(_rows(low))
    # today's M for y <= 1 over the box: 19 * 1.01
    assert "19.19" not in text
    assert "4.04" in text and "8.08" in text


@pytest.mark.parametrize("build", [_three_slabs, _blocks])
@pytest.mark.parametrize(
    "obj, truth",
    [
        (lambda x, y: -y, -9.0),  # third slab, y = 9
        (lambda x, y: -x - y, -19.0),  # third slab, x = 10, y = 9
        (lambda x, y: x - y, -1.0),  # second slab, x = 4, y = 5
        (lambda x, y: x + y, 0.0),  # first slab
    ],
)
def test_conditioned_solves_to_the_enumerated_optimum(build, obj, truth, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m, x, y = build()
    m.minimize(obj(x, y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = m.solve(gdp_method="mbigm", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-6)


def test_each_conditioned_m_is_at_most_todays(monkeypatch):
    """The invariant that makes ON at least as tight as OFF, on every row."""
    from discopt.solvers.lp_backend import get_exact_lp_solver

    m, x, y = _three_slabs()
    (dc,) = [c for c in m._constraints if isinstance(c, gr._DisjunctiveConstraint)]
    lp_data = gr._precompute_lp_relaxation(m)
    box = gr._lp_box(m, lp_data[4])
    rows = [gr._disjunct_linear_rows(d, m, lp_data[4]) for d in dc.disjuncts]
    solve_lp = get_exact_lp_solver()
    checked = 0
    for k, d in enumerate(dc.disjuncts):
        for con in d:
            today = gr._compute_big_m_lp(con, m, lp_data)
            up, down = gr._conditioned_ms(con, k, rows, m, lp_data, box, solve_lp)
            assert up[k] == 0.0 and down[k] == 0.0
            assert all(0.0 <= v <= today for v in up + down)
            checked += 1
    assert checked == 7  # 2 + 3 + 2 rows


def test_or_and_nested_disjunctions_keep_todays_route(monkeypatch):
    """The route declines what it does not cover; those lower exactly as before."""
    from discopt.modeling.core import _DisjunctiveConstraint

    m = dm.Model("nested")
    z = m.continuous("z", lb=0, ub=10)
    inner = _DisjunctiveConstraint(disjuncts=[[z <= 1], [z >= 4]], name="inner")
    m.either_or([[z <= 6, inner], [z >= 8]], name="outer")
    assert (
        gr._reformulate_disjunction_mbigm_conditioned(
            m._constraints[0], m, None, gr._precompute_lp_relaxation(m)
        )
        is None
    )


def _implied():
    """A: [x <= 5, y <= 3]; B: [x <= 2]. B implies A's ``x <= 5``: its conditioned M
    against B is exactly 0, so the row carries no selector term at all."""
    m = dm.Model("implied")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 5, y <= 3], [x <= 2]], name="imp")
    m.minimize(-x - y)
    return m


def test_a_row_implied_by_the_other_disjunct_drops_its_term(monkeypatch):
    """No tiny coefficient: a floor of 1e-6 here instead of 0 put 940 such terms into
    med_term_purchasing and HiGHS's root LP then failed the certificate check."""
    monkeypatch.setenv(FLAG, "1")
    low = reformulate_gdp(_implied(), "mbigm")
    rows = [c for c in low._constraints if (c.name or "").startswith("_gdp_mbc_imp_d0_c0")]
    assert len(rows) == 1
    assert "_gdp_aux" not in str(rows[0].body)  # x - 5 <= 0, no selector term
    r = _implied().solve(gdp_method="mbigm", time_limit=60)
    # B (x <= 2, y free up to 10) gives -12, better than A's -8.
    assert r.status == "optimal" and r.objective == pytest.approx(-12.0, abs=1e-6)


def test_no_conditioned_coefficient_is_tiny(monkeypatch):
    calls = []
    real = gr._conditioned_ms

    def spy(*a, **k):
        up, down = real(*a, **k)
        calls.append(up + down)
        return up, down

    monkeypatch.setattr(gr, "_conditioned_ms", spy)
    _lower(_three_slabs, monkeypatch, on=True)
    _lower(lambda: (_implied(), None, None), monkeypatch, on=True)
    ms = [v for c in calls for v in c]
    assert ms and all(v == 0.0 or v >= 1e-3 for v in ms)


def test_spent_budget_keeps_todays_m(monkeypatch):
    """Past the LP budget the rows keep today's unconditioned M (deterministic)."""
    monkeypatch.setattr(gr, "_MBIGM_CONDITIONED_LP_BUDGET", 0)
    text = "\n".join(_rows(_lower(_three_slabs, monkeypatch, on=True)))
    assert "19.19" in text  # today's M for y <= 1
