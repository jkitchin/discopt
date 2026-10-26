"""#1984: LOA's global subproblems had no time cap, and LOA logged nothing per iteration.

* ``LOA._global_step`` called ``solve_subproblem`` without ``time_cap``, so one
  nested ``Model.solve`` got the whole remaining budget (measured: 849 s of a
  3600 s run on one flowsheet configuration). The cap is now LBB's / LD-SDA's
  node budget, a tenth of the run within [1 s, 30 s] (#1153), settable as
  ``subproblem_time_limit`` / ``gdp_options["subproblem_time_limit"]``.
* The module docstring called ``"nlp"`` the default; ``"global"`` is.
* The old LOA logged one INFO line per iteration to
  ``discopt.solvers.gdpopt_loa``; the rewrite logged nothing. The line is back.
* OA/ECP's iteration line printed ``LB`` (certified or heuristic), while the
  gap stop reads only the certified bound, so the log showed ``LB = UB`` one
  master before the run stopped. The line now also prints the certified bound.
"""

from __future__ import annotations

import logging
import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms import loa as loa_mod
from discopt.gdp.algorithms.loa import LOA, SUBPROBLEM_TIME_LIMIT
from discopt.gdp.algorithms.ric import RIC

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _nonconvex():
    m = dm.Model("nonconvex")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x**2 + y**2 >= 9], [x + y >= 5]], name="d")
    m.minimize(x + y)
    return m


@pytest.fixture
def global_limits(monkeypatch):
    """Record the time limit every global subproblem is handed."""
    import discopt.gdp.algorithms.subproblem as sp

    limits: list[float] = []
    real = sp._global

    def spy(sub, limit, tol):
        limits.append(limit)
        return real(sub, limit, tol)

    monkeypatch.setattr(sp, "_global", spy)
    return limits


@pytest.mark.parametrize(
    ("time_limit", "cap"),
    [(None, SUBPROBLEM_TIME_LIMIT), (3600.0, 30.0), (20.0, 2.0), (5.0, 1.0)],
)
def test_default_cap_is_the_1153_node_budget(time_limit, cap):
    a = LOA(_nonconvex(), time_limit=time_limit)
    assert a.subproblem_time_limit == pytest.approx(cap)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_nonpositive_cap_is_refused(bad):
    with pytest.raises(ValueError, match="subproblem_time_limit"):
        LOA(_nonconvex(), time_limit=60, subproblem_time_limit=bad)


@pytest.mark.parametrize("cls", [LOA, RIC])
def test_global_subproblem_gets_the_cap(cls, global_limits):
    r = _quiet(cls(_nonconvex(), time_limit=600, subproblem_time_limit=0.75).run)
    assert global_limits, "no global subproblem ran"
    assert max(global_limits) <= 0.75
    assert r.objective == pytest.approx(3.0, abs=1e-5)


def test_default_cap_reaches_the_subproblem(global_limits):
    # time_limit 3600 s: the subproblem used to be handed ~3600 s.
    _quiet(LOA(_nonconvex(), time_limit=3600).run)
    assert global_limits and max(global_limits) <= SUBPROBLEM_TIME_LIMIT


def test_gdp_options_key_reaches_loa(global_limits):
    r = _quiet(
        _nonconvex().solve,
        gdp_method="loa",
        time_limit=600,
        gdp_options={"subproblem_time_limit": 0.6},
    )
    assert global_limits and max(global_limits) <= 0.6
    assert r.status == "optimal"


@pytest.mark.parametrize("bad", [0, -2.0, True, "5", float("nan")])
def test_gdp_options_key_is_validated(bad):
    with pytest.raises(ValueError, match="subproblem_time_limit"):
        _nonconvex().solve(gdp_method="loa", gdp_options={"subproblem_time_limit": bad})


def test_gdp_options_key_is_refused_by_a_method_that_does_not_read_it():
    with pytest.raises(ValueError, match="does not read"):
        _nonconvex().solve(gdp_method="enumerate", gdp_options={"subproblem_time_limit": 5})


def test_docstring_names_the_real_default():
    import inspect

    assert inspect.signature(LOA).parameters["subproblem"].default == "global"
    assert 'The default ``"nlp"``' not in (loa_mod.__doc__ or "")


def test_one_info_line_per_iteration(caplog):
    with caplog.at_level(logging.INFO, logger="discopt.solvers.gdpopt_loa"):
        a = LOA(_nonconvex(), time_limit=600)
        r = _quiet(a.run)
    lines = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "discopt.solvers.gdpopt_loa" and " iter " in rec.getMessage()
    ]
    assert r.status == "optimal"
    assert len(lines) == a.iterations >= 1
    first = lines[0]
    assert first.startswith("LOA iter 1: master bound ")
    assert "configuration [" in first and "subproblem global " in first
    assert " bound " in first and " incumbent " in first


def test_oa_iteration_line_shows_the_certified_bound(caplog):
    m = dm.Model("ecp_lin")
    x = m.continuous("x", lb=0, ub=3)
    y = m.binary("y")
    m.minimize(x + 10 * y)
    m.subject_to(x**2 - 4 * y <= 0)
    m.subject_to(x >= 0.5)
    with caplog.at_level(logging.INFO, logger="discopt.solvers.oa"):
        _quiet(m.solve, solver="mip-nlp", mip_nlp_method="ecp", time_limit=60)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("OA-ECP iter")]
    assert lines, "ECP logged no iteration"
    assert all("certified_LB=" in ln for ln in lines)
