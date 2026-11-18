"""#2237: the convex-MINLP route's OA master, and the fallback a HiGHS default needs.

highs-parity D3 (2026-11-07) made the in-house lazy master the route's default, and
the #2059 SOC outer block, which ``oa.solve_lp_nlp_bb`` builds on the HiGHS master
only, is not built on the default route. #2237 measured a HiGHS default (following
``DISCOPT_LP_ENGINE``) and it failed bar 2 on the 47 routed syn/rsyn instances
(certified 31 in-house / 32 HiGHS+SCIP / 30 HiGHS without SCIP; wall 537 / 624 /
640 s), so the default stays on the in-house master:

* unset or ``simplex`` -> the in-house master, whatever ``DISCOPT_LP_ENGINE`` says;
* ``highs`` -> HiGHS, the documented opt-in (``MissingHighsError`` without
  ``highspy``); it never falls back.

A failed ``=highs`` master is reported and answered by the #1059 fallback to the
default path, never by an in-house re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from discopt._optional_highs import MissingHighsError
from discopt.modeling.core import from_nl
from discopt.solver import (
    _convex_minlp_auto_route,
    _convex_route_oa_master,
)

NL = Path(__file__).parent / "data" / "minlplib_nl" / "gbd.nl"
MASTER_ENV = "DISCOPT_CONVEX_ROUTE_OA_MASTER"
ENGINE_ENV = "DISCOPT_LP_ENGINE"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        MASTER_ENV,
        ENGINE_ENV,
        "DISCOPT_CONVEX_MINLP_ROUTE",
        "DISCOPT_CONVEX_ROUTE_METHOD",
        "DISCOPT_LP_MILP_BACKEND",
        "DISCOPT_LP_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)


def _block_highspy(monkeypatch):
    """Simulate a HiGHS-less install (``highs_available`` honours a ``None`` entry)."""
    monkeypatch.setitem(sys.modules, "highspy", None)


def test_default_master_is_the_in_house_master():
    """#2237 kept D3: unset, the route master is the in-house one, and the route
    fires on it."""
    assert _convex_route_oa_master() == "simplex"
    method, reason, opts = _convex_minlp_auto_route(from_nl(str(NL)))
    assert method == "lp_nlp_bb" and opts == {"milp_solver": "simplex"}, reason
    assert "master=simplex" in reason


@pytest.mark.parametrize(
    "engine, var, expected",
    [
        # DISCOPT_LP_ENGINE does not govern the route master.
        ("rust", None, "simplex"),
        ("highs", None, "simplex"),
        ("auto", None, "simplex"),
        # The variable is the only way in, in both directions.
        ("rust", "highs", "highs"),
        ("auto", "highs", "highs"),
        ("highs", "simplex", "simplex"),
        # Blank means unset (``_env_selector``), as on every other seam.
        ("highs", "  ", "simplex"),
    ],
)
@pytest.mark.requires_highs
def test_engine_switch_and_override(monkeypatch, engine, var, expected):
    monkeypatch.setenv(ENGINE_ENV, engine)
    if var is not None:
        monkeypatch.setenv(MASTER_ENV, var)
    assert _convex_route_oa_master() == expected


def test_default_needs_no_highspy(monkeypatch):
    """No default path needs HiGHS: on a HiGHS-less install the route still fires on
    the in-house master, even under ``DISCOPT_LP_ENGINE=highs``."""
    _block_highspy(monkeypatch)
    for engine in ("auto", "highs"):
        monkeypatch.setenv(ENGINE_ENV, engine)
        assert _convex_route_oa_master() == "simplex"
    method, reason, opts = _convex_minlp_auto_route(from_nl(str(NL)))
    assert method == "lp_nlp_bb" and opts == {"milp_solver": "simplex"}, reason


def test_explicit_highs_without_highspy_raises(monkeypatch):
    _block_highspy(monkeypatch)
    monkeypatch.setenv(MASTER_ENV, "highs")
    with pytest.raises(MissingHighsError):
        _convex_route_oa_master()


@pytest.mark.parametrize("value", ["auto", "gurobi"])
def test_unknown_master_values_still_raise(monkeypatch, value):
    monkeypatch.setenv(MASTER_ENV, value)
    with pytest.raises(ValueError, match=MASTER_ENV):
        _convex_route_oa_master()


def _break_highs_master(monkeypatch):
    """Make the HiGHS lazy master raise, as on a ``passModel`` refusal. Returns the
    per-engine call counters."""
    import discopt.solvers.milp_highs as milp_highs
    import discopt.solvers.milp_simplex as milp_simplex

    calls = {"highs": 0, "simplex": 0}
    real_s = milp_simplex.solve_milp_with_lazy_cuts

    def broken(*a, **kw):
        calls["highs"] += 1
        raise RuntimeError("HiGHS rejected the master model")

    def counted(*a, **kw):
        calls["simplex"] += 1
        return real_s(*a, **kw)

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", broken)
    monkeypatch.setattr(milp_simplex, "solve_milp_with_lazy_cuts", counted)
    return calls


@pytest.mark.requires_highs
def test_explicit_highs_master_failure_is_not_answered_by_the_in_house_master(monkeypatch):
    """``=highs`` means "HiGHS, and tell me when it fails": no in-house re-run; the
    #1059 fallback to the default path still answers soundly."""
    monkeypatch.setenv(MASTER_ENV, "highs")
    calls = _break_highs_master(monkeypatch)
    with pytest.warns(RuntimeWarning, match="routed solve raised"):
        r = from_nl(str(NL)).solve(time_limit=60)
    route = r.algorithm_route or ""
    assert calls["highs"] >= 1 and calls["simplex"] == 0, calls
    assert "engine fallback" not in route and "fell back" in route, route
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.2, abs=1e-6)


@pytest.mark.requires_highs
def test_caller_milp_solver_highs_is_never_answered_by_the_in_house_master(
    monkeypatch,
):
    """An explicit ``milp_solver="highs"`` is the caller's pick, not the router's: a
    failed HiGHS master is reported and the #1059 fallback answers, never a re-run."""
    calls = _break_highs_master(monkeypatch)
    with pytest.warns(RuntimeWarning, match="routed solve raised"):
        r = from_nl(str(NL)).solve(time_limit=60, milp_solver="highs")
    route = r.algorithm_route or ""
    assert calls["highs"] >= 1 and calls["simplex"] == 0, calls
    assert "engine fallback" not in route and "fell back" in route, route
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.2, abs=1e-6)
