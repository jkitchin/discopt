"""#2237: the convex-MINLP route's OA master follows ``DISCOPT_LP_ENGINE``.

highs-parity D3 (2026-11-07) made the in-house lazy master the route's default, and
the #2059 SOC outer block, which ``oa.solve_lp_nlp_bb`` builds on the HiGHS master
only, went dead on the default route (``portfol_classical050_1`` stopped certifying on
the route and fell back to NLP-BB). Owner decision 2026-11-18: with
``DISCOPT_CONVEX_ROUTE_OA_MASTER`` unset the master follows the global engine switch,
as the pure LP/MILP/IIS seams do since the 2026-11-17 engine switch:

* ``auto`` -> HiGHS where ``highspy`` imports, else the in-house master (silently);
* ``highs`` -> HiGHS (``MissingHighsError`` without ``highspy``);
* ``rust`` -> the in-house master.

The explicit variable still overrides. Under ``auto`` a routed solve whose HiGHS
master failed (raised, or ended ``error``) is re-run once on the in-house master, the
seams' engine-fallback convention; an explicit ``highs`` anywhere reports the failure.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import pytest
from discopt._optional_highs import MissingHighsError
from discopt.modeling.core import from_nl
from discopt.solver import (
    _convex_minlp_auto_route,
    _convex_route_oa_master,
    _convex_route_oa_master_falls_back,
    _route_master_failed,
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


@pytest.mark.requires_highs
def test_default_master_is_highs_where_highspy_imports():
    """The regression: unset, the route master is HiGHS (it was ``simplex`` under D3,
    which left the #2059 SOC block dead on the default route)."""
    assert _convex_route_oa_master() == "highs"
    method, reason, opts = _convex_minlp_auto_route(from_nl(str(NL)))
    assert method == "lp_nlp_bb" and opts == {"milp_solver": "highs"}, reason
    assert "master=highs" in reason


@pytest.mark.parametrize(
    "engine, var, expected",
    [
        ("rust", None, "simplex"),
        ("highs", None, "highs"),
        ("auto", None, "highs"),
        # The per-seam variable outranks the switch in both directions.
        ("rust", "highs", "highs"),
        ("highs", "simplex", "simplex"),
        ("auto", "simplex", "simplex"),
        # Blank means unset (``_env_selector``), as on every other seam.
        ("rust", "  ", "simplex"),
    ],
)
@pytest.mark.requires_highs
def test_engine_switch_and_override(monkeypatch, engine, var, expected):
    monkeypatch.setenv(ENGINE_ENV, engine)
    if var is not None:
        monkeypatch.setenv(MASTER_ENV, var)
    assert _convex_route_oa_master() == expected


def test_auto_without_highspy_is_the_in_house_master_silently(monkeypatch):
    """No default path needs HiGHS: ``auto`` on a HiGHS-less install routes to the
    in-house master without raising, and the route still fires."""
    _block_highspy(monkeypatch)
    assert _convex_route_oa_master() == "simplex"
    method, reason, opts = _convex_minlp_auto_route(from_nl(str(NL)))
    assert method == "lp_nlp_bb" and opts == {"milp_solver": "simplex"}, reason


@pytest.mark.parametrize("env", [(ENGINE_ENV, "highs"), (MASTER_ENV, "highs")])
def test_explicit_highs_without_highspy_raises(monkeypatch, env):
    _block_highspy(monkeypatch)
    monkeypatch.setenv(*env)
    with pytest.raises(MissingHighsError):
        _convex_route_oa_master()


@pytest.mark.parametrize("value", ["auto", "gurobi"])
def test_unknown_master_values_still_raise(monkeypatch, value):
    monkeypatch.setenv(MASTER_ENV, value)
    with pytest.raises(ValueError, match=MASTER_ENV):
        _convex_route_oa_master()


def test_fallback_applies_only_under_auto_with_the_variable_unset(monkeypatch):
    assert _convex_route_oa_master_falls_back()
    monkeypatch.setenv(MASTER_ENV, "highs")
    assert not _convex_route_oa_master_falls_back()
    monkeypatch.delenv(MASTER_ENV)
    monkeypatch.setenv(ENGINE_ENV, "highs")
    assert not _convex_route_oa_master_falls_back()


def test_route_master_failed_reads_only_raises_and_error_status():
    class R:
        def __init__(self, statuses):
            self.mip_nlp_trace = {"iterations": [{"master_status": s} for s in statuses]}

    assert _route_master_failed(None, RuntimeError("x"))
    assert _route_master_failed(R(["optimal", "error"]), None)
    for statuses in (["optimal"], ["time_limit"], ["infeasible"], []):
        assert not _route_master_failed(R(statuses), None)
    assert not _route_master_failed(None, None)


def _break_highs_master(monkeypatch, mode):
    """Make the HiGHS lazy master fail: raise as on a ``passModel`` refusal, or return
    its real result with status ERROR. Returns the per-engine call counters."""
    import discopt.solvers.milp_highs as milp_highs
    import discopt.solvers.milp_simplex as milp_simplex
    from discopt.solvers import SolveStatus

    calls = {"highs": 0, "simplex": 0}
    real_h = milp_highs.solve_milp_with_lazy_cuts
    real_s = milp_simplex.solve_milp_with_lazy_cuts

    def broken(*a, **kw):
        calls["highs"] += 1
        if mode == "raise":
            raise RuntimeError("HiGHS rejected the master model")
        return dataclasses.replace(real_h(*a, **kw), status=SolveStatus.ERROR)

    def counted(*a, **kw):
        calls["simplex"] += 1
        return real_s(*a, **kw)

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", broken)
    monkeypatch.setattr(milp_simplex, "solve_milp_with_lazy_cuts", counted)
    return calls


@pytest.mark.requires_highs
@pytest.mark.parametrize("mode", ["raise", "error"])
def test_auto_reruns_a_failed_highs_master_on_the_in_house_master(monkeypatch, mode):
    calls = _break_highs_master(monkeypatch, mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = from_nl(str(NL)).solve(time_limit=60)
    route = r.algorithm_route or ""
    assert calls["highs"] >= 1 and calls["simplex"] >= 1, calls
    assert "engine fallback: HiGHS OA master" in route, route
    assert route.startswith("mip-nlp/lp_nlp_bb:") and "fell back" not in route, route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(2.2, abs=1e-6)


@pytest.mark.requires_highs
def test_explicit_highs_master_failure_is_not_answered_by_the_in_house_master(monkeypatch):
    """``=highs`` means "HiGHS, and tell me when it fails": no in-house re-run; the
    #1059 fallback to the default path still answers soundly."""
    monkeypatch.setenv(MASTER_ENV, "highs")
    calls = _break_highs_master(monkeypatch, "raise")
    with pytest.warns(RuntimeWarning, match="routed solve raised"):
        r = from_nl(str(NL)).solve(time_limit=60)
    route = r.algorithm_route or ""
    assert calls["highs"] >= 1 and calls["simplex"] == 0, calls
    assert "engine fallback" not in route and "fell back" in route, route
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.2, abs=1e-6)
