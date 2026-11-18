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

The engine-fallback machinery (re-run a failed router-picked HiGHS master once on the
in-house master, merge the two attempts, warn on a recovered raise) is reachable on no
shipped configuration; the tests below exercise it by simulating a router whose
default is HiGHS (``_router_defaults_to_highs``), the configuration a future flip
would ship.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import discopt.solver as solver_mod
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


def test_fallback_applies_only_under_auto_with_the_variable_unset(monkeypatch):
    assert _convex_route_oa_master_falls_back()
    monkeypatch.setenv(MASTER_ENV, "highs")
    assert not _convex_route_oa_master_falls_back()
    monkeypatch.delenv(MASTER_ENV)
    monkeypatch.setenv(ENGINE_ENV, "highs")
    assert not _convex_route_oa_master_falls_back()


def test_fallback_never_applies_under_highs_options():
    """``highs_options`` is a request for HiGHS: its failure is reported, not answered
    by an engine that never saw the options (the seams' #1620 rule)."""
    assert _convex_route_oa_master_falls_back()
    token = solver_mod._HIGHS_OPTIONS_REQUEST.set(((("presolve", "off"),), {"used": False}))
    try:
        assert not _convex_route_oa_master_falls_back()
    finally:
        solver_mod._HIGHS_OPTIONS_REQUEST.reset(token)
    assert _convex_route_oa_master_falls_back()


@pytest.fixture
def _router_defaults_to_highs(monkeypatch):
    """Simulate a router whose default master is HiGHS (the variable stays unset),
    the only configuration in which the #2237 engine fallback applies."""
    pytest.importorskip("highspy")
    monkeypatch.setattr(solver_mod, "_convex_route_oa_master", lambda: "highs")


class _R:
    """A routed result: a master trace plus the fields ``_route_result_is_certified``
    reads."""

    def __init__(self, statuses, *, status="feasible", gap_certified=False, gap=None):
        self.mip_nlp_trace = {"iterations": [{"master_status": s} for s in statuses]}
        self.status = status
        self.gap_certified = gap_certified
        self.gap = gap


def test_route_master_failed_reads_only_raises_and_error_status():
    assert _route_master_failed(None, RuntimeError("x"), 1e-4)
    assert _route_master_failed(_R(["optimal", "error"]), None, 1e-4)
    for statuses in (["optimal"], ["time_limit"], ["infeasible"], []):
        assert not _route_master_failed(_R(statuses), None, 1e-4)
    assert not _route_master_failed(None, None, 1e-4)


def test_route_master_failed_never_discards_a_finished_result():
    """Review of #2241: ``solve_lp_nlp_bb`` can close the gap after a master call that
    ended ``error`` (``converged_early`` with an incumbent). Such a result finished the
    model; re-running it would throw the certificate away."""
    trace = ["optimal", "error"]
    closed = _R(trace, status="optimal", gap_certified=True, gap=0.0)
    assert not _route_master_failed(closed, None, 1e-4)
    for terminal in ("infeasible", "unbounded"):
        assert not _route_master_failed(_R(trace, status=terminal), None, 1e-4)
    # A *valid* but open gap did not finish the model: still a failure.
    open_gap = _R(trace, status="feasible", gap_certified=True, gap=0.3)
    assert _route_master_failed(open_gap, None, 1e-4)
    # ... and whether it is open is the caller's tolerance, not a fixed one.
    assert not _route_master_failed(open_gap, None, 0.5)


def _break_highs_master(monkeypatch, mode):
    """Make the HiGHS lazy master fail: raise as on a ``passModel`` refusal
    (``raise``), return its real result relabelled ERROR (``error``: the point and
    bound survive, so OA can still close the gap), or return ERROR with no point and
    no bound (``error_bare``). Returns the per-engine call counters."""
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
        res = dataclasses.replace(real_h(*a, **kw), status=SolveStatus.ERROR)
        if mode == "error_bare":
            res = dataclasses.replace(res, x=None, objective=None, bound=None)
        return res

    def counted(*a, **kw):
        calls["simplex"] += 1
        return real_s(*a, **kw)

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", broken)
    monkeypatch.setattr(milp_simplex, "solve_milp_with_lazy_cuts", counted)
    return calls


@pytest.mark.requires_highs
@pytest.mark.parametrize("mode", ["raise", "error_bare"])
def test_auto_reruns_a_failed_highs_master_on_the_in_house_master(
    monkeypatch, _router_defaults_to_highs, mode
):
    calls = _break_highs_master(monkeypatch, mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = from_nl(str(NL)).solve(time_limit=60)
    route = r.algorithm_route or ""
    recovered = [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "re-run on the in-house" in str(w.message)
    ]
    if mode == "raise":
        # Review of #2241 (#1614 E-01(c)): a raise the re-run recovers from is still
        # reported, with its message, both as a warning and in the route note.
        assert len(recovered) == 1, [str(w.message) for w in caught]
        assert "RuntimeError: HiGHS rejected the master model" in recovered[0]
        assert "HiGHS rejected the master model" in route, route
    else:
        assert recovered == [], recovered
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


@pytest.mark.requires_highs
def test_caller_milp_solver_highs_is_never_answered_by_the_in_house_master(
    monkeypatch, _router_defaults_to_highs
):
    """An explicit ``milp_solver="highs"`` is the caller's pick, not the router's: a
    failed HiGHS master is reported and the #1059 fallback answers, never a re-run."""
    calls = _break_highs_master(monkeypatch, "raise")
    with pytest.warns(RuntimeWarning, match="routed solve raised"):
        r = from_nl(str(NL)).solve(time_limit=60, milp_solver="highs")
    route = r.algorithm_route or ""
    assert calls["highs"] >= 1 and calls["simplex"] == 0, calls
    assert "engine fallback" not in route and "fell back" in route, route
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.2, abs=1e-6)


@pytest.mark.requires_highs
def test_rerun_that_fails_too_keeps_the_first_attempt(monkeypatch, _router_defaults_to_highs):
    """Review of #2241: the failed HiGHS attempt is merged into what the #1059
    fallback sees, never dropped, even when the in-house re-run raises."""
    import discopt.solvers.milp_simplex as milp_simplex

    calls = _break_highs_master(monkeypatch, "error_bare")

    def simplex_raises(*a, **kw):
        calls["simplex"] += 1
        raise RuntimeError("in-house master broke too")

    monkeypatch.setattr(milp_simplex, "solve_milp_with_lazy_cuts", simplex_raises)
    merges = []
    real_merge = solver_mod._merge_route_and_fallback

    def spy(route, fallback, is_maximize):
        merges.append((route, fallback))
        return real_merge(route, fallback, is_maximize)

    monkeypatch.setattr(solver_mod, "_merge_route_and_fallback", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = from_nl(str(NL)).solve(time_limit=60)
    route = r.algorithm_route or ""
    assert calls["highs"] >= 1 and calls["simplex"] >= 1, calls
    assert "engine fallback: HiGHS OA master returned error" in route, route
    # First merge: the two attempts (the re-run raised, so it contributes nothing).
    first_attempt, rerun = merges[0]
    assert first_attempt is not None and rerun is None
    assert first_attempt.mip_nlp_trace["iterations"], "first attempt has no trace"
    # Whatever happened next, the answer is still sound and certified.
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.2, abs=1e-6)


@pytest.mark.requires_highs
def test_an_error_master_call_the_route_recovers_from_is_not_rerun(
    monkeypatch, _router_defaults_to_highs
):
    """Review of #2241, end to end: a master call relabelled ERROR that still carries
    its point and bound lets ``solve_lp_nlp_bb`` close the gap, so the route
    *finished* the model and its certificate is returned, not re-run."""
    calls = _break_highs_master(monkeypatch, "error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = from_nl(str(NL)).solve(time_limit=60)
    route = r.algorithm_route or ""
    trace = r.mip_nlp_trace["iterations"]
    # Anti-vacuity: the trace really did record an ``error`` master call.
    assert any(it.get("master_status") == "error" for it in trace), trace
    assert calls["highs"] >= 1 and calls["simplex"] == 0, calls
    assert "engine fallback" not in route and "fell back" not in route, route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(2.2, abs=1e-6)
