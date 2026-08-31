"""#1066: an auto-routed MIP-NLP solve that RAISES must fall back, not propagate.

The convex-MINLP router retargeted from ``oa`` to ``lp_nlp_bb`` on the HiGHS
master. That master refuses -- loudly, and correctly -- to accept a lazy cut
whose coefficients it cannot represent on an unbounded column, rather than
silently dropping terms. But a refusal from an algorithm the *router* picked is
not an answer to give the caller: on the 104-instance route panel (2026-08-29)
``st_test1`` and ``st_test5`` were ``optimal`` on the default path and came back
as ``HiGHS rejected the master model`` once the retarget sent them to the
single-tree master.

So an auto-routed raise is treated exactly like an auto-routed non-certificate:
spend the rest of the budget on the default path. An EXPLICIT
``solver="mip-nlp"`` still raises -- the caller chose the algorithm.
"""

import discopt.modeling as dm
import discopt.solver as solver_mod
import discopt.solvers.mip_nlp as mip_nlp_mod
import pytest


def _convex_minlp():
    """A small convex MINLP the auto-route accepts (miqp, integral, convex)."""
    m = dm.Model("route_raise")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.binary("y")
    m.subject_to(x >= 2.0 * y)
    m.subject_to(x + y >= 1.5)
    m.minimize((x - 3.0) ** 2 + y)
    return m


def _routed(model, monkeypatch):
    """Confirm the router actually picks a MIP-NLP method for this model."""
    monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", "1")
    method, reason, _opts = solver_mod._convex_minlp_auto_route(model)
    return method, reason


class _Boom(RuntimeError):
    pass


def test_auto_routed_raise_falls_back_to_the_default_path(monkeypatch):
    model = _convex_minlp()
    method, reason = _routed(model, monkeypatch)
    # Anti-vacuity (CLAUDE.md §6): if the router declines this model the rest of
    # the test exercises nothing at all.
    assert method is not None, f"router declined the fixture: {reason}"

    calls = []

    def _raising(*args, **kwargs):
        calls.append(1)
        raise _Boom("HiGHS rejected the master model")

    monkeypatch.setattr(mip_nlp_mod, "solve_mip_nlp", _raising)

    result = model.solve(time_limit=30.0, gap_tolerance=1e-4)

    assert calls, "the routed solve never ran, so nothing raised"
    assert result is not None
    assert result.status == "optimal", result.status
    # The fallback is recorded, not silent -- the whole point of #1059's
    # ``algorithm_route`` field.
    assert result.algorithm_route is not None
    assert "_Boom" in result.algorithm_route or "raised" in result.algorithm_route
    # y=0 admits x=3 (x >= 1.5 binds loosely), so the optimum is 0.0.
    assert result.objective == pytest.approx(0.0, abs=1e-4)


def test_explicit_mip_nlp_still_raises(monkeypatch):
    """No fallback when the caller named the algorithm."""
    model = _convex_minlp()

    def _raising(*args, **kwargs):
        raise _Boom("HiGHS rejected the master model")

    monkeypatch.setattr(mip_nlp_mod, "solve_mip_nlp", _raising)

    with pytest.raises(_Boom):
        model.solve(solver="mip-nlp", time_limit=30.0, gap_tolerance=1e-4)
