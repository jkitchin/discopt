"""#1775 G6.2: an incumbent the exit gate refuses is never reported.

Owner decision (2026-10-23). Before, a point that failed the exit gate even after
the repair polish (#1916) was returned with its objective and ``x``, merely
uncertified -- on GDPlib ``mod_hens`` such a point was 5.05e-5 outside a Pyomo
row. The routes still mark it and return it, and ``Model.solve`` withdraws it
AFTER its #1537 repair, unless that repair produced a point ``verify_point``
accepts. Withdrawing inside the route was tried first and was wrong: on #1836's
robust row the tree's point is 5.7e-6 outside, the exit polish errors on the
``max`` kink, and the #1537 repair lands on ``0.5 + sqrt(10)`` exactly -- an
early withdrawal discarded that answer (``test_issue1836_ro_sign_monotone``).
"""

from __future__ import annotations

import linecache
import sys
import warnings

import discopt.modeling as dm
import discopt.solver as SV
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_status_without_incumbent():
    assert SV._status_without_incumbent(5.0, 10.0, 100, 100) == "node_limit"
    assert SV._status_without_incumbent(10.2, 10.0, 3, 100) == "time_limit"
    assert SV._status_without_incumbent(1.0, 10.0, 3, 100) == "unknown"
    assert SV._status_without_incumbent(1.0, None, 3, None) == "unknown"


def _spatial_gdp():
    m = dm.Model("g62_never_spatial")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 2], [x + y >= 5, x <= 3.5]], name="d")
    m.subject_to(x * y >= 1)
    m.minimize(-x * y + 0.3 * x**2 - y)
    return m


def _convex_gdp():
    """Convex with a nonlinear row: routes to NLP-B&B (as in the G6.2 defer test)."""
    m = dm.Model("g62_never_nlpbb")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 1], [x + y >= 3], [x - y >= 2]], name="d")
    m.subject_to(x**2 + y**2 <= 12)
    m.minimize((x - 1.6) ** 2 + (y - 0.4) ** 2)
    return m


def _refuse_at_exit(monkeypatch):
    """Spatial: every exit-gate verdict refuses. NLP-B&B: only the #1561 integral
    gate refuses (its #954 gate raises rather than reports, so it is left real),
    reached by making the integral realisation differ from the point by 1e-9."""
    calls = []
    real_excess = SV._nonlinear_point_excess
    real_round = SV._round_incumbent_integers

    def excess(ev, x, *a, **k):
        f = sys._getframe(1)
        if f.f_code.co_name == "_solve_nlp_bb":
            if x is not f.f_locals.get("_integral_point"):
                return real_excess(ev, x, *a, **k)
        calls.append(f.f_code.co_name)
        return 1.0, "row 0 (forced)", 1

    def rnd(x, *a, **k):
        f = sys._getframe(1)
        out = real_round(x, *a, **k)
        if f.f_code.co_name == "_solve_nlp_bb" and "_integral_point" in linecache.getline(
            f.f_code.co_filename, f.f_lineno
        ):
            pert = np.array(out[0], dtype=float)
            pert[0] += 1e-9
            return pert, out[1]
        return out

    monkeypatch.setattr(SV, "_nonlinear_point_excess", excess)
    monkeypatch.setattr(SV, "_round_incumbent_integers", rnd)
    return calls


@pytest.mark.parametrize(
    "build, stat",
    [
        (_spatial_gdp, "spatial/exit_gate_refused"),
        (_convex_gdp, "nlpbb/integral_incumbent_unverified"),
    ],
)
def test_a_refused_incumbent_is_withdrawn(build, stat, monkeypatch):
    """The repair is switched off, so nothing can verify the refused point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = build().solve(time_limit=30)
    assert ok.status == "optimal" and ok.x is not None  # the unforced control
    calls = _refuse_at_exit(monkeypatch)
    monkeypatch.setenv("DISCOPT_INCUMBENT_REPAIR", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = build().solve(time_limit=30)
    assert calls, "the exit gate never ran"
    stats = r.solver_stats or {}
    assert stats.get(stat) == 1.0 and stats.get("certificate/incumbent_withdrawn") == 1.0, stats
    assert r.x is None and r.objective is None and r.gap is None
    assert r.status in ("unknown", "time_limit", "node_limit")
    assert not r.gap_certified


def test_a_refused_point_the_repair_verifies_is_kept():
    """#1836's robust row: refused by the spatial exit gate, repaired by #1537."""
    from discopt.ro import BoxUncertaintySet, RobustCounterpart

    m = dm.Model("even_var")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=1.0)
    m.subject_to((x - p) ** 2 <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve()
    stats = r.solver_stats or {}
    if not stats.get("spatial/exit_gate_refused"):
        pytest.skip("the route accepted its own point on this run; nothing to repair")
    assert stats.get("certificate/incumbent_repaired") == 1.0, stats
    assert "certificate/incumbent_withdrawn" not in stats
    assert r.objective == pytest.approx(0.5 + np.sqrt(10.0), abs=1e-9)


def test_withdraw_is_a_no_op_without_a_refusal():
    from discopt.modeling.core import SolveResult, _withdraw_refused_incumbent

    r = SolveResult(status="feasible", objective=1.0, x={"x": np.array(1.0)})
    _withdraw_refused_incumbent(r, 10.0, None)
    assert r.objective == 1.0 and r.x is not None
