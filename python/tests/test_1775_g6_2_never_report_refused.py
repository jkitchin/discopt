"""#1775 G6.2: an incumbent the exit gate refuses is never reported.

Owner decision (2026-10-23). Before, a point that failed the exit gate even after
the repair polish (#1916) was returned with its objective and ``x``, merely
uncertified -- on GDPlib ``mod_hens`` such a point was 5.05e-5 outside a Pyomo
row. It is now withdrawn on both routes: no ``x``, no objective, the status names
the resource that ran out (``"unknown"`` if none did), and the dual bound stands.
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
    [(_spatial_gdp, "spatial/incumbent_withdrawn"), (_convex_gdp, "nlpbb/incumbent_withdrawn")],
)
def test_a_refused_incumbent_is_withdrawn(build, stat, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = build().solve(time_limit=30)
    assert ok.status == "optimal" and ok.x is not None  # the unforced control
    calls = _refuse_at_exit(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = build().solve(time_limit=30)
    assert calls, "the exit gate never ran"
    assert (r.solver_stats or {}).get(stat) == 1.0, r.solver_stats
    assert r.x is None and r.objective is None and r.gap is None
    assert r.status in ("unknown", "time_limit", "node_limit")
    assert not r.gap_certified
