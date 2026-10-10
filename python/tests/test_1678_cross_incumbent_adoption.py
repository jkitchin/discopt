"""#1678 II.19: the #1634 cross-check discarded the presolve-free solve's better point.

``_weaker_bound`` lowers a certified ``optimal`` bound to the presolve-free solve's
bound and re-tests the gap against the PRIMARY incumbent. The cross bound is certified
relative to the cross solve's own incumbent, which is often better (measured on a
multi-knapsack panel, ``NV=30, NR=5, gap_tolerance=0.01``, seed 103: primary -126.0943,
cross -126.5926 -- the route published the worse point). Re-testing the cross bound
against the worse point can reopen a gap that is closed, and the certificate was then
declined; and the better, route-verified point was thrown away either way.

The fix adopts the cross incumbent when it passes ``_verified_mip_point`` and is
better. No bound is adopted from it. ``milp_highs._cross_check_presolve`` (the OA/GDP
master path) already returned the better verified point; this aligns the LP/MILP
route with it.
"""

from __future__ import annotations

import discopt.solvers.lp_milp_highs as L
import numpy as np
import pytest
from discopt import Model

OPT = -7.0
PRIMARY_OBJ = -6.9996  # inside the 1e-4 relative gap of the true bound -7
CROSS_BOUND = -7.0005  # closes against -7, reopens against -6.9996


def _model():
    m = Model("adopt_1678")
    x = m.continuous("x", lb=0.0, ub=1.0)
    z = m.binary("z")
    m.subject_to(x <= z)
    m.minimize(-7.0 * x)
    return m


def _forge(monkeypatch, *, cross_x_better: bool):
    """Primary: a feasible but slightly worse point, its bound the true -7. Cross: the
    optimum, with a bound below -7 that a 1e-4 rel gap allows around it. The #1640
    tight re-run gets no verdict, so it cannot rescue the certificate."""
    real = L._solve_milp_scaled
    calls: list[str] = []

    def fake(sf, *, presolve=True, **kw):
        if kw.get("feasibility_tolerance", 1e-6) < 1e-6:
            calls.append("tight")
            return L.HighsOutcome("time_limit", message="forced no verdict")
        out = real(sf, presolve=presolve, **kw)
        assert out.status == "optimal" and out.x is not None
        col = int(np.flatnonzero(sf.c == -7.0)[0])
        if presolve:
            calls.append("primary")
            x = out.x.copy()
            x[col] = PRIMARY_OBJ / -7.0
            out.x, out.objective, out.bound = x, PRIMARY_OBJ, OPT
        else:
            calls.append("cross")
            if not cross_x_better:
                x = out.x.copy()
                x[col] = PRIMARY_OBJ / -7.0
                out.x, out.objective = x, PRIMARY_OBJ
            out.bound = CROSS_BOUND
        return out

    monkeypatch.setattr(L, "_solve_milp_scaled", fake)
    monkeypatch.setattr(L, "presolve_exact_class", lambda sf: False)
    return calls


def test_better_cross_point_is_adopted_and_the_certificate_stands(monkeypatch):
    calls = _forge(monkeypatch, cross_x_better=True)
    r = _model().solve(time_limit=30, gap_tolerance=1e-4)
    assert calls[:2] == ["primary", "cross"], calls
    assert r.objective == pytest.approx(OPT, abs=1e-9)
    assert r.status == "optimal" and r.gap_certified
    # The published bound is the weaker (cross) one, never an unconfirmed claim.
    assert r.bound == pytest.approx(CROSS_BOUND, abs=1e-9)
    assert r.bound <= r.objective
    assert r.solver_stats["milp/presolve_cross_incumbent_adopted"] == 1.0


def test_without_a_better_point_the_decline_stands(monkeypatch):
    """Control: the same weaker bound with no better verified point is still declined."""
    calls = _forge(monkeypatch, cross_x_better=False)
    r = _model().solve(time_limit=30, gap_tolerance=1e-4)
    assert "cross" in calls and "tight" in calls, calls
    assert r.status == "feasible" and not r.gap_certified
    assert r.objective == pytest.approx(PRIMARY_OBJ, abs=1e-9)
    assert r.bound == pytest.approx(CROSS_BOUND, abs=1e-9)
    assert "milp/presolve_cross_incumbent_adopted" not in r.solver_stats


def test_unverifiable_cross_point_is_not_adopted(monkeypatch):
    """A cross point failing the route's verifier never replaces the incumbent."""
    calls = _forge(monkeypatch, cross_x_better=True)
    real_vp = L._verified_mip_point
    seen: list = []

    def reject_optimum(sf, x):
        pt = real_vp(sf, x)
        if "cross" in calls and pt is not None and pt[1] < PRIMARY_OBJ - 1e-9:
            seen.append(pt[1])
            return None
        return pt

    monkeypatch.setattr(L, "_verified_mip_point", reject_optimum)
    r = _model().solve(time_limit=30, gap_tolerance=1e-4)
    assert seen, "the verifier was never asked about the better point"
    assert "cross" in calls
    assert r.objective == pytest.approx(PRIMARY_OBJ, abs=1e-9)
    assert not r.gap_certified


def test_tolerance_level_gain_is_not_adopted(monkeypatch):
    """A gain within the equality yardstick is verifier tolerance, not a better point:
    on the #1640 piecewise case the cross point verifies at -7.3e-7 against the exact
    optimum 0, and adopting it published a super-optimal incumbent."""
    from test_1640_cross_check_tolerance_bound import _log_max

    seen: list = []
    real = L._weaker_bound

    def spy(sf, out, cross, kw):
        pt = L._verified_mip_point(sf, cross.x)
        mine = L._verified_mip_point(sf, out.x)
        res = real(sf, out, cross, kw)
        seen.append((pt, mine, dict(res.stats)))
        return res

    monkeypatch.setattr(L, "_weaker_bound", spy)
    m, truth = _log_max()
    r = m.solve(time_limit=30)
    assert len(seen) == 1
    pt, mine, stats = seen[0]
    assert pt is not None and mine is not None
    assert 0.0 < mine[1] - pt[1] <= L.CERT_ABS  # the cross point IS marginally better
    assert "milp/presolve_cross_incumbent_adopted" not in stats
    assert r.status == "optimal" and r.objective == pytest.approx(truth, abs=1e-6)
