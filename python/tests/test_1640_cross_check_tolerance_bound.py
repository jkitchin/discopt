"""#1640: the #1634 presolve-free cross-solve's bound is a feasibility-tolerance artifact.

On the #1494 piecewise ``log``/max case at a large offset the presolved HiGHS solve
certifies 0 exactly, while the presolve-free cross-solve stops at bound -1.5e-6:
HiGHS's tree bound over node LPs solved to ``mip_feasibility_tolerance = 1e-6``. The
bound scales with that tolerance (-1.5e-8 at 1e-8) and does not move with
``mip_abs_gap = mip_rel_gap = 0``. #1638 published the weaker bound, which reopened the
1e-6 absolute gap and lost the certificate. The fix re-asks the cross-solve at
:data:`CROSS_TIGHT_FEAS_TOL` only when its bound reopened the gap.
"""

from __future__ import annotations

import discopt.solvers.lp_milp_highs as L
import numpy as np
import pytest
from discopt import Model

_SHAPE = np.array([0.0, 5.0, -5.0, 0.0, 3.0])


def _log_max(offset: float = 1e4, h: float = 1.0, frac: float = 1.5):
    b = offset + h * np.arange(_SHAPE.size)
    m = Model("pwl_1640")
    x = m.continuous("x", lb=float(b[0]), ub=float(b[-1]))
    y = m.piecewise(x, b, _SHAPE, method="log", name="y")
    m.subject_to(x == offset + frac * h)
    m.maximize(y)
    return m, float(np.interp(offset + frac * h, b, _SHAPE))


def _spy(monkeypatch):
    seen: list = []
    real = L._weaker_bound

    def spy(sf, out, cross, kw):
        res = real(sf, out, cross, kw)
        seen.append(dict(res.stats))
        return res

    monkeypatch.setattr(L, "_weaker_bound", spy)
    return seen


def test_tight_cross_solve_restores_the_certificate(monkeypatch):
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    seen = _spy(monkeypatch)
    m, truth = _log_max()
    r = m.solve(time_limit=30)
    assert len(seen) == 1, "the #1634 cross-check did not run"
    st = seen[0]
    assert st.get("milp/presolve_cross_bound_lowered") == 1.0
    assert st.get("milp/presolve_cross_tight_confirmed") == 1.0
    # The tight bound is ~100x closer than the 1e-6 one, not merely inside the gap.
    assert abs(st["milp/presolve_cross_tight_bound"]) < 1e-7
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(truth, abs=1e-6)
    # Maximize: the published bound is an upper bound, never below the optimum.
    assert r.bound >= truth - 1e-9


def test_untightened_cross_solve_still_declines(monkeypatch):
    """Mechanism pin: with the re-run at the original 1e-6 tolerance the cross bound
    is unchanged, the gap stays open, and the result is declined -- never certified."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    monkeypatch.setattr(L, "CROSS_TIGHT_FEAS_TOL", 1e-6)
    seen = _spy(monkeypatch)
    m, truth = _log_max()
    r = m.solve(time_limit=30)
    assert len(seen) == 1 and seen[0].get("milp/presolve_cross_tight_ran") == 1.0
    assert "milp/presolve_cross_tight_confirmed" not in seen[0]
    assert r.status == "feasible" and not r.gap_certified
    assert r.objective == pytest.approx(truth, abs=1e-6)


def test_tight_cross_without_a_verdict_keeps_the_decline(monkeypatch):
    """The tight solve must earn the confirmation: one with no certified verdict leaves
    the certificate declined (the #1309 rule)."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    real = L._solve_milp_scaled

    def no_verdict(sf, *, presolve=True, **kw):
        if kw.get("feasibility_tolerance", 1e-6) < 1e-6:
            return L.HighsOutcome("time_limit", message="forced no verdict")
        return real(sf, presolve=presolve, **kw)

    monkeypatch.setattr(L, "_solve_milp_scaled", no_verdict)
    seen = _spy(monkeypatch)
    m, _ = _log_max()
    r = m.solve(time_limit=30)
    assert len(seen) == 1 and seen[0].get("milp/presolve_cross_tight_ran") == 1.0
    assert not r.gap_certified and r.status == "feasible"


def test_tight_cross_point_refutes_the_claim(monkeypatch):
    """A verified point of the tight solve below the primary's bound by more than the
    equality yardstick refutes it, exactly as the first cross-solve's point would."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    real = L._solve_milp_scaled

    def lifted(sf, *, presolve=True, **kw):
        out = real(sf, presolve=presolve, **kw)
        if kw.get("feasibility_tolerance", 1e-6) < 1e-6:
            fired.append(1)
        return out

    fired: list = []
    monkeypatch.setattr(L, "_solve_milp_scaled", lifted)
    real_vp = L._verified_mip_point

    def low_point(sf, x):
        pt = real_vp(sf, x)
        if fired and pt is not None:
            return pt[0], pt[1] - 1.0
        return pt

    monkeypatch.setattr(L, "_verified_mip_point", low_point)
    seen = _spy(monkeypatch)
    m, _ = _log_max()
    r = m.solve(time_limit=30)
    assert fired, "the tight cross-solve never ran"
    assert "milp/presolve_cross_tight_confirmed" not in seen[0]
    assert not r.gap_certified


def test_tight_bound_far_above_the_cross_bound_keeps_the_decline(monkeypatch):
    """Confirming drops the cross bound, so the tight solve may only explain it as
    tolerance error: a tight bound above it by more than the tolerance-scaled slack is
    two certified solves disagreeing, and the decline stands (review of #1649)."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    real = L._solve_milp_scaled

    def jumped(sf, *, presolve=True, **kw):
        out = real(sf, presolve=presolve, **kw)
        if kw.get("feasibility_tolerance", 1e-6) < 1e-6 and out.bound is not None:
            out.bound = float(out.bound) + 1e3
        return out

    monkeypatch.setattr(L, "_solve_milp_scaled", jumped)
    seen = _spy(monkeypatch)
    m, _ = _log_max()
    r = m.solve(time_limit=30)
    assert seen[0].get("milp/presolve_cross_tight_ran") == 1.0
    assert seen[0]["milp/presolve_cross_tight_shift"] > 1e2
    assert "milp/presolve_cross_tight_confirmed" not in seen[0]
    assert not r.gap_certified and r.status == "feasible"


def test_tight_solve_budget_charges_only_the_cross_solve(monkeypatch):
    """The caller already took the primary's wall time off ``kw``; charging it again
    starved the tight solve of budget it had (review of #1649)."""
    got: list = []

    def capture(sf, *, presolve=True, **kw):
        got.append(kw["time_limit"])
        return L.HighsOutcome("time_limit")

    monkeypatch.setattr(L, "_solve_milp_scaled", capture)
    out = L.HighsOutcome("optimal", objective=0.0, bound=-1.5e-6, gap_certified=True)
    out.wall_time = 5.0
    cross = L.HighsOutcome("optimal", objective=0.0, bound=-1.5e-6, gap_certified=True)
    cross.wall_time = 4.0
    kw = dict(time_limit=6.0, gap_tolerance=1e-4, abs_gap_tolerance=None)
    assert L._tight_cross_bound(None, out, cross, kw, 0.0) is False
    assert got == [pytest.approx(2.0)]


def test_no_budget_for_the_tight_solve_keeps_the_decline():
    out = L.HighsOutcome("optimal", objective=0.0, bound=-1.5e-6, gap_certified=True)
    out.wall_time = 5.0
    cross = L.HighsOutcome("optimal", objective=0.0, bound=-1.5e-6, gap_certified=True)
    cross.wall_time = 5.0
    kw = dict(time_limit=5.0, gap_tolerance=1e-4, abs_gap_tolerance=None)
    assert L._tight_cross_bound(None, out, cross, kw, 0.0) is False
    assert out.stats.get("milp/presolve_cross_tight_skipped") == 1.0
    assert out.bound == -1.5e-6
