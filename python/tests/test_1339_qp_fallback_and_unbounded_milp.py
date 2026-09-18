"""The two gaps PR #1339 measured and left open, now closed.

**A. A feasible convex QP came back `error`.** ``_solve_qp_matrix`` guards every
QP answer twice -- primal feasibility, and KKT stationarity for a backend that
reports a residual. POUNCE was the only QP backend, so a point the stationarity
guard rejected left the route with nothing at all. Measured on
``min (w - 0.5)**2`` beside an unrelated row of activity ~1e14: POUNCE's KKT
residual is 9.2e-05 against a 1e-06 bar, the guard correctly refuses it, and a
problem whose optimum is 0 at ``w = 0.5`` reported `error`.

``QPResult.kkt_error`` has always documented the remedy -- it is ``None`` "for
vertex solvers like HiGHS that reach an exact optimum", so a caller "can degrade
to a vertex solver instead of trusting a drifted objective". ``qp_highs`` is that
solver, invoked THROUGH the same guards rather than around them. That is the
difference from the JAX QP IPM rescue #359 removed from this exact position,
which "degraded past the guard" by issuing its own certificate from its own
convergence flag.

**B. A genuinely unbounded MILP came back `error`.** Certifying `unbounded` needs
a verified recession ray of the relaxation AND an integer-feasible point (Meyer:
``rec(conv(S)) = rec(P)`` for rational data with ``S`` nonempty). HiGHS supplies
the ray but reports ``kUnbounded`` with NO incumbent -- it stopped because the
objective ran away, not because the system is empty -- so the point was missing
only because nobody asked for it. Asking is #1337's move again: drop the
objective, keep the rows, box and integrality, and solve for feasibility alone.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_milp_highs import StdForm, solve_milp_std

# No marker: fast regression fences on a certification defect, like #1319/#1320's.


# ─────────────────────────────────────────────────────────────
# A. the QP fallback
# ─────────────────────────────────────────────────────────────


def _feasible_qp(huge: bool):
    """``min (w - 0.5)**2`` -- optimum 0 at w = 0.5 -- with an unrelated,
    perfectly satisfiable row of activity ~1e14 that is what defeats POUNCE."""
    m = dm.Model("i1339_qp")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.subject_to(a - b <= 2)
    m.subject_to(a - b >= 1)
    if huge:
        L = 1e14
        z = m.continuous("z", lb=L, ub=2 * L)
        v = m.continuous("v", lb=0, ub=2 * L)
        m.subject_to(z - v <= 0)
    w = m.continuous("w", lb=-1, ub=1)
    m.minimize((w - 0.5) ** 2)
    return m


def test_a_feasible_convex_qp_is_solved_not_errored():
    """The repro. Pre-fix: ``error`` (POUNCE's point failed the KKT gate and there
    was no second engine)."""
    r = _feasible_qp(huge=True).solve(time_limit=60)
    assert r.status == "optimal", f"got {r.status!r}"
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert float(np.asarray(r.x["w"])) == pytest.approx(0.5, abs=1e-5)


def test_the_small_sibling_is_unchanged():
    """Control: without the huge row POUNCE succeeds and nothing degrades."""
    r = _feasible_qp(huge=False).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-9)


def test_the_fallback_does_not_run_when_pounce_succeeds(monkeypatch):
    """The fallback is a LAST resort: a QP the primary engine solves must never
    reach it, so nothing that works today can change."""
    from discopt import solver as S

    calls = []
    real = S._solve_qp_highs

    def _spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(S, "_solve_qp_highs", _spy)
    r = _feasible_qp(huge=False).solve(time_limit=60)
    assert r.status == "optimal"
    assert not calls, "the HiGHS fallback ran on a QP POUNCE already solved"


def test_an_infeasible_qp_is_never_certified_optimal():
    """The direction that would be a false certificate."""
    m = dm.Model("i1339_qp_infeas")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.subject_to(a - b <= 0)
    m.subject_to(a - b >= 1)
    w = m.continuous("w", lb=-1, ub=1)
    m.minimize((w - 0.5) ** 2)
    r = m.solve(time_limit=60)
    assert r.status not in ("optimal", "feasible"), f"got {r.status!r}"


# ── unit: the backend's refusals are the load-bearing part ──────────────


def test_backend_never_emits_an_uncrosschecked_verdict():
    """``_solve_qp_matrix`` maps a backend's INFEASIBLE/UNBOUNDED straight to a
    certificate. POUNCE earns that by running the mandatory Phase-1 cross-check
    inside its own ``solve_qp`` (#1319); a raw HiGHS label carries no such check,
    so this backend must return ERROR for both rather than let one through."""
    from discopt.solvers.qp_highs import solve_qp

    checked = 0
    # genuinely infeasible: x <= 1 and x >= 2
    r = solve_qp(
        Q=np.eye(1),
        c=np.zeros(1),
        A_ub=np.array([[1.0], [-1.0]]),
        b_ub=np.array([1.0, -2.0]),
        bounds=[(-10.0, 10.0)],
    )
    assert r.status is SolveStatus.ERROR, f"emitted {r.status!r} for an infeasible QP"
    checked += 1
    # genuinely unbounded: min -x over x >= 0
    r2 = solve_qp(Q=np.zeros((1, 1)), c=np.array([-1.0]), bounds=[(0.0, np.inf)])
    assert r2.status is SolveStatus.ERROR, f"emitted {r2.status!r} for an unbounded QP"
    checked += 1
    assert checked == 2


def test_backend_refuses_a_nonconvex_hessian():
    """HiGHS solves convex QPs only; a nonconvex one answered as though convex is a
    false certificate, so the backend refuses rather than returning a local point."""
    from discopt.solvers.qp_highs import solve_qp

    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_qp(Q=np.array([[-2.0]]), c=np.array([0.0]), bounds=[(-1.0, 1.0)])


def test_backend_refuses_integrality():
    from discopt.solvers.qp_highs import solve_qp

    with pytest.raises(ValueError, match="integrality"):
        solve_qp(Q=np.eye(1), c=np.zeros(1), bounds=[(0.0, 1.0)], integrality=np.array([1]))


def test_backend_solves_a_convex_qp_with_rows():
    """The positive half, on a problem with a known closed-form answer:
    ``min 0.5(x^2 + y^2)`` s.t. ``x + y >= 2`` has its optimum at ``x = y = 1``."""
    from discopt.solvers.qp_highs import solve_qp

    r = solve_qp(
        Q=np.eye(2),
        c=np.zeros(2),
        A_ub=np.array([[-1.0, -1.0]]),
        b_ub=np.array([-2.0]),
        bounds=[(-10.0, 10.0)] * 2,
    )
    assert r.status is SolveStatus.OPTIMAL
    assert r.x == pytest.approx([1.0, 1.0], abs=1e-6)
    assert r.objective == pytest.approx(1.0, abs=1e-6)
    assert r.kkt_error is None, "a vertex solver reports no residual (the gate skips it)"


# ─────────────────────────────────────────────────────────────
# B. the unbounded MILP
# ─────────────────────────────────────────────────────────────


def _unbounded_milp_sf() -> StdForm:
    """``min -y`` s.t. ``2x == 2``, x integer in [0,5], y in [0, 1e20]."""
    return StdForm.from_arrays(
        c=np.array([0.0, -1.0]),
        A=np.array([[2.0, 0.0]]),
        b=np.array([2.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([5.0, 1e20]),
        int_idx=np.array([0]),
    )


def test_an_unbounded_milp_is_certified_unbounded():
    """The repro. Pre-fix: ``error`` -- HiGHS returns kUnbounded with no incumbent,
    so Meyer's theorem had no integer-feasible point to stand on."""
    m = dm.Model("i1339_milp")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 2)
    y = m.continuous("y", lb=0, ub=1e20)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    assert r.status == "unbounded", f"got {r.status!r}"


def test_the_witness_is_produced_and_is_integer_feasible():
    out = solve_milp_std(
        _unbounded_milp_sf(), time_limit=60.0, gap_tolerance=1e-4, max_nodes=10_000
    )
    assert out.status == "unbounded"
    assert out.stats.get("milp/unbounded_feasibility_probe") == 1.0
    assert out.labels.get("milp/unbounded_provenance") == "meyer-ray-plus-integer-point"
    assert out.x is not None, "an `unbounded` certificate must carry its witness"
    x = np.asarray(out.x, dtype=np.float64)
    # 2x == 2 -> x = 1, and it must actually be an integer.
    assert x[0] == pytest.approx(1.0, abs=1e-6)
    assert abs(x[0] - round(float(x[0]))) <= 1e-5


def test_a_bounded_sibling_is_unchanged():
    m = dm.Model("i1339_milp_bounded")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 2)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-10.0, abs=1e-6)


def test_an_integrally_infeasible_sibling_is_still_infeasible():
    """#1337's case must be untouched: ``2x == 1`` has no integer solution, and the
    probe must not turn that into an ``unbounded``."""
    m = dm.Model("i1339_milp_infeas")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 1)
    y = m.continuous("y", lb=0, ub=1e20)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    assert r.status == "infeasible", f"got {r.status!r}"
    assert r.status != "unbounded"


def test_the_probe_declines_with_no_budget(monkeypatch):
    """No budget is not a verdict: the route reports ``error`` exactly as before."""
    from discopt.solvers import lp_milp_highs as L

    out = L._integer_feasible_point(_unbounded_milp_sf(), 0.0, {})
    assert out is None


def test_the_probe_verifies_the_point_it_returns(monkeypatch):
    """The witness is the whole evidence for the certificate, so a probe that
    returns a point failing re-checking yields no point at all."""
    from discopt.solvers import lp_milp_highs as L
    from discopt.solvers.lp_milp_highs import HighsOutcome

    def _bogus(sf, **kw):
        # integral, but violates 2x == 2 outright
        return HighsOutcome("optimal", x=np.array([4.0, 0.0]), objective=0.0)

    monkeypatch.setattr(L, "solve_milp_std", _bogus)
    assert L._integer_feasible_point(_unbounded_milp_sf(), 30.0, {}) is None


def test_the_probe_asks_a_zero_objective_question(monkeypatch):
    """It must differ from the original solve in the objective alone."""
    from discopt.solvers import lp_milp_highs as L

    seen = []
    real = L.solve_milp_std

    def _record(sf, **kw):
        seen.append(sf)
        return real(sf, **kw)

    monkeypatch.setattr(L, "solve_milp_std", _record)
    sf = _unbounded_milp_sf()
    L._integer_feasible_point(sf, 30.0, {})
    assert len(seen) == 1, f"expected exactly the probe solve, got {len(seen)}"
    probe = seen[0]
    assert np.all(probe.c == 0.0), "the probe must drop the objective"
    assert probe.obj_const == 0.0
    assert np.array_equal(probe.b, sf.b)
    assert np.array_equal(probe.xl, sf.xl)
    assert np.array_equal(probe.xu, sf.xu)
    assert np.array_equal(probe.int_idx, sf.int_idx)
    assert np.array_equal(probe.A.toarray(), sf.A.toarray())
