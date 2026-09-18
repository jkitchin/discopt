"""A genuinely unbounded MILP came back `error` -- the second gap #1339 left open.

Certifying `unbounded` needs a verified recession ray of the relaxation AND an
integer-feasible point (Meyer: ``rec(conv(S)) = rec(P)`` for rational data with
``S`` nonempty). HiGHS supplies the ray but reports ``kUnbounded`` with NO
incumbent -- it stopped because the objective ran away, not because the system is
empty -- so the point was missing only because nobody asked for it. Asking is
#1337's move again: drop the objective, keep the rows, the box and the
integrality, and solve for feasibility alone.

The OTHER gap #1339 named -- a feasible convex QP returning `error` because
POUNCE is the only QP backend and its point failed the KKT-stationarity guard --
is NOT fixed here. A HiGHS convex-QP fallback was built and reverted: HiGHS's QP
solver hangs or dies with SIGFPE on large-magnitude models, which is exactly the
class the gap is about (POUNCE fails there because the model is ill-scaled).
Measured on ``min 0.5*2*(w-0.5)^2 + y`` with ``y in [-L, 0]``: fine at L = 1e3,
hangs at L >= 1e12, and `test_bound_beyond_pounce_infinity_is_still_relaxed`
(L = 5e19) either hangs or takes the process down with a floating-point
exception. A fallback that can kill the process is worse than the `error` it
replaces.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers.lp_milp_highs import StdForm, solve_milp_std

# No marker: fast regression fences on a certification defect, like #1319/#1320's.


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
