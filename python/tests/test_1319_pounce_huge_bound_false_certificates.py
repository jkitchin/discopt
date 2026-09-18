"""#1319: the POUNCE LP/QP route certified wrong statuses on huge-bound problems.

Three sibling defects, all follow-ups to #1309:

1. **LP Phase-1 threshold.** #1309 made Ipopt code 2 depend on the elastic Phase-1
   LP, on the grounds that a positive minimal violation is a Farkas certificate.
   That holds only if Phase-1 is solved exactly -- but it is solved by the same IPM
   on the same relaxed box, and ``_is_infeasible_violation`` scaled its threshold
   only with the right-hand side, never with ``|A|·|x|``. On a box reaching ~1e15
   those differ by twenty orders of magnitude, so pure cancellation noise read as a
   Farkas certificate and a trivially feasible LP came back certified ``infeasible``.
2. **QP code 2.** ``qp_pounce.solve_qp`` still trusted raw Ipopt code 2 on the
   *default* route: the ``elif certificate and ... INFEASIBLE`` branch was left
   unconverted when the ``lp_pounce`` one was fixed.
3. **QP unbounded.** The LP route has the #850 guard for a declared finite bound in
   ``[1e15, 1e20)`` that the IPM relaxes to its own infinity; ``_solve_qp_matrix``
   had none and trusted the verdict, calling a bounded QP ``unbounded``.

A wrong *certified* status is the one thing CLAUDE.md §1 admits no slack on. Where
an engine honoring the declared box exists (the exact simplex, case 1) the right
answer is recovered; where none does (the QP route is POUNCE-only by design, #359)
the honest outcome is ``error``, never an unverified certificate -- the #850/#937
precedent.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_pounce import (
    _INF,
    _is_infeasible_violation,
    _phase1_min_violation,
    _stack_constraints,
)
from discopt.solvers.qp_pounce import solve_qp

# No marker: these are fast regression fences on a *certification* defect, so they
# belong in the default PR suite (same as #1309's).


# ---------------------------------------------------------------------------
# 1. The LP Phase-1 cross-check must not certify a false ``infeasible``
# ---------------------------------------------------------------------------


def _pinch_model(L: float) -> dm.Model:
    """``max x1 + x2`` over a thin but nonempty sliver at magnitude ``L``.

    ``x1 - x2 <= 1`` and ``x2 - x1 <= -1`` force ``x1 - x2 == 1`` exactly. The
    sliver is narrower than the floating-point resolution of the difference at
    ``L ~ 1e15``, which is what makes the IPM's Phase-1 residual look like a
    violation. The optimum is ``x1 = 3L, x2 = 3L - 1``, i.e. ``6L - 1``.
    """
    m = dm.Model("pinch")
    m.continuous("x1", lb=L, ub=3 * L)
    m.continuous("x2", lb=-3 * L, ub=3 * L)
    x1, x2 = m._variables
    m.subject_to(x1 - x2 <= 1)
    m.subject_to(x2 - x1 <= -1)
    m.maximize(x1 + x2)
    return m


@pytest.mark.parametrize("declared_box", ["1", "0"])
@pytest.mark.parametrize("L", [9e14, 2e15, 5e15])
def test_huge_bound_lp_is_not_certified_infeasible(monkeypatch, L, declared_box):
    """The repro: a feasible LP over ~1e15 bounds came back certified infeasible.

    ``(L + 1, L)`` is exactly feasible, so ``infeasible`` is simply wrong. Run on
    BOTH box thresholds: the Phase-1 row-activity fix is what holds this line, and
    it must hold whether or not ``DISCOPT_POUNCE_DECLARED_BOX`` is engaged -- the
    §1 guarantee cannot be contingent on a performance flag.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", declared_box)
    r = _pinch_model(L).solve()
    assert r.status != "infeasible", "certified a false 'infeasible' on a feasible LP"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(6 * L - 1, rel=1e-9)


# ---------------------------------------------------------------------------
# 1b. The declared-box threshold itself
# ---------------------------------------------------------------------------


def test_declared_box_threshold_matches_pounce_own_infinity(monkeypatch):
    """The threshold is POUNCE's documented limit, not a discopt guess.

    Ipopt's ``nlp_{lower,upper}_bound_inf`` default to ∓1e19, and #1319 measured
    POUNCE returning the exact optimum up to 9.9e18 and flipping to UNBOUNDED at
    exactly 1e19. ``=0`` restores the legacy 1e15.
    """
    from discopt.solvers.lp_pounce import (
        _LEGACY_BOUND_THRESHOLD,
        _POUNCE_BOUND_INF,
        finite_bound_threshold,
    )

    assert _POUNCE_BOUND_INF == 1e19
    assert _LEGACY_BOUND_THRESHOLD == 1e15

    # Default-OFF pending the §5 graduation gate; flip this assertion together
    # with the default when the differential panel graduates the flag.
    monkeypatch.delenv("DISCOPT_POUNCE_DECLARED_BOX", raising=False)
    assert finite_bound_threshold() == _LEGACY_BOUND_THRESHOLD, (
        "default must be OFF until the §5 panel graduates the flag"
    )
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "0")
    assert finite_bound_threshold() == _LEGACY_BOUND_THRESHOLD, "=0 must opt out"
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "1")
    assert finite_bound_threshold() == _POUNCE_BOUND_INF


def test_declared_box_window_tracks_the_live_threshold(monkeypatch):
    """The #850 deferral window and the box marshaling must not drift apart: a
    hardcoded 1e15 in the guard would defer verdicts the IPM no longer relaxes."""
    from discopt.solver import _declared_box_relaxed_to_ipm_inf

    box = [(0.0, 1.0), (-1e16, 0.0)]  # 1e16 is inside the legacy window only
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "0")
    assert _declared_box_relaxed_to_ipm_inf(box) is True
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "1")
    assert _declared_box_relaxed_to_ipm_inf(box) is False
    # A bound at/above POUNCE's own infinity is relaxed under either setting.
    assert _declared_box_relaxed_to_ipm_inf([(0.0, 1.0), (-5e19, 0.0)]) is True


def test_phase1_roundoff_is_not_read_as_a_violation():
    """Unit-level: the Phase-1 slack on the feasible sliver is below the
    row-activity noise floor, so it must not certify infeasibility.

    Asserted on the measured quantities rather than on the verdict alone, so a
    future change that makes the verdict right for the wrong reason still fails.
    """
    L = 9e14
    A, cl, cu = _stack_constraints(
        np.array([[1.0, -1.0], [-1.0, 1.0]]), np.array([1.0, -1.0]), None, None, 2
    )
    lb = np.array([L, -3 * L])
    ub = np.array([3 * L, 3 * L])
    opts = {"print_level": 0, "constr_viol_tol": 1e-8, "bound_relax_factor": 0.0}

    phase1 = _phase1_min_violation(A, cl, cu, lb, ub, opts)
    assert phase1 is not None, "the well-posed elastic Phase-1 LP must solve"
    total = float(phase1.slacks.sum())
    activity = float(np.max(phase1.row_activity))
    # The residual is sub-epsilon relative to the arithmetic that produced it.
    assert activity > 1e14
    assert total / activity < 1e-14
    assert not _is_infeasible_violation(phase1.slacks, cl, cu, phase1.row_activity)


def test_phase1_still_certifies_a_genuine_infeasibility_at_huge_magnitude():
    """The guard must not be a blanket amnesty: a data-significant violation at
    the same magnitude is still certified, so the #1319 fix does not cost the
    Phase-1 certificate the class it exists for."""
    L = 9e14
    # x1 + x2 == 0 and x1 + x2 == L: minimal total violation is L.
    A, cl, cu = _stack_constraints(
        None, None, np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.0, L]), 2
    )
    lb = np.full(2, -3 * L)
    ub = np.full(2, 3 * L)
    opts = {"print_level": 0, "constr_viol_tol": 1e-8, "bound_relax_factor": 0.0}

    phase1 = _phase1_min_violation(A, cl, cu, lb, ub, opts)
    assert phase1 is not None
    assert _is_infeasible_violation(phase1.slacks, cl, cu, phase1.row_activity)


def test_no_phase1_point_means_no_certificate():
    """``row_activity=None`` (no Phase-1 point) can never certify: there is
    nothing to measure the violation against."""
    assert not _is_infeasible_violation(
        np.array([1e9, 1e9]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), None
    )


# ---------------------------------------------------------------------------
# 2. The QP route must cross-check raw Ipopt code 2 the way the LP route does
# ---------------------------------------------------------------------------


def _witness_qp_model() -> dm.Model:
    """The #1319 part-2 model. Exact feasible witness: v0=-0.375, v1=5e15,
    v3=-5e15, w=0.5 gives a row value of exactly -2.0 >= -2, objective 1.5e16."""
    m = dm.Model("q")
    m.continuous("v0", lb=-1e16, ub=1e16)
    m.continuous("v1", lb=5e15, ub=2e18)
    m.continuous("v3", lb=-5e15, ub=1e16)
    m.continuous("w", lb=0, ub=1)
    v0, v1, v3, w = m._variables
    m.subject_to(5 * v0 - v1 - v3 >= -2)
    m.minimize(5 * v1 + 2 * v3 + (w - 0.5) ** 2)
    return m


def test_huge_bound_qp_is_not_certified_infeasible(monkeypatch):
    """The QP route certified ``infeasible`` on this feasible model.

    With the declared box honored it is solved outright to the witness objective
    5*5e15 + 2*(-5e15) + 0 = 1.5e16.
    """
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "1")
    r = _witness_qp_model().solve()
    assert r.status != "infeasible", "certified a false 'infeasible' on a feasible QP"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.5e16, rel=1e-9)


def test_huge_bound_qp_never_certifies_infeasible_with_the_legacy_box(monkeypatch):
    """Opt-out arm: with ``DISCOPT_POUNCE_DECLARED_BOX=0`` the IPM goes back to
    discarding the bound, so POUNCE still raises its numerical code 2 -- but the
    code-2 cross-check must keep that from becoming a certificate. The honest
    outcome is ``error`` (there is no second QP engine to degrade to, #359).

    This is the guard that holds the §1 line independently of the flag.
    """
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "0")
    r = _witness_qp_model().solve()
    assert r.status != "infeasible", "certified a false 'infeasible' on a feasible QP"
    assert not r.gap_certified


def test_solve_qp_code2_not_confirmed_by_phase1_reports_error(monkeypatch):
    """Unit-level counterpart on the legacy box: called directly, the same system
    must not come back ``INFEASIBLE`` -- CLAUDE.md §1 takes ``ERROR`` over a false
    certificate."""
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "0")
    n = 4
    Q = np.zeros((n, n))
    Q[3, 3] = 2.0
    c = np.array([0.0, 5.0, 2.0, -1.0])
    # -(5 v0 - v1 - v3) <= 2
    A_ub = np.array([[-5.0, 1.0, 1.0, 0.0]])
    b_ub = np.array([2.0])
    bounds = [(-1e16, 1e16), (5e15, 2e18), (-5e15, 1e16), (0.0, 1.0)]

    res = solve_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, certificate=True)
    assert res.status != SolveStatus.INFEASIBLE


def test_solve_qp_still_certifies_a_genuine_infeasibility():
    """A real conflict is still reported infeasible, with its witness attached --
    the code-2 cross-check tightens the certificate, it does not remove it."""
    Q = np.zeros((2, 2))
    c = np.array([1.0, 1.0])
    A_eq = np.array([[1.0, 1.0], [1.0, 1.0]])
    b_eq = np.array([1.0, 5.0])
    res = solve_qp(Q=Q, c=c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, _INF)] * 2, certificate=True)
    assert res.status == SolveStatus.INFEASIBLE
    cert = res.infeasibility_certificate
    assert cert is not None
    assert cert.total_violation == pytest.approx(4.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. The QP route needs the #850 huge-bound unbounded guard
# ---------------------------------------------------------------------------


def _bounded_qp_model(L: float) -> dm.Model:
    """``min (w - 0.5)^2 + y`` with ``w in [0,1]``, ``y in [-L, 0]``, ``w + y <= 1``.

    Bounded below: the analytic optimum is ``-L`` at ``w = 0.5, y = -L``.
    """
    m = dm.Model("q3")
    m.continuous("w", lb=0, ub=1)
    m.continuous("y", lb=-L, ub=0)
    w, y = m._variables
    m.subject_to(w + y <= 1)
    m.minimize((w - 0.5) ** 2 + y)
    return m


@pytest.mark.parametrize("L", [1e15, 1e16, 1e18])
def test_bounded_qp_over_huge_bounds_solves_to_its_analytic_optimum(monkeypatch, L):
    """A QP bounded below by its declared box came back ``unbounded``.

    POUNCE honors a finite bound up to its own 1e19 infinity, so the declared box
    reaches it and the analytic optimum ``-L`` is recovered across the whole window
    the legacy 1e15 threshold used to discard.
    """
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "1")
    r = _bounded_qp_model(L).solve()
    assert r.status != "unbounded", f"reported 'unbounded' for a QP bounded below by -{L:g}"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-L, rel=1e-9)


def test_bounded_qp_is_never_reported_unbounded_with_the_legacy_box(monkeypatch):
    """Opt-out arm: on the legacy 1e15 box the IPM still discards ``y >= -1e15``,
    so its verdict describes a larger box than the declared one. The #850/#1319
    guard must refuse to certify it -- ``error``, not ``unbounded``."""
    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "0")
    with pytest.warns(RuntimeWarning, match=r"relaxed a declared"):
        r = _bounded_qp_model(1e15).solve()
    assert r.status != "unbounded", "reported 'unbounded' for a QP bounded below by -1e15"
    assert r.status == "error"


def test_qp_below_the_relaxation_window_is_unaffected():
    """Regression fence: a box well below either threshold is untouched by both."""
    L = 9e14
    r = _bounded_qp_model(L).solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-L, rel=1e-9)


def test_bound_beyond_pounce_infinity_is_still_relaxed(monkeypatch):
    """The upper fence: at/above POUNCE's own 1e19 infinity the bound genuinely is
    infinite to the engine, so the guard must still decline to certify rather than
    trusting a verdict about a box POUNCE could not see."""
    from discopt.solvers.lp_pounce import _POUNCE_BOUND_INF, finite_bound_threshold

    monkeypatch.setenv("DISCOPT_POUNCE_DECLARED_BOX", "1")
    assert finite_bound_threshold() == _POUNCE_BOUND_INF == 1e19
    with pytest.warns(RuntimeWarning, match=r"relaxed a declared"):
        r = _bounded_qp_model(5e19).solve()
    assert r.status != "unbounded"


# ── the regression the #1319 fix itself introduced ────────────────────────
#
# Defect 1 above raised the infeasibility bar by a row-activity roundoff floor.
# The bar moved, but the READING of "under the bar" did not: the code took it as
# proof of FEASIBILITY. It is not. A genuine conflict whose minimal violation is
# small in absolute terms sits under that floor whenever the Phase-1 point is
# large, and calling it feasible let the raw Ipopt code-3/4 ``UNBOUNDED`` survive
# the ray check -- certifying an INFEASIBLE problem as ``unbounded``, trading
# #1319's false `infeasible` for a false `unbounded`.
#
# The measured case: ``x - y <= 1`` with ``x - y >= 10`` over ``x, y in
# [1e15, 3e15]``. True minimal total violation 9; row activity 4e15; floor
# 1e-12 * 4e15 = 4000. main certified `infeasible` here; the first #1319 fix did
# not. The answer is a third verdict -- undecided is not feasible.


def test_small_genuine_violation_at_huge_activity_is_not_read_as_feasible():
    """The exact numbers from the regression: a real violation of 9 under a
    4000 roundoff floor must come back UNDECIDED, never FEASIBLE."""
    from discopt.solvers.lp_pounce import PHASE1_UNDECIDED, _phase1_verdict

    slacks = np.array([0.0, 9.0])
    cl = np.array([-1e20, 10.0])
    cu = np.array([1.0, 1e20])
    activity = np.array([2e15, 2e15])
    assert _phase1_verdict(slacks, cl, cu, activity) == PHASE1_UNDECIDED


def test_phase1_verdict_separates_all_three_cases():
    """Feasible / undecided / infeasible are three distinct answers, and the
    clean-zero and clearly-violated ends must keep their old verdicts."""
    from discopt.solvers.lp_pounce import (
        PHASE1_FEASIBLE,
        PHASE1_INFEASIBLE,
        PHASE1_UNDECIDED,
        _phase1_verdict,
    )

    cl = np.array([-1e20, 10.0])
    cu = np.array([1.0, 1e20])
    activity = np.array([2e15, 2e15])
    assert _phase1_verdict(np.array([0.0, 0.0]), cl, cu, activity) == PHASE1_FEASIBLE
    assert _phase1_verdict(np.array([0.0, 9.0]), cl, cu, activity) == PHASE1_UNDECIDED
    assert _phase1_verdict(np.array([0.0, 1e14]), cl, cu, activity) == PHASE1_INFEASIBLE
    # No Phase-1 point certifies nothing in EITHER direction (guard kept from the
    # original fix -- reading it against the rhs alone restores the false
    # `infeasible`, reading it as feasible restores the false `unbounded`).
    assert _phase1_verdict(np.array([1e9, 1e9]), cl, cu, None) == PHASE1_UNDECIDED


def test_infeasible_lp_at_huge_magnitude_is_never_certified_unbounded():
    """End to end: the conflicting system above must not come back ``unbounded``
    on any route. ``infeasible`` (the exact simplex decides it) or ``error`` (no
    engine could) are both acceptable; a certified ``unbounded`` is not."""
    L = 1e15
    m = dm.Model("infeas_at_scale")
    x = m.continuous("x", lb=L, ub=3 * L)
    y = m.continuous("y", lb=L, ub=3 * L)
    m.subject_to(x - y <= 1.0)
    m.subject_to(x - y >= 10.0)
    m.minimize(x + y)
    r = m.solve(time_limit=60)
    assert r.status != "unbounded", f"infeasible LP certified {r.status!r}"
