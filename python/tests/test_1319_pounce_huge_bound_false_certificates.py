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


@pytest.mark.parametrize("L", [9e14, 2e15, 5e15])
def test_huge_bound_lp_is_not_certified_infeasible(monkeypatch, L):
    """The repro: a feasible LP over ~1e15 bounds came back certified infeasible.

    ``(L + 1, L)`` is exactly feasible, so ``infeasible`` is simply wrong. The
    Phase-1 threshold now accounts for the row activity, POUNCE's unconfirmed
    code 2 degrades to ``error``, and the exact simplex -- which honors the
    declared box -- supplies the certificate.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    r = _pinch_model(L).solve()
    assert r.status != "infeasible", "certified a false 'infeasible' on a feasible LP"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(6 * L - 1, rel=1e-9)


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


def test_huge_bound_qp_is_not_certified_infeasible():
    """The default (no env var) QP route certified ``infeasible`` on a model with
    an exact feasible witness: v0=-0.375, v1=5e15, v3=-5e15, w=0.5 gives a row
    value of exactly -2.0 >= -2."""
    m = dm.Model("q")
    m.continuous("v0", lb=-1e16, ub=1e16)
    m.continuous("v1", lb=5e15, ub=2e18)
    m.continuous("v3", lb=-5e15, ub=1e16)
    m.continuous("w", lb=0, ub=1)
    v0, v1, v3, w = m._variables
    m.subject_to(5 * v0 - v1 - v3 >= -2)
    m.minimize(5 * v1 + 2 * v3 + (w - 0.5) ** 2)

    r = m.solve()
    assert r.status != "infeasible", "certified a false 'infeasible' on a feasible QP"
    assert not (r.status == "infeasible" and r.gap_certified)


def test_solve_qp_code2_not_confirmed_by_phase1_reports_error():
    """Unit-level counterpart: called directly, the same system must not come back
    ``INFEASIBLE``. There is no second QP engine to degrade to (#359), so the
    honest status is ``ERROR`` -- CLAUDE.md §1 takes that over a false certificate.
    """
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


def test_bounded_qp_over_huge_bounds_is_not_reported_unbounded():
    """A QP bounded below by its declared box must never come back ``unbounded``.

    The IPM relaxed ``y >= -1e15`` to its own infinity, so its verdict describes a
    larger box than the declared one. With no second QP engine to defer to, the
    guard reports ``error`` -- honest, and not a certificate about a box nobody
    posed.
    """
    with pytest.warns(RuntimeWarning, match=r"\[1e15, 1e20\)"):
        r = _bounded_qp_model(1e15).solve()
    assert r.status != "unbounded", "reported 'unbounded' for a QP bounded below by -1e15"
    assert r.status == "error"


def test_qp_below_the_relaxation_window_is_unaffected():
    """Regression fence: the guard keys on the ``[1e15, 1e20)`` window only, so the
    same QP one order smaller still solves to its analytic optimum."""
    L = 9e14
    r = _bounded_qp_model(L).solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-L, rel=1e-9)
