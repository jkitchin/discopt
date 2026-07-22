"""Regression tests for #850: cross-backend certificate consistency on the pure
LP fast path.

Three related cross-backend inconsistencies were reported by the adversary sweep
(2026-07-22), all on the model-level LP fast path where ``nlp_solver`` only
selects the LP *engine* and must therefore not change the certificate:

* **Obs 1** — ``min -x`` with the *default* continuous box ``x ∈ [0, 9.999e19]``.
  The exact simplex (whose infinity threshold is exactly ``1e20``) honors
  ``9.999e19`` as a finite bound and returns ``optimal`` at the corner; the
  interior-point engine (POUNCE) relaxes any ``|bound| ≥ 1e15`` to ``±∞`` (its
  barrier cannot condition so huge a finite bound) and returns ``unbounded``.
  Same model, contradictory certificates depending on ``nlp_solver``.

* **Obs 2** — the same direction on the *integer* default box (``1e6``), which is
  below the ``1e15`` relaxation threshold, so every backend already agrees on
  ``optimal`` at ``-1e6``. Kept here as a consistency pin.

* **Obs 3** — ``min -x`` s.t. the *general inequality* ``x ≤ 1e4`` (not a bound).
  POUNCE returns a point ``1e-4`` on the infeasible side of the row (its fixed
  Ipopt ``constr_viol_tol`` floor), i.e. a super-optimal, constraint-infeasible
  primal, which the LP-level feasibility guard accepted because its tolerance
  scaled with the *global* largest variable (``1e-6·1e4 = 1e-2``) rather than the
  row's own term magnitude.

Decision taken (documented on the issue): the default box is the problem **as
posed** — a real (if huge) finite bound — so the sound certificate is ``optimal``
at the corner, which the exact simplex, ``ipm`` and ``ipopt`` already produce.
The two fixes make POUNCE agree with that exact oracle rather than emit a
contradicting certificate:

1. ``_matrix_solution_feasible`` now uses the per-row term-scaled tolerance
   ``|viol_i| ≤ tol + rtol·Σ_j|A_ij||x_j|`` (the same convention as
   ``primal_heuristics._check_constraint_feasibility``), so a super-optimal
   infeasible primal is rejected and the caller degrades to the exact simplex
   (Obs 3).
2. ``_solve_lp_matrix`` declines to certify ``unbounded`` when the engine relaxed
   a declared finite bound in ``[1e15, 1e20)`` to the IPM infinity, deferring to
   the exact simplex, which honors the declared box (Obs 1 & 2).

POUNCE is not required: the fixes are exercised with stub LP engines that mimic
POUNCE's two failure modes, plus end-to-end ``ipm``/``ipopt`` assertions. When
POUNCE *is* installed, the final test pins full cross-backend agreement.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402
import warnings  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.solvers import LPResult, SolveStatus  # noqa: E402

pytestmark = pytest.mark.unit


# ── Unit: the per-row term-scaled feasibility guard (Obs 3 root cause) ──


def test_matrix_guard_rejects_super_optimal_infeasible_point():
    """A point 1e-4 on the infeasible side of ``x ≤ 1e4`` is NOT feasible: the
    per-row threshold is ``1e-6 + 1e-9·1e4 ≈ 1.1e-5``, far below the 1e-4
    violation. The pre-fix global scaling gave ``1e-6·(1+1e4) ≈ 1e-2`` and wrongly
    accepted it."""
    x = np.array([10000.0000999])
    A_ub = np.array([[1.0]])
    b_ub = np.array([1e4])
    assert not S._matrix_solution_feasible(x, A_ub, b_ub, None, None, [(0.0, 1e6)])


def test_matrix_guard_accepts_exact_and_noise_points():
    """The exact vertex ``x = 1e4`` (what the simplex returns) is feasible, and so
    is a point carrying only cancellation noise proportional to the row terms."""
    A_ub = np.array([[1.0]])
    b_ub = np.array([1e4])
    assert S._matrix_solution_feasible(np.array([1e4]), A_ub, b_ub, None, None, [(0.0, 1e6)])
    # noise ~1e-9·|x| on a large-term row is forgiven (rtol term)
    x_noise = np.array([1e4 + 5e-6])  # 5e-6 > 1e-6 abs, but < 1e-6 + 1e-9·1e4
    assert S._matrix_solution_feasible(x_noise, A_ub, b_ub, None, None, [(0.0, 1e6)])


def test_matrix_guard_still_catches_gross_mislabel():
    """The guard's original purpose — rejecting a grossly infeasible point flagged
    optimal (the ~7.5 HiGHS QP case) — is preserved."""
    x = np.array([1e6])
    A_ub = np.array([[1.0]])
    b_ub = np.array([1e6 - 7.5])
    assert not S._matrix_solution_feasible(x, A_ub, b_ub, None, None, [(0.0, 2e6)])


def test_matrix_guard_equality_and_bound_rows():
    """Equality-row and bound-row violations use the same per-row scaling."""
    # equality x == 1e4, point off by 1e-3 -> infeasible
    assert not S._matrix_solution_feasible(
        np.array([1e4 + 1e-3]), None, None, np.array([[1.0]]), np.array([1e4]), [(0.0, 1e6)]
    )
    # bound violation 1e-3 above ub -> infeasible
    assert not S._matrix_solution_feasible(
        np.array([1e4 + 1e-3]), None, None, None, None, [(0.0, 1e4)]
    )
    # exactly on the bound -> feasible
    assert S._matrix_solution_feasible(np.array([1e4]), None, None, None, None, [(0.0, 1e4)])


# ── Unit: the relaxed-bound detector (Obs 1 root cause) ──


def test_declared_box_relaxed_detector():
    """Only bounds in ``[1e15, 1e20)`` — finite to the simplex, ``±∞`` to the IPM —
    are flagged; genuine infinities and ordinary bounds are not."""
    assert S._declared_box_relaxed_to_ipm_inf([(0.0, 9.999e19)])  # continuous default
    assert S._declared_box_relaxed_to_ipm_inf([(-9.999e19, 9.999e19)])
    assert S._declared_box_relaxed_to_ipm_inf([(0.0, 1e15)])  # at the threshold
    assert not S._declared_box_relaxed_to_ipm_inf([(0.0, 1e20)])  # genuine inf (simplex too)
    assert not S._declared_box_relaxed_to_ipm_inf([(0.0, np.inf)])  # genuine inf
    assert not S._declared_box_relaxed_to_ipm_inf([(0.0, 1e6)])  # integer default, finite
    assert not S._declared_box_relaxed_to_ipm_inf([(0.0, 10.0)])
    assert not S._declared_box_relaxed_to_ipm_inf(None)
    assert not S._declared_box_relaxed_to_ipm_inf([])


# ── Integration: _solve_lp_matrix degrades on the two POUNCE failure modes ──


def _stub_unbounded(**kwargs):
    return LPResult(status=SolveStatus.UNBOUNDED)


def _stub_super_optimal_infeasible(
    c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, **kw
):
    # x = 1e4 + 1e-4 : optimal-labelled but 1e-4 past the x<=1e4 row (super-optimal)
    return LPResult(
        status=SolveStatus.OPTIMAL, x=np.array([10000.0000999]), objective=-10000.0000999
    )


def _stub_exact(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, **kw):
    return LPResult(status=SolveStatus.OPTIMAL, x=np.array([1e4]), objective=-1e4)


def test_obs1_unbounded_with_relaxed_bound_degrades():
    """Obs 1: an engine that reports UNBOUNDED on the default continuous box (a
    bound it silently relaxed to ∞) must not be certified — ``_solve_lp_matrix``
    returns None so the caller falls through to the exact simplex."""
    m = dm.Model("u")
    x = m.continuous("x", lb=0)  # default ub = 9.999e19
    m.minimize(-x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = S._solve_lp_matrix(m, time.perf_counter(), None, _stub_unbounded, "STUB-IPM")
    assert res is None


def test_obs1_genuine_unbounded_still_reported():
    """A genuinely infinite bound (ub = 1e20, infinite to the simplex too) is NOT a
    relaxation, so an UNBOUNDED verdict is passed through, never masked."""
    m = dm.Model("u2")
    x = m.continuous("x", lb=0, ub=1e20)
    m.minimize(-x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = S._solve_lp_matrix(m, time.perf_counter(), None, _stub_unbounded, "STUB-IPM")
    assert res is not None and res.status == "unbounded"


def test_obs3_super_optimal_infeasible_degrades():
    """Obs 3: an optimal-labelled point violating a general inequality past the
    per-row tolerance is rejected -> ``_solve_lp_matrix`` returns None (degrade)."""
    m = dm.Model("km")
    x = m.continuous("x", lb=0, ub=1e6)
    m.minimize(-x)
    m.subject_to(x <= 1e4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = S._solve_lp_matrix(
            m, time.perf_counter(), None, _stub_super_optimal_infeasible, "STUB-IPM"
        )
    assert res is None


def test_obs3_exact_point_accepted():
    """Control: the exact feasible vertex the simplex returns is accepted."""
    m = dm.Model("km2")
    x = m.continuous("x", lb=0, ub=1e6)
    m.minimize(-x)
    m.subject_to(x <= 1e4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = S._solve_lp_matrix(m, time.perf_counter(), None, _stub_exact, "STUB-IPM")
    assert res is not None and res.status == "optimal"
    assert res.objective == pytest.approx(-1e4, abs=1e-6)


# ── End-to-end: the certificate every backend must produce ──


@pytest.mark.parametrize("backend", ["ipm", "ipopt"])
def test_obs1_default_box_optimal_at_corner(backend):
    """min -x on the default continuous box returns ``optimal`` at the sentinel
    corner on the exact/ipm/ipopt paths — the certificate POUNCE must now match."""
    m = dm.Model("unb")
    x = m.continuous("x", lb=0)
    m.minimize(-x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(nlp_solver=backend)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-9.999e19, rel=1e-9)


@pytest.mark.parametrize("backend", ["ipm", "ipopt"])
def test_obs2_default_integer_box_optimal(backend):
    """Obs 2 pin: the integer default box (1e6, below the relaxation threshold) is
    a real finite bound on every backend -> optimal at -1e6."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = dm.Model("intunb")
        x = m.integer("x", lb=0)
        m.minimize(-x)
        r = m.solve(nlp_solver=backend)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1e6, rel=1e-9)


@pytest.mark.parametrize("backend", ["ipm", "ipopt"])
def test_obs3_general_inequality_feasible_optimum(backend):
    """Obs 3 end-to-end: the returned point sits exactly on ``x ≤ 1e4`` (no
    super-optimal violation) with objective -10000 on every backend."""
    m = dm.Model("km")
    x = m.continuous("x", lb=0, ub=1e6)
    m.minimize(-x)
    m.subject_to(x <= 1e4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(nlp_solver=backend)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1e4, abs=1e-4)
    assert r.value(x) <= 1e4 + 1e-5  # feasible, not super-optimal


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda m: (m.continuous("x", lb=0), m.minimize(-m._variables[0])), id="obs1"),
        pytest.param(
            lambda m: (
                m.continuous("x", lb=0, ub=1e6),
                m.minimize(-m._variables[0]),
                m.subject_to(m._variables[0] <= 1e4),
            ),
            id="obs3",
        ),
    ],
)
def test_cross_backend_agreement_when_pounce_present(build):
    """When POUNCE is installed, all of ipm/ipopt/pounce must return the SAME
    status and objective — the core cross-backend-consistency property of #850.
    Skipped where POUNCE is absent (the stub tests above cover its behavior)."""
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE

    if not POUNCE_AVAILABLE:
        pytest.skip("pounce not installed; stub tests cover the degrade path")

    results = {}
    for backend in ["ipm", "ipopt", "pounce"]:
        m = dm.Model(f"xb_{backend}")
        build(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(nlp_solver=backend)
        results[backend] = (r.status, r.objective)

    statuses = {s for s, _ in results.values()}
    assert len(statuses) == 1, f"cross-backend status disagreement: {results}"
    objs = [o for _, o in results.values() if o is not None]
    if objs:
        assert max(objs) - min(objs) <= 1e-4 * (1 + abs(objs[0])), (
            f"cross-backend objective disagreement: {results}"
        )
