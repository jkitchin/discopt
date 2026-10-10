"""#1683: power-of-two column equilibration of the Model -> POUNCE qp-ipm hand-off.

qp-ipm's iteration count depended on the units of the variables: the linear MPC
QP below is the same problem for every ``UA``, yet took 13 iterations at
``UA = 1`` and 115 at ``UA = 6e4`` (94 vs 14 at N = 1200). The route now hands
the engine ``D P D``, ``A D``, ``G D``, ``lb/d``, ``ub/d`` with every ``d_j`` a
power of two, maps ``x = d*xt`` and ``z_b = zt_b / d`` back exactly, and judges
the result on residuals recomputed in the caller's units.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt.solvers import convex_ipm_pounce as cvx

pytestmark = [pytest.mark.requires_pounce]


def _mpc(N=200, UA=1.0):
    m = Model("mpc")
    x = m.continuous("x", shape=(N + 1,), lb=-50, ub=50)
    u = m.continuous("u", shape=(N,), lb=-2.0 * UA, ub=2.0 * UA)
    m.subject_to(x[0] == 10.0)
    m.subject_to(x[1:] == 0.95 * x[:-1] + (0.1 / UA) * u)
    m.minimize(dm.sum(x**2) + 0.01 * dm.sum(u**2) / UA**2)
    return m


def _mpc_matrices(N=200, UA=1.0):
    """The same QP as matrix data: columns ``[x_0..x_N, u_0..u_{N-1}]``."""
    nx, nu = N + 1, N
    n = nx + nu
    Q = sp.diags(np.r_[np.full(nx, 2.0), np.full(nu, 0.02 / UA**2)]).tocsr()
    c = np.zeros(n)
    A = sp.lil_matrix((N + 1, n))
    A[0, 0] = 1.0  # x_0 == 10
    for k in range(N):  # x_{k+1} - 0.95 x_k - (0.1/UA) u_k == 0
        A[k + 1, k + 1] = 1.0
        A[k + 1, k] = -0.95
        A[k + 1, nx + k] = -0.1 / UA
    A_eq = A.tocsr()
    b_eq = np.r_[10.0, np.zeros(N)]
    bounds = [(-50.0, 50.0)] * nx + [(-2.0 * UA, 2.0 * UA)] * nu
    return Q, c, A_eq, b_eq, bounds


def _pounce(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(solver="pounce", **kw)


def _iters(res):
    return res.solve_report["statistics"]["iteration_count"]


# ----------------------------------------------------------------- the measurement


@pytest.mark.parametrize("UA", [1.0, 6e4, 1e3, 1e6, 1e-3])
def test_unit_scaled_mpc_iterations_do_not_depend_on_units(UA, monkeypatch):
    """Measured before the fix (N=200): 13 / 115 / 15 / 14 / 13 iterations; after:
    13 at every ``UA``.

    ``UA < 1`` makes ``u``'s coefficients *large*; the one-sided scale leaves that
    column alone (``test_column_scale_is_one_sided``), so that case is the
    unscaled solve, bit for bit: 13 iterations, objective 637.89256986 against
    637.89254939, with a valid certified bound 637.89253911 below both."""
    monkeypatch.delenv(cvx.COLSCALE_ENV, raising=False)
    ref = _pounce(_mpc(UA=1.0))
    res = _pounce(_mpc(UA=UA))
    assert res.status == "optimal" and res.algorithm_route == "pounce:qp-ipm"
    assert res.objective == pytest.approx(ref.objective, rel=1e-9 if UA >= 1 else 1e-7)
    assert res.bound <= ref.objective + 1e-9
    assert _iters(res) <= 16, _iters(res)


def test_opt_out_restores_the_unscaled_hand_off(monkeypatch):
    """``DISCOPT_POUNCE_QP_COLSCALE=0`` hands POUNCE the caller's matrices unchanged."""
    import pounce.qp as pq

    seen = []
    orig = pq.solve_qp

    def spy(*a, **kw):
        seen.append(kw)
        return orig(*a, **kw)

    monkeypatch.setattr(pq, "solve_qp", spy)
    Q, c, A_eq, b_eq, bounds = _mpc_matrices(N=20, UA=6e4)
    lb = np.array([b[0] for b in bounds])
    monkeypatch.setenv(cvx.COLSCALE_ENV, "0")
    cvx.solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    assert seen and all(np.array_equal(kw["lb"], lb) for kw in seen)
    seen.clear()
    monkeypatch.delenv(cvx.COLSCALE_ENV)
    cvx.solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    assert seen and not np.array_equal(seen[0]["lb"], lb)


# ------------------------------------------------------------------ exactness


def test_column_scale_is_powers_of_two_and_exact():
    Q, c, A_eq, b_eq, bounds = _mpc_matrices(N=30, UA=6e4)
    n = len(c)
    d = cvx.column_scale(Q, A_eq, None, n)
    assert d is not None
    mant, _ = np.frexp(d)
    assert np.all(mant == 0.5), "every d_j must be a power of two"
    Qs = cvx._scale_sym(Q, d)
    As = cvx._scale_cols(A_eq, d)
    # Exact round trip: dividing back by the same powers of two returns the data.
    np.testing.assert_array_equal(cvx._scale_sym(Qs, 1.0 / d).toarray(), Q.toarray())
    np.testing.assert_array_equal(cvx._scale_cols(As, 1.0 / d).toarray(), A_eq.toarray())
    lb = np.array([b[0] for b in bounds])
    np.testing.assert_array_equal((lb / d) * d, lb)
    # Equilibrated: every column's largest entry within a factor ~2 of 1.
    nrm = np.maximum(cvx._col_absmax(Qs, n), cvx._col_absmax(As, n))
    assert np.all((nrm > 0.25) & (nrm < 4.0)), (nrm.min(), nrm.max())


def test_unit_scaled_qp_is_handed_over_unchanged():
    Q = np.diag([2.0, 1.0])
    A = np.array([[1.0, 1.0]])
    assert cvx.column_scale(Q, A, None, 2) is None


def test_caller_residuals_match_the_engines_own_definition(monkeypatch):
    """The recomputation reproduces POUNCE's ``QpResiduals`` on an unscaled solve."""
    import pounce.qp as pq

    Q, c, A_eq, b_eq, bounds = _mpc_matrices(N=40, UA=1.0)
    c = c + 0.3  # a nonzero linear term, so every block of the residual is exercised
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    G = sp.csr_matrix(np.ones((1, len(c))))
    h = np.array([5.0])
    r = pq.solve_qp(P=Q, c=c, A=A_eq, b=b_eq, G=G, h=h, lb=lb, ub=ub)
    assert r.status == "optimal"
    mine = cvx.caller_residuals(r.x, r.y, r.z, r.z_lb, r.z_ub, Q, c, A_eq, b_eq, G, h, lb, ub)
    eng = r.residuals
    executed = 0
    for key in ("primal_infeasibility", "dual_infeasibility", "complementarity"):
        assert mine[key] == pytest.approx(eng[key], rel=1e-6, abs=1e-14), key
        executed += 1
    assert executed == 3


def test_scaled_and_unscaled_solves_agree_and_meet_kkt_in_caller_units(monkeypatch):
    Q, c, A_eq, b_eq, bounds = _mpc_matrices(N=60, UA=6e4)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    monkeypatch.setenv(cvx.COLSCALE_ENV, "0")
    off = cvx.solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    monkeypatch.setenv(cvx.COLSCALE_ENV, "1")
    on = cvx.solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    assert on.status == off.status == cvx.SolveStatus.OPTIMAL
    assert on.objective == pytest.approx(off.objective, rel=1e-8)
    assert on.iterations < off.iterations
    # KKT of the returned (unscaled) point and multipliers, on the caller's data.
    st = on.warm_start_state
    resid = cvx.caller_residuals(
        on.x, st["y"], None, st["z_lb"], st["z_ub"], Q, c, A_eq, b_eq, None, None, lb, ub
    )
    assert resid["primal_infeasibility"] <= 1e-8
    assert resid["bound_violation"] == 0.0
    scale = max(1.0, float(np.max(np.abs(Q @ on.x))))
    assert resid["dual_infeasibility"] <= 1e-8 * scale
    assert resid["complementarity"] <= 1e-8 * scale


def test_column_scale_is_one_sided():
    """``d >= 1`` always: a column whose entries are large keeps ``d_j = 1``.

    A ``d_j < 1`` would map the engine's dual residual back as ``r_j / d_j`` --
    amplified -- so the engine's ``tol`` would no longer hold in caller units.
    Measured with a two-sided scale on QPLIB_8938: the caller-unit dual residual
    came back 5.0e-8 against 3.9e-12 unscaled, the ``tol*sigma*min(d)`` re-solve
    then asked for 4.9e-12, below the 1.1e-11 primal-residual floor, and ran
    18 -> 219 iterations to the limit."""
    # A large-coefficient column (``k*j**2``, ``1e4*j`` in a row) is left alone ...
    P = sp.csc_matrix(np.diag([2.0, 1e6]))
    A = np.array([[1.0, -1e4]])
    assert cvx.column_scale(P, A, None, 2) is None
    # ... while a tiny-coefficient one is scaled up, and only it.
    A = np.array([[1.0, 1e-4]])
    d = cvx.column_scale(np.diag([2.0, 1e-8]), A, None, 2)
    assert d is not None and d[0] == 1.0 and d[1] > 1.0


def test_large_coefficient_retry_arm_is_unchanged(monkeypatch):
    """The #1537/#1658 ``tol*sigma`` re-solve still runs, with the column scale on."""
    monkeypatch.delenv(cvx.COLSCALE_ENV, raising=False)
    P = sp.csc_matrix(np.diag([2.0, 6400.0]))
    c = np.array([-0.8, -1.6])
    A_eq = np.array([[1.0, -1.0]])
    b_eq = np.array([0.4])
    lb, ub = np.array([0.0, -1e3]), np.array([1.0, 1e3])
    status, res, *_ = cvx._solve(P, c, None, None, A_eq, b_eq, lb, ub, None, None)
    assert status == "optimal"
    assert getattr(res, "retried_at_scaled_tol", False) is True
    assert cvx.caller_unit_converged(res, P, c, cvx._ENGINE_DEFAULT_TOL)


# ----------------------------------------------------------- warm start, parity


def test_warm_start_under_the_column_scale(monkeypatch):
    monkeypatch.delenv(cvx.COLSCALE_ENV, raising=False)
    m = _mpc(N=60, UA=6e4)
    r1 = _pounce(m)
    r2 = _pounce(m, warm_start=r1)
    assert r2.status == "optimal"
    assert r2.objective == pytest.approx(r1.objective, rel=1e-9)
    assert _iters(r2) <= 3, _iters(r2)


@pytest.mark.parametrize("UA", [1.0, 6e4])
def test_matrix_solve_qp_and_model_route_agree(UA, monkeypatch):
    """``convex_ipm_pounce.solve_qp`` on the matrices and ``Model.solve(solver="pounce")``
    on the model reach the same optimum in the same number of iterations (#1679 I.13)."""
    monkeypatch.delenv(cvx.COLSCALE_ENV, raising=False)
    Q, c, A_eq, b_eq, bounds = _mpc_matrices(N=200, UA=UA)
    mat = cvx.solve_qp(Q, c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    res = _pounce(_mpc(N=200, UA=UA))
    assert res.objective == pytest.approx(mat.objective, rel=1e-9)
    assert abs(_iters(res) - mat.iterations) <= 1, (_iters(res), mat.iterations)
