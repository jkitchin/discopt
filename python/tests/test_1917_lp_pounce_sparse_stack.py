"""#1917: ``lp_pounce`` keeps a sparse constraint system sparse.

``_stack_constraints`` densified ``A_ub``/``A_eq`` with ``np.vstack`` of full-width
rows. The convexity classifier's ``LinearContext.affine_range`` hands it the
model's whole linear system (CSR), and on the hull of GDPlib biofuel (~15k rows)
that one stack took the solve from 2.4 GB to 15.3 GB peak RSS and an OOM kill.
The phase-1 elastic LP added an ``m x m`` dense identity on top.

These tests fail on the dense stack: it returns an ``ndarray``, and a
6000 x 6000 system (288 MB dense, ~0.4 MB sparse) blows the allocation bound.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pounce")

from discopt.solvers import (
    SolveStatus,  # noqa: E402
    lp_pounce,  # noqa: E402
)

_N = 6000  # dense float64 n x n = 288 MB


def _banded_system(n: int):
    """``x_i - x_{i+1} <= 1`` chain plus ``sum`` of three neighbours ``== 1`` rows."""
    i = np.arange(n - 1)
    A_ub = sp.csr_matrix(
        (np.r_[np.ones(n - 1), -np.ones(n - 1)], (np.r_[i, i], np.r_[i, i + 1])),
        shape=(n - 1, n),
    )
    b_ub = np.ones(n - 1)
    k = np.arange(0, n - 2, 3)
    rows = np.repeat(np.arange(k.size), 3)
    cols = (k[:, None] + np.arange(3)).ravel()
    A_eq = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(k.size, n))
    b_eq = np.ones(k.size)
    return A_ub, b_ub, A_eq, b_eq


def test_stack_of_sparse_rows_is_sparse():
    A_ub, b_ub, A_eq, b_eq = _banded_system(50)
    A, cl, cu = lp_pounce._stack_constraints(A_ub, b_ub, A_eq, b_eq, 50)
    assert sp.issparse(A), type(A)
    assert A.shape == (A_ub.shape[0] + A_eq.shape[0], 50)
    assert A.nnz == A_ub.nnz + A_eq.nnz
    np.testing.assert_array_equal(A.toarray(), np.vstack([A_ub.toarray(), A_eq.toarray()]))
    np.testing.assert_array_equal(cu, np.r_[b_ub, b_eq])
    np.testing.assert_array_equal(cl[A_ub.shape[0] :], b_eq)


def test_dense_input_gives_the_same_solve_as_sparse():
    A_ub, b_ub, A_eq, b_eq = _banded_system(60)
    c = np.linspace(-1.0, 1.0, 60)
    bounds = [(-5.0, 5.0)] * 60
    rs = lp_pounce.solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
    rd = lp_pounce.solve_lp(c, A_ub.toarray(), b_ub, A_eq.toarray(), b_eq, bounds=bounds)
    assert rs.status == rd.status == SolveStatus.OPTIMAL
    assert rs.objective == pytest.approx(rd.objective, abs=1e-9)


def test_large_sparse_solve_stays_sparse_in_memory():
    A_ub, b_ub, A_eq, b_eq = _banded_system(_N)
    c = np.zeros(_N)
    c[0] = 1.0
    bounds = [(-10.0, 10.0)] * _N
    tracemalloc.start()
    try:
        res = lp_pounce.solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, time_limit=60)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.status == SolveStatus.OPTIMAL
    # A dense stack alone is 8 * rows * n bytes (~384 MB here).
    assert peak < 64 * 2**20, f"peak traced allocation {peak / 2**20:.0f} MB"


def test_phase1_and_point_check_accept_sparse():
    # An infeasible pair: x0 - x1 <= -1 and x1 - x0 <= -1.
    A = sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    cl = np.full(2, -lp_pounce._INF)
    cu = np.array([-1.0, -1.0])
    lb, ub = np.zeros(2), np.full(2, 5.0)
    p1 = lp_pounce._phase1_min_violation(A, cl, cu, lb, ub, {})
    assert p1 is not None and p1.slacks.sum() > 0.5
    assert lp_pounce._simplex_feasibility_verdict(A, cl, cu, lb, ub) == lp_pounce.PHASE1_INFEASIBLE
    assert not lp_pounce._point_satisfies(A, cl, cu, lb, ub, np.zeros(2))
