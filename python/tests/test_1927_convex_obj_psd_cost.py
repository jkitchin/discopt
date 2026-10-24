"""#1927: the convex-objective PSD test does not run an unbounded dense eigensolve.

``_objective_is_convex_quadratic`` evaluated two dense ``n x n`` objective Hessians
and ran ``eigvalsh`` with no size guard: on ``nd_netgen-2000-2-5-a-a-ns_7``
(n=9,999) it took 172.6 s and on ``popdynm200`` (n=11,215) 217.0 s, for the
verdict ``False`` both times. The test now works on the sparse Hessian: exact
O(nnz) checks first (a diagonal entry below the margin bounds lambda_min below
it; Gershgorin proves it), the dense eigensolve only up to
``_CONVEX_OBJ_DENSE_EIG_MAX_N``, and abstention above that.
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import solver
from discopt._tape_nlp_evaluator import build_evaluator


def _gate(model):
    n = sum(v.size for v in model._variables)
    return solver._objective_is_convex_quadratic(model, build_evaluator(model, lambda: None), n)


@pytest.fixture
def eig_calls(monkeypatch):
    calls = []
    real = np.linalg.eigvalsh

    def spy(a, *args, **kw):
        calls.append(np.shape(a))
        return real(a, *args, **kw)

    monkeypatch.setattr(np.linalg, "eigvalsh", spy)
    return calls


@pytest.mark.unit
def test_a_linear_only_column_decides_without_an_eigensolve(eig_calls):
    n = 5000
    m = dm.Model("big_linear_col")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    m.minimize(dm.sum(x[1:] * x[1:]) + x[0])
    assert _gate(m) is False
    assert eig_calls == []


@pytest.mark.unit
def test_a_diagonally_dominant_objective_is_proved_by_gershgorin(eig_calls):
    n = 5000
    m = dm.Model("big_dd")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    m.minimize(dm.sum(2.0 * x * x) + x[0] * x[1])
    assert _gate(m) is True
    assert eig_calls == []


@pytest.mark.unit
def test_a_small_pd_objective_still_takes_the_dense_eigensolve(eig_calls):
    m = dm.Model("small_pd")
    x = m.continuous("x", lb=-1, ub=1)
    y = m.continuous("y", lb=-1, ub=1)
    # PD but not diagonally dominant: Hessian [[2, 3], [3, 5]], det 1.
    m.minimize(x * x + 3 * x * y + 2.5 * y * y)
    assert _gate(m) is True
    assert eig_calls == [(2, 2)]


@pytest.mark.unit
def test_above_the_cap_an_undecided_matrix_abstains(eig_calls):
    # PD, not diagonally dominant, larger than the dense cap: abstain, no eigensolve.
    n = solver._CONVEX_OBJ_DENSE_EIG_MAX_N + 2
    k = n // 2
    rows = np.arange(k)
    # 2x2 blocks [[1, 1.9], [1.9, 4]]: det 0.39 > 0 (PD), row 1 not dominant.
    H = sp.lil_matrix((n, n))
    H[2 * rows, 2 * rows] = 1.0
    H[2 * rows + 1, 2 * rows + 1] = 4.0
    H[2 * rows, 2 * rows + 1] = 1.9
    H[2 * rows + 1, 2 * rows] = 1.9
    assert solver._sparse_hessian_is_pd_with_margin(H.tocsr()) is False
    assert eig_calls == []
    # The same 2x2 block structure under the cap is decided by the eigensolve.
    small = sp.csr_matrix(np.array([[1.0, 1.9], [1.9, 4.0]]))
    assert solver._sparse_hessian_is_pd_with_margin(small) is True
    assert eig_calls == [(2, 2)]
