"""POUNCE QP/LP callbacks declare only the structural nonzeros.

``qp_pounce._QPCallbacks`` used to declare every ``m*n`` Jacobian entry
(``np.meshgrid``) and the whole lower triangle of ``Q`` (``np.tril_indices``) as
structural nonzeros, so POUNCE factored a dense KKT system. On the COPS 3.0
``torsion`` convex QP (two single-variable rows per grid cell), peak RSS was
0.25 GB at 144 variables and 5.41 GB at 484 (97.4 s), and 1,024 and 2,704
variables were OOM-killed at 15 GB. With only the nonzeros declared: 0.14 GB / 1.5 s
at 484 variables and 1.10 GB / 22.7 s at 2,704, with the objective unchanged to
1e-15. ``lp_pounce._LPCallbacks`` had the same dense Jacobian.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers.lp_pounce import _LPCallbacks
from discopt.solvers.qp_pounce import _QPCallbacks


def _sparse_problem(n: int = 6):
    rng = np.random.default_rng(0)
    A = np.zeros((2 * n, n))
    A[np.arange(n), np.arange(n)] = 1.0  # x_i <= D_i rows
    A[n + np.arange(n), np.arange(n)] = -1.0  # -x_i <= D_i rows
    Q = np.diag(rng.uniform(1.0, 2.0, n))
    Q[0, 1] = Q[1, 0] = 0.5
    return Q, rng.normal(size=n), A


def test_qp_callbacks_declare_only_nonzeros():
    Q, c, A = _sparse_problem()
    cb = _QPCallbacks(Q, c, A)
    rows, cols = cb.jacobianstructure()
    assert len(rows) == np.count_nonzero(A) == 2 * Q.shape[0]
    hr, hc = cb.hessianstructure()
    assert len(hr) == np.count_nonzero(np.tril(Q))
    assert np.all(hr >= hc)  # lower triangle

    # The declared entries carry exactly the matrix values.
    J = np.zeros_like(A)
    J[rows, cols] = cb.jacobian(np.zeros(A.shape[1]))
    np.testing.assert_array_equal(J, A)
    H = np.zeros_like(Q)
    H[hr, hc] = cb.hessian(np.zeros(Q.shape[0]), np.zeros(A.shape[0]), 2.0)
    np.testing.assert_array_equal(H, 2.0 * np.tril(Q))


def test_lp_callbacks_declare_only_nonzeros():
    _, c, A = _sparse_problem()
    cb = _LPCallbacks(c, A)
    rows, cols = cb.jacobianstructure()
    assert len(rows) == np.count_nonzero(A)
    J = np.zeros_like(A)
    J[rows, cols] = cb.jacobian(np.zeros(A.shape[1]))
    np.testing.assert_array_equal(J, A)


def test_all_zero_rows_and_q_are_accepted():
    n = 3
    cb = _QPCallbacks(np.zeros((n, n)), np.ones(n), np.zeros((2, n)))
    assert len(cb.jacobianstructure()[0]) == 0
    assert len(cb.hessianstructure()[0]) == 0


@pytest.mark.timeout(120)
def test_box_rowed_convex_qp_solves_at_size():
    """COPS-torsion shape: a convex QP whose rows are all single-variable.

    At 30x30 (1,024 variables) the dense declaration was OOM-killed at 15 GB.
    """
    nx = ny = 30
    hx, hy = 1.0 / (nx + 1), 1.0 / (ny + 1)
    D = np.array(
        [
            [min(min(i, nx - i + 1) * hx, min(j, ny - j + 1) * hy) for j in range(ny + 2)]
            for i in range(nx + 2)
        ]
    )
    m = dm.Model("torsion_shape")
    v = m.continuous("v", shape=(nx + 2, ny + 2), lb=-np.inf, ub=np.inf)
    quad = dm.sum(((v[1:, :] - v[:-1, :]) / hx) ** 2) + dm.sum(((v[:, 1:] - v[:, :-1]) / hy) ** 2)
    m.minimize(0.5 * hx * hy * quad - 5.0 * hx * hy * dm.sum(v))
    for i in range(nx + 2):
        for j in range(ny + 2):
            m.subject_to(v[i, j] >= -float(D[i, j]))
            m.subject_to(v[i, j] <= float(D[i, j]))
    r = m.solve(time_limit=90, initial_solution={v: D})
    assert r.objective is not None and np.isfinite(r.objective)
    x = np.asarray(r.x["v"]).reshape(nx + 2, ny + 2)
    assert np.all(np.abs(x) <= D + 1e-6)
