"""#1813: the Rust builder perturbed fast-API coefficients with absolute 1e-15 tests.

``ModelBuilder`` (``crates/discopt-core/src/expr.rs``) dropped any ``Q_ij``, ``c_j``
or constraint coefficient with ``|v| < 1e-15`` and snapped one within ``1e-15`` of
1.0 to exactly 1.0. The model the solver and the convexity certificate saw was
then not the model the user wrote: an indefinite ``Q`` came out PSD and was
certified convex. Coefficients are now stored exactly; only an exact 0.0 is
skipped and only an exact 1.0 drops its multiply.
"""

from __future__ import annotations

from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.convexity.certificate import certify_quadratic_objective_convex
from discopt._relax.problem_classifier import extract_qp_data

pytestmark = pytest.mark.smoke


def _qp(Q, c=None, *, lb=-1.0, ub=1.0):
    n = Q.shape[0]
    m = dm.Model("t")
    x = m.continuous("x", shape=(n,), lb=lb, ub=ub)
    m.add_quadratic_objective(sp.csr_matrix(Q), np.zeros(n) if c is None else c, x)
    return m


def _extracted(m):
    qd = extract_qp_data(m)
    Q = qd.Q.toarray() if sp.issparse(qd.Q) else np.asarray(qd.Q)
    return Q, np.asarray(qd.c)


def _laplacian_with_dent(n=60, k=30):
    """Path-graph Laplacian (PSD, singular) with ``L[k, k] = 2 - 2^-50``.

    Lowering one diagonal of a singular PSD matrix makes it indefinite; in the
    builder the coefficient ``0.5 * L[k, k] = 1 - 2^-51`` was within 1e-15 of 1
    and was snapped to 1, restoring the PSD matrix.
    """
    L = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    L[0, 0] = L[-1, -1] = 1.0
    L[k, k] = 2.0 - 2.0**-50
    return L


def test_indefinite_laplacian_is_kept_exactly_and_not_certified():
    L = _laplacian_with_dent()
    # The input really is indefinite: ``1' L 1 = -2^-50``, in exact rational
    # arithmetic. Not ``eigvalsh``: the true lambda_min is about -1.5e-17, below its
    # resolution, and Accelerate's LAPACK returns +1.1e-16 for it.
    assert sum(Fraction(v) for v in L.flat) == -Fraction(1, 2**50)
    m = _qp(L)
    Q, _ = _extracted(m)
    assert np.array_equal(Q, L)
    assert certify_quadratic_objective_convex(m) is False


def test_tiny_off_diagonal_is_kept_and_not_certified():
    Q0 = np.array([[0.0, 1e-20], [1e-20, 1.0]])  # det < 0: indefinite
    m = _qp(Q0)
    Q, _ = _extracted(m)
    assert np.array_equal(Q, Q0)
    assert certify_quadratic_objective_convex(m) is False


def test_tiny_objective_is_not_wiped():
    Q0 = 1e-16 * np.array([[1.0, 3.0], [3.0, 1.0]])
    m = _qp(Q0)
    Q, _ = _extracted(m)
    assert np.array_equal(Q, Q0)
    assert certify_quadratic_objective_convex(m) is False


def test_linear_objective_vector_is_exact():
    c0 = np.array([1e-16, 1.0 - 2.0**-52, 1.0, 0.0])
    m = _qp(np.eye(4), c=c0)
    _, c = _extracted(m)
    assert np.array_equal(c, c0)

    m2 = dm.Model("lin")
    x = m2.continuous("x", shape=(4,), lb=0, ub=1)
    m2.add_linear_objective(c0, x)
    assert np.array_equal(np.asarray(extract_qp_data(m2).c), c0)


def test_linear_constraint_coefficient_near_one_is_exact():
    a = 1.0 - 2.0**-52
    m = dm.Model("row")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.add_linear_constraints(sp.csr_matrix(np.array([[a, 1e-16]])), x, "==", np.array([0.5]))
    m.add_quadratic_objective(sp.eye(2, format="csr"), np.zeros(2), x)
    A = extract_qp_data(m).A_eq
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    assert A[0, 0] == a
    assert A[0, 1] == 1e-16
