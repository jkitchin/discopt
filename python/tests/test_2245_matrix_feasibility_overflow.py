"""#2245: the matrix-form feasibility gate accepted a point whose row activity overflows.

``_matrix_solution_feasible`` is the arbiter that decides whether an LP/QP
engine's point may be returned. On ``2**900 * x <= 1`` at ``x = finfo.max`` the
cheap activity, the ``fsum`` re-sum AND the representability floor
``|a|·2ε·|x|`` all overflow to ``+inf``, and the final test was ``inf > inf`` --
False -- so a point violating the row by ~1e600 was accepted. The same held for
the equality row.

A second hole of the same kind: a row whose float activity is ``nan`` (``inf -
inf`` from overflowing terms) or an overflowed ``-inf`` that hides cancellation
was never flagged by the cheap pass at all, because ``nan > thresh`` and ``-inf >
thresh`` are both False.

Rows whose float evaluation leaves the ``f64`` range are now decided in exact
rationals (every ``f64`` is a dyadic rational), so the verdict is the true one in
both directions: overflowing-but-feasible rows are still accepted.
"""

import numpy as np
import pytest
from discopt import solver as S

pytestmark = pytest.mark.smoke

M = float(np.finfo(np.float64).max)
BIG = float(2**900)
FREE = np.array([[-np.inf, np.inf]])


def _feas(x, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    with np.errstate(over="ignore", invalid="ignore"):
        return S._matrix_solution_feasible(
            np.asarray(x, dtype=np.float64),
            None if A_ub is None else np.asarray(A_ub, dtype=np.float64),
            None if b_ub is None else np.asarray(b_ub, dtype=np.float64),
            None if A_eq is None else np.asarray(A_eq, dtype=np.float64),
            None if b_eq is None else np.asarray(b_eq, dtype=np.float64),
            bounds,
        )


def test_issue_reproducer_inequality_rejected():
    assert _feas([M], A_ub=[[BIG]], b_ub=[1.0], bounds=np.array([[0.0, np.inf]])) is False


def test_issue_reproducer_equality_rejected():
    assert _feas([M], A_eq=[[BIG]], b_eq=[1.0], bounds=np.array([[0.0, np.inf]])) is False


def test_overflowing_but_satisfied_inequality_accepted():
    # -2**900 * M <= 1 holds: the exact path must not reject on overflow alone.
    assert _feas([M], A_ub=[[-BIG]], b_ub=[1.0], bounds=FREE) is True


def test_nan_activity_from_cancelling_overflow_is_retested():
    # Terms (-M, -M, M, M): BLAS gives nan or -inf, the true activity is 0.
    x = [M, M, M, M]
    A = [[-1.0, -1.0, 1.0, 1.0]]
    bounds = np.array([[-np.inf, np.inf]] * 4)
    assert _feas(x, A_ub=A, b_ub=[-1.0], bounds=bounds) is False  # 0 <= -1 false
    assert _feas(x, A_ub=A, b_ub=[1.0], bounds=bounds) is True  # 0 <= 1 true
    assert _feas(x, A_eq=A, b_eq=[0.0], bounds=bounds) is True
    assert _feas(x, A_eq=A, b_eq=[1.0], bounds=bounds) is False


def test_nan_activity_from_infinite_products_is_retested():
    # Products BIG*M and -BIG*M are each +-inf in f64; their exact sum is 0.
    x = [M, M]
    A = [[BIG, -BIG]]
    bounds = np.array([[-np.inf, np.inf]] * 2)
    assert _feas(x, A_ub=A, b_ub=[-1.0], bounds=bounds) is False
    assert _feas(x, A_ub=A, b_ub=[1.0], bounds=bounds) is True


def test_exact_path_decides_row_directly():
    a = np.array([BIG])
    x = np.array([M])
    assert S._row_truly_violated_exact(a, x, 1.0, 1e-6, signed=True) is True
    assert S._row_truly_violated_exact(-a, x, 1.0, 1e-6, signed=True) is False
    assert S._row_truly_violated_exact(a, x, np.inf, 1e-6, signed=True) is False
    assert S._row_truly_violated_exact(a, x, np.inf, 1e-6, signed=False) is True


def test_sparse_operand_same_verdict():
    sp = pytest.importorskip("scipy.sparse")
    A = sp.csr_matrix(np.array([[BIG]]))
    with np.errstate(over="ignore", invalid="ignore"):
        assert (
            S._matrix_solution_feasible(np.array([M]), A, np.array([1.0]), None, None, None)
            is False
        )
