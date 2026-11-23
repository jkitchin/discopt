"""#2315: the matrix gate keeps its term-scaled tolerance when the term sum overflows.

``_matrix_row_threshold`` computes ``min(tol + rtol*row_scale, cap)`` in floats.
When ``sum_j |A_ij x_j|`` overflows although every term is finite, the first
argument is ``inf`` and only the (much larger) distance cap remained. The re-test
then compared the exactly recomputed residual against that cap.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solver import _matrix_solution_feasible

A_BIG = float(2**1023)
X = np.array([1.0, 1.0, 0.0])


def _rows(fmt):
    A = np.array([[A_BIG, -A_BIG, 1.0]])
    return sp.csr_matrix(A) if fmt == "csr" else A


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["dense", "csr"])
def test_inequality_row_with_overflowing_term_sum(fmt):
    # Activity is exactly 0; allowance is tol + rtol * 2**1024 ~ 1.8e299.
    with np.errstate(over="ignore", invalid="ignore"):
        assert _matrix_solution_feasible(X, _rows(fmt), [-1e298], None, None, None)
        assert not _matrix_solution_feasible(X, _rows(fmt), [-1e301], None, None, None)


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["dense", "csr"])
def test_equality_row_with_overflowing_term_sum(fmt):
    with np.errstate(over="ignore", invalid="ignore"):
        assert _matrix_solution_feasible(X, None, None, _rows(fmt), [-1e298], None)
        assert not _matrix_solution_feasible(X, None, None, _rows(fmt), [-1e301], None)


@pytest.mark.unit
def test_finite_term_sum_is_the_reference_behaviour():
    A = np.array([[float(2**1022), -float(2**1022), 1.0]])
    assert _matrix_solution_feasible(X, A, [-1e298], None, None, None)
    assert not _matrix_solution_feasible(X, A, [-1e301], None, None, None)


@pytest.mark.unit
def test_exact_threshold_only_touches_overflowed_rows_and_never_loosens():
    from discopt.solver import _exact_overflowed_thresholds

    A = np.array([[A_BIG, -A_BIG, 1.0], [1.0, 1.0, 0.0]])
    thresh = np.array([9.0e303, 5.0e-6])
    row_scale = np.array([np.inf, 2.0])
    out = _exact_overflowed_thresholds(A, X, thresh, row_scale, np.array([0, 1]), 1e-6, 1e-9)
    assert out[0] == pytest.approx(1e-6 + 2e-9 * 2.0**1023, rel=1e-12)
    assert out[1] == thresh[1]
    assert np.all(out <= thresh)
    # The input array is not modified in place.
    assert thresh[0] == 9.0e303
