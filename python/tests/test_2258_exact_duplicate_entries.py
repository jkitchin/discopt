"""#2258: duplicate entries of an ``add_linear_constraints`` matrix are summed exactly.

A sparse matrix may store several entries for one ``(row, column)``; by sparse-matrix
semantics they are summed. ``add_linear_constraints`` passed them through unchanged and
every path summed them in binary64 in its own order: ``2**53, 1, -2**53`` in one cell
(exactly ``x <= 1``) read as coefficient 0 in the LP, so ``solve`` certified x = 10,
while ``verify_point`` accepted x = 2 and rejected x = 5. The matrix is now coalesced
exactly at build time, so every path reads the coefficient 1.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.modeling.core import _csr_with_exact_duplicates
from discopt.validation.feasibility import verify_point

BIG = 2.0**53


def _cancelling_csr():
    # One row, three entries in column 0 whose exact sum is 1.
    return sp.csr_matrix(([BIG, 1.0, -BIG], [0, 0, 0], [0, 3]), shape=(1, 1))


@pytest.mark.smoke
@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
def test_coalesce_is_exact_in_every_format(fmt):
    A = _cancelling_csr()
    if fmt == "coo":
        A = sp.coo_matrix(([BIG, 1.0, -BIG], ([0, 0, 0], [0, 0, 0])), shape=(1, 1))
    elif fmt == "csc":
        A = sp.csc_matrix(([BIG, 1.0, -BIG], [0, 0, 0], [0, 3]), shape=(1, 1))
    out = _csr_with_exact_duplicates(A)
    assert sp.isspmatrix_csr(out)
    assert out.nnz == 1
    assert out.data.tolist() == [1.0]
    # The binary64 reductions this replaces get it wrong, which is the point.
    assert float(np.float64(BIG) + 1.0 - BIG) == 0.0


@pytest.mark.smoke
def test_coalesce_keeps_layout_and_other_entries():
    # Row 0: duplicates in col 2 (exact sum 3), single in col 0; row 1 empty;
    # row 2: unsorted singles. Stored order is scrambled on purpose.
    rows = [2, 0, 0, 2, 0, 0]
    cols = [1, 2, 0, 0, 2, 2]
    vals = [5.0, BIG, 7.0, 4.0, 3.0, -BIG]
    A = sp.coo_matrix((vals, (rows, cols)), shape=(3, 3))
    out = _csr_with_exact_duplicates(A)
    assert out.has_canonical_format
    dense = out.toarray()
    np.testing.assert_array_equal(dense, [[7.0, 0.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 0.0]])
    # Exact even where an intermediate binary64 sum would overflow.
    B = sp.coo_matrix(([1e308, 1e308, -1e308], ([0, 0, 0], [0, 0, 0])), shape=(1, 1))
    assert _csr_with_exact_duplicates(B).data.tolist() == [1e308]
    # Integer data is added exactly before the one rounding.
    C = sp.coo_matrix(
        (np.array([2**60, 1, -(2**60)], dtype=np.int64), ([0, 0, 0], [0, 0, 0])), shape=(1, 1)
    )
    assert _csr_with_exact_duplicates(C).data.tolist() == [1.0]


@pytest.mark.smoke
@pytest.mark.parametrize("block", [False, True])
def test_solve_and_verify_agree_on_exact_row(block):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    if block:
        y = m.continuous("y", lb=0, ub=10)
        A = sp.csr_matrix(([BIG, 1.0, -BIG, 1.0], [0, 0, 0, 1], [0, 4]), shape=(1, 2))
        m.add_linear_constraints(A, [x, y], "<=", [1.0])
        m.maximize(x + 0.0 * y)
    else:
        m.add_linear_constraints(_cancelling_csr(), x, "<=", [1.0])
        m.maximize(x)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-6)
    assert r.x["x"] == pytest.approx(1.0, abs=1e-6)
    n = 2 if block else 1
    assert verify_point(m, [1.0] + [0.0] * (n - 1)).ok
    for v in (2.0, 5.0, 10.0):
        assert not verify_point(m, [v] + [0.0] * (n - 1)).ok, v
