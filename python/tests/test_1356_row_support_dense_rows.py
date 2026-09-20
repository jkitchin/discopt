"""Issue #1356: the envelope-row claim scanned every dense row once per product term.

``build_spatial_kernel_spec`` and ``IncrementalMcCormick`` identify a term's envelope
rows as the rows touching its aux column whose support lies inside the term's operand
columns. They rebuilt a set of the whole row for every (term, row) pair, so a
quadratic constraint aggregating N products cost O(N^2): QPLIB_10003 (51k products)
spent >5 min there and overran a 30 s ``time_limit`` by 5x. ``row_support_within``
rejects a row with more above-tolerance entries than the operand set in O(1).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.incremental_mccormick import row_support_within


def _brute(A: sp.csr_matrix, r: int, allowed: set, tol: float) -> bool:
    lo, hi = A.indptr[r], A.indptr[r + 1]
    return {int(A.indices[t]) for t in range(lo, hi) if abs(A.data[t]) > tol} <= allowed


@pytest.mark.smoke
def test_row_support_within_matches_set_containment():
    rng = np.random.default_rng(1356)
    checks = 0
    for tol in (1e-12, 1e-7):
        for _ in range(20):
            A = sp.random(40, 12, density=0.15, format="csr", random_state=rng)
            # Entries at/below tol are not support; one empty row; one dense row.
            A.data[rng.random(A.data.size) < 0.2] = tol / 2
            A = sp.vstack([A, sp.csr_matrix((1, 12)), sp.csr_matrix(np.ones((1, 12)))]).tocsr()
            A.sort_indices()
            within = row_support_within(A, tol)
            for r in range(A.shape[0]):
                for allowed in ({0, 1}, {2, 3, 4}, set(range(12)), set()):
                    assert within(r, allowed) == _brute(A, r, allowed, tol)
                    checks += 1
    assert checks == 2 * 20 * 42 * 4


@pytest.mark.smoke
def test_dense_row_is_rejected_without_scanning_it():
    """One row of 20k entries touched by 20k terms: the old per-call set rebuild was
    4e8 element visits (minutes). The count test answers each call in O(1)."""
    n = 20_000
    A = sp.csr_matrix(np.ones((1, n)))
    within = row_support_within(A, 1e-12)
    t = time.perf_counter()
    assert not any(within(0, {k, (k + 1) % n, (k + 2) % n}) for k in range(n))
    assert time.perf_counter() - t < 2.0
