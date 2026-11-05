"""highs-parity P2.5 (re-scoped): the exact dual correction's rational solve is sparse.

``perold`` was the one L-panel LP the Rust route left uncertified: its simplex reaches the
optimum, but the Rust dual's roundoff puts ~2.9e-13 reduced costs on open sides, and the
exact dual correction (``exact_ns_bound``) ran its dense rational elimination out of the
4e8 ``EXACT_MAX_TOTAL_WORK`` budget. The blocks it solves are LP columns with a few
nonzeros each; ``_exact_solve`` now eliminates sparsely (Markowitz-style pivots) and needs
1.9e7. A nonsingular system has one rational solution, so every system the dense form
solved gets the identical answer -- pinned here against a dense reference.
"""

from __future__ import annotations

import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

H = pytest.importorskip("discopt.solvers.lp_milp_highs")

_FIX = Path(__file__).parent / "data" / "highs_parity" / "p25_perold_rust_dual.npz"


def _dense_reference(M, rhs):
    """Plain dense Gaussian elimination over the rationals (no work cap)."""
    n = len(rhs)
    T = [list(row) + [rhs[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = next((r for r in range(col, n) if T[r][col] != 0), None)
        if piv is None:
            return None
        T[col], T[piv] = T[piv], T[col]
        for r in range(col + 1, n):
            if T[r][col] != 0:
                f = T[r][col] / T[col][col]
                T[r] = [a - f * b for a, b in zip(T[r], T[col])]
    z = [Fraction(0)] * n
    for i in reversed(range(n)):
        z[i] = (T[i][n] - sum((T[i][k] * z[k] for k in range(i + 1, n)), Fraction(0))) / T[i][i]
    return z


def test_sparse_solve_matches_dense_reference_including_singular():
    rnd = random.Random(2025)
    vals = [0, 0, 0, 0, 1, -2, 3, Fraction(1, 2), Fraction(-7, 3)]
    solved = singular = 0
    for _ in range(300):
        n = rnd.randint(1, 30)
        M = [[Fraction(rnd.choice(vals)) for _ in range(n)] for _ in range(n)]
        rhs = [Fraction(rnd.randint(-9, 9), rnd.randint(1, 4)) for _ in range(n)]
        want = _dense_reference(M, rhs)
        got = H._exact_solve(M, rhs, None)
        assert got == want
        solved += want is not None
        singular += want is None
    assert solved >= 50 and singular >= 50  # both outcomes exercised


def _perold():
    d = np.load(_FIX)
    A = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    sf = H.StdForm.from_arrays(d["c"], A, d["b"], d["xl"], d["xu"], float(d["obj_const"]))
    return sf, d["y"], float(d["oracle"])


def test_perold_rust_dual_certifies_within_the_work_budget():
    sf, y, oracle = _perold()
    spent_total = []
    real = H._exact_solve

    def spy(M, rhs, deadline, spent=None):
        z = real(M, rhs, deadline, spent)
        spent_total.append(spent[0] if spent else 0)
        return z

    H._exact_solve = spy
    try:
        bound, why = H.exact_ns_bound(y, H.fbbt_box(sf))
    finally:
        H._exact_solve = real
    assert bound is not None, why
    # A rigorous lower bound: at most the optimum, and tight on this LP.
    assert bound <= oracle
    assert oracle - bound <= 1e-6 * max(1.0, abs(oracle))
    assert spent_total and max(spent_total) < H.EXACT_MAX_TOTAL_WORK / 10
