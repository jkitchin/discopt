"""``milp_simplex._fbbt_eq_bounds`` must contain the feasible set in exact arithmetic.

Its box feeds every ``milp_simplex`` Neumaier-Shcherbina bound (std, sharp,
basis-verified) and ``_farkas_certified_std``. Without outward widening, a small
term absorbed by a large activity sum is lost: the box excluded feasible points,
the NS bound exceeded the exact optimum, and a feasible system was "proved"
infeasible (found by the #1956 adjacent-bug sweep).
"""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import milp_simplex as ms


@pytest.mark.smoke
def test_absorbed_term_keeps_lower_bound_below_exact_optimum():
    # x0 + z1 + z2 = 1e16, x0 free, z1 in [0, 1e16], z2 in [0, 1]; min x0 -> -1.
    A = sp.csc_matrix(np.array([[1.0, 1.0, 1.0]]))
    b = np.array([1e16])
    lb = np.array([-np.inf, 0.0, 0.0])
    ub = np.array([np.inf, 1e16, 1.0])
    lo, _ = ms._fbbt_eq_bounds(A, b, lb, ub)
    assert F(lo[0]) <= -1
    c = np.array([1.0, 0.0, 0.0])
    y = np.zeros(1)
    for bound in (ms._safe_lp_lower_bound_std, ms._safe_lp_lower_bound_sharp):
        g = bound(y, c, A, b, np.array([-1e20, 0, 0]), np.array([1e20, 1e16, 1.0]))
        assert g is None or F(g) <= -1, (bound.__name__, g)


@pytest.mark.smoke
def test_feasible_system_is_not_proved_infeasible():
    B = 2.0**54
    A = sp.csc_matrix(np.array([[1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, -1]], dtype=float))
    b = np.array([B, 0.0, -18.0])
    lb = np.array([-1e20, 0, 0, -10.0])
    ub = np.array([1e20, B, 9.0, 10.0])
    pt = [F(-9), F(int(B)), F(9), F(9)]  # exactly feasible
    Ad = A.toarray()
    assert all(sum(F(Ad[i, j]) * pt[j] for j in range(4)) == F(b[i]) for i in range(3))
    lo, hi = ms._fbbt_eq_bounds(
        A, b, np.where(lb <= -1e20, -np.inf, lb), np.where(ub >= 1e20, np.inf, ub)
    )
    assert all(F(lo[j]) <= pt[j] <= F(hi[j]) for j in range(4)), (lo, hi)
    assert ms._farkas_certified_std(np.array([0.0, 1.0, 1.0]), A, b, lb, ub) is False
