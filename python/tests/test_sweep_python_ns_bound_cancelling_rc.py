"""The Python Neumaier-Shcherbina bounds must be proved, not margin-guarded (#1956 class).

``obbt._ns_safe_lp_lower_bound`` and ``milp_simplex._safe_lp_lower_bound_std`` formed
the reduced costs ``c ± Aᵀy`` in float64. ``min -x1 - t x2  s.t.  3 x1 + x2 <= 0`` with
``t = fl(1/3)`` and ``y = t`` has the exact reduced cost ``-1 + 3t = -5.6e-17``, which
rounds to ``0``: the column's ``rc·H`` term was dropped and both returned ``~-1e-10``
for an LP whose minimum is ``-0.0555`` at ``H = 1e15``. Zero tolerance against the
exact (rational) optimum.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.obbt import _ns_safe_lp_lower_bound
from discopt.solvers import milp_simplex as ms

T = 1 / 3


def _exact(H):
    return -Fraction(H) + 3 * Fraction(T) * Fraction(H)


@pytest.mark.smoke
@pytest.mark.parametrize("H", [1e6, 1e12, 1e15])
@pytest.mark.parametrize("sparse", [False, True])
def test_obbt_ns_bound_never_exceeds_exact_minimum(H, sparse):
    c = np.array([-1.0, -T])
    A = np.array([[3.0, 1.0]])
    A = sp.csr_matrix(A) if sparse else A
    g = _ns_safe_lp_lower_bound(
        c, np.array([-T]), A, np.array([0.0]), np.array([0.0, -3 * H]), np.array([H, 3 * H])
    )
    assert g is not None and Fraction(g) <= _exact(H), (g, float(_exact(H)))


@pytest.mark.smoke
def test_obbt_ns_bound_equality_row_multiplier_stays_free():
    # Same LP as an equality: the multiplier is free-sign, the slack fixed at 0.
    H = 1e15
    g = _ns_safe_lp_lower_bound(
        np.array([-1.0, -T]),
        np.array([-T]),
        np.array([[3.0, 1.0]]),
        np.array([0.0]),
        np.array([0.0, -3 * H]),
        np.array([H, 3 * H]),
        n_eq=1,
    )
    assert g is not None and Fraction(g) <= _exact(H)


@pytest.mark.smoke
@pytest.mark.parametrize("H", [1e6, 1e12, 1e15])
def test_std_ns_bound_never_exceeds_exact_minimum(H):
    a = sp.csc_matrix(np.array([[3.0, 1.0, 1.0]]))
    g = ms._safe_lp_lower_bound_std(
        np.array([-T]),
        np.array([-1.0, -T, 0.0]),
        a,
        np.array([0.0]),
        np.array([0.0, -3 * H, 0.0]),
        np.array([H, 3 * H, 1e20]),
    )
    assert g is not None and Fraction(g) <= _exact(H), (g, float(_exact(H)))


@pytest.mark.smoke
def test_well_conditioned_bound_is_unchanged():
    # min x1 + x2  s.t. -x1 - x2 <= -1, x in [0, 5]: duals exact, g = 1 - margin.
    g = _ns_safe_lp_lower_bound(
        np.array([1.0, 1.0]),
        np.array([-1.0]),
        np.array([[-1.0, -1.0]]),
        np.array([-1.0]),
        np.zeros(2),
        np.full(2, 5.0),
    )
    assert g is not None and 1.0 - 1e-9 <= g <= 1.0
