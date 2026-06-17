"""Tests for targeted RLT bound-factor cuts.

Validity is the correctness-critical property: the RLT product of a constraint
factor and a bound factor is non-negative at every feasible point, so the
linearized cut never removes one. These use a synthetic lifted-column layout so
the moment values can be set exactly.
"""

from __future__ import annotations

import numpy as np
from discopt._jax.rlt_cuts import rlt_constraint_bound_cut

# 2 original variables (cols 0,1); squares X_00,X_11 and product X_01 lifted.
_INFO = {
    "original": {0: 0, 1: 1},
    "monomial": {(0, 2): 2, (1, 2): 3},
    "bilinear": {(0, 1): 4},
}
_N = 5


def _pack(x: np.ndarray) -> np.ndarray:
    z = np.zeros(_N)
    z[0], z[1] = x[0], x[1]
    z[2], z[3] = x[0] ** 2, x[1] ** 2
    z[4] = x[0] * x[1]
    return z


def _slack(cut, x: np.ndarray) -> float:
    return float(cut.coeffs @ _pack(x) - cut.rhs)  # ">=" cut: >= 0 means satisfied


def test_cut_valid_for_all_feasible_points():
    # Constraint x0 + x1 <= 1; lower bound factor x0 - 0 >= 0.
    a = {0: 1.0, 1: 1.0}
    b = 1.0
    # A relaxation point with X != x x^T that violates the RLT product.
    x_full = np.zeros(_N)
    x_full[0], x_full[1] = 0.5, 0.5
    x_full[2], x_full[3], x_full[4] = 0.5, 0.5, 0.5  # inflated moments
    cut = rlt_constraint_bound_cut(a, b, 0, 0.0, True, _INFO, x_full, _N)
    assert cut is not None
    # Every true feasible point (x0+x1<=1, x>=0) satisfies the cut.
    rng = np.random.default_rng(0)
    worst = np.inf
    n_ok = 0
    while n_ok < 4000:
        x = rng.uniform(0, 1, 2)
        if x[0] + x[1] <= 1.0:
            worst = min(worst, _slack(cut, x))
            n_ok += 1
    assert worst >= -1e-9, f"RLT cut violated a feasible point by {worst}"


def test_upper_bound_factor_valid():
    a = {0: 1.0, 1: 1.0}
    b = 1.0
    x_full = np.zeros(_N)
    x_full[0], x_full[1] = 0.5, 0.5
    x_full[2], x_full[3], x_full[4] = 0.5, 0.5, -0.5
    cut = rlt_constraint_bound_cut(a, b, 1, 1.0, False, _INFO, x_full, _N)
    if cut is not None:
        rng = np.random.default_rng(1)
        n_ok = 0
        while n_ok < 2000:
            x = rng.uniform(0, 1, 2)
            if x[0] + x[1] <= 1.0:
                assert _slack(cut, x) >= -1e-9
                n_ok += 1


def test_no_cut_when_satisfied():
    a = {0: 1.0, 1: 1.0}
    b = 1.0
    # Consistent moment point (X = x x^T): the RLT product is exactly >= 0.
    x_full = _pack(np.array([0.3, 0.3]))
    assert rlt_constraint_bound_cut(a, b, 0, 0.0, True, _INFO, x_full, _N) is None


def test_none_when_product_column_missing():
    info = {"original": {0: 0, 1: 1}, "monomial": {(0, 2): 2}, "bilinear": {}}
    # X_01 not lifted -> cannot form the product -> no cut.
    x_full = np.zeros(3)
    assert rlt_constraint_bound_cut({0: 1.0, 1: 1.0}, 1.0, 0, 0.0, True, info, x_full, 3) is None
