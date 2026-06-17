"""Tests for second-order-cone (SOC) outer-approximation cuts.

Validity is the correctness-critical property: an SOC gradient cut must hold for
every point in the cone ``||y|| <= t`` (so it never removes a feasible point),
while separating a point outside it.
"""

from __future__ import annotations

import numpy as np
from discopt._jax.soc_cuts import soc_gradient_cut


def _layout(k: int):
    # y in cols 0..k-1, t in col k.
    return list(range(k)), k, k + 1


def _eval(cut, y: np.ndarray, t: float, y_cols, t_col, n_total) -> float:
    z = np.zeros(n_total)
    z[y_cols] = y
    z[t_col] = t
    # "<=" cut: satisfied iff coeffs @ z <= rhs, i.e. slack = rhs - coeffs@z >= 0.
    return float(cut.rhs - cut.coeffs @ z)


def test_no_cut_inside_cone():
    y_cols, t_col, n = _layout(3)
    # ||y|| = 3 <= t = 5: inside the cone.
    assert soc_gradient_cut(np.array([1.0, 2.0, 2.0]), 5.0, y_cols, t_col, n) is None


def test_no_cut_at_origin():
    y_cols, t_col, n = _layout(2)
    assert soc_gradient_cut(np.zeros(2), -1.0, y_cols, t_col, n) is None


def test_cut_separates_point_outside_cone():
    y_cols, t_col, n = _layout(2)
    y = np.array([3.0, 4.0])  # ||y|| = 5
    t = 1.0  # outside: 5 > 1
    cut = soc_gradient_cut(y, t, y_cols, t_col, n)
    assert cut is not None
    # The violating point is cut off (negative slack).
    assert _eval(cut, y, t, y_cols, t_col, n) < -1e-9


def test_cut_valid_for_all_cone_points():
    y_cols, t_col, n = _layout(4)
    rng = np.random.default_rng(0)
    y0 = np.array([2.0, -3.0, 1.0, 0.5])
    cut = soc_gradient_cut(y0, 0.5, y_cols, t_col, n)  # ||y0||~3.8 > 0.5
    assert cut is not None
    worst = np.inf
    for _ in range(5000):
        y = rng.uniform(-3, 3, 4)
        t = np.linalg.norm(y) + rng.uniform(0, 2)  # strictly inside the cone
        worst = min(worst, _eval(cut, y, t, y_cols, t_col, n))
    assert worst >= -1e-9, f"SOC cut violated a cone point by {worst}"


def test_cut_is_supporting_on_the_boundary():
    # On the boundary point that generated it, the cut is tight (slack ~ 0).
    y_cols, t_col, n = _layout(2)
    y = np.array([3.0, 4.0])
    cut = soc_gradient_cut(y, 1.0, y_cols, t_col, n)
    # At (y, t=||y||) the constraint u^T y = ||y|| = t holds with equality.
    assert abs(_eval(cut, y, float(np.linalg.norm(y)), y_cols, t_col, n)) < 1e-9
