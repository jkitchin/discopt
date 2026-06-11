"""Tests for multivariate superposition cuts (M8 of issue #81).

Covers the standalone cut generator in :mod:`discopt._jax.superposition`:

* **Soundness** — every generated cut is a valid global bound on the lifted
  graph ``w = f(x) y`` over the box (checked on > 10^4 samples), so the
  relaxation can never exclude a true point. This is the non-negotiable
  rigorous-bound invariant.
* **Monotone tightening** — adding interior references never loosens the
  corner-only envelope and strictly tightens it somewhere.
* **Strict LP-gap closure** — on an equality-coupled ``exp(x) y == k`` instance,
  the LP bound from the superposition family strictly exceeds the corner-only
  (McCormick-like) family while both remain valid lower bounds on the true
  optimum. This is the multivariate strict win that univariate OA cannot show.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.superposition import (
    BilinearNonlinearTerm,
    bilinear_nonlinear_cuts,
    interior_references,
    mccormick_references,
    superposition_references,
)
from scipy.optimize import linprog

# A representative bilinear-of-nonlinear term: w = exp(x) * y.
F = lambda t: jnp.exp(t)  # noqa: E731
X_BOX = (-1.0, 1.0)
Y_BOX = (0.5, 2.0)


def _grid(nx: int = 121, ny: int = 121):
    xs = np.linspace(X_BOX[0], X_BOX[1], nx)
    ys = np.linspace(Y_BOX[0], Y_BOX[1], ny)
    X, Y = np.meshgrid(xs, ys)
    W = np.exp(X) * Y
    return X, Y, W


def _envelopes(cuts, X, Y):
    """Pointwise lower/upper bound surfaces implied by the cuts (over x,y,w)."""
    lo = np.full_like(X, -np.inf)
    hi = np.full_like(X, np.inf)
    for cut in cuts:
        ax, ay, aw = cut.coeffs  # ax*x + ay*y + aw*w  {sense}  rhs, aw == 1
        # w {sense} rhs - ax*x - ay*y
        plane = cut.rhs - ax * X - ay * Y
        if cut.sense == ">=":
            lo = np.maximum(lo, plane)
        elif cut.sense == "<=":
            hi = np.minimum(hi, plane)
    return lo, hi


def _term() -> BilinearNonlinearTerm:
    return BilinearNonlinearTerm(F, X_BOX, Y_BOX, degree=10)


# ---------------------------------------------------------------------------
# Soundness — the rigorous-bound invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refs",
    [
        mccormick_references(X_BOX, Y_BOX),
        superposition_references(X_BOX, Y_BOX, nx=3, ny=3),
        superposition_references(X_BOX, Y_BOX, nx=5, ny=5),
    ],
)
def test_cuts_are_sound_on_dense_grid(refs):
    """Every cut bounds the true surface on > 10^4 samples (no point excluded)."""
    cuts = bilinear_nonlinear_cuts(_term(), refs)
    X, Y, W = _grid(121, 121)  # 14641 samples > 1e4
    lo, hi = _envelopes(cuts, X, Y)
    assert (lo <= W + 1e-9).all(), f"underestimator exceeds surface by {(lo - W).max():.2e}"
    assert (W <= hi + 1e-9).all(), f"overestimator below surface by {(W - hi).max():.2e}"


def test_each_cut_individually_valid():
    """No single cut shaves the surface anywhere on the grid."""
    cuts = bilinear_nonlinear_cuts(_term(), superposition_references(X_BOX, Y_BOX))
    X, Y, W = _grid(81, 81)
    for cut in cuts:
        ax, ay, _ = cut.coeffs
        plane = cut.rhs - ax * X - ay * Y
        if cut.sense == ">=":
            assert (plane <= W + 1e-9).all()
        else:
            assert (plane >= W - 1e-9).all()


# ---------------------------------------------------------------------------
# Monotone tightening — interior references never loosen, sometimes tighten
# ---------------------------------------------------------------------------


def test_superposition_never_looser_and_strictly_tighter():
    corners = bilinear_nonlinear_cuts(_term(), mccormick_references(X_BOX, Y_BOX))
    full = bilinear_nonlinear_cuts(_term(), superposition_references(X_BOX, Y_BOX))
    X, Y, _ = _grid(121, 121)
    lo_c, hi_c = _envelopes(corners, X, Y)
    lo_f, hi_f = _envelopes(full, X, Y)
    # never looser: superposition lower >= corner lower; upper <= corner upper
    assert (lo_f >= lo_c - 1e-9).all()
    assert (hi_f <= hi_c + 1e-9).all()
    # strictly tighter somewhere in the interior
    assert (lo_f > lo_c + 1e-6).any() or (hi_f < hi_c - 1e-6).any()


# ---------------------------------------------------------------------------
# Strict LP-gap closure — the multivariate strict win
# ---------------------------------------------------------------------------


def _lp_bound(cuts, k: float, c_x: float, c_y: float) -> float:
    """min c_x*x + c_y*y  s.t. cuts, w == k, (x,y) in box.

    Variables ordered [x, y, w]. Returns the LP optimum (a valid lower bound on
    the true constrained optimum of  min c_x x + c_y y  s.t. exp(x) y == k).
    """
    A_ub, b_ub = [], []
    for cut in cuts:
        ax, ay, aw = cut.coeffs
        if cut.sense == "<=":
            A_ub.append([ax, ay, aw])
            b_ub.append(cut.rhs)
        elif cut.sense == ">=":
            A_ub.append([-ax, -ay, -aw])
            b_ub.append(-cut.rhs)
    res = linprog(
        c=[c_x, c_y, 0.0],
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array([[0.0, 0.0, 1.0]]),
        b_eq=np.array([k]),
        bounds=[X_BOX, Y_BOX, (None, None)],
        method="highs",
    )
    assert res.success, res.message
    return float(res.fun)


def _true_optimum(k: float, c_x: float, c_y: float) -> float:
    X, Y, W = _grid(401, 401)
    feas = np.abs(W - k) < 0.01
    obj = c_x * X + c_y * Y
    return float(obj[feas].min())


def test_superposition_lp_strictly_beats_corner_and_stays_valid():
    """Superposition LP bound > corner-only bound, both <= true optimum."""
    k, c_x, c_y = 1.307, np.cos(0.524), np.sin(0.524)
    corner_cuts = bilinear_nonlinear_cuts(_term(), mccormick_references(X_BOX, Y_BOX))
    super_cuts = bilinear_nonlinear_cuts(_term(), superposition_references(X_BOX, Y_BOX))

    lb_corner = _lp_bound(corner_cuts, k, c_x, c_y)
    lb_super = _lp_bound(super_cuts, k, c_x, c_y)
    true_opt = _true_optimum(k, c_x, c_y)

    # both are valid lower bounds
    assert lb_corner <= true_opt + 1e-6
    assert lb_super <= true_opt + 1e-6
    # superposition strictly tightens the gap
    assert lb_super > lb_corner + 1e-3, (lb_corner, lb_super, true_opt)


# ---------------------------------------------------------------------------
# Reference-set helpers
# ---------------------------------------------------------------------------


def test_reference_helpers():
    assert len(mccormick_references(X_BOX, Y_BOX)) == 4
    assert len(interior_references(X_BOX, Y_BOX, nx=3, ny=4)) == 12
    assert len(interior_references(X_BOX, Y_BOX, nx=0, ny=0)) == 0
    assert len(superposition_references(X_BOX, Y_BOX, nx=2, ny=2)) == 8
    # interior points are strictly inside the box
    for x, y in interior_references(X_BOX, Y_BOX, 3, 3):
        assert X_BOX[0] < x < X_BOX[1]
        assert Y_BOX[0] < y < Y_BOX[1]
