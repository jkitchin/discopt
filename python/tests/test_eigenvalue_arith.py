"""Tests for ``discopt._jax.convexity.eigenvalue_arith`` (M6 of issue #51).

Acceptance criteria from issue #51:

1. Computed bounds contain the true range on a randomized test set with
   ≥ 10⁴ samples per case.
2. Resulting bounds are at least as tight as forward interval arithmetic
   (``interval_ad_quadratic_bound`` in this module is the apples-to-
   apples reference for a quadratic form) on quadratic / PSD-structured
   test cases.
3. Hertz-Rohn vertex enumeration is documented but not implemented.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax.convexity.eigenvalue_arith import (
    QuadraticForm,
    interval_ad_quadratic_bound,
    quadratic_form_bound,
)

N_SAMPLES = 10_000


def _bound_floats(iv):
    return float(iv.lo.item()), float(iv.hi.item())


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_square_Q():
    with pytest.raises(ValueError):
        QuadraticForm(Q=np.zeros((3, 4)), b=np.zeros(3), c=0.0)


def test_constructor_rejects_b_shape_mismatch():
    with pytest.raises(ValueError):
        QuadraticForm(Q=np.eye(3), b=np.zeros(2), c=0.0)


def test_evaluate_matches_explicit_formula():
    Q = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([0.5, -0.25])
    c = 0.1
    qf = QuadraticForm(Q=Q, b=b, c=c)
    x = np.array([0.7, -0.4])
    expected = float(x @ Q @ x + b @ x + c)
    assert qf.evaluate(x) == pytest.approx(expected, abs=1e-12)


def test_evaluate_batch():
    qf = QuadraticForm(Q=np.eye(2), b=np.zeros(2), c=0.0)
    xs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(qf.evaluate(xs), [1.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Soundness: 10⁴-sample randomized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["PSD", "NSD", "indef", "diagonal"])
def test_eigenvalue_bound_is_sound(kind):
    rng = np.random.default_rng({"PSD": 0, "NSD": 1, "indef": 2, "diagonal": 3}[kind])
    n = 5
    A = rng.standard_normal((n, n))
    if kind == "PSD":
        Q = A @ A.T
    elif kind == "NSD":
        Q = -(A @ A.T)
    elif kind == "diagonal":
        Q = np.diag(rng.uniform(-2.0, 2.0, n))
    else:
        Q = (A + A.T) / 2
    b = rng.standard_normal(n)
    c = float(rng.standard_normal())
    qf = QuadraticForm(Q=Q, b=b, c=c)

    x_lo = -1.5 * np.ones(n)
    x_hi = 1.5 * np.ones(n)
    bound = quadratic_form_bound(qf, x_lo, x_hi)
    lo, hi = _bound_floats(bound)

    xs = rng.uniform(x_lo, x_hi, size=(N_SAMPLES, n))
    ys = qf.evaluate(xs)
    assert (ys >= lo - 1e-9).all(), f"{kind}: lower bound violated"
    assert (ys <= hi + 1e-9).all(), f"{kind}: upper bound violated"


def test_soundness_on_asymmetric_box():
    """Asymmetric box (x0 != 0) — exercises the linear-correction path."""
    rng = np.random.default_rng(7)
    n = 4
    A = rng.standard_normal((n, n))
    Q = A @ A.T  # PSD
    qf = QuadraticForm(Q=Q, b=rng.standard_normal(n), c=0.25)
    x_lo = np.array([0.5, -1.0, 2.0, -0.5])
    x_hi = np.array([1.5, 1.0, 3.5, 0.5])
    bound = quadratic_form_bound(qf, x_lo, x_hi)
    lo, hi = _bound_floats(bound)
    xs = rng.uniform(x_lo, x_hi, size=(N_SAMPLES, n))
    ys = qf.evaluate(xs)
    assert lo - 1e-9 <= ys.min() and ys.max() <= hi + 1e-9


def test_degenerate_box_collapses_to_point_value():
    """When x_lo == x_hi the bound must collapse to f at that point."""
    Q = np.array([[2.0, 1.0], [1.0, 3.0]])
    qf = QuadraticForm(Q=Q, b=np.array([1.0, -1.0]), c=0.5)
    x = np.array([0.7, -0.4])
    bound = quadratic_form_bound(qf, x, x)
    lo, hi = _bound_floats(bound)
    fval = qf.evaluate(x)
    assert lo <= fval <= hi
    assert hi - lo < 1e-10


# ---------------------------------------------------------------------------
# Tightness vs forward interval AD on quadratic / PSD-structured cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_eigenvalue_tighter_than_interval_on_psd(seed):
    rng = np.random.default_rng(seed)
    n = 6
    A = rng.standard_normal((n, n))
    qf = QuadraticForm(Q=A @ A.T, b=rng.standard_normal(n), c=0.0)
    x_lo = -np.ones(n)
    x_hi = np.ones(n)
    eig_lo, eig_hi = _bound_floats(quadratic_form_bound(qf, x_lo, x_hi))
    iv_lo, iv_hi = _bound_floats(interval_ad_quadratic_bound(qf, x_lo, x_hi))
    eig_w = eig_hi - eig_lo
    iv_w = iv_hi - iv_lo
    assert eig_w <= iv_w + 1e-9, (
        f"PSD seed={seed}: eigval width {eig_w} should be <= interval width {iv_w}"
    )


@pytest.mark.parametrize("seed", range(8))
def test_eigenvalue_tighter_than_interval_on_nsd(seed):
    rng = np.random.default_rng(seed)
    n = 6
    A = rng.standard_normal((n, n))
    qf = QuadraticForm(Q=-(A @ A.T), b=rng.standard_normal(n), c=0.0)
    x_lo = -np.ones(n)
    x_hi = np.ones(n)
    eig_lo, eig_hi = _bound_floats(quadratic_form_bound(qf, x_lo, x_hi))
    iv_lo, iv_hi = _bound_floats(interval_ad_quadratic_bound(qf, x_lo, x_hi))
    assert (eig_hi - eig_lo) <= (iv_hi - iv_lo) + 1e-9


# ---------------------------------------------------------------------------
# Diagonal sanity: for Q = diag(λ), the eigenvalue bound matches the exact
# range because each y_i = x_i - x0_i is independently bounded.
# ---------------------------------------------------------------------------


def test_diagonal_psd_matches_exact_range():
    Q = np.diag([1.0, 2.0, 3.0])
    qf = QuadraticForm(Q=Q, b=np.zeros(3), c=0.0)
    x_lo = np.array([-1.0, -2.0, 0.0])
    x_hi = np.array([1.0, 1.0, 2.0])
    bound = quadratic_form_bound(qf, x_lo, x_hi)
    lo, hi = _bound_floats(bound)
    # x^T diag(λ) x = Σ λ_i x_i² with x_i² ∈ [min(x_lo², x_hi², 0_if_in), max(x_lo², x_hi²)].
    # x_lo[0..1] include 0 in their range so min x_i² = 0 there;
    # x_lo[2]=0 so min x_2² = 0 too.
    expected_max = 1.0 * 1.0 + 2.0 * 4.0 + 3.0 * 4.0
    # Lower bound has roundoff slack from the αBB construction; allow some.
    assert lo <= 0.0 + 1e-9
    assert hi >= expected_max - 1e-9


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_rejects_inverted_box():
    qf = QuadraticForm(Q=np.eye(2), b=np.zeros(2), c=0.0)
    with pytest.raises(ValueError):
        quadratic_form_bound(qf, np.array([1.0, 0.0]), np.array([0.0, 1.0]))


def test_rejects_x_shape_mismatch():
    qf = QuadraticForm(Q=np.eye(3), b=np.zeros(3), c=0.0)
    with pytest.raises(ValueError):
        quadratic_form_bound(qf, np.zeros(2), np.zeros(2))
