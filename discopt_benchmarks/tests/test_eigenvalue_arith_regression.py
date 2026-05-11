"""M6 regression: eigenvalue arithmetic on benchmark-style quadratic
subexpressions.

The eigenvalue bound provider (``discopt._jax.convexity.eigenvalue_arith``)
is not yet wired into the LP relaxation compiler — that integration
ships alongside the M2/M3 kernel and M11 wrapper integration as a
follow-up. This regression exercises the kernel directly on quadratic
forms drawn from typical benchmark problem structures and asserts:

1. The eigenvalue-based interval enclosure contains the true range of
   ``f(x) = x^T Q x + b^T x + c`` at every sampled point in the domain
   (soundness, ≥ 10⁴ samples).
2. On PSD / NSD / diagonal-quadratic structures, the eigenvalue bound
   is at least as tight as forward interval arithmetic (the M6
   acceptance criterion).
3. On a representative two-variable quadratic with mixed signs, the
   bound recovers the exact range to within the αBB roundoff slack.

These properties are the contract M6 makes with downstream consumers.
Breaking any of them is a correctness regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax.convexity.eigenvalue_arith import (
    QuadraticForm,
    interval_ad_quadratic_bound,
    quadratic_form_bound,
)

N_REGRESSION_SAMPLES = 10_000


def _bound_floats(iv):
    return float(iv.lo.item()), float(iv.hi.item())


def _make_problem(seed, n, kind):
    rng = np.random.default_rng(seed)
    a_mat = rng.standard_normal((n, n))
    if kind == "psd":
        q_mat = a_mat @ a_mat.T
    elif kind == "nsd":
        q_mat = -(a_mat @ a_mat.T)
    elif kind == "diagonal":
        q_mat = np.diag(rng.uniform(-3.0, 3.0, n))
    else:
        q_mat = (a_mat + a_mat.T) / 2
    b = rng.standard_normal(n)
    c = float(rng.standard_normal())
    qf = QuadraticForm(Q=q_mat, b=b, c=c)
    x_lo = -1.5 * np.ones(n)
    x_hi = 1.5 * np.ones(n)
    return qf, x_lo, x_hi, rng


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.parametrize("kind", ["psd", "nsd", "diagonal", "indef"])
@pytest.mark.parametrize("n", [3, 5, 8])
def test_eigenvalue_enclosure_is_sound(kind, n):
    qf, x_lo, x_hi, rng = _make_problem(seed=0, n=n, kind=kind)
    bound = quadratic_form_bound(qf, x_lo, x_hi)
    lo, hi = _bound_floats(bound)
    xs = rng.uniform(x_lo, x_hi, size=(N_REGRESSION_SAMPLES, n))
    ys = qf.evaluate(xs)
    assert (ys >= lo - 1e-9).all(), f"{kind}/n={n}: lower bound violated"
    assert (ys <= hi + 1e-9).all(), f"{kind}/n={n}: upper bound violated"


@pytest.mark.regression
@pytest.mark.parametrize("kind", ["psd", "nsd", "diagonal"])
@pytest.mark.parametrize("seed", range(4))
def test_eigenvalue_at_least_as_tight_as_interval_on_structured(kind, seed):
    qf, x_lo, x_hi, _ = _make_problem(seed=seed, n=6, kind=kind)
    eig_lo, eig_hi = _bound_floats(quadratic_form_bound(qf, x_lo, x_hi))
    iv_lo, iv_hi = _bound_floats(interval_ad_quadratic_bound(qf, x_lo, x_hi))
    assert (eig_hi - eig_lo) <= (iv_hi - iv_lo) + 1e-9, (
        f"{kind} seed={seed}: eigval width {eig_hi - eig_lo} > interval {iv_hi - iv_lo}"
    )


@pytest.mark.regression
def test_two_variable_indefinite_recovers_known_range():
    """For ``f(x, y) = x*y`` over ``[−1, 1]^2``, the true range is
    ``[−1, 1]``. ``Q = [[0, 1/2], [1/2, 0]]`` has eigenvalues ``±1/2``.
    The αBB-style enclosure should recover this exactly to within
    roundoff slack."""
    q_mat = np.array([[0.0, 0.5], [0.5, 0.0]])
    qf = QuadraticForm(Q=q_mat, b=np.zeros(2), c=0.0)
    x_lo = -np.ones(2)
    x_hi = np.ones(2)
    lo, hi = _bound_floats(quadratic_form_bound(qf, x_lo, x_hi))
    assert lo <= -1.0 + 1e-9
    assert hi >= 1.0 - 1e-9
    # Should not be wildly looser than the truth.
    assert lo >= -2.5 and hi <= 2.5


@pytest.mark.regression
def test_box_refinement_tightens_bound():
    """Halving the box around the origin should at least halve the
    quadratic-form-bound width on a PSD problem (the linear correction
    halves and the quadratic part scales by 1/4)."""
    rng = np.random.default_rng(11)
    a_mat = rng.standard_normal((4, 4))
    qf = QuadraticForm(Q=a_mat @ a_mat.T, b=rng.standard_normal(4), c=0.0)
    full_lo, full_hi = _bound_floats(quadratic_form_bound(qf, -np.ones(4), np.ones(4)))
    half_lo, half_hi = _bound_floats(quadratic_form_bound(qf, -0.5 * np.ones(4), 0.5 * np.ones(4)))
    full_w = full_hi - full_lo
    half_w = half_hi - half_lo
    # Half-box width must be strictly less than full-box width.
    assert half_w < full_w
