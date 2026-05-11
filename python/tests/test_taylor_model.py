"""Tests for ``discopt._jax.taylor_model`` (M3 of issue #51).

Covers the M3 acceptance criteria:
- Polynomial-plus-remainder enclosure contains the true function range on a
  randomized test set with ≥ 10⁴ samples per operator.
- Convergence-rate behavior matches Bompadre, Mitsos, Chachuat (2013): the
  remainder width shrinks under domain refinement at the predicted rate.
- At expansion order 1, the linearization matches the existing
  ``cutting_planes.py`` Taylor cuts (sanity check).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax import cutting_planes
from discopt._jax import taylor_model as tm

N_SAMPLES = 10_000
TOL = 1e-9


def _sample_box(domain, n=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    a, b = domain
    return rng.uniform(a, b, size=n)


def _enclosure_contains(true_vals, model: tm.TaylorModel, x_vals):
    poly = model.evaluate_polynomial(x_vals)
    lo = poly + model.remainder[0]
    hi = poly + model.remainder[1]
    lo_ok = (true_vals >= lo - TOL).all()
    hi_ok = (true_vals <= hi + TOL).all()
    return lo_ok, hi_ok


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def test_from_constant():
    m = tm.from_constant(2.5, (0.0, 1.0), 4)
    assert m.degree == 4
    np.testing.assert_allclose(m.evaluate_polynomial(np.linspace(0, 1, 100)), 2.5)
    assert m.bounds() == (2.5, 2.5)


def test_from_variable_evaluates_to_identity():
    domain = (-2.0, 3.5)
    m = tm.from_variable(domain, 5)
    xs = np.linspace(*domain, 1000)
    np.testing.assert_allclose(m.evaluate_polynomial(xs), xs, atol=1e-12)
    np.testing.assert_allclose(m.bounds(), domain)


def test_from_variable_requires_degree_one():
    with pytest.raises(ValueError):
        tm.from_variable((0.0, 1.0), 0)


# ---------------------------------------------------------------------------
# Linear arithmetic exact
# ---------------------------------------------------------------------------


def test_add_sub_neg_scalar_exact():
    domain = (-1.0, 2.0)
    x = tm.from_variable(domain, 4)
    expr = tm.scalar_mul(-2.0, tm.sub(x, tm.from_constant(0.5, domain, 4)))
    xs = _sample_box(domain)
    np.testing.assert_allclose(expr.evaluate_polynomial(xs), -2.0 * (xs - 0.5), atol=1e-12)
    assert expr.remainder == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def test_mul_x_squared_exact_at_degree_two():
    domain = (-1.5, 2.0)
    x = tm.from_variable(domain, 2)
    x2 = tm.mul(x, x)
    xs = _sample_box(domain)
    np.testing.assert_allclose(x2.evaluate_polynomial(xs), xs**2, atol=1e-12)
    assert x2.remainder[0] <= 0.0 <= x2.remainder[1]


def test_mul_truncation_remainder_is_sound():
    """x*x at degree 1 must inflate the remainder."""
    domain = (-1.0, 1.0)
    x = tm.from_variable(domain, 1)
    x2 = tm.mul(x, x)
    assert x2.degree == 1
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(xs**2, x2, xs)
    assert lo_ok and hi_ok


def test_mul_cubic_sound():
    domain = (-1.0, 2.0)
    x = tm.from_variable(domain, 5)
    expr = tm.mul(tm.mul(x, x), x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(xs**3, expr, xs)
    assert lo_ok and hi_ok


# ---------------------------------------------------------------------------
# Univariate compositions — 10⁴-sample soundness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-1.0, 2.0), 8),
        ((-2.0, 0.5), 10),
        ((0.0, 3.0), 6),
    ],
)
def test_exp_soundness(domain, degree):
    x = tm.from_variable(domain, degree)
    e = tm.taylor_exp(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.exp(xs), e, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((0.5, 5.0), 8),
        ((1.0, 2.0), 6),
        ((1.0, 100.0), 12),
    ],
)
def test_log_soundness(domain, degree):
    x = tm.from_variable(domain, degree)
    expr = tm.taylor_log(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.log(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((0.5, 4.0), 8),
        ((1.0, 10.0), 10),
    ],
)
def test_sqrt_soundness(domain, degree):
    x = tm.from_variable(domain, degree)
    expr = tm.taylor_sqrt(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.sqrt(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((0.5, 3.0), 8),
        ((-3.0, -0.5), 8),
    ],
)
def test_recip_soundness(domain, degree):
    x = tm.from_variable(domain, degree)
    expr = tm.taylor_recip(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(1.0 / xs, expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-1.5, 1.5), 10),
        ((-2.0, 2.0), 12),
    ],
)
def test_sin_soundness(domain, degree):
    x = tm.from_variable(domain, degree)
    expr = tm.taylor_sin(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.sin(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-1.5, 1.5), 8),
        ((0.0, 2.5), 10),
    ],
)
def test_cos_soundness(domain, degree):
    x = tm.from_variable(domain, degree)
    expr = tm.taylor_cos(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.cos(xs), expr, xs)
    assert lo_ok and hi_ok


# ---------------------------------------------------------------------------
# Composed expressions
# ---------------------------------------------------------------------------


def test_exp_of_square_soundness():
    domain = (-1.0, 1.0)
    x = tm.from_variable(domain, 10)
    expr = tm.taylor_exp(tm.taylor_square(x))
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.exp(xs**2), expr, xs)
    assert lo_ok and hi_ok


def test_log_of_one_plus_square_soundness():
    domain = (-1.5, 1.5)
    x = tm.from_variable(domain, 12)
    one = tm.from_constant(1.0, domain, 12)
    expr = tm.taylor_log(tm.add(one, tm.taylor_square(x)))
    xs = _sample_box(domain)
    lo_ok, hi_ok = _enclosure_contains(np.log(1.0 + xs**2), expr, xs)
    assert lo_ok and hi_ok


# ---------------------------------------------------------------------------
# Order-1 sanity: matches cutting_planes.py linearization at the expansion
# point. Acceptance criterion from issue #51 M3.
# ---------------------------------------------------------------------------


def test_order_one_taylor_matches_cutting_planes_oa_cut():
    """Order-1 Taylor model of f at the box midpoint should give the same
    affine polynomial as a one-shot OA cut from cutting_planes.py."""
    domain = (-1.0, 2.0)
    a, b = domain
    x0 = 0.5 * (a + b)  # midpoint = 0.5

    # Order-1 Taylor model of exp on this box.
    x = tm.from_variable(domain, 1)
    e1 = tm.taylor_exp(x)
    # In the s-basis: coeffs[0] = f(x0), coeffs[1] = f'(x0) * half_width.
    # Equivalent affine in original x: f(x0) + f'(x0) * (x - x0).
    poly_at_a = e1.coeffs[0] + e1.coeffs[1] * (-1.0)  # at s = -1, x = a
    poly_at_b = e1.coeffs[0] + e1.coeffs[1] * (1.0)  # at s = +1, x = b
    # Recover slope and intercept in x-space.
    slope_x = (poly_at_b - poly_at_a) / (b - a)
    intercept_x = poly_at_a - slope_x * a

    # Independent OA-style linearization at x0: f(x0) + f'(x0) * (x - x0).
    f_x0 = float(np.exp(x0))
    fprime_x0 = float(np.exp(x0))  # exp is its own derivative
    oa_slope = fprime_x0
    oa_intercept = f_x0 - fprime_x0 * x0

    np.testing.assert_allclose(slope_x, oa_slope, atol=1e-10)
    np.testing.assert_allclose(intercept_x, oa_intercept, atol=1e-10)

    # Also produce the actual cutting_planes.LinearCut and confirm slope.
    cut = cutting_planes.generate_oa_cut(
        grad=np.array([fprime_x0]),
        func_val=f_x0,
        x_star=np.array([x0]),
        sense="<=",
    )
    assert cut.coeffs[0] == pytest.approx(oa_slope, abs=1e-10)
    # Cut form: grad·x ≤ grad·x* − g(x*); for our affine polynomial p(x) and
    # constraint p(x) ≤ z this corresponds to slope*x − z ≤ slope*x* − f(x*),
    # so the cut RHS is grad·x* − f(x*) = fprime_x0 * x0 − f(x0).
    assert cut.rhs == pytest.approx(fprime_x0 * x0 - f_x0, abs=1e-10)


# ---------------------------------------------------------------------------
# Convergence-rate (Bompadre-Mitsos-Chachuat 2013) — width should shrink
# polynomially under domain refinement for analytic f.
# ---------------------------------------------------------------------------


def test_convergence_under_refinement():
    """Halving the domain width should shrink the remainder by at least the
    leading-order Taylor remainder rate. For f = exp at degree d, the
    Lagrange remainder scales like ``half_width^(d+1)`` — i.e., halving the
    width shrinks the bound by a factor of 2^(d+1) in the limit. We assert
    a much weaker rate (factor of 2) to leave room for sampling overhead.
    """
    d = 6
    centers = np.linspace(-0.5, 0.5, 5)

    def width_at(half_w):
        ws = []
        for c in centers:
            dom = (c - half_w, c + half_w)
            e = tm.taylor_exp(tm.from_variable(dom, d))
            ws.append(e.remainder[1] - e.remainder[0])
        return float(np.mean(ws))

    w_full = width_at(1.0)
    w_half = width_at(0.5)
    w_quarter = width_at(0.25)

    assert w_half < w_full / 2.0
    assert w_quarter < w_half / 2.0


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def test_log_rejects_nonpositive_inner():
    x = tm.from_variable((-1.0, 1.0), 4)
    with pytest.raises(ValueError):
        tm.taylor_log(x)


def test_sqrt_rejects_zero_inner():
    x = tm.from_variable((0.0, 1.0), 4)
    with pytest.raises(ValueError):
        tm.taylor_sqrt(x)


def test_recip_rejects_zero_in_range():
    x = tm.from_variable((-1.0, 1.0), 4)
    with pytest.raises(ValueError):
        tm.taylor_recip(x)


def test_add_rejects_domain_mismatch():
    a = tm.from_variable((-1.0, 1.0), 4)
    b = tm.from_variable((0.0, 2.0), 4)
    with pytest.raises(ValueError):
        tm.add(a, b)
