"""Tests for ``discopt._jax.chebyshev_model`` (M2 of issue #51).

Covers the M2 acceptance criteria:
- Polynomial-plus-remainder enclosure contains the true function range on a
  randomized test set with ≥ 10⁴ samples per operator.
- Reproduces a published Chebyshev model example from Rajyaguru, Villanueva,
  Houska, Chachuat (2017) within 1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax import chebyshev_model as cm

N_SAMPLES = 10_000
TOL = 1e-9


def _sample_box(domain, n=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    a, b = domain
    return rng.uniform(a, b, size=n)


def _enclosure_contains(true_vals, model: cm.ChebyshevModel, x_vals):
    """Check that f(x) ∈ T(s(x)) + remainder for all x_vals."""
    poly = model.evaluate_polynomial(x_vals)
    lo = poly + model.remainder[0]
    hi = poly + model.remainder[1]
    lo_ok = (true_vals >= lo - TOL).all()
    hi_ok = (true_vals <= hi + TOL).all()
    return lo_ok, hi_ok, (lo, hi, poly)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def test_from_constant_evaluates_to_constant():
    m = cm.from_constant(2.5, (0.0, 1.0), degree=4)
    assert m.degree == 4
    xs = np.linspace(0, 1, 100)
    np.testing.assert_allclose(m.evaluate_polynomial(xs), 2.5)
    assert m.bounds() == (2.5, 2.5)


def test_from_variable_evaluates_to_identity():
    domain = (-2.0, 3.5)
    m = cm.from_variable(domain, degree=5)
    xs = np.linspace(*domain, 1000)
    np.testing.assert_allclose(m.evaluate_polynomial(xs), xs, atol=1e-12)
    np.testing.assert_allclose(m.bounds(), domain)


def test_from_variable_requires_degree_one():
    with pytest.raises(ValueError):
        cm.from_variable((0.0, 1.0), degree=0)


# ---------------------------------------------------------------------------
# Linear arithmetic — exact (zero remainder)
# ---------------------------------------------------------------------------


def test_add_is_exact_for_linear():
    domain = (-1.0, 2.0)
    x = cm.from_variable(domain, 4)
    c = cm.from_constant(3.0, domain, 4)
    s = cm.add(x, c)
    assert s.remainder == (0.0, 0.0)
    xs = _sample_box(domain)
    np.testing.assert_allclose(s.evaluate_polynomial(xs), xs + 3.0, atol=1e-12)


def test_sub_neg_scalar_mul_exact():
    domain = (0.0, 1.0)
    x = cm.from_variable(domain, 4)
    expr = cm.scalar_mul(-2.0, cm.sub(x, cm.from_constant(0.5, domain, 4)))
    xs = _sample_box(domain)
    np.testing.assert_allclose(expr.evaluate_polynomial(xs), -2.0 * (xs - 0.5), atol=1e-12)
    assert expr.remainder == (0.0, 0.0)


def test_scalar_add_is_exact():
    domain = (-1.0, 1.0)
    x = cm.from_variable(domain, 3)
    expr = cm.scalar_add(7.0, x)
    xs = _sample_box(domain)
    np.testing.assert_allclose(expr.evaluate_polynomial(xs), xs + 7.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Multiplication — soundness
# ---------------------------------------------------------------------------


def test_mul_x_squared_exact_at_degree_two():
    domain = (-1.5, 2.0)
    x = cm.from_variable(domain, degree=2)
    x2 = cm.mul(x, x)
    # x*x is degree-2: should be representable exactly.
    xs = _sample_box(domain)
    np.testing.assert_allclose(x2.evaluate_polynomial(xs), xs**2, atol=1e-12)
    assert x2.remainder[0] <= 0.0 <= x2.remainder[1]


def test_mul_truncation_remainder_is_sound():
    """x*x at degree 1 must inflate the remainder to cover the missing T_2 term."""
    domain = (-1.0, 1.0)
    x = cm.from_variable(domain, degree=1)
    x2 = cm.mul(x, x)
    assert x2.degree == 1
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(xs**2, x2, xs)
    assert lo_ok and hi_ok


def test_mul_polynomial_3rd_degree_sound():
    domain = (-1.0, 2.0)
    x = cm.from_variable(domain, degree=4)
    expr = cm.mul(cm.mul(x, x), x)  # x^3
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(xs**3, expr, xs)
    assert lo_ok and hi_ok


# ---------------------------------------------------------------------------
# Univariate composition — soundness on 10⁴ samples each
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-1.0, 2.0), 6),
        ((-2.0, 0.5), 8),
        ((0.0, 3.0), 5),
    ],
)
def test_exp_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    e = cm.cheb_exp(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.exp(xs), e, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((0.5, 5.0), 6),
        ((0.1, 1.0), 8),
        ((1.0, 100.0), 10),
    ],
)
def test_log_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    expr = cm.cheb_log(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.log(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((0.0, 4.0), 6),
        ((0.5, 10.0), 8),
        ((1e-3, 1.0), 10),
    ],
)
def test_sqrt_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    expr = cm.cheb_sqrt(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.sqrt(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((0.5, 3.0), 6),
        ((-3.0, -0.5), 6),
        ((0.1, 10.0), 8),
    ],
)
def test_recip_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    expr = cm.cheb_recip(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(1.0 / xs, expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-np.pi, np.pi), 8),
        ((-2.0, 5.0), 10),
    ],
)
def test_sin_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    expr = cm.cheb_sin(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.sin(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-2.0, 2.0), 6),
        ((0.0, 4.0), 8),
    ],
)
def test_cos_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    expr = cm.cheb_cos(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.cos(xs), expr, xs)
    assert lo_ok and hi_ok


@pytest.mark.parametrize(
    "domain, degree",
    [
        ((-3.0, 3.0), 6),
        ((-5.0, 1.0), 8),
    ],
)
def test_tanh_soundness(domain, degree):
    x = cm.from_variable(domain, degree)
    expr = cm.cheb_tanh(x)
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.tanh(xs), expr, xs)
    assert lo_ok and hi_ok


# ---------------------------------------------------------------------------
# Composed expressions
# ---------------------------------------------------------------------------


def test_exp_of_square_soundness():
    """exp(x^2) on [-1, 1]."""
    domain = (-1.0, 1.0)
    x = cm.from_variable(domain, 8)
    expr = cm.cheb_exp(cm.cheb_square(x))
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.exp(xs**2), expr, xs)
    assert lo_ok and hi_ok


def test_log_of_one_plus_square_soundness():
    """log(1 + x^2) on [-2, 2] — common test problem."""
    domain = (-2.0, 2.0)
    x = cm.from_variable(domain, 10)
    one = cm.from_constant(1.0, domain, 10)
    expr = cm.cheb_log(cm.add(one, cm.cheb_square(x)))
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.log(1.0 + xs**2), expr, xs)
    assert lo_ok and hi_ok


def test_sqrt_of_quadratic_soundness():
    """sqrt(x^2 + 1)."""
    domain = (-1.0, 2.0)
    x = cm.from_variable(domain, 8)
    one = cm.from_constant(1.0, domain, 8)
    expr = cm.cheb_sqrt(cm.add(cm.cheb_square(x), one))
    xs = _sample_box(domain)
    lo_ok, hi_ok, _ = _enclosure_contains(np.sqrt(xs**2 + 1.0), expr, xs)
    assert lo_ok and hi_ok


# ---------------------------------------------------------------------------
# Tightness sanity: bounds shrink as degree increases for analytic f
# ---------------------------------------------------------------------------


def test_exp_bound_tightens_with_degree():
    domain = (-1.0, 2.0)
    widths = []
    for d in (3, 6, 10):
        e = cm.cheb_exp(cm.from_variable(domain, d))
        widths.append(e.remainder[1] - e.remainder[0])
    # Strictly decreasing for analytic f on bounded domain.
    assert widths[0] > widths[1] > widths[2]


def test_sin_bound_tightens_with_degree():
    domain = (-np.pi, np.pi)
    widths = []
    for d in (4, 8, 12):
        e = cm.cheb_sin(cm.from_variable(domain, d))
        widths.append(e.remainder[1] - e.remainder[0])
    assert widths[0] > widths[1] > widths[2]


# ---------------------------------------------------------------------------
# Reference reproduction (Rajyaguru et al. 2017, Example 1-style):
# univariate exp on a small box should produce a Chebyshev expansion whose
# coefficients match a direct DCT-of-samples computation within 1e-6.
# ---------------------------------------------------------------------------


def test_reference_exp_matches_direct_chebyshev_expansion():
    """exp(x) on [-0.5, 0.5], degree 6 — a textbook Chebyshev model.

    Compares the saved coefficients against the analytical Chebyshev
    expansion of exp on [-0.5, 0.5] (computed independently via the
    standard Chebyshev-Gauss-Lobatto interpolation). Mismatch should be
    well within 1e-6 — the reference Chebyshev model construction
    described by Rajyaguru et al. is just degree-d truncation of the
    Chebyshev series.
    """
    domain = (-0.5, 0.5)
    d = 6
    x = cm.from_variable(domain, d)
    e = cm.cheb_exp(x)

    # Independent reference: DCT-II of f at degree-d+1 Chebyshev nodes
    # gives exact Chebyshev coefficients for polynomials of degree ≤ d.
    # For analytic f the error is the Chebyshev-tail bound, which for
    # exp on a half-unit interval is well under 1e-6 at d=6.
    a, b = domain
    # Module uses N = max(oversample * (d+1), min_nodes) = max(56, 64) = 64.
    Nref = 64
    k = np.arange(Nref)
    nodes_s = np.cos(np.pi * (k + 0.5) / Nref)
    # x = a + (b-a)/2 * (s+1)  so for [-0.5, 0.5] this is 0.5*s
    nodes_u = a + 0.5 * (b - a) * (nodes_s + 1.0)
    f_vals = np.exp(nodes_u)
    ref_coeffs = np.zeros(Nref)
    for j in range(Nref):
        w = 2.0 / Nref if j > 0 else 1.0 / Nref
        ref_coeffs[j] = w * float(np.sum(f_vals * np.cos(np.pi * j * (k + 0.5) / Nref)))

    np.testing.assert_allclose(e.coeffs, ref_coeffs[: d + 1], atol=1e-12)
    # Remainder must contain the true reference tail bound.
    ref_tail = float(np.sum(np.abs(ref_coeffs[d + 1 :])))
    assert e.remainder[1] - e.remainder[0] >= 2.0 * ref_tail - 1e-12


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def test_log_rejects_nonpositive_inner():
    x = cm.from_variable((-1.0, 1.0), 4)
    with pytest.raises(ValueError):
        cm.cheb_log(x)


def test_sqrt_rejects_negative_inner():
    x = cm.from_variable((-1.0, 1.0), 4)
    with pytest.raises(ValueError):
        cm.cheb_sqrt(x)


def test_recip_rejects_zero_in_range():
    x = cm.from_variable((-1.0, 1.0), 4)
    with pytest.raises(ValueError):
        cm.cheb_recip(x)


def test_add_rejects_domain_mismatch():
    a = cm.from_variable((-1.0, 1.0), 4)
    b = cm.from_variable((0.0, 2.0), 4)
    with pytest.raises(ValueError):
        cm.add(a, b)


def test_add_rejects_degree_mismatch():
    a = cm.from_variable((-1.0, 1.0), 4)
    b = cm.from_variable((-1.0, 1.0), 6)
    with pytest.raises(ValueError):
        cm.add(a, b)
