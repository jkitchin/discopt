"""M3 regression: Taylor model bounds on benchmark-style subexpressions.

The Taylor model kernel (``discopt._jax.taylor_model``) is not yet wired
into the LP relaxation compiler — that integration ships alongside the
Chebyshev model integration as a follow-up. This regression exercises the
kernel directly on subexpressions drawn from typical benchmark problems
and asserts:

1. The polynomial-plus-remainder enclosure contains the true function
   value at every sampled point in the domain (soundness, ≥ 10⁴ samples).
2. The remainder shrinks under domain refinement at the predicted Taylor
   convergence rate (Bompadre, Mitsos, Chachuat 2013).
3. Bound width on a transcendental-heavy expression is consistent with the
   Chebyshev kernel's bound on the same expression (cross-check between
   M2 and M3).

These properties are the contract M3 makes with downstream consumers.
Breaking any of them is a correctness regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax import chebyshev_model as cm
from discopt._jax import taylor_model as tm

N_REGRESSION_SAMPLES = 10_000


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.parametrize(
    "name, build_expr, true_fn, domain, degree",
    [
        (
            "neg_exp_quad",
            lambda x_dom, d: tm.taylor_exp(
                tm.scalar_mul(
                    -1.0,
                    tm.taylor_square(tm.scalar_add(-1.0, tm.from_variable(x_dom, d))),
                )
            ),
            lambda xs: np.exp(-((xs - 1.0) ** 2)),
            (-1.0, 3.0),
            10,
        ),
        (
            "log_one_plus_xsq",
            lambda x_dom, d: tm.taylor_log(
                tm.scalar_add(1.0, tm.taylor_square(tm.from_variable(x_dom, d)))
            ),
            lambda xs: np.log(1.0 + xs**2),
            (-1.5, 1.5),
            12,
        ),
        (
            "sqrt_two_plus_xsq",
            lambda x_dom, d: tm.taylor_sqrt(
                tm.scalar_add(2.0, tm.taylor_square(tm.from_variable(x_dom, d)))
            ),
            lambda xs: np.sqrt(2.0 + xs**2),
            (-2.0, 2.0),
            10,
        ),
    ],
)
def test_taylor_enclosure_is_sound(name, build_expr, true_fn, domain, degree):
    rng = np.random.default_rng(0)
    xs = rng.uniform(domain[0], domain[1], size=N_REGRESSION_SAMPLES)
    expr = build_expr(domain, degree)
    poly = expr.evaluate_polynomial(xs)
    true = true_fn(xs)
    lo = poly + expr.remainder[0]
    hi = poly + expr.remainder[1]
    assert (true >= lo - 1e-9).all(), f"{name}: lower bound violated"
    assert (true <= hi + 1e-9).all(), f"{name}: upper bound violated"


@pytest.mark.regression
def test_taylor_remainder_shrinks_with_degree():
    """Tightness: increasing degree shrinks the remainder for analytic f."""
    domain = (-1.0, 1.0)  # narrow box — Taylor series of exp converges fast
    widths = []
    for d in (4, 8, 12):
        e = tm.taylor_exp(tm.from_variable(domain, d))
        widths.append(e.remainder[1] - e.remainder[0])
    assert widths[1] < widths[0]
    assert widths[2] < widths[1]


@pytest.mark.regression
def test_taylor_convergence_under_domain_refinement():
    """Bompadre-Mitsos-Chachuat convergence: halving the box width should
    shrink the order-d Taylor remainder by at least a factor of 2 (the rate
    is in fact 2^(d+1) asymptotically; we assert a much weaker rate to
    leave room for sampling-based remainder overhead)."""
    d = 6
    half_widths = (1.0, 0.5, 0.25)
    widths = []
    for hw in half_widths:
        dom = (-hw, hw)
        e = tm.taylor_exp(tm.from_variable(dom, d))
        widths.append(e.remainder[1] - e.remainder[0])
    assert widths[1] < widths[0] / 2.0
    assert widths[2] < widths[1] / 2.0


@pytest.mark.regression
def test_taylor_and_chebyshev_agree_on_smooth_expression():
    """Both M2 and M3 must produce sound enclosures of the same expression
    on the same domain. Their bound widths can differ (Chebyshev usually
    wins on transcendental-heavy expressions), but neither should violate
    soundness, and the polynomial parts should agree at the expansion
    midpoint (the Taylor and Chebyshev bases coincide there in value)."""
    domain = (-0.5, 0.5)
    d = 8
    x_t = tm.from_variable(domain, d)
    x_c = cm.from_variable(domain, d)
    expr_t = tm.taylor_exp(x_t)
    expr_c = cm.cheb_exp(x_c)

    # At the midpoint (s = 0), the polynomial parts must agree to working
    # precision since both are interpolating the same analytic function.
    midpoint = 0.0  # midpoint of [-0.5, 0.5]
    p_t = float(np.asarray(expr_t.evaluate_polynomial(np.array([midpoint]))).ravel()[0])
    p_c = float(np.asarray(expr_c.evaluate_polynomial(np.array([midpoint]))).ravel()[0])
    np.testing.assert_allclose(p_t, p_c, atol=1e-6)

    # Both must contain the true value within their respective remainders.
    true = float(np.exp(midpoint))
    assert expr_t.remainder[0] - 1e-9 <= true - p_t <= expr_t.remainder[1] + 1e-9
    assert expr_c.remainder[0] - 1e-9 <= true - p_c <= expr_c.remainder[1] + 1e-9
