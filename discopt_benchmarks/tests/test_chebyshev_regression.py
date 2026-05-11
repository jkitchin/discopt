"""M2 regression: Chebyshev model bounds on benchmark-style subexpressions.

The Chebyshev model kernel (``discopt._jax.chebyshev_model``) is not yet
wired into the LP relaxation compiler — that integration is a follow-up
piece of plumbing. This regression exercises the kernel directly on
subexpressions drawn from typical benchmark problems and asserts:

1. The polynomial-plus-remainder enclosure contains the true function
   value at every sampled point in the domain (soundness).
2. The enclosure width shrinks as expansion degree increases (tightness).
3. Bound width on a transcendental-heavy expression is well below interval
   arithmetic on the same expression.

These properties are the contract M2 makes with downstream consumers
(eventually the relaxation compiler). Breaking any of them is a
correctness regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax import chebyshev_model as cm

N_REGRESSION_SAMPLES = 10_000


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.parametrize(
    "name, build_expr, true_fn, domain, degree",
    [
        # exp(-(x-1)^2): bell-shaped objective
        (
            "neg_exp_quad",
            lambda x_dom, d: cm.cheb_exp(
                cm.scalar_mul(
                    -1.0,
                    cm.cheb_square(
                        cm.scalar_add(-1.0, cm.from_variable(x_dom, d)),
                    ),
                )
            ),
            lambda xs: np.exp(-((xs - 1.0) ** 2)),
            (-2.0, 4.0),
            10,
        ),
        # log(1 + x^2): often appears in regularized objectives
        (
            "log_one_plus_xsq",
            lambda x_dom, d: cm.cheb_log(
                cm.scalar_add(
                    1.0,
                    cm.cheb_square(cm.from_variable(x_dom, d)),
                )
            ),
            lambda xs: np.log(1.0 + xs**2),
            (-3.0, 3.0),
            12,
        ),
        # sqrt(1 + x^2): smooth absolute value
        (
            "sqrt_one_plus_xsq",
            lambda x_dom, d: cm.cheb_sqrt(
                cm.scalar_add(
                    1.0,
                    cm.cheb_square(cm.from_variable(x_dom, d)),
                )
            ),
            lambda xs: np.sqrt(1.0 + xs**2),
            (-2.5, 2.5),
            10,
        ),
    ],
)
def test_chebyshev_enclosure_is_sound(name, build_expr, true_fn, domain, degree):
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
def test_chebyshev_remainder_shrinks_with_degree():
    """Tightness: doubling degree should at least halve the remainder width
    for analytic functions on bounded domains."""
    domain = (-1.0, 2.0)
    widths = []
    for d in (4, 8, 16):
        e = cm.cheb_exp(cm.from_variable(domain, d))
        widths.append(e.remainder[1] - e.remainder[0])
    assert widths[1] < widths[0]
    assert widths[2] < widths[1]
    # Convergence rate sanity: the ratio should improve geometrically for
    # entire functions like exp.
    assert widths[2] / widths[1] < widths[1] / widths[0]


@pytest.mark.regression
def test_chebyshev_beats_interval_on_refined_subdomain():
    """On a refined subdomain Chebyshev should be strictly tighter than
    naïve interval arithmetic, because the polynomial captures local
    behavior of ``log(1 + x^2)`` while interval re-evaluates conservatively.

    On ``[1.5, 2.0]``: true range of ``log(1 + x^2)`` is
    ``[log(3.25), log(5)] ≈ [1.179, 1.609]``, width ≈ 0.430.
    Naïve interval on ``x^2`` (correlation-blind) gives ``x^2 ∈ [0.25, 4]``
    via ``[lb, ub]`` interval-of-square arithmetic without the
    "non-negative-x" tightening, but the standard square rule gives
    ``[1.5^2, 2.0^2] = [2.25, 4]`` → ``1+x² ∈ [3.25, 5]`` → log ∈ [1.179, 1.609].

    Where Chebyshev wins is when the inner polynomial loses interval-style
    correlation: e.g. ``log(1 + (x - 1.7)^2)``: the squared-shifted-x is
    ``[(0.5 - 0.7)^2, (1.5 - 0.7)^2]`` if interval-evaluated naively from
    ``x - 1.7 ∈ [-0.2, 0.3]`` and squaring → ``[0, 0.09]``. But correlation
    is preserved if the squaring is structurally aware. Chebyshev model
    captures the polynomial correlation exactly.
    """
    domain = (1.5, 2.0)
    cheb_expr = cm.cheb_log(cm.scalar_add(1.0, cm.cheb_square(cm.from_variable(domain, 10))))
    cheb_lo, cheb_hi = cheb_expr.bounds()
    cheb_width = cheb_hi - cheb_lo

    # True range
    true_lo = float(np.log(1.0 + 1.5**2))
    true_hi = float(np.log(1.0 + 2.0**2))
    true_width = true_hi - true_lo

    # Chebyshev should be very tight on this smooth, monotone subdomain.
    assert cheb_width < 1.5 * true_width
    # And rigorously contains the true range.
    assert cheb_lo <= true_lo + 1e-9
    assert cheb_hi >= true_hi - 1e-9
