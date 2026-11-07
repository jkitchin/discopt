"""Regression tests for #2191 and #2192: the convexity layer must not treat a
*nearly* special structure as the special structure itself.

* #2191 -- an exponent within ``np.isclose`` of 0, 1 or an integer was treated
  as exactly that value (``g**p`` replaced by ``1``, ``g`` or ``g**round(p)``).
  That is a different function, not a rounding error, so the interval Hessian
  could exclude the true one and ``certify_convex`` proved a nonconvex model
  convex. The same ``np.isclose`` decision sat in the syntactic classifier
  (``rules._classify_power``), the forward interval evaluator, the reverse
  (FBBT) power rule, and a ``1e-12`` variant in the quadratic-cut polynomial
  extractor and AMP's univariate-objective bound.
* #2192 -- ``_hess_is_exactly_zero`` accepted any Hessian with entries below
  ``1e-300`` as affine, so ``_rank1_quotient`` dropped a real (tiny) denominator
  Hessian ``-(g**2/h**2) H_h`` that a large scale factor makes decisive.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import interval_ad
from discopt._relax.convexity.certificate import certify_convex
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_ad import interval_hessian
from discopt._relax.convexity.interval_ad_reverse import tighten_box
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt._relax.convexity.rules import Curvature, classify_expr
from discopt._relax.cutting_planes import _quadratic_polynomial
from discopt.solvers.amp import _flatten_objective_power_terms


def _hess00(expr, model):
    h = interval_hessian(expr, model).hess
    return float(np.asarray(h.lo)[0, 0]), float(np.asarray(h.hi)[0, 0])


# ── #2191: near-special exponents ──────────────────────────────────────


@pytest.mark.smoke
def test_near_one_exponent_hessian_is_not_zero():
    """``u**p`` with ``p = 1 - 2**-20`` is strictly concave: H = p(p-1)u**(p-2) < 0."""
    p = 1.0 - 2.0**-20
    m = dm.Model("near_one")
    u = m.continuous("u", lb=1.0, ub=2.0)
    lo, hi = _hess00(u**p, m)
    true_at_1 = p * (p - 1.0)  # ~ -9.5e-7
    assert lo <= true_at_1 < 0.0, (lo, hi)


@pytest.mark.smoke
def test_near_one_exponent_composition_not_certified_convex():
    """``(u**p)**2 / u == u**(2p-1)``, strictly concave; the issue's Hessian probe."""
    p = 1.0 - 2.0**-20
    r = 2.0 * p - 1.0
    m = dm.Model("near_one_comp")
    u = m.continuous("u", lb=1.0, ub=2.0)
    j = (u**p) ** 2 / u
    lo, _ = _hess00(j, m)
    assert lo <= r * (r - 1.0), lo  # encloses j''(1) ~ -1.9e-6
    assert certify_convex(j, m) != Curvature.CONVEX


@pytest.mark.smoke
@pytest.mark.parametrize("p", [1.0 - 1e-7, 1.0 + 1e-7])
def test_rules_near_one_exponent_is_not_affine(p):
    """The syntactic classifier returned the *base* (AFFINE) for ``x**p``, p ~ 1."""
    m = dm.Model("rules_near_one")
    x = m.continuous("x", lb=1.0, ub=2.0)
    assert classify_expr(x**p, m) != Curvature.AFFINE
    # Concave for p < 1, convex for p > 1 on a positive base -- never the reverse.
    verdict = classify_expr(x**p, m)
    wrong = Curvature.CONVEX if p < 1.0 else Curvature.CONCAVE
    assert verdict != wrong


@pytest.mark.smoke
def test_interval_eval_near_integer_exponent_encloses_true_value():
    """``x**(2 + 1e-6)`` on [2, 3] reaches ``3**(2+1e-6) > 9``; ``x**2`` does not."""
    p = 2.0 + 1e-6
    assert np.isclose(p, 2.0)  # the old rule took this as exactly 2
    m = dm.Model("eval_near_int")
    x = m.continuous("x", lb=2.0, ub=3.0)
    enc = evaluate_interval(x**p, m)
    assert float(enc.hi) >= 3.0**p > 9.0 + 1e-6
    assert float(enc.lo) <= 2.0**p


@pytest.mark.smoke
def test_reverse_propagation_near_integer_exponent_keeps_feasible_points():
    """``x**(2 + 1e-7) >= 100`` holds for ``x >= 100**(1/p) ~ 10 - 1.15e-6``.

    The old ``isclose`` rule inverted the integer square, giving ``x >= 10`` and
    cutting off feasible points just below 10.
    """
    p = 2.0 + 1e-7
    assert np.isclose(p, round(p))  # the old rule took this as exactly 2
    m = dm.Model("reverse_near_int")
    x = m.continuous("x", lb=1.0, ub=20.0)
    box = tighten_box([(x**p, Interval(np.float64(100.0), np.float64(np.inf)))], m)
    true_root = 100.0 ** (1.0 / p)
    assert true_root < 10.0 - 1e-6
    assert float(np.asarray(box[x].lo)) <= true_root, box[x]
    assert float(np.asarray(box[x].hi)) >= 20.0


@pytest.mark.smoke
@pytest.mark.parametrize("p", [2.0 + 1e-13, 2.0 - 1e-13, 1.0 + 1e-13])
def test_cutting_plane_quadratic_extractor_rejects_near_integer_exponent(p):
    m = dm.Model("cp_near_int")
    x = m.continuous("x", lb=1.0, ub=8.0)
    assert _quadratic_polynomial(x**p, m) is None
    assert _quadratic_polynomial(x**2.0, m) is not None  # exact control


@pytest.mark.smoke
@pytest.mark.parametrize("p", [2.0 + 1e-13, 2.0 - 1e-13, 1.0 + 1e-13])
def test_amp_power_term_flattening_rejects_near_integer_exponent(p):
    m = dm.Model("amp_near_int")
    x = m.continuous("x", lb=1.0, ub=8.0)
    assert _flatten_objective_power_terms(x**p, m, 1.0, {}) is None
    groups: dict[int, dict[int, float]] = {}
    assert _flatten_objective_power_terms(x**2.0, m, 1.0, groups) == 0.0  # exact control
    assert groups[0] == {2: 1.0}


@pytest.mark.smoke
def test_exact_special_exponents_unchanged():
    """Exact 0, 1 and integer exponents keep their structural fast paths."""
    m = dm.Model("exact")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    assert classify_expr(x**1.0, m) == Curvature.AFFINE
    assert classify_expr(x**2.0, m) == Curvature.CONVEX
    assert _hess00(x**1.0, m) == (0.0, 0.0)
    lo, hi = _hess00(x**2.0, m)
    assert lo <= 2.0 <= hi and hi - lo < 1e-12  # outward-rounded point 2
    # Integer path works on a base that crosses zero (fractional would abstain).
    assert np.isfinite(_hess00(x**3.0, m)).all()


@pytest.mark.slow
@pytest.mark.correctness
def test_near_one_exponent_model_solves_to_true_optimum():
    # ~7 s / 3349 nodes on a dev machine: the B&B needed once the false convex
    # certificate is gone. The certificate checks above are the fast-lane probe.
    """End-to-end reproducer from #2191: optimum 54 at w = 1, not 59.19 at w = 4.79."""
    p = 1.0 - 2.0**-17
    m = dm.Model("near_one_exponent")
    w = m.continuous("w", lb=1.0, ub=7.0)
    f = 1e6 * ((w**p) ** 2 / w) - (1e6 - 36.0) * w + 2.0 * (w - 4.0) ** 2
    m.minimize(f)
    assert certify_convex(f, m) != Curvature.CONVEX
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.bound <= 54.0 + 1e-4
    assert r.objective == pytest.approx(54.0, abs=1e-3)


# ── #2192: tiny-but-real denominator curvature ─────────────────────────


@pytest.mark.smoke
def test_hess_is_exactly_zero_rejects_tiny_curvature():
    H = interval_ad._HMap
    keys = np.array([0], dtype=np.int64)
    assert interval_ad._hess_is_exactly_zero(interval_ad._H0)
    assert interval_ad._hess_is_exactly_zero(H(keys, np.array([0.0]), np.array([0.0])))
    tiny = 2.0 * 2.0**-1000
    assert not interval_ad._hess_is_exactly_zero(H(keys, np.array([tiny]), np.array([tiny])))
    # An enclosure of 0 that is not the point 0 is not a proof of affinity.
    sub = 5e-324
    assert not interval_ad._hess_is_exactly_zero(H(keys, np.array([-sub]), np.array([sub])))


@pytest.mark.smoke
def test_tiny_denominator_hessian_is_kept():
    """``2**1000 (1 + 2**-502 x)**2 / (1 + 2**-1000 x**2)``: exact f''(0) = -15/8."""
    m = dm.Model("tiny_den")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    f = 2.0**1000 * ((1.0 + 2.0**-502 * x) ** 2 / (1.0 + 2.0**-1000 * x**2))
    lo, _ = _hess00(f, m)
    assert lo <= -15.0 / 8.0, lo
    assert certify_convex(f, m) != Curvature.CONVEX


@pytest.mark.smoke
def test_rank1_perspective_still_certifies():
    """The rank-1 path must still fire for a genuinely affine denominator."""
    m = dm.Model("perspective")
    x = m.continuous("x", lb=0.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=4.0)
    f = (x - 2.0 * y + 1.0) ** 2 / (y + 0.5)
    assert certify_convex(f, m) == Curvature.CONVEX


@pytest.mark.smoke
@pytest.mark.correctness
def test_tiny_denominator_model_solves_to_true_optimum():
    """End-to-end reproducer from #2192: optimum 0.1 at x = -3, not 0.41 at x = 0.68."""
    k = 2.0**-1000
    m = dm.Model("tiny_denominator_curvature")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    f = k * ((x - 1.0) ** 2 / (k * (1.0 + x**2))) + 0.5 * x
    m.minimize(f)
    assert certify_convex(f, m) != Curvature.CONVEX
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.bound <= 0.1 + 1e-4
    assert r.objective == pytest.approx(0.1, abs=1e-3)
