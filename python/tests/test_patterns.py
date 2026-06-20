"""Certification tests for the relaxation/cut pattern catalog.

Each implemented pattern in ``symbolic.patterns`` is certified numerically
(containment + curvature for relaxations, validity for cuts) to back the analytic
proof in its docstring / ``design/relaxation-patterns.md``.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from discopt._jax.symbolic import patterns as P  # noqa: E402

pytestmark = pytest.mark.relaxation


def _convexity_ok(fn, lo, hi, n=400, kind="convex", tol=1e-6):
    """Jensen check over random chords in the box (vectorized in 1-D or n-D)."""
    rng = np.random.default_rng(0)
    lo = np.atleast_1d(lo)
    hi = np.atleast_1d(hi)
    a = rng.uniform(lo, hi, size=(n, lo.size))
    b = rng.uniform(lo, hi, size=(n, lo.size))
    mid = 0.5 * (a + b)
    fa = np.array([float(fn(*pt)) for pt in a])
    fb = np.array([float(fn(*pt)) for pt in b])
    fm = np.array([float(fn(*pt)) for pt in mid])
    if kind == "convex":
        return bool(np.all(fm <= 0.5 * (fa + fb) + tol))
    return bool(np.all(fm >= 0.5 * (fa + fb) - tol))


# ---------------------------------------------------------------------------
# P3. Bilinear hull
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("box", [(-3, 4, -2, 5), (0, 10, 0, 7), (-5, -1, 2, 6)])
def test_bilinear_envelope_sound_and_curved(box):
    xl, xu, yl, yu = box
    rng = np.random.default_rng(1)
    xs = rng.uniform(xl, xu, 3000)
    ys = rng.uniform(yl, yu, 3000)
    cv, cc = P.bilinear_envelope(xs, ys, xl, xu, yl, yu)
    f = xs * ys
    assert np.all(np.asarray(cv) <= f + 1e-9)
    assert np.all(np.asarray(cc) >= f - 1e-9)
    assert _convexity_ok(
        lambda x, y: P.bilinear_envelope(x, y, xl, xu, yl, yu)[0], (xl, yl), (xu, yu)
    )
    assert _convexity_ok(
        lambda x, y: P.bilinear_envelope(x, y, xl, xu, yl, yu)[1],
        (xl, yl),
        (xu, yu),
        kind="concave",
    )


def test_bilinear_corner_exact():
    """The hull is tight (cv == cc == xy) at the box corners."""
    xl, xu, yl, yu = -2.0, 3.0, 1.0, 4.0
    for x in (xl, xu):
        for y in (yl, yu):
            cv, cc = P.bilinear_envelope(x, y, xl, xu, yl, yu)
            assert float(cv) == pytest.approx(x * y, abs=1e-9)
            assert float(cc) == pytest.approx(x * y, abs=1e-9)


# ---------------------------------------------------------------------------
# P4. Reciprocal + linear-fractional
# ---------------------------------------------------------------------------


def test_reciprocal_envelope_sound():
    yl, yu = 0.5, 5.0
    ys = np.linspace(yl, yu, 500)
    cv, cc = P.reciprocal_envelope(ys, yl, yu)
    f = 1.0 / ys
    assert np.all(np.asarray(cv) <= f + 1e-9)
    assert np.all(np.asarray(cc) >= f - 1e-9)


def test_linear_fractional_lifted_sound():
    xl, xu, yl, yu = 0.0, 8.0, 1.0, 6.0
    rng = np.random.default_rng(2)
    xs = rng.uniform(xl, xu, 4000)
    ys = rng.uniform(yl, yu, 4000)
    recip = 1.0 / ys  # the lifted reciprocal at its exact value
    cv, cc = P.linear_fractional_lifted(xs, recip, xl, xu, yl, yu)
    f = xs / ys
    assert np.all(np.asarray(cv) <= f + 1e-9)
    assert np.all(np.asarray(cc) >= f - 1e-9)


# ---------------------------------------------------------------------------
# P6. RLT sum-to-constant cut
# ---------------------------------------------------------------------------


def test_rlt_sum_constant_cut_valid():
    """sum_j w_{ij} == C * x_i holds at every feasible point of sum_j x_j = C."""
    C = 10.0
    rng = np.random.default_rng(3)
    for _ in range(2000):
        x0 = rng.uniform(0, C)
        x1 = C - x0
        x = np.array([x0, x1])
        for i in (0, 1):
            products = [x[i] * x[0], x[i] * x[1]]  # w_{i0}, w_{i1}
            lhs, rhs = P.rlt_sum_constant_cut(products, x[i], C)
            assert float(lhs) == pytest.approx(float(rhs), abs=1e-9)


def test_rlt_cut_not_implied_by_mccormick():
    """The RLT equality cuts off points the per-term McCormick hull admits.

    Pick a McCormick-feasible (x0, x1, w00, w01) that violates the RLT equality,
    showing the cut adds information beyond the bilinear hulls.
    """
    C = 10.0
    x0, x1 = 5.0, 5.0  # feasible: x0 + x1 = C
    # McCormick admits any w0j in [cv, cc] of x0*xj; choose w within the hull but
    # not on the surface so the RLT equality is violated.
    xl, xu = 0.0, C
    cv00, cc00 = P.bilinear_envelope(x0, x0, xl, xu, xl, xu)
    cv01, cc01 = P.bilinear_envelope(x0, x1, xl, xu, xl, xu)
    w00 = float(cv00)  # underestimate, McCormick-feasible
    w01 = float(cv01)
    lhs, rhs = float(w00 + w01), C * x0
    assert abs(lhs - rhs) > 1e-3  # RLT equality is violated -> cut is informative


# ---------------------------------------------------------------------------
# P7. Posynomial log-convexity
# ---------------------------------------------------------------------------


def test_posynomial_value_matches_monomial():
    c, a = 2.5, np.array([0.7, -1.3, 2.0])
    x = np.array([3.0, 1.5, 0.8])
    val = P.posynomial_logconvex(np.log(c), a, np.log(x))
    assert float(val) == pytest.approx(c * np.prod(x**a), rel=1e-9)


def test_posynomial_is_convex_in_log_domain():
    c, a = 1.3, np.array([0.5, 1.2])
    fn = lambda u0, u1: P.posynomial_logconvex(np.log(c), a, np.array([u0, u1]))  # noqa: E731
    # convex in u over a log-box
    assert _convexity_ok(fn, (-1.0, -1.0), (1.5, 1.5), kind="convex")


# ---------------------------------------------------------------------------
# P10. Fortet/Glover binary-product linearization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_binary_product_exact_on_binaries(n):
    import itertools

    for bits in itertools.product([0.0, 1.0], repeat=n):
        cv, cc = P.binary_product_linearization(list(bits))
        prod = float(np.prod(bits))
        assert float(cv) == pytest.approx(prod, abs=1e-9)
        assert float(cc) == pytest.approx(prod, abs=1e-9)


def test_binary_product_bounds_multilinear_relaxation():
    """Over [0,1]^n the Fortet bounds enclose the multilinear product."""
    rng = np.random.default_rng(7)
    for n in (2, 3):
        b = rng.uniform(0, 1, size=(5000, n))
        cv, cc = P.binary_product_linearization([b[:, i] for i in range(n)])
        prod = np.prod(b, axis=1)
        assert np.all(np.asarray(cv) <= prod + 1e-9)
        assert np.all(np.asarray(cc) >= prod - 1e-9)


# ---------------------------------------------------------------------------
# P12. Complementarity cut
# ---------------------------------------------------------------------------


def test_complementarity_cut_valid():
    """x/x_ub + y/y_ub <= 1 holds for every (x,y) with x*y=0, x,y in box."""
    x_ub, y_ub = 8.0, 5.0
    rng = np.random.default_rng(8)
    for _ in range(2000):
        if rng.random() < 0.5:
            x, y = 0.0, rng.uniform(0, y_ub)  # x = 0 branch
        else:
            x, y = rng.uniform(0, x_ub), 0.0  # y = 0 branch
        lhs, rhs = P.complementarity_cut(x, y, x_ub, y_ub)
        assert float(lhs) <= rhs + 1e-9


def test_complementarity_cut_separates_interior():
    """An interior (x,y) with xy>0 can violate the cut (so it is informative)."""
    x_ub, y_ub = 8.0, 5.0
    lhs, rhs = P.complementarity_cut(7.0, 4.0, x_ub, y_ub)  # xy = 28 > 0
    assert float(lhs) > rhs  # 7/8 + 4/5 = 1.675 > 1 -> cut excludes this point


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_covers_fields_and_status():
    done = set(P.available("done"))
    assert {
        "bilinear",
        "linear_fractional",
        "rlt_sum_constant",
        "posynomial",
        "binary_product",
        "complementarity",
    } <= done
    # every entry has fields and a citation
    for name in P.available():
        pat = P.PATTERNS[name]
        assert pat.fields and pat.citation
