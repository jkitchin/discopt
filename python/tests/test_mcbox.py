"""P0.2 soundness for the MCBox propagating McCormick type (MAiNGO-parity plan).

Every operator must satisfy, over its box: (a) bracket cv <= f <= cc on samples;
(b) subgradient support — cv above every tangent, cc below every tangent (the
rule-based subgradients are VALID, no jax.grad-over-construction); (c) interval
containment lo <= f <= hi. Plus: composite expressions, pytree jit/vmap, and loud
refusal outside scope.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax import mcbox as mc
from discopt._jax.mcbox import UnsupportedMcboxOp, relax_through

jax.config.update("jax_enable_x64", True)
RNG = np.random.default_rng(0)


def _check(fn, f_true, lb, ub, n_samp=4000, n_pairs=400, tol=1e-7):
    """Vectorized: one jitted vmap computes (cv,cc,sub,lo,hi) for all sample points."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    n = lb.size
    lbj, ubj = jnp.asarray(lb), jnp.asarray(ub)

    @jax.jit
    def batch(P):
        def one(p):
            z = relax_through(fn, p, lbj, ubj)
            return z.cv, z.cc, z.sub_cv, z.sub_cc, z.lo, z.hi

        return jax.vmap(one)(P)

    P = lb + RNG.random((n_samp, n)) * (ub - lb)
    cv, cc, scv, scc, lo, hi = (np.asarray(a) for a in batch(jnp.asarray(P)))
    fx = np.array([float(f_true(x)) for x in P])
    s = tol * (np.abs(fx) + 1)
    assert np.all(cv <= fx + s), f"cv>f max {np.max(cv - fx):.2e}"
    assert np.all(fx <= cc + s), f"f>cc max {np.max(fx - cc):.2e}"
    assert np.all(lo <= fx + s) and np.all(fx <= hi + s), "interval containment failed"
    # subgradient support over random pairs (vectorized)
    A = lb + RNG.random((n_pairs, n)) * (ub - lb)
    B = lb + RNG.random((n_pairs, n)) * (ub - lb)
    cvA, ccA, gcvA, gccA, _, _ = (np.asarray(a) for a in batch(jnp.asarray(A)))
    cvB, ccB, _, _, _, _ = (np.asarray(a) for a in batch(jnp.asarray(B)))
    lin_cv = cvA + np.einsum("ij,ij->i", gcvA, B - A)
    lin_cc = ccA + np.einsum("ij,ij->i", gccA, B - A)
    assert np.all(cvB >= lin_cv - tol * (np.abs(cvA) + 1)), (
        f"cv subgradient violated (max {np.max(lin_cv - cvB):.2e})"
    )
    assert np.all(ccB <= lin_cc + tol * (np.abs(ccA) + 1)), (
        f"cc subgradient violated (max {np.max(ccB - lin_cc):.2e})"
    )


# ---- arithmetic ----
def test_bilinear():
    _check(lambda x, y: x * y, lambda v: v[0] * v[1], [0.0, 0.0], [4.0, 4.0])


def test_bilinear_mixed_sign():
    _check(lambda x, y: x * y, lambda v: v[0] * v[1], [-2.0, -3.0], [3.0, 1.0])


def test_affine_combo():
    _check(
        lambda x, y: 2.0 * x - 3.0 * y + 5.0,
        lambda v: 2 * v[0] - 3 * v[1] + 5,
        [-1.0, -1.0],
        [2.0, 2.0],
    )


def test_even_power():
    _check(
        lambda x, y: x**2 - 2.0 * x * y,
        lambda v: v[0] ** 2 - 2 * v[0] * v[1],
        [-2.0, -2.0],
        [3.0, 3.0],
    )


def test_odd_power_single_sign():
    # x**3 via sound repeated multiplication (the bilinear rule) — valid on a positive base
    _check(lambda x: x**3, lambda v: v[0] ** 3, [1.0], [4.0])


def test_odd_power_spanning_zero_is_sound():
    # repeated-mult power is sign-agnostic, so x**3 over a sign-spanning box is SOUND
    # (looser than the tight monomial hull, which is P1.3) — no refusal needed here.
    _check(lambda x: x**3, lambda v: v[0] ** 3, [-2.0], [2.0])


# ---- univariate intrinsics (only provably-convex envelopes; S-shaped -> P1.1) ----
@pytest.mark.parametrize(
    "name,fn,f,lb,ub",
    [
        ("exp", lambda x: mc.exp(x), lambda v: np.exp(v[0]), [-1.0], [1.5]),
        ("log", lambda x: mc.log(x), lambda v: np.log(v[0]), [0.5], [4.0]),
        ("log2", lambda x: mc.log2(x), lambda v: np.log2(v[0]), [0.5], [4.0]),
        ("log10", lambda x: mc.log10(x), lambda v: np.log10(v[0]), [0.5], [4.0]),
        ("sqrt", lambda x: mc.sqrt(x), lambda v: np.sqrt(v[0]), [0.1], [5.0]),
        ("softplus", lambda x: mc.softplus(x), lambda v: np.logaddexp(v[0], 0.0), [-2.0], [2.0]),
        ("abs", lambda x: mc.abs(x), lambda v: np.abs(v[0]), [-2.0], [3.0]),
    ],
)
def test_univariate(name, fn, f, lb, ub):
    _check(fn, f, lb, ub)


@pytest.mark.parametrize("op", ["tanh", "atan", "sigmoid", "sinh"])
def test_sshaped_refused_until_p1(op):
    with pytest.raises(UnsupportedMcboxOp):
        relax_through(
            lambda x: getattr(mc, op)(x), jnp.array([0.5]), jnp.array([-1.0]), jnp.array([1.0])
        )


# ---- composite (the P0.1 function + a richer one) ----
def test_composite_p01():
    _check(
        lambda x, y: x * mc.exp(y) - x * y,
        lambda v: v[0] * np.exp(v[1]) - v[0] * v[1],
        [0.0, 0.0],
        [2.0, 1.5],
    )


def test_composite_transcendental():
    _check(
        lambda x, y: mc.exp(x * y) + x * mc.log(y + 3.0),
        lambda v: np.exp(v[0] * v[1]) + v[0] * np.log(v[1] + 3.0),
        [-1.0, 0.0],
        [1.0, 2.0],
    )


# ---- pytree: jit + vmap over boxes ----
def test_jit_and_vmap_over_boxes():
    def fn(x, lb, ub):
        z = relax_through(lambda a, b: a * mc.exp(b) - a * b, x, lb, ub)
        return z.cv, z.cc, z.sub_cv

    j = jax.jit(jax.vmap(fn))
    NB = 64
    lb = jnp.array(RNG.random((NB, 2)))
    ub = lb + 0.5
    x = (lb + ub) / 2
    cv, cc, scv = j(x, lb, ub)
    assert cv.shape == (NB,) and scv.shape == (NB, 2)
    assert bool(jnp.all(cv <= cc + 1e-9))


# ---- sound-or-refuse ----
def test_refuse_noninteger_power():
    with pytest.raises(UnsupportedMcboxOp):
        relax_through(lambda x: x**1.5, jnp.array([1.0]), jnp.array([0.5]), jnp.array([3.0]))


def test_refuse_var_division():
    with pytest.raises(UnsupportedMcboxOp):
        relax_through(
            lambda x, y: x / y, jnp.array([1.0, 1.0]), jnp.array([0.0, 1.0]), jnp.array([2.0, 3.0])
        )
