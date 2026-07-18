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


# ---- univariate intrinsics (convex-envelope ops + non-spanning S-shaped, P1.1) ----
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
        # S-shaped over NON-spanning boxes (P1.1: kernel-chain valid there)
        ("tanh+", lambda x: mc.tanh(x), lambda v: np.tanh(v[0]), [0.2], [1.8]),
        ("tanh-", lambda x: mc.tanh(x), lambda v: np.tanh(v[0]), [-1.8], [-0.2]),
        ("atan+", lambda x: mc.atan(x), lambda v: np.arctan(v[0]), [0.2], [2.0]),
        ("sigmoid-", lambda x: mc.sigmoid(x), lambda v: 1 / (1 + np.exp(-v[0])), [-2.0], [-0.2]),
        ("sinh+", lambda x: mc.sinh(x), lambda v: np.sinh(v[0]), [0.2], [1.5]),
    ],
)
def test_univariate(name, fn, f, lb, ub):
    _check(fn, f, lb, ub)


@pytest.mark.parametrize(
    "op,f",
    [
        ("tanh", np.tanh),
        ("atan", np.arctan),
        ("sigmoid", lambda t: 1 / (1 + np.exp(-t))),
        ("sinh", np.sinh),
    ],
)
def test_sshaped_spanning_sound_but_loose(op, f):
    # P1.1: over a sign-spanning box the S-shaped cv/cc fall back to the sound
    # constant envelope (jnp.where) — valid bracket + valid (zero) subgradient.
    _check(lambda x: getattr(mc, op)(x), lambda v: f(v[0]), [-1.5], [1.5])


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


# ---- variable division via sign-definite reciprocal (P1.2) ----
def test_division_positive_denominator():
    _check(lambda x, y: x / y, lambda v: v[0] / v[1], [0.5, 1.0], [3.0, 4.0])


def test_division_negative_denominator():
    _check(lambda x, y: x / y, lambda v: v[0] / v[1], [0.5, -4.0], [3.0, -1.0])


def test_division_nonaffine_denominator():
    # (x+1)/(y*y+1): non-affine MCBox denominator with a provably-positive interval
    # (y non-spanning, so the y*y bilinear interval stays >= 0 and the reciprocal is
    # well-defined; a spanning y would make the loose y*y interval cross 0 and the
    # reciprocal correctly returns a no-information bracket instead of a wrong bound).
    _check(
        lambda x, y: (x + 1.0) / (y * y + 1.0),
        lambda v: (v[0] + 1.0) / (v[1] * v[1] + 1.0),
        [-1.0, 0.5],
        [2.0, 2.0],
    )


def test_division_zero_crossing_denominator_noinfo():
    # denominator interval spans 0 -> reciprocal is unbounded -> no-information
    # bracket (non-finite cv/cc), jit-safe and never a wrong finite bound.
    z = relax_through(
        lambda x, y: x / y, jnp.array([1.0, 1.0]), jnp.array([1.0, -2.0]), jnp.array([2.0, 2.0])
    )
    assert not bool(jnp.isfinite(z.cv)) and not bool(jnp.isfinite(z.cc))


# ---- subgradient validity AT THE BOX FACES/CORNERS ----
# The interior-random ``_check`` never lands a base point exactly on a box face, but a
# Kelley/LP iterate always sits on one (an LP optimum is a polytope vertex). There the
# intrinsic envelopes used to hand back the ``jnp.clip`` tie subgradient (a 0.5-halved,
# INVALID slope), so the McCormick cut could exclude the true optimum -> too-high bound.
# These probe the support inequality with the base point pinned to corners/faces.
@pytest.mark.parametrize(
    "name,fn,lb,ub",
    [
        ("exp", lambda x: mc.exp(x), [0.0], [2.0]),
        ("exp-neg-slope", lambda x: mc.exp(-0.6 * x), [0.2], [2.0]),
        ("log", lambda x: mc.log(x), [0.5], [3.0]),
        ("sqrt", lambda x: mc.sqrt(x), [0.1], [4.0]),
        ("softplus", lambda x: mc.softplus(x), [-2.0], [2.0]),
        ("abs", lambda x: mc.abs(x), [-2.0], [3.0]),
        ("recip-of-affine", lambda x, y: x / y, [0.5, 1.0], [3.0, 4.0]),
        ("composite", lambda x, y: y - mc.exp(-0.6 * x), [0.2, 0.3], [2.0, 0.89]),
    ],
)
def test_subgradient_valid_at_box_faces(name, fn, lb, ub):
    import itertools

    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    n = lb.size

    @jax.jit
    def one(p):
        z = relax_through(fn, p, jnp.asarray(lb), jnp.asarray(ub))
        return z.cv, z.cc, z.sub_cv, z.sub_cc

    batch = jax.jit(jax.vmap(one))
    corners = np.array(list(itertools.product(*zip(lb, ub))), dtype=float)
    faces = []  # face-centre points: one coord pinned to a bound, others random
    rng = np.random.default_rng(1)
    for i in range(n):
        for bnd in (lb[i], ub[i]):
            for _ in range(15):
                p = lb + rng.random(n) * (ub - lb)
                p[i] = bnd
                faces.append(p)
    bases = np.vstack([corners, np.array(faces)]) if faces else corners
    probes = lb + rng.random((2000, n)) * (ub - lb)
    probes = np.vstack([probes, corners])
    cvB, ccB, _, _ = (np.asarray(a) for a in batch(jnp.asarray(probes)))
    cvA, ccA, scvA, sccA = (np.asarray(a) for a in batch(jnp.asarray(bases)))
    tol = 1e-7
    for k in range(len(bases)):
        lin_cv = cvA[k] + (probes - bases[k]) @ scvA[k]
        lin_cc = ccA[k] + (probes - bases[k]) @ sccA[k]
        assert np.all(cvB >= lin_cv - tol * (abs(cvA[k]) + 1)), (
            f"{name}: cv subgradient invalid at face base {bases[k]} "
            f"(max {float(np.max(lin_cv - cvB)):.2e})"
        )
        assert np.all(ccB <= lin_cc + tol * (abs(ccA[k]) + 1)), (
            f"{name}: cc subgradient invalid at face base {bases[k]} "
            f"(max {float(np.max(ccB - lin_cc)):.2e})"
        )


# ---- P1.4: fractional powers (signomials) over a positive base ----
@pytest.mark.parametrize(
    "a,lb,ub",
    [
        (0.5, 0.1, 4.0),  # concave increasing
        (1.5, 0.2, 3.0),  # convex increasing
        (2.5, 0.5, 2.0),  # convex increasing
        (-0.5, 0.3, 3.0),  # convex decreasing
        (-1.5, 0.5, 2.0),  # convex decreasing
        (0.25, 1.0, 5.0),
    ],
)
def test_fractional_power_relaxes_soundly(a, lb, ub):
    _check(lambda x: x**a, lambda v: v[0] ** a, [lb], [ub])


def test_fractional_power_of_affine_combo():
    # composition: (2x + y + 1)**0.5 over a box keeping the base positive.
    _check(
        lambda x, y: (2.0 * x + y + 1.0) ** 0.5,
        lambda v: (2.0 * v[0] + v[1] + 1.0) ** 0.5,
        [0.0, 0.0],
        [2.0, 3.0],
    )


def test_fractional_power_face_subgradients_valid():
    # box faces (where a Kelley/LP iterate sits) must have valid subgradients.
    for a, lb, ub in [(0.5, 0.1, 4.0), (1.5, 0.2, 3.0), (-0.5, 0.3, 3.0)]:

        @jax.jit
        def one(p, _a=a):
            z = relax_through(lambda x: x**_a, p, jnp.array([lb]), jnp.array([ub]))
            return z.cv, z.cc, z.sub_cv, z.sub_cc

        batch = jax.jit(jax.vmap(one))
        probes = np.linspace(lb, ub, 400).reshape(-1, 1)
        bases = np.array([[lb], [ub], [0.5 * (lb + ub)]])
        cvB, ccB, _, _ = (np.asarray(v) for v in batch(jnp.asarray(probes)))
        cvA, ccA, scvA, sccA = (np.asarray(v) for v in batch(jnp.asarray(bases)))
        for k in range(len(bases)):
            lin_cv = cvA[k] + (probes - bases[k]) @ scvA[k]
            lin_cc = ccA[k] + (probes - bases[k]) @ sccA[k]
            assert np.all(cvB >= lin_cv - 1e-7 * (abs(cvA[k]) + 1)), f"a={a} cv face"
            assert np.all(ccB <= lin_cc + 1e-7 * (abs(ccA[k]) + 1)), f"a={a} cc face"


def test_fractional_power_nonpositive_base_no_info():
    # x**a (non-integer a) is undefined for x <= 0 -> no-information bracket, not a
    # wrong finite bound (jit-safe). Any consumer of a non-finite bracket refuses.
    z = relax_through(lambda x: x**0.5, jnp.array([0.5]), jnp.array([-1.0]), jnp.array([2.0]))
    assert not bool(jnp.isfinite(z.cv)) and not bool(jnp.isfinite(z.cc))


# ---- sound-or-refuse ----
def test_refuse_nonpositive_integer_power():
    # x**0 / x**-2 (non-positive INTEGER exponent) still refuse (reciprocals go via /).
    with pytest.raises(UnsupportedMcboxOp):
        relax_through(lambda x: x**0, jnp.array([1.0]), jnp.array([0.5]), jnp.array([3.0]))
    with pytest.raises(UnsupportedMcboxOp):
        relax_through(lambda x: x**-2, jnp.array([1.0]), jnp.array([0.5]), jnp.array([3.0]))
