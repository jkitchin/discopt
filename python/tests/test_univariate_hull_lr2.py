"""Unit tests for the LR-2 rigorous 1-D hull envelope (``univariate_hull``).

Task ``cert:LR-2`` (``docs/dev/lever-a-root-tightness-plan.md`` §4, H-UNI). These
assert the *soundness* invariants the envelope must satisfy on their own, without
the solver: every underestimator line lies at or below ``f`` and every
overestimator at or above — now proven **between** grid nodes, not merely at them
(#632 review finding 2), so the check samples densely and to a tight tolerance.

Each envelope is built through the real build bridge: the second-order remainder
that certifies the facets between nodes comes from a sound interval Hessian
(:func:`milp_relaxation._diag_hessian_enclosure`), exactly as the solver builds it.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.milp_relaxation import _diag_hessian_enclosure
from discopt._jax.univariate_hull import univariate_hull_envelope

# Enclosure never reached (box/finiteness abstention happens first).
_UNUSED_ENCL = lambda a, b: (0.0, 0.0)  # noqa: E731


def _build(build_expr, lo, hi):
    """Return ``(env, value_batch)`` for ``build_expr(x)`` on ``x∈[lo,hi]``.

    Routes through the same primitives the solver uses: a compiled+vmapped value
    batch and the interval-Hessian curvature enclosure.
    """
    m = dm.Model()
    x = m.continuous("x", lb=lo, ub=hi)
    e = build_expr(x)
    m.minimize(e)
    f = compile_expression(e, m)
    fb = jax.vmap(lambda xv: jnp.reshape(f(xv), ()))

    def value_batch(xarr):
        pts = np.asarray(xarr, dtype=float).reshape(-1, 1)
        return np.asarray(fb(jnp.asarray(pts))).ravel()

    encl = _diag_hessian_enclosure(e, m, x, 0)
    env = univariate_hull_envelope(lo, hi, value_batch, encl)
    return env, value_batch


def _assert_sound(build_expr, lo, hi, tol=1e-7, n=500_000, seed=0):
    env, value_batch = _build(build_expr, lo, hi)
    assert env is not None, f"envelope abstained on [{lo},{hi}]"
    lower, upper, (clo, chi) = env
    rng = np.random.default_rng(seed)
    # DENSE adversarial sampling — the whole point of the rigor fix is that a bound
    # must hold BETWEEN the 40010 build-grid nodes, so sample far more densely.
    X = np.concatenate([[lo, hi, 0.5 * (lo + hi)], np.sort(rng.uniform(lo, hi, n))])
    F = value_batch(X)
    under = np.max(np.array([s * X + b for s, b in lower]), axis=0)
    over = np.min(np.array([s * X + b for s, b in upper]), axis=0)
    assert np.all(under <= F + tol), f"underestimator cut f (max viol {np.max(under - F):.2e})"
    assert np.all(over >= F - tol), f"overestimator below f (max viol {np.max(F - over):.2e})"
    assert clo - tol <= float(np.min(F)) and float(np.max(F)) <= chi + tol, "col bounds exclude f"
    return env


# ---- soundness on a spread of curvatures (incl. the non-convex/non-concave case) --


def test_sound_convex():
    _assert_sound(lambda x: x * x, -1.0, 2.0)


def test_sound_concave():
    _assert_sound(lambda x: dm.log(x), 0.5, 8.0)


def test_sound_nonconvex_nonconcave_nvs09_composite():
    # g(x) = (ln(x-2))^2 + (ln(10-x))^2 on [3,9] — min f'' ≈ -0.097 (neither
    # convex nor concave), the exact H-UNI target that certifies nvs09.
    _assert_sound(lambda x: dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2, 3.0, 9.0)


def test_sound_cubic_inflection():
    # x^3 has an interior inflection at 0 (convex for x>0, concave for x<0) — the
    # neither-convex-nor-concave case, with f'' = 6x enclosed exactly by interval AD.
    _assert_sound(lambda x: x**3, -2.0, 2.0)


def test_unsupported_curvature_abstains_tanh():
    # tanh'' is outside the interval-AD grammar → unbounded enclosure → the envelope
    # must REFUSE (return None), never ship a facet it cannot certify between nodes.
    env, _ = _build(lambda x: dm.tanh(x), -3.0, 3.0)
    assert env is None


def test_sound_sharp_gaussian_well():
    # The #632 review's pathological case: a needle so sharply curved the OLD
    # grid-sampled shift left a ~1e-6 violation between nodes. The rigorous
    # second-order remainder must close it.
    _assert_sound(lambda x: -dm.exp(-10000.0 * (x - 0.05) ** 2), 0.0, 1.0)


# ---- tightness: the composite recovers its exact minimum ---------------------


def test_nvs09_composite_tightness():
    def g(x):
        return math.log(x - 2.0) ** 2 + math.log(10.0 - x) ** 2

    lo, hi = 3.0, 9.0
    env, _ = _build(lambda x: dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2, lo, hi)
    assert env is not None
    lower, _, _ = env
    xs = np.linspace(lo, hi, 200_001)
    under = np.max(np.array([[s * x + b for x in xs] for s, b in lower]), axis=0)
    exact_min = float(np.min([g(float(x)) for x in xs]))
    # sound (at or below the true min) and tight (small loss — the rigorous O(h^2)
    # remainder is ~1e-6, far inside the certificate budget).
    assert float(under.min()) <= exact_min + 1e-9
    assert exact_min - float(under.min()) < 1e-2


# ---- monotone-shrink: a sub-box never yields a worse (looser) local band -----


def test_monotone_shrink_sound_on_subbox():
    # a shrunk box must still be sound (per-node recomputation relies on this).
    _assert_sound(lambda x: dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2, 4.0, 7.0)


# ---- degenerate / abstention paths ------------------------------------------


def test_degenerate_box_abstains():
    def batch(xa):
        return np.asarray(xa, dtype=float).reshape(-1) ** 2

    assert univariate_hull_envelope(3.0, 3.0, batch, _UNUSED_ENCL) is None
    assert univariate_hull_envelope(5.0, 3.0, batch, _UNUSED_ENCL) is None


def test_non_finite_values_abstain():
    # f blows up inside the box → non-finite samples → abstain (never ship a NaN row)
    def batch(xa):
        x = np.asarray(xa, dtype=float).reshape(-1)
        with np.errstate(divide="ignore"):
            return 1.0 / x

    assert univariate_hull_envelope(-1.0, 1.0, batch, _UNUSED_ENCL) is None


def test_unbounded_hessian_enclosure_abstains():
    # If the curvature enclosure is unbounded on a cell, no O(h^2) bound exists →
    # the envelope must refuse rather than ship an uncertified facet.
    def batch(xa):
        return np.asarray(xa, dtype=float).reshape(-1) ** 2

    inf_encl = lambda a, b: (float("-inf"), float("inf"))  # noqa: E731
    assert univariate_hull_envelope(-1.0, 2.0, batch, inf_encl) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
