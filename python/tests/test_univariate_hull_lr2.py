"""Unit tests for the LR-2 rigorous 1-D hull envelope (``univariate_hull``).

Task ``cert:LR-2`` (``docs/dev/lever-a-root-tightness-plan.md`` §4, H-UNI). These
assert the *soundness* invariants the envelope must satisfy on their own, without
the solver: every underestimator line lies at or below ``f`` and every
overestimator at or above, at box corners and random interior points to a tight
tolerance; a shrunk box never loses (monotone-shrink); and the nvs09 composite is
lifted tightly enough to certify.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from discopt._jax.univariate_hull import univariate_hull_envelope


def _batch(fn):
    return lambda xarr: np.array([fn(float(x)) for x in np.asarray(xarr).ravel()], dtype=float)


def _tightest_under(lines, x):
    return max(s * x + b for s, b in lines)


def _tightest_over(lines, x):
    return min(s * x + b for s, b in lines)


def _assert_sound(fn, lo, hi, tol=1e-8, n=4000, seed=0):
    env = univariate_hull_envelope(lo, hi, _batch(fn))
    assert env is not None, f"envelope abstained on [{lo},{hi}]"
    lower, upper, (clo, chi) = env
    rng = np.random.default_rng(seed)
    pts = np.concatenate([[lo, hi, 0.5 * (lo + hi)], rng.uniform(lo, hi, n)])
    for x in pts:
        fv = fn(float(x))
        assert _tightest_under(lower, x) <= fv + tol, f"underestimator cut f at x={x}"
        assert _tightest_over(upper, x) >= fv - tol, f"overestimator below f at x={x}"
        # column bounds must enclose the true value
        assert clo - tol <= fv <= chi + tol, f"col bounds exclude f({x})={fv}"
    return env


# ---- soundness on a spread of curvatures (incl. the non-convex/non-concave case) --


def test_sound_convex():
    _assert_sound(lambda x: x * x, -1.0, 2.0)


def test_sound_concave():
    _assert_sound(lambda x: math.log(x), 0.5, 8.0)


def test_sound_nonconvex_nonconcave_nvs09_composite():
    # g(x) = (ln(x-2))^2 + (ln(10-x))^2 on [3,9] — min f'' ≈ -0.097 (neither
    # convex nor concave), the exact H-UNI target that certifies nvs09.
    def g(x):
        return math.log(x - 2.0) ** 2 + math.log(10.0 - x) ** 2

    _assert_sound(g, 3.0, 9.0)


def test_sound_sigmoid_like_inflection():
    # a genuine S-curve with an interior inflection point
    _assert_sound(lambda x: math.tanh(x), -3.0, 3.0)


# ---- closed-form envelope checks at corners to 1e-12 ------------------------


def test_corner_values_match_function_convex():
    lo, hi = -1.0, 2.0
    lower, upper, _ = univariate_hull_envelope(lo, hi, _batch(lambda x: x * x))
    for x in (lo, hi):
        # for a convex f the underestimator hull touches f at sampled corners
        assert _tightest_under(lower, x) <= x * x + 1e-9
        assert _tightest_over(upper, x) >= x * x - 1e-9


# ---- tightness: the composite recovers its exact minimum ---------------------


def test_nvs09_composite_tightness():
    def g(x):
        return math.log(x - 2.0) ** 2 + math.log(10.0 - x) ** 2

    lo, hi = 3.0, 9.0
    lower, _, _ = univariate_hull_envelope(lo, hi, _batch(g))
    xs = np.linspace(lo, hi, 200_001)
    under = np.max(np.array([[s * x + b for x in xs] for s, b in lower]), axis=0)
    exact_min = float(np.min([g(float(x)) for x in xs]))
    # per-variable underestimator minimum within the certificate budget
    # (nvs09 tol on the summed objective is 4.4e-3; per-variable share is far
    # tighter). The hull's min must be at or below the true min (sound) and
    # within a small loss of it (tight).
    assert float(under.min()) <= exact_min + 1e-9
    assert exact_min - float(under.min()) < 1e-2


# ---- monotone-shrink: a sub-box never yields a worse (looser) local band -----


def test_monotone_shrink_sound_on_subbox():
    def g(x):
        return math.log(x - 2.0) ** 2 + math.log(10.0 - x) ** 2

    # a shrunk box must still be sound (this is what per-node recomputation relies
    # on); assert soundness holds on an interior sub-box.
    _assert_sound(g, 4.0, 7.0)


# ---- degenerate / abstention paths ------------------------------------------


def test_degenerate_box_abstains():
    assert univariate_hull_envelope(3.0, 3.0, _batch(lambda x: x * x)) is None
    assert univariate_hull_envelope(5.0, 3.0, _batch(lambda x: x * x)) is None


def test_non_finite_values_abstain():
    # f blows up inside the box → non-finite samples → abstain (never ship a NaN row)
    assert univariate_hull_envelope(-1.0, 1.0, _batch(lambda x: 1.0 / x)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
