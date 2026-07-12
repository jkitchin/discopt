"""Unit tests for the LR-2 analytical 1-D hull envelope (``univariate_hull``).

Task ``cert:LR-2`` (lever-a §4/§0.1, H-UNI). The envelope is built analytically —
verified-curvature tangent/secant lines, NO sampling — so these assert the
*soundness* invariant STRICTLY: every underestimator line lies at or below ``f`` and
every overestimator at or above, checked by dense adversarial sampling at tolerance
0 (the lines are shifted outward by the construction's internal margin, so a true
bound must hold with zero slack). Plus tightness on the nvs09 composite, and the
abstention paths.
"""

from __future__ import annotations

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.univariate_hull import univariate_hull_envelope


def _env(build_expr, lo, hi):
    """Return ``(env, value_batch)`` for ``build_expr(x)`` on ``x∈[lo,hi]``."""
    m = dm.Model()
    x = m.continuous("x", lb=lo, ub=hi)
    e = build_expr(x)
    m.minimize(e)
    env = univariate_hull_envelope(e, m, x, 0, lo, hi)
    f = compile_expression(e, m)
    fb = jax.vmap(lambda xv: jnp.reshape(f(xv), ()))

    def value_batch(xarr):
        return np.asarray(fb(jnp.asarray(np.asarray(xarr, float).reshape(-1, 1)))).ravel()

    return env, value_batch


def _assert_sound(build_expr, lo, hi, n=500_000, seed=0):
    env, vb = _env(build_expr, lo, hi)
    assert env is not None, f"envelope abstained on [{lo},{hi}]"
    lower, upper, (clo, chi) = env
    rng = np.random.default_rng(seed)
    X = np.concatenate([[lo, hi, 0.5 * (lo + hi)], np.sort(rng.uniform(lo, hi, n))])
    F = vb(X)
    under = np.max(np.array([s * X + b for s, b in lower]), axis=0)
    over = np.min(np.array([s * X + b for s, b in upper]), axis=0)
    # STRICT: no violation at all (lines carry their own outward safety margin).
    assert np.all(under <= F), f"underestimator cut f (max viol {float(np.max(under - F)):.2e})"
    assert np.all(over >= F), f"overestimator below f (max viol {float(np.max(F - over)):.2e})"
    assert clo <= float(np.min(F)) and float(np.max(F)) <= chi, "col bounds exclude f"
    return env


# ---- soundness across curvature classes (incl. the multi-inflection target) --


def test_sound_convex():
    _assert_sound(lambda x: x * x, -1.0, 2.0)


def test_sound_concave():
    _assert_sound(lambda x: dm.log(x), 0.5, 8.0)


def test_sound_single_inflection_cubic():
    # x^3: one interior inflection at 0 (concave then convex).
    _assert_sound(lambda x: x**3, -2.0, 2.0)


def test_sound_multi_inflection_nvs09_composite():
    # g(x) = (ln(x-2))^2 + (ln(10-x))^2 on [3,9] — convex->concave->convex (TWO
    # inflections), the exact H-UNI target that certifies nvs09.
    _assert_sound(lambda x: dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2, 3.0, 9.0)


def test_sound_single_log_square():
    _assert_sound(lambda x: dm.log(x - 2.0) ** 2, 3.0, 9.0)


# ---- tightness: the nvs09 composite recovers its minimum closely ------------


def test_nvs09_composite_tightness():
    lo, hi = 3.0, 9.0
    env, vb = _env(lambda x: dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2, lo, hi)
    assert env is not None
    lower, _, _ = env
    xs = np.linspace(lo, hi, 200_001)
    under = np.max(np.array([[s * x + b for x in xs] for s, b in lower]), axis=0)
    exact_min = float(np.min(vb(xs)))
    # sound (at or below true min) and tight (small loss — well inside the
    # per-variable certificate budget the plan cites).
    assert float(under.min()) <= exact_min + 1e-9
    assert exact_min - float(under.min()) < 0.15


# ---- monotone-shrink: a sub-box stays sound (per-node recomputation) --------


def test_monotone_shrink_sound_on_subbox():
    _assert_sound(lambda x: dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2, 4.0, 7.0)


# ---- degenerate / abstention paths ------------------------------------------


def test_degenerate_box_abstains():
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    e = x**2
    m.minimize(e)
    assert univariate_hull_envelope(e, m, x, 0, 3.0, 3.0) is None
    assert univariate_hull_envelope(e, m, x, 0, 5.0, 3.0) is None


def test_randomized_soundness_sweep():
    """Adversarial fuzz: many random single-variable composites over random boxes;
    every built envelope must be sound under dense sampling (a certificate component
    can't be validated by hand-picked cases alone). Abstention is fine; a violation
    is not."""
    rng = np.random.default_rng(7)
    ops = [
        lambda e: dm.log(e),
        lambda e: dm.exp(e),
        lambda e: e**2,
        lambda e: e**3,
        lambda e: dm.sqrt(e),
        lambda e: dm.log(e) ** 2,
    ]
    built = 0
    for _ in range(30):
        lo = float(rng.uniform(0.5, 3.0))
        hi = lo + float(rng.uniform(1.0, 6.0))
        m = dm.Model()
        x = m.continuous("x", lb=lo, ub=hi)
        e = x * float(rng.uniform(0.3, 2.0)) + float(rng.uniform(0.5, 3.0))
        for _ in range(int(rng.integers(1, 3))):
            e = ops[int(rng.integers(len(ops)))](e)
            e = e * float(rng.uniform(0.5, 1.5)) + float(rng.uniform(-1.0, 2.0))
        m.minimize(e)
        try:
            env = univariate_hull_envelope(e, m, x, 0, lo, hi)
        except Exception:
            continue
        if env is None:
            continue
        lower, upper, _ = env
        f = compile_expression(e, m)
        fb = jax.vmap(lambda xv: jnp.reshape(f(xv), ()))
        X = np.sort(rng.uniform(lo, hi, 200_000))
        F = np.asarray(fb(jnp.asarray(X.reshape(-1, 1)))).ravel()
        if not np.all(np.isfinite(F)):
            continue
        built += 1
        under = np.max(np.array([s * X + b for s, b in lower]), axis=0)
        over = np.min(np.array([s * X + b for s, b in upper]), axis=0)
        assert np.all(under <= F), (
            f"underestimator cut f on [{lo},{hi}] ({float(np.max(under - F)):.2e})"
        )
        assert np.all(over >= F), (
            f"overestimator below f on [{lo},{hi}] ({float(np.max(F - over)):.2e})"
        )
    assert built >= 10, f"sweep built too few envelopes ({built})"


def test_pathological_curvature_abstains():
    # A needle far too sharply curved to resolve within the subdivision budget:
    # the envelope must REFUSE (None) rather than churn or ship a loose band.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    e = -dm.exp(-10000.0 * (x - 0.05) ** 2)
    m.minimize(e)
    assert univariate_hull_envelope(e, m, x, 0, 0.0, 1.0) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
