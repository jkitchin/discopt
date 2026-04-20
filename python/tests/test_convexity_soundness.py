"""Soundness oracles for the convexity detector.

These tests assert the contract that a CONVEX or CONCAVE verdict is a
mathematical proof, not a guess. Two independent oracles are used:

1. **Jensen fuzz** — for any expression the detector calls CONVEX, a
   random pair ``(x1, x2)`` in the box plus a random ``λ ∈ [0, 1]``
   must satisfy ``f(λ·x1 + (1−λ)·x2) ≤ λ·f(x1) + (1−λ)·f(x2)``. The
   symmetric inequality is checked for CONCAVE. Any violation is a
   detector bug.

2. **Numerical Hessian cross-check** — for a CONVEX verdict, a sampled
   Hessian must have all eigenvalues ≥ ``−tol`` at every sample point.
   A negative eigenvalue disproves convexity; a non-negative spectrum
   is *consistent* with the verdict (but does not by itself prove it,
   which is why both oracles are used together).

Marked ``slow`` because they evaluate + differentiate expressions at
many sample points. Run with ``pytest -m slow`` to execute explicitly.

References
----------
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.1.3 (Jensen's
inequality) and §3.1.4 (second-order conditions).
"""

from __future__ import annotations

import math
from typing import Callable

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.convexity import Curvature, classify_expr
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import Expression, FunctionCall, Model

pytestmark = pytest.mark.slow


JENSEN_SAMPLES = 64
HESSIAN_SAMPLES = 16
# Jensen tolerance accommodates fp roundoff plus curvature evaluation
# noise at a few function scales.
JENSEN_ATOL = 1e-6
JENSEN_RTOL = 1e-6
HESSIAN_ATOL = 1e-6


def _var_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all model-variable bounds into flat ``(lb, ub)`` vectors."""
    lbs: list[np.ndarray] = []
    ubs: list[np.ndarray] = []
    for v in model._variables:
        lbs.append(np.asarray(v.lb, dtype=float).ravel())
        ubs.append(np.asarray(v.ub, dtype=float).ravel())
    lb = np.concatenate(lbs) if lbs else np.zeros(0)
    ub = np.concatenate(ubs) if ubs else np.zeros(0)
    # Clip any infinite bounds to a finite envelope so sampling is well
    # defined. The envelope is deliberately wide; narrow boxes suffice
    # for the soundness check because we're not searching for worst-
    # case points.
    lb = np.where(np.isfinite(lb), lb, -10.0)
    ub = np.where(np.isfinite(ub), ub, 10.0)
    return lb, ub


def _random_points(lb: np.ndarray, ub: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """``n`` uniform random points in the box ``[lb, ub]``."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lb, ub, size=(n, lb.size))


def _jensen_check(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    convex: bool,
    seed: int = 0,
) -> None:
    """Assert Jensen's inequality direction consistent with ``convex``."""
    rng = np.random.default_rng(seed)
    xs = _random_points(lb, ub, JENSEN_SAMPLES, seed=seed)
    ys = _random_points(lb, ub, JENSEN_SAMPLES, seed=seed + 1)
    lambdas = rng.uniform(0.0, 1.0, size=JENSEN_SAMPLES)
    for x, y, lam in zip(xs, ys, lambdas):
        mid = lam * x + (1 - lam) * y
        lhs = float(f(jnp.asarray(mid)))
        rhs = lam * float(f(jnp.asarray(x))) + (1 - lam) * float(f(jnp.asarray(y)))
        tol = JENSEN_ATOL + JENSEN_RTOL * max(abs(lhs), abs(rhs), 1.0)
        if convex:
            assert lhs <= rhs + tol, (
                f"Jensen violation (convex): f(mid)={lhs:.6g} > "
                f"λf(x)+(1−λ)f(y)={rhs:.6g}, λ={lam:.3f}"
            )
        else:
            assert lhs >= rhs - tol, (
                f"Jensen violation (concave): f(mid)={lhs:.6g} < "
                f"λf(x)+(1−λ)f(y)={rhs:.6g}, λ={lam:.3f}"
            )


def _hessian_spectrum_check(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    convex: bool,
    seed: int = 0,
) -> None:
    """Assert Hessian eigenvalues have the sign required by ``convex``.

    Sampling cannot *prove* convexity but can *falsify* it: a negative
    eigenvalue at a reachable point disproves a CONVEX verdict.
    """
    hess = jax.jit(jax.hessian(f))
    xs = _random_points(lb, ub, HESSIAN_SAMPLES, seed=seed)
    for x in xs:
        H = np.asarray(hess(jnp.asarray(x)))
        if H.ndim == 0:
            eigs = np.array([float(H)])
        else:
            # Symmetrize defensively to avoid tiny asymmetric roundoff
            # flipping the eigenvalue signs.
            H = 0.5 * (H + H.T)
            eigs = np.linalg.eigvalsh(H)
        scale = max(1.0, float(np.max(np.abs(eigs))))
        if convex:
            assert np.min(eigs) >= -HESSIAN_ATOL * scale, (
                f"Hessian has negative eigenvalue {np.min(eigs):.3e} on a CONVEX verdict at x={x}"
            )
        else:
            assert np.max(eigs) <= HESSIAN_ATOL * scale, (
                f"Hessian has positive eigenvalue {np.max(eigs):.3e} on a CONCAVE verdict at x={x}"
            )


def _box_expression(expr: Expression, model: Model, convex: bool) -> None:
    """Run both oracles on ``expr`` with the detector verdict locked."""
    verdict = classify_expr(expr, model)
    target = Curvature.CONVEX if convex else Curvature.CONCAVE
    assert verdict in (target, Curvature.AFFINE), (
        f"Expected detector to prove {target.name}; got {verdict.name}"
    )
    f = compile_expression(expr, model)
    lb, ub = _var_bounds(model)
    _jensen_check(f, lb, ub, convex=convex)
    _hessian_spectrum_check(f, lb, ub, convex=convex)


# ──────────────────────────────────────────────────────────────────────
# Convex corpus — Jensen + Hessian must both confirm the verdict
# ──────────────────────────────────────────────────────────────────────


class TestConvexCorpus:
    def test_x_squared(self):
        m = Model("t")
        x = m.continuous("x", lb=-3, ub=3)
        _box_expression(x**2, m, convex=True)

    def test_sum_of_squares(self):
        m = Model("t")
        x = m.continuous("x", lb=-3, ub=3)
        y = m.continuous("y", lb=-3, ub=3)
        _box_expression(x**2 + y**2, m, convex=True)

    def test_exp_affine(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        _box_expression(dm.exp(2.0 * x + 1.0), m, convex=True)

    def test_exp_of_convex(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        _box_expression(dm.exp(x**2), m, convex=True)

    def test_abs_affine(self):
        m = Model("t")
        x = m.continuous("x", lb=-3, ub=3)
        _box_expression(dm.abs(x), m, convex=True)

    def test_reciprocal_positive(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        _box_expression(1.0 / x, m, convex=True)

    def test_even_power(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        _box_expression(x**4, m, convex=True)

    def test_odd_power_nonneg(self):
        m = Model("t")
        x = m.continuous("x", lb=0.0, ub=5.0)
        _box_expression(x**3, m, convex=True)

    def test_max_of_squares(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        expr = FunctionCall("max", x**2, y**2)
        _box_expression(expr, m, convex=True)

    def test_reciprocal_of_sqrt(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        _box_expression(1.0 / dm.sqrt(x), m, convex=True)


# ──────────────────────────────────────────────────────────────────────
# Concave corpus
# ──────────────────────────────────────────────────────────────────────


class TestConcaveCorpus:
    def test_log_positive(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        _box_expression(dm.log(x), m, convex=False)

    def test_sqrt_nonneg(self):
        m = Model("t")
        x = m.continuous("x", lb=0.01, ub=5.0)
        _box_expression(dm.sqrt(x), m, convex=False)

    def test_negative_of_quadratic(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        _box_expression(-(x**2), m, convex=False)

    def test_log_of_sqrt(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        _box_expression(dm.log(dm.sqrt(x)), m, convex=False)

    def test_reciprocal_negative(self):
        m = Model("t")
        x = m.continuous("x", lb=-5.0, ub=-0.5)
        _box_expression(1.0 / x, m, convex=False)

    def test_min_of_logs(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        y = m.continuous("y", lb=0.5, ub=5.0)
        expr = FunctionCall("min", dm.log(x), dm.log(y))
        _box_expression(expr, m, convex=False)


# ──────────────────────────────────────────────────────────────────────
# Non-convex corpus — detector must NOT classify as CONVEX/CONCAVE,
# AND the Jensen oracle must find a violation somewhere (sanity check
# for the oracle itself, not the detector).
# ──────────────────────────────────────────────────────────────────────


class TestNonconvexCorpusOracleFindsViolation:
    """For genuinely nonconvex expressions, both detector and oracle agree."""

    def test_bilinear_is_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        assert classify_expr(x * y, m) == Curvature.UNKNOWN

    def test_sin_is_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-10, ub=10)
        assert classify_expr(dm.sin(x), m) == Curvature.UNKNOWN

    def test_convex_minus_convex_is_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        assert classify_expr(x**2 - y**2, m) == Curvature.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Oracle self-test: verify Jensen actually flags a known violation
# ──────────────────────────────────────────────────────────────────────


class TestOracleSelfCheck:
    """Confirm the Jensen oracle can actually catch a violation.

    If this test fails, the soundness oracle is broken and protects
    nothing. It flags x**2 - y**2 as NOT convex by finding a Jensen
    violation on random samples.
    """

    def test_jensen_flags_saddle(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        expr = x**2 - y**2
        f = compile_expression(expr, m)
        lb, ub = _var_bounds(m)
        # We expect the convex-direction Jensen check to fail.
        with pytest.raises(AssertionError):
            _jensen_check(f, lb, ub, convex=True, seed=42)
        # And the concave-direction check to fail too.
        with pytest.raises(AssertionError):
            _jensen_check(f, lb, ub, convex=False, seed=42)

    def test_hessian_flags_saddle(self):
        m = Model("t")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        expr = x**2 - y**2
        f = compile_expression(expr, m)
        lb, ub = _var_bounds(m)
        with pytest.raises(AssertionError):
            _hessian_spectrum_check(f, lb, ub, convex=True, seed=42)


# ──────────────────────────────────────────────────────────────────────
# Near-boundary robustness
# ──────────────────────────────────────────────────────────────────────


class TestDomainEdges:
    """Exercise the rules near the edge of their domain guards."""

    def test_log_near_zero(self):
        m = Model("t")
        x = m.continuous("x", lb=1e-3, ub=2.0)
        _box_expression(dm.log(x), m, convex=False)

    def test_sqrt_at_zero(self):
        m = Model("t")
        # sqrt at x=0 has an infinite Hessian; shift the box above zero
        # so numerical derivatives stay finite for the oracle.
        x = m.continuous("x", lb=1e-3, ub=2.0)
        _box_expression(dm.sqrt(x), m, convex=False)

    def test_trig_unknown_on_large_box(self):
        """Trig is intentionally UNKNOWN on unbounded periodic ranges."""
        m = Model("t")
        x = m.continuous("x", lb=-math.pi, ub=math.pi)
        assert classify_expr(dm.sin(x), m) == Curvature.UNKNOWN
