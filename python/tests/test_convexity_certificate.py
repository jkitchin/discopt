"""Soundness + consistency tests for the box-local convexity certificate.

Three layers of checks ride on this certificate:

1. **Consistency with the syntactic walker.** PR 1's CONVEX verdicts
   must remain CONVEX under the certificate; CONCAVE likewise.
   Otherwise the two layers would disagree on cases both can reason
   about.

2. **Box-local tightening.** Expressions the syntactic walker flags
   UNKNOWN but which are convex on a tightened box must be
   successfully certified (e.g., ``x^3`` on the full line is UNKNOWN
   but on ``[0, ∞)`` it is CONVEX).

3. **Soundness — Jensen fuzz.** Every certificate CONVEX verdict is
   independently validated against random convex-combination tests
   on the box. A violation would indicate the certificate asserted a
   convexity claim that isn't true.
"""

from __future__ import annotations

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.convexity import Curvature, certify_convex, classify_expr
from discopt._jax.convexity.interval import Interval
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import Model


def _jensen_consistent(expr, model, *, convex: bool, seed: int = 0):
    f = compile_expression(expr, model)
    rng = np.random.default_rng(seed)
    lbs = np.concatenate([np.asarray(v.lb).ravel() for v in model._variables])
    ubs = np.concatenate([np.asarray(v.ub).ravel() for v in model._variables])
    for _ in range(48):
        x = rng.uniform(lbs, ubs)
        y = rng.uniform(lbs, ubs)
        lam = rng.uniform()
        mid = lam * x + (1 - lam) * y
        lhs = float(f(jnp.asarray(mid)))
        rhs = lam * float(f(jnp.asarray(x))) + (1 - lam) * float(f(jnp.asarray(y)))
        tol = 1e-6 + 1e-6 * max(abs(lhs), abs(rhs), 1.0)
        if convex:
            assert lhs <= rhs + tol, f"Jensen (convex) violation at x={x}, y={y}"
        else:
            assert lhs >= rhs - tol, f"Jensen (concave) violation at x={x}, y={y}"


# ──────────────────────────────────────────────────────────────────────
# Consistency with the syntactic walker
# ──────────────────────────────────────────────────────────────────────


class TestConsistencyWithSyntactic:
    """Expressions the syntactic walker proves CONVEX must also be
    certified CONVEX (or abstained) — never contradicted."""

    def test_convex_quadratic(self):
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        assert classify_expr(x**2, m) == Curvature.CONVEX
        assert certify_convex(x**2, m) == Curvature.CONVEX

    def test_convex_exp_of_square(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        assert classify_expr(dm.exp(x**2), m) == Curvature.CONVEX
        assert certify_convex(dm.exp(x**2), m) == Curvature.CONVEX

    def test_concave_log(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        assert classify_expr(dm.log(x), m) == Curvature.CONCAVE
        assert certify_convex(dm.log(x), m) == Curvature.CONCAVE

    def test_concave_sqrt(self):
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=5.0)
        assert classify_expr(dm.sqrt(x), m) == Curvature.CONCAVE
        assert certify_convex(dm.sqrt(x), m) == Curvature.CONCAVE

    def test_reciprocal_positive(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=3.0)
        assert classify_expr(1.0 / x, m) == Curvature.CONVEX
        assert certify_convex(1.0 / x, m) == Curvature.CONVEX


# ──────────────────────────────────────────────────────────────────────
# Box-local tightening — certificate beats syntactic walker
# ──────────────────────────────────────────────────────────────────────


class TestBoxLocalTightening:
    """Cases where the syntactic walker stays UNKNOWN but the
    certificate produces a proof on a tight enough box."""

    def test_cubic_on_nonneg_box(self):
        """``x^3`` on [0.5, 3] is convex; syntactic rule requires the
        base to be AFFINE, which it is, and NONNEG — which it is. The
        certificate independently confirms it."""
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=3.0)
        # The syntactic walker already handles this case — the goal
        # here is for the certificate to agree, not to break new
        # ground.
        assert classify_expr(x**3, m) == Curvature.CONVEX
        assert certify_convex(x**3, m) == Curvature.CONVEX

    def test_quartic_on_narrow_box_gives_certificate(self):
        """``x^4 − 2 x^2`` is nonconvex on a wide domain but convex
        on a narrow box with ``|x|`` bounded above ``1/√2``."""
        m = Model("t")
        # Wide box: detector says UNKNOWN.
        x = m.continuous("x", lb=-3.0, ub=3.0)
        assert classify_expr(x**4 - 2.0 * x**2, m) == Curvature.UNKNOWN
        # On a narrow box well away from the inflection point the
        # certificate should prove CONVEX using the interval Hessian.
        narrow = {x: Interval.from_bounds(1.0, 2.0)}
        verdict = certify_convex(x**4 - 2.0 * x**2, m, box=narrow)
        assert verdict == Curvature.CONVEX


# ──────────────────────────────────────────────────────────────────────
# Abstention on unsupported or indefinite cases
# ──────────────────────────────────────────────────────────────────────


class TestAbstention:
    def test_bilinear_remains_indefinite(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        # Hessian of xy is [[0, 1], [1, 0]] — eigenvalues ±1.
        assert certify_convex(x * y, m) is None

    def test_abs_not_certified(self):
        """|x| is convex in truth but has no well-defined Hessian at 0."""
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        # Certificate abstains because the interval Hessian is unbounded.
        assert certify_convex(dm.abs(x), m) is None

    def test_odd_power_mixed_sign_abstains(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        # x^3 on [-1, 1] is neither convex nor concave — the
        # Hessian 6x straddles zero.
        assert certify_convex(x**3, m) is None


# ──────────────────────────────────────────────────────────────────────
# Jensen soundness audit — every CONVEX certificate passes Jensen
# ──────────────────────────────────────────────────────────────────────


class TestCrossConsistencyWithSyntactic:
    """Enforce the global invariant: the certificate never disagrees
    with a proven syntactic verdict. It may abstain (return ``None``)
    when the interval Hessian is too loose for Gershgorin, but it
    must never emit the opposite verdict."""

    @pytest.mark.parametrize(
        "builder,lb,ub,syntactic",
        [
            (lambda x: x**2 + 1.0, -2.0, 2.0, Curvature.CONVEX),
            (lambda x: dm.exp(x), -2.0, 2.0, Curvature.CONVEX),
            (lambda x: dm.exp(x**2), -1.0, 1.0, Curvature.CONVEX),
            (lambda x: 1.0 / x, 0.5, 3.0, Curvature.CONVEX),
            (lambda x: dm.log(x), 0.5, 5.0, Curvature.CONCAVE),
            (lambda x: dm.sqrt(x), 0.1, 5.0, Curvature.CONCAVE),
            (lambda x: -(x**2), -2.0, 2.0, Curvature.CONCAVE),
        ],
    )
    def test_certificate_never_disagrees(self, builder, lb, ub, syntactic):
        m = Model("t")
        x = m.continuous("x", lb=lb, ub=ub)
        expr = builder(x)
        assert classify_expr(expr, m) == syntactic
        cert = certify_convex(expr, m)
        if cert is not None:
            assert cert == syntactic, f"Certificate {cert} disagrees with syntactic {syntactic}"


@pytest.mark.slow
class TestJensenAuditOnCertificate:
    """Every CONVEX/CONCAVE verdict from the certificate is cross-
    checked via Jensen's inequality."""

    @pytest.mark.parametrize(
        "builder,lb,ub,convex",
        [
            (lambda x: x**2, -2.0, 2.0, True),
            (lambda x: dm.exp(x**2), -1.0, 1.0, True),
            (lambda x: 1.0 / x, 0.5, 3.0, True),
            (lambda x: dm.log(x), 0.5, 5.0, False),
            (lambda x: dm.sqrt(x), 0.1, 5.0, False),
            (lambda x: x**4 - 2.0 * x**2, 1.1, 2.5, True),
        ],
    )
    def test_scalar_certificate_passes_jensen(self, builder, lb, ub, convex):
        m = Model("t")
        x = m.continuous("x", lb=lb, ub=ub)
        expr = builder(x)
        verdict = certify_convex(expr, m)
        assert verdict == (Curvature.CONVEX if convex else Curvature.CONCAVE)
        _jensen_consistent(expr, m, convex=convex)
