"""#814: `quadratic_curvature` restricts `eigvalsh` to the nonzero SUPPORT of Q.

`_quadratic_data` returns Q as a full ``n_total x n_total`` dense matrix, so a
quadratic that touches only a few variables inside a large model previously
triggered an O(n_total^3) ``eigvalsh`` — which ground the root relaxation build
for 75s+ on gastrans582 (2186 vars) before B&B ever started. The fix restricts the
eigenproblem to the nonzero support; the omitted rows/cols contribute exactly-zero
eigenvalues that cannot flip the CONVEX (min >= 0) / CONCAVE (max <= 0) sign tests,
so the classification is identical.

These tests pin the classification correctness on small-support-in-large-space
quadratics (the case that exercises the support-restriction branch).
"""

from __future__ import annotations

import discopt.modeling as dm
from discopt._jax.convexity import Curvature
from discopt._jax.convexity import patterns as pat


def _big_model_with_extra_vars(n_extra: int = 60):
    """A model with n_extra continuous vars so the full Q is (n_extra+3)-dim while
    any quadratic below touches only x/y/z — support strictly smaller than Q."""
    m = dm.Model("q814")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    z = m.continuous("z", lb=-2.0, ub=2.0)
    for i in range(n_extra):
        m.continuous(f"w{i}", lb=-1.0, ub=1.0)
    return m, x, y, z


def test_814_convex_quadratic_small_support_in_large_model():
    m, x, y, z = _big_model_with_extra_vars()
    # touches only x, y (support 2) inside a 63-var model
    assert pat.quadratic_curvature(x * x + 2.0 * (y * y), m) == Curvature.CONVEX


def test_814_concave_quadratic_small_support_in_large_model():
    m, x, y, z = _big_model_with_extra_vars()
    assert pat.quadratic_curvature(-(x * x) - 3.0 * (z * z), m) == Curvature.CONCAVE


def test_814_indefinite_quadratic_small_support_in_large_model():
    m, x, y, z = _big_model_with_extra_vars()
    # x^2 - y^2 is indefinite -> UNKNOWN (both signs present in the support)
    assert pat.quadratic_curvature(x * x - y * y, m) == Curvature.UNKNOWN


def test_814_affine_and_full_support_unchanged():
    m, x, y, z = _big_model_with_extra_vars(n_extra=0)
    # 3-var model, quadratic over all 3 (support == full): must still classify
    assert pat.quadratic_curvature(x * x + y * y + z * z, m) == Curvature.CONVEX
    # a purely linear expr is AFFINE (Q == 0)
    assert pat.quadratic_curvature(2.0 * x + y, m) == Curvature.AFFINE
