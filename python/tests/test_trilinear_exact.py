"""Tests for relax_trilinear_exact (permutation-symmetric nested McCormick).

The exact convex hull of x*y*z on a box (Rikun 1997 / Meyer-Floudas 2004)
requires explicit facet inequalities not yet implemented; ``relax_trilinear_exact``
ships a sound, strict improvement over the original single-ordering nested
bilinear by considering all three orderings ((xy)z, (xz)y, (yz)x) and merging
with max-of-cvs / min-of-ccs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.envelopes import relax_trilinear, relax_trilinear_exact

# ─────────────────────────────────────────────────────────────────────
# Soundness — cv ≤ x*y*z ≤ cc on a wide variety of boxes/points
# ─────────────────────────────────────────────────────────────────────


_SIGN_PATTERN_BOXES: list[tuple[float, float, float, float, float, float]] = [
    # All positive
    (0.5, 2.0, 1.0, 3.0, 0.2, 1.5),
    # All negative
    (-2.0, -0.5, -3.0, -1.0, -1.5, -0.2),
    # x mixed, y/z positive
    (-1.0, 2.0, 0.5, 2.5, 1.0, 3.0),
    # y mixed
    (1.0, 3.0, -1.5, 1.5, 0.5, 2.0),
    # z mixed
    (0.5, 2.0, 1.0, 2.5, -2.0, 1.0),
    # All mixed sign
    (-1.5, 2.0, -2.0, 1.0, -1.0, 1.5),
    # Tight around zero
    (-0.1, 0.1, -0.1, 0.1, -0.1, 0.1),
    # Asymmetric mixed
    (-3.0, 0.5, 0.2, 1.0, -0.5, 2.0),
]


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_soundness_random_points(box):
    """For random points in the box, cv ≤ x*y*z ≤ cc."""
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    key = jax.random.PRNGKey(7)
    n = 256
    kx, ky, kz = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    cv_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_exact(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    cvs, ccs = cv_fn(xs, ys, zs)
    truths = xs * ys * zs

    eps = 1e-9
    assert jnp.all(cvs <= truths + eps), (
        f"cv violated on box {box}: max gap = {float(jnp.max(cvs - truths))}"
    )
    assert jnp.all(ccs >= truths - eps), (
        f"cc violated on box {box}: max gap = {float(jnp.max(truths - ccs))}"
    )


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_corner_tightness(box):
    """At box corners, cv = cc = corner_value (within tolerance)."""
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    for xc in (x_lb, x_ub):
        for yc in (y_lb, y_ub):
            for zc in (z_lb, z_ub):
                cv, cc = relax_trilinear_exact(xc, yc, zc, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
                truth = xc * yc * zc
                np.testing.assert_allclose(float(cv), truth, atol=1e-9)
                np.testing.assert_allclose(float(cc), truth, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Strict tightness vs single-ordering nested bilinear
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_at_least_as_tight_as_nested(box):
    """For all sample points in the box, exact cv >= nested cv and cc <= nested cc."""
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    key = jax.random.PRNGKey(13)
    n = 128
    kx, ky, kz = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    nested_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    exact_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_exact(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    nested_cv, nested_cc = nested_fn(xs, ys, zs)
    exact_cv, exact_cc = exact_fn(xs, ys, zs)

    eps = 1e-9
    assert jnp.all(exact_cv >= nested_cv - eps), "exact cv looser than nested"
    assert jnp.all(exact_cc <= nested_cc + eps), "exact cc looser than nested"


def test_strictly_tighter_on_some_box():
    """On at least one mixed-sign box and point, exact is strictly tighter."""
    x_lb, x_ub = -1.5, 2.0
    y_lb, y_ub = -2.0, 1.0
    z_lb, z_ub = -1.0, 1.5

    key = jax.random.PRNGKey(99)
    kx, ky, kz = jax.random.split(key, 3)
    n = 1024
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    nested_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    exact_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_exact(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    nested_cv, nested_cc = nested_fn(xs, ys, zs)
    exact_cv, exact_cc = exact_fn(xs, ys, zs)

    cv_gap = jnp.max(exact_cv - nested_cv)
    cc_gap = jnp.max(nested_cc - exact_cc)
    # At least one of cv or cc should improve by a measurable margin.
    assert max(float(cv_gap), float(cc_gap)) > 1e-6, (
        "exact relaxation provided no improvement on a mixed-sign box "
        f"(cv_gap={float(cv_gap)}, cc_gap={float(cc_gap)})"
    )


# ─────────────────────────────────────────────────────────────────────
# JAX compatibility — jit, vmap, grad
# ─────────────────────────────────────────────────────────────────────


def test_jit_compiles_and_runs():
    f = jax.jit(relax_trilinear_exact)
    cv, cc = f(0.5, 1.0, 1.5, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
    truth = 0.5 * 1.0 * 1.5
    assert float(cv) <= truth + 1e-9
    assert float(cc) >= truth - 1e-9


def test_vmap_over_points():
    n = 32
    key = jax.random.PRNGKey(0)
    xs = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    ys = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    zs = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    f = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_exact(xv, yv, zv, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    )
    cvs, ccs = f(xs, ys, zs)
    assert cvs.shape == (n,)
    assert ccs.shape == (n,)


def test_grad_through_cv():
    def loss(xv):
        cv, _ = relax_trilinear_exact(xv, 1.0, 1.5, -1.0, 2.0, 0.0, 2.0, 0.0, 3.0)
        return cv

    g = jax.grad(loss)(0.5)
    assert jnp.isfinite(g)


# ─────────────────────────────────────────────────────────────────────
# Compiler-level dispatch — Model x*y*z routes through trilinear-exact
# ─────────────────────────────────────────────────────────────────────


def _build_xyz_relaxation(x_lb, x_ub, y_lb, y_ub, z_lb, z_ub):
    """Build a relaxation callable for f = x*y*z over the given box."""
    from discopt._jax.relaxation_compiler import _compile_relax_node
    from discopt.modeling.core import Model

    m = Model("trilinear_test")
    x = m.continuous("x", lb=x_lb, ub=x_ub)
    y = m.continuous("y", lb=y_lb, ub=y_ub)
    z = m.continuous("z", lb=z_lb, ub=z_ub)
    expr = x * y * z

    fn = _compile_relax_node(expr, m)
    return fn


def test_compiler_dispatch_default_uses_exact(monkeypatch):
    """Without DISCOPT_TRILINEAR=nested, the compiler picks trilinear-exact.

    Tests the LP-relaxation entry point: x_cv = lb, x_cc = ub. Under this
    convention compositional McCormick yields a non-trivial bound interval,
    and the exact path should produce strictly tighter bounds than the
    single-ordering nested path on at least one mixed-sign box.
    """
    monkeypatch.delenv("DISCOPT_TRILINEAR", raising=False)
    box = (-1.5, 2.0, -2.0, 1.0, -1.0, 1.5)
    fn_exact = _build_xyz_relaxation(*box)

    monkeypatch.setenv("DISCOPT_TRILINEAR", "nested")
    fn_nested = _build_xyz_relaxation(*box)
    monkeypatch.delenv("DISCOPT_TRILINEAR", raising=False)

    lb_arr = jnp.array([box[0], box[2], box[4]])
    ub_arr = jnp.array([box[1], box[3], box[5]])

    # LP relaxation: x_cv = lb, x_cc = ub. The compositional convention
    # evaluates cv/cc at the *midpoint* of [x_cv, x_cc] (here, the box
    # midpoint), so soundness means cv ≤ f(midpoint) ≤ cc.
    cv_e, cc_e = fn_exact(lb_arr, ub_arr, lb_arr, ub_arr)
    cv_n, cc_n = fn_nested(lb_arr, ub_arr, lb_arr, ub_arr)

    mid = (lb_arr + ub_arr) * 0.5
    f_mid = float(mid[0] * mid[1] * mid[2])
    eps = 1e-7
    assert float(cv_e) <= f_mid + eps
    assert float(cc_e) >= f_mid - eps
    assert float(cv_n) <= f_mid + eps
    assert float(cc_n) >= f_mid - eps

    # Exact at least as tight as nested.
    assert float(cv_e) >= float(cv_n) - eps
    assert float(cc_e) <= float(cc_n) + eps
    # And strictly tighter on at least one of cv / cc for this box.
    cv_tightening = float(cv_e) - float(cv_n)
    cc_tightening = float(cc_n) - float(cc_e)
    assert max(cv_tightening, cc_tightening) > 1e-6, (
        "Exact and nested produced identical LP bounds — dispatch not engaging."
    )


def test_compiler_dispatch_skips_repeated_variables(monkeypatch):
    """x*x*y should NOT route to trilinear-exact (repeated variable)."""
    from discopt._jax.relaxation_compiler import _try_extract_trilinear_chain
    from discopt.modeling.core import Model

    m = Model("repeated")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    expr = x * x * y
    assert _try_extract_trilinear_chain(expr, m) is None


def test_compiler_dispatch_three_distinct_vars():
    """x*y*z with three distinct vars matches the trilinear detector."""
    from discopt._jax.relaxation_compiler import _try_extract_trilinear_chain
    from discopt.modeling.core import Model

    m = Model("three_distinct")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    expr = x * y * z
    offsets = _try_extract_trilinear_chain(expr, m)
    assert offsets is not None
    assert len(set(offsets)) == 3  # all distinct
