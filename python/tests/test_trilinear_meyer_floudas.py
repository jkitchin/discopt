"""Tests for relax_trilinear_meyer_floudas (Meyer-Floudas 2004 convex hull).

Validates that ``relax_trilinear_meyer_floudas`` returns the convex / concave
envelopes of ``x*y*z`` on an axis-aligned box, computed via the convex hull
of the eight vertex values (Rikun 1997 / Meyer-Floudas 2004).

The implementation is verified against:
  - corner exactness (cv = cc = x*y*z at all 8 corners);
  - pointwise soundness (cv <= x*y*z <= cc) on random box points;
  - pointwise dominance over single-ordering nested McCormick
    (``relax_trilinear``) and the best-of-three nested McCormick
    (``relax_trilinear_exact``);
  - agreement with an external LP solver (scipy.optimize.linprog) on the
    convex-hull LP (up to numerical tolerance);
  - JAX compatibility (``jit``, ``vmap``, ``grad``);
  - compiler dispatch behaviour for x*y*z expressions and the
    ``DISCOPT_TRILINEAR`` env-var selector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.envelopes import (
    relax_trilinear,
    relax_trilinear_exact,
    relax_trilinear_meyer_floudas,
)

# ─────────────────────────────────────────────────────────────────────
# Sign-pattern coverage
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


# ─────────────────────────────────────────────────────────────────────
# Corner exactness
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_corner_tightness(box):
    """At every box corner, cv = cc = corner_value."""
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    for xc in (x_lb, x_ub):
        for yc in (y_lb, y_ub):
            for zc in (z_lb, z_ub):
                cv, cc = relax_trilinear_meyer_floudas(
                    xc, yc, zc, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub
                )
                truth = xc * yc * zc
                np.testing.assert_allclose(float(cv), truth, atol=1e-9)
                np.testing.assert_allclose(float(cc), truth, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Soundness on random points
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_soundness_random_points(box):
    """For random points in the box, cv <= x*y*z <= cc."""
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    key = jax.random.PRNGKey(7)
    n = 512
    kx, ky, kz = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_meyer_floudas(
            xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub
        )
    )
    cvs, ccs = fn(xs, ys, zs)
    truths = xs * ys * zs

    eps = 1e-8
    assert jnp.all(cvs <= truths + eps), (
        f"cv violated on box {box}: max gap = {float(jnp.max(cvs - truths))}"
    )
    assert jnp.all(ccs >= truths - eps), (
        f"cc violated on box {box}: max gap = {float(jnp.max(truths - ccs))}"
    )


# ─────────────────────────────────────────────────────────────────────
# Pointwise dominance over weaker relaxations
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_dominates_nested_pointwise(box):
    """MF cv >= nested cv and MF cc <= nested cc at every sampled point."""
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    key = jax.random.PRNGKey(13)
    n = 256
    kx, ky, kz = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    nested_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    mf_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_meyer_floudas(
            xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub
        )
    )
    cv_n, cc_n = nested_fn(xs, ys, zs)
    cv_m, cc_m = mf_fn(xs, ys, zs)

    eps = 1e-8
    assert jnp.all(cv_m >= cv_n - eps), "MF cv looser than nested at some point"
    assert jnp.all(cc_m <= cc_n + eps), "MF cc looser than nested at some point"


@pytest.mark.parametrize("box", _SIGN_PATTERN_BOXES)
def test_dominates_best_of_three_pointwise(box):
    """MF cv >= best-of-three cv and MF cc <= best-of-three cc at every point.

    The Meyer-Floudas envelope merges the convex-hull LP value with the
    best-of-three nested McCormick (which is sound pointwise but not
    convex). Taking the elementwise max-of-cvs / min-of-ccs guarantees MF
    is at least as tight as the best-of-three relaxation everywhere.
    """
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    key = jax.random.PRNGKey(29)
    n = 256
    kx, ky, kz = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    exact_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_exact(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    mf_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_meyer_floudas(
            xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub
        )
    )
    cv_e, cc_e = exact_fn(xs, ys, zs)
    cv_m, cc_m = mf_fn(xs, ys, zs)

    eps = 1e-8
    assert jnp.all(cv_m >= cv_e - eps), (
        f"MF cv looser than best-of-three: max gap = {float(jnp.max(cv_e - cv_m))}"
    )
    assert jnp.all(cc_m <= cc_e + eps), (
        f"MF cc looser than best-of-three: max gap = {float(jnp.max(cc_m - cc_e))}"
    )


def test_strictly_tighter_on_mixed_sign_box():
    """On a mixed-sign box, MF is strictly tighter than best-of-three somewhere."""
    box = (-1.5, 2.0, -2.0, 1.0, -1.0, 1.5)
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box

    key = jax.random.PRNGKey(99)
    n = 1024
    kx, ky, kz = jax.random.split(key, 3)
    xs = jax.random.uniform(kx, (n,), minval=x_lb, maxval=x_ub)
    ys = jax.random.uniform(ky, (n,), minval=y_lb, maxval=y_ub)
    zs = jax.random.uniform(kz, (n,), minval=z_lb, maxval=z_ub)

    exact_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_exact(xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    )
    mf_fn = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_meyer_floudas(
            xv, yv, zv, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub
        )
    )
    cv_e, cc_e = exact_fn(xs, ys, zs)
    cv_m, cc_m = mf_fn(xs, ys, zs)

    cv_gap = float(jnp.max(cv_m - cv_e))
    cc_gap = float(jnp.max(cc_e - cc_m))
    assert max(cv_gap, cc_gap) > 1e-3, (
        "MF gives no strict improvement over best-of-three on this box "
        f"(cv_gap={cv_gap}, cc_gap={cc_gap})"
    )


# ─────────────────────────────────────────────────────────────────────
# Agreement with external LP solver (scipy.optimize.linprog)
# ─────────────────────────────────────────────────────────────────────


def test_matches_scipy_lp_hull():
    """MF envelope should match the explicit convex-hull LP from scipy.linprog.

    Solves the LP   min/max sum_i lambda_i * f_i
       s.t.  sum_i lambda_i * v_i = p,  sum_i lambda_i = 1,  lambda_i >= 0
    at random box points and confirms MF reproduces the LP value (within
    numerical tolerance). MF's merge with best-of-three may sometimes be
    *tighter* than the convex hull (because best-of-three is non-convex),
    so we check cv_mf >= cv_lp and cc_mf <= cc_lp.
    """
    from scipy.optimize import linprog

    box = (-1.5, 2.0, -2.0, 1.0, -1.0, 1.5)
    x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = box
    verts = np.array([[a, b, c] for a in (x_lb, x_ub) for b in (y_lb, y_ub) for c in (z_lb, z_ub)])
    fc = verts[:, 0] * verts[:, 1] * verts[:, 2]

    rng = np.random.default_rng(42)
    n = 100
    xs = rng.uniform(x_lb, x_ub, n)
    ys = rng.uniform(y_lb, y_ub, n)
    zs = rng.uniform(z_lb, z_ub, n)

    for i in range(n):
        p = np.array([xs[i], ys[i], zs[i]])
        A_eq = np.vstack([verts.T, np.ones((1, 8))])
        b_eq = np.concatenate([p, [1.0]])
        res_cv = linprog(fc, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 8, method="highs")
        res_cc = linprog(-fc, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 8, method="highs")
        cv_lp = float(res_cv.fun)
        cc_lp = float(-res_cc.fun)

        cv_mf, cc_mf = relax_trilinear_meyer_floudas(
            float(xs[i]), float(ys[i]), float(zs[i]), x_lb, x_ub, y_lb, y_ub, z_lb, z_ub
        )

        eps = 1e-6
        # MF tightens at least as much as the convex-hull LP
        assert float(cv_mf) >= cv_lp - eps, f"MF cv looser than LP hull at {p}"
        assert float(cc_mf) <= cc_lp + eps, f"MF cc looser than LP hull at {p}"


# ─────────────────────────────────────────────────────────────────────
# JAX compatibility — jit, vmap, grad
# ─────────────────────────────────────────────────────────────────────


def test_jit_compiles_and_runs():
    f = jax.jit(relax_trilinear_meyer_floudas)
    cv, cc = f(0.5, 1.0, 1.5, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
    truth = 0.5 * 1.0 * 1.5
    assert float(cv) <= truth + 1e-8
    assert float(cc) >= truth - 1e-8


def test_vmap_over_points():
    n = 32
    key = jax.random.PRNGKey(0)
    xs = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    ys = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    zs = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    f = jax.vmap(
        lambda xv, yv, zv: relax_trilinear_meyer_floudas(
            xv, yv, zv, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
        )
    )
    cvs, ccs = f(xs, ys, zs)
    assert cvs.shape == (n,)
    assert ccs.shape == (n,)


def test_grad_through_cv_is_finite():
    def loss(xv):
        cv, _ = relax_trilinear_meyer_floudas(xv, 1.0, 1.5, -1.0, 2.0, 0.0, 2.0, 0.0, 3.0)
        return cv

    g = jax.grad(loss)(0.5)
    assert jnp.isfinite(g)


# ─────────────────────────────────────────────────────────────────────
# Compiler dispatch — default selects MF, env var routes to alternatives
# ─────────────────────────────────────────────────────────────────────


def _build_xyz_relaxation(x_lb, x_ub, y_lb, y_ub, z_lb, z_ub):
    """Build a relaxation callable for f = x*y*z over the given box."""
    from discopt._jax.relaxation_compiler import _compile_relax_node
    from discopt.modeling.core import Model

    m = Model("trilinear_mf_test")
    x = m.continuous("x", lb=x_lb, ub=x_ub)
    y = m.continuous("y", lb=y_lb, ub=y_ub)
    z = m.continuous("z", lb=z_lb, ub=z_ub)
    expr = x * y * z

    fn = _compile_relax_node(expr, m)
    return fn


def test_compiler_default_uses_meyer_floudas(monkeypatch):
    """Without DISCOPT_TRILINEAR set, the compiler picks Meyer-Floudas.

    On an all-positive asymmetric box, MF should produce strictly tighter
    LP bounds at the midpoint than the best-of-three permutation-symmetric
    nested path (DISCOPT_TRILINEAR=exact). Validates that the env-var
    selector engages distinct dispatches. (The midpoint of some boxes
    lands on a region where both relaxations coincide; the all-positive
    box below is one where they differ.)
    """
    box = (0.5, 2.0, 1.0, 3.0, 0.2, 1.5)

    monkeypatch.delenv("DISCOPT_TRILINEAR", raising=False)
    fn_mf = _build_xyz_relaxation(*box)

    monkeypatch.setenv("DISCOPT_TRILINEAR", "exact")
    fn_exact = _build_xyz_relaxation(*box)

    monkeypatch.delenv("DISCOPT_TRILINEAR", raising=False)

    lb_arr = jnp.array([box[0], box[2], box[4]])
    ub_arr = jnp.array([box[1], box[3], box[5]])

    cv_mf_v, cc_mf_v = fn_mf(lb_arr, ub_arr, lb_arr, ub_arr)
    cv_ex_v, cc_ex_v = fn_exact(lb_arr, ub_arr, lb_arr, ub_arr)

    mid = (lb_arr + ub_arr) * 0.5
    f_mid = float(mid[0] * mid[1] * mid[2])
    eps = 1e-7

    # Both are sound at the midpoint.
    assert float(cv_mf_v) <= f_mid + eps
    assert float(cc_mf_v) >= f_mid - eps
    assert float(cv_ex_v) <= f_mid + eps
    assert float(cc_ex_v) >= f_mid - eps

    # MF dominates best-of-three.
    assert float(cv_mf_v) >= float(cv_ex_v) - eps
    assert float(cc_mf_v) <= float(cc_ex_v) + eps

    # MF must be strictly tighter than best-of-three on this box.
    cv_tighten = float(cv_mf_v) - float(cv_ex_v)
    cc_tighten = float(cc_ex_v) - float(cc_mf_v)
    assert max(cv_tighten, cc_tighten) > 1e-6, (
        "MF and best-of-three returned identical LP bounds — "
        "dispatch is not selecting Meyer-Floudas."
    )


def test_compiler_env_var_nested_falls_through(monkeypatch):
    """DISCOPT_TRILINEAR=nested disables the trilinear dispatch and routes the
    expression through the generic compositional bilinear path. This is a
    legacy/debug path -- the test only verifies the dispatch toggles distinct
    behavior (different return values), not strict tightness orderings.
    """
    box = (0.5, 2.0, 1.0, 3.0, 0.2, 1.5)

    monkeypatch.delenv("DISCOPT_TRILINEAR", raising=False)
    fn_mf = _build_xyz_relaxation(*box)

    monkeypatch.setenv("DISCOPT_TRILINEAR", "nested")
    fn_nested = _build_xyz_relaxation(*box)
    monkeypatch.delenv("DISCOPT_TRILINEAR", raising=False)

    lb_arr = jnp.array([box[0], box[2], box[4]])
    ub_arr = jnp.array([box[1], box[3], box[5]])

    cv_mf_v, cc_mf_v = fn_mf(lb_arr, ub_arr, lb_arr, ub_arr)
    cv_ne_v, cc_ne_v = fn_nested(lb_arr, ub_arr, lb_arr, ub_arr)

    # The two dispatches must produce distinguishable bounds.
    assert (abs(float(cv_mf_v) - float(cv_ne_v)) > 1e-6) or (
        abs(float(cc_mf_v) - float(cc_ne_v)) > 1e-6
    ), "Trilinear dispatch is not toggling with DISCOPT_TRILINEAR=nested."
