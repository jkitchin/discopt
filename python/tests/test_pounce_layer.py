"""POUNCE-backed differentiable JAX layers (roadmap: retire-JAX-IPM P0).

Validates the building blocks that let POUNCE live in a JAX autodiff graph:
- ``solve_lp_kkt`` / ``solve_qp_kkt`` return a valid interior-point KKT point in
  the differentiable-layer sign convention (``c - Aᵀy - z_l + z_u = 0``).
- ``pounce_layer.make_nlp_layer`` is a first-order-differentiable JAX layer whose
  gradients match the sIPOPT sensitivity (``pounce_sensitivity``) and a black-box
  finite difference, and which composes under ``jit``/``vmap``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from discopt.solvers.lp_pounce import solve_lp_kkt  # noqa: E402
from discopt.solvers.qp_pounce import solve_qp_kkt  # noqa: E402


class TestKKTPoint:
    @pytest.mark.parametrize("seed", range(5))
    def test_lp_kkt_residual(self, seed):
        rng = np.random.default_rng(seed)
        n, m = 6, 2
        A = rng.standard_normal((m, n))
        b = A @ rng.uniform(0.2, 0.8, n)  # interior-feasible RHS
        c = rng.standard_normal(n)
        x_l, x_u = np.zeros(n), np.ones(n)
        _obj, x, y, z_l, z_u = solve_lp_kkt(c, A, b, x_l, x_u)
        assert np.abs(c - A.T @ y - z_l + z_u).max() < 1e-6  # stationarity
        assert np.abs(A @ x - b).max() < 1e-6  # primal feasibility
        assert z_l.min() > -1e-6 and z_u.min() > -1e-6  # dual feasibility

    @pytest.mark.parametrize("seed", range(5))
    def test_qp_kkt_residual(self, seed):
        rng = np.random.default_rng(100 + seed)
        n, m = 5, 2
        M = rng.standard_normal((n, n))
        Q = M @ M.T + 0.5 * np.eye(n)  # SPD
        A = rng.standard_normal((m, n))
        b = A @ rng.uniform(0.2, 0.8, n)
        c = rng.standard_normal(n)
        x_l, x_u = np.zeros(n), np.ones(n)
        _obj, x, y, z_l, z_u = solve_qp_kkt(Q, c, A, b, x_l, x_u)
        assert np.abs(Q @ x + c - A.T @ y - z_l + z_u).max() < 1e-6
        assert np.abs(A @ x - b).max() < 1e-6


def _build(pval):
    import discopt.modeling as dm

    m = dm.Model("t")
    p = m.parameter("p", value=pval)
    x0 = m.continuous("x0", lb=0.0, ub=2.0)
    x1 = m.continuous("x1", lb=0.0, ub=2.0)
    m.minimize((x0 - p) ** 2 + x1**2)
    m.subject_to(x0 + x1 == 1.0)
    return m, p


class TestNLPLayer:
    def test_forward(self):
        from discopt._jax.pounce_layer import make_nlp_layer

        m, p = _build(0.3)
        layer = make_nlp_layer(m, [p])
        obj, x, _lam = layer(jnp.array([0.3]))
        # min (x0-0.3)^2 + x1^2 s.t. x0+x1=1 -> x0=0.65, x1=0.35, obj=0.245.
        assert abs(float(obj) - 0.245) < 1e-6
        assert np.allclose(np.asarray(x), [0.65, 0.35], atol=1e-5)

    def test_dx_dp_matches_sipopt(self):
        from discopt._jax.pounce_layer import make_nlp_layer
        from discopt.solvers.sipopt import pounce_sensitivity

        m, p = _build(0.3)
        layer = make_nlp_layer(m, [p])
        dxdp = np.asarray(jax.jacobian(lambda pp: layer(pp)[1])(jnp.array([0.3])))
        sens = pounce_sensitivity(m, [p])
        assert np.allclose(dxdp.ravel(), sens.dx_dp.ravel(), atol=1e-4, rtol=1e-3)

    def test_dobj_dp_matches_fd(self):
        from discopt._jax.pounce_layer import make_nlp_layer

        m, p = _build(0.3)
        layer = make_nlp_layer(m, [p])
        dobj = float(jax.grad(lambda pp: layer(pp)[0])(jnp.array([0.3]))[0])

        eps = 1e-5

        def solved_obj(pv):
            mm, pp = _build(pv)
            return float(make_nlp_layer(mm, [pp])(jnp.array([pv]))[0])

        fd = (solved_obj(0.3 + eps) - solved_obj(0.3 - eps)) / (2 * eps)
        assert abs(dobj - fd) < 1e-4

    def test_jit_and_vmap(self):
        from discopt._jax.pounce_layer import make_nlp_layer

        m, p = _build(0.3)
        layer = make_nlp_layer(m, [p])
        assert abs(float(jax.jit(layer)(jnp.array([0.3]))[0]) - 0.245) < 1e-6
        batched = jax.vmap(lambda pp: layer(pp)[0])(jnp.array([[0.2], [0.4], [0.6]]))
        assert np.asarray(batched).shape == (3,)
