"""Two tape lowerings that kept ``discopt.ml.train()`` on JAX (#1873 follow-up).

``docs/dev/jax-optional-inventory.md`` §1 listed both as the reason trainable
surrogates were ``[jax]``-only:

1. **Stacked matmul.** ``TrainableNetwork`` writes each layer as
   ``(N, 1, d_in) @ (d_in, d_out)``. ``_nl_expr_compiler._matmul`` refused any
   operand with more than two axes, so the whole model fell back to the JAX
   evaluator. It now follows ``np.matmul``: batch axes broadcast, a 1-D operand is
   promoted and its axis dropped.
2. **Array-valued Gauss-Newton residuals.** ``train()`` sets ``gauss_newton=True``,
   and a collocation fit's residual (``prediction - data``) is an array.
   ``TapeNLPEvaluator._build_residual_exprs`` used the scalar-only
   ``compile_to_nl_expr`` and refused it. It now contributes one residual row per
   element, in the JAX arm's ``reshape(-1)``-and-concatenate order.

Neither changes what is computed, only which engine computes it. Measured on the
four ``test_nn_trainable`` / ``test_dae_fit`` training models, tape vs JAX: 120
comparisons of objective, gradient, constraints, Jacobian and (Gauss-Newton)
Lagrangian Hessian at random points, worst relative difference 1.0e-15; and
``train()`` on each backend, 3 interleaved repetitions, gives IDENTICAL iteration
counts and objectives (to every printed digit) on all four.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_array
from discopt._tape_nlp_evaluator import TapeNLPEvaluator

pytestmark = [pytest.mark.smoke]


def _flat_point(model, rng):
    vals = {v: rng.uniform(-1.0, 1.0, size=v.shape) for v in model._variables}
    x = np.concatenate([np.asarray(vals[v], dtype=float).reshape(-1) for v in model._variables])
    return vals, x


def _eval(arr, x):
    return np.vectorize(lambda node: float(node.eval(x.tolist())), otypes=[float])(arr)


# (left shape, right shape): batch @ matrix, matrix @ batch, 1-D promotion on each
# side, a broadcast batch axis on both sides, and the TrainableNetwork layer shape.
SHAPES = [
    ((4, 2, 3), (3, 2)),
    ((3, 2), (4, 2, 5)),
    ((3,), (5, 3, 2)),
    ((4, 2, 3), (3,)),
    ((4, 1, 2, 3), (5, 3, 2)),
    ((2, 1, 3), (2, 3, 4)),
    ((20, 1, 6), (6, 1)),
]


@pytest.mark.parametrize("ls,rs", SHAPES)
def test_stacked_matmul_matches_numpy(ls, rs):
    m = dm.Model("mm")
    a = m.continuous("a", shape=ls, lb=-1, ub=1)
    b = m.continuous("b", shape=rs, lb=-1, ub=1)
    expr = a @ b
    arr = compile_to_nl_array(expr, m)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(3):
        vals, x = _flat_point(m, rng)
        ref = np.matmul(vals[a], vals[b])
        assert arr.shape == ref.shape, (arr.shape, ref.shape)
        np.testing.assert_allclose(_eval(arr, x), ref, rtol=1e-14, atol=1e-14)
        checked += ref.size
    assert checked > 0


@pytest.mark.parametrize(
    "ls,rs",
    [((4, 2, 3), (5, 3, 2)), ((4, 2, 3), (2, 2)), ((3, 2, 3), (4,))],
)
def test_stacked_matmul_refuses_what_numpy_refuses(ls, rs):
    with pytest.raises(ValueError):
        np.matmul(np.zeros(ls), np.zeros(rs))
    m = dm.Model("mm")
    a = m.continuous("a", shape=ls, lb=-1, ub=1)
    b = m.continuous("b", shape=rs, lb=-1, ub=1)
    with pytest.raises(UnsupportedForTape):
        compile_to_nl_array(a @ b, m)


def _array_residual_model():
    """``min sum((A x - y)**2) + (x0 - 1)**2``: one array residual and one scalar.

    The Gauss-Newton Hessian is exactly ``2 (AᵀA + e0 e0ᵀ)``, so the tape result
    can be checked in closed form, with no second engine.
    """
    rng = np.random.default_rng(3)
    a_mat = rng.standard_normal((6, 3))
    y = rng.standard_normal(6)
    m = dm.Model("gn")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    m.minimize(dm.sum((a_mat @ x - y) ** 2) + (x[0] - 1.0) ** 2)
    m._gauss_newton_hessian = True
    return m, a_mat


def test_array_residual_gauss_newton_builds_on_the_tape():
    m, a_mat = _array_residual_model()
    ev = TapeNLPEvaluator(m)
    assert ev.is_gauss_newton
    assert len(ev._residual_exprs) == a_mat.shape[0] + 1
    e0 = np.zeros((1, 3))
    e0[0, 0] = 1.0
    expected = 2.0 * (a_mat.T @ a_mat + e0.T @ e0)
    checked = 0
    for x in np.random.default_rng(4).uniform(-2, 2, size=(3, 3)):
        h = ev.evaluate_lagrangian_hessian(x, 1.0, np.zeros(ev.n_constraints))
        assert np.abs(h).max() > 0.0
        np.testing.assert_allclose(h, expected, rtol=1e-13, atol=1e-13)
        checked += 1
    assert checked == 3


@pytest.mark.requires_jax
def test_array_residual_gauss_newton_matches_the_jax_arm():
    """The row ORDER matters only through ``JᵀJ``, which is order-invariant, so
    compare the residual Jacobian rows themselves against JAX's
    ``residual_vector`` order on a model with two array residuals."""
    from discopt._relax.nlp_evaluator import NLPEvaluator

    rng = np.random.default_rng(5)
    a1, a2 = rng.standard_normal((4, 3)), rng.standard_normal((2, 2, 3))
    m = dm.Model("gn2")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    m.minimize(dm.sum((a1 @ x - 1.0) ** 2) + dm.sum((a2 @ x) ** 2))
    m._gauss_newton_hessian = True
    tape = TapeNLPEvaluator(m)
    jx = NLPEvaluator(m, gauss_newton=True)
    assert tape.is_gauss_newton and jx.is_gauss_newton
    assert len(tape._residual_exprs) == 4 + 4
    for pt in rng.uniform(-2, 2, size=(3, 3)):
        lam = np.zeros(tape.n_constraints)
        np.testing.assert_allclose(
            tape.evaluate_lagrangian_hessian(pt, 1.0, lam),
            jx.evaluate_lagrangian_hessian(pt, 1.0, lam),
            rtol=1e-13,
            atol=1e-13,
        )
        rows = np.array([[float(g) for g in r.gradient(pt.tolist())] for r in tape._residual_exprs])
        ref = np.vstack([a1, a2.reshape(-1, 3)])
        np.testing.assert_allclose(rows, ref, rtol=1e-13, atol=1e-13)
