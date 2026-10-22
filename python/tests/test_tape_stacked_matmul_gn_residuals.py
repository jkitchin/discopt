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


# --------------------------------------------------------------------------- #
# Matrix 1-/inf-norms, reductions past NlExpr.max_depth, empty sums
# --------------------------------------------------------------------------- #
# Three more constructs the tape used to refuse, sending the whole model to the
# JAX evaluator (``jax-optional-inventory.md`` §1). Each is checked against numpy
# here, so the test runs without JAX; ``lower_diff`` in the PR compared them with
# the JAX evaluator too (values and gradients, worst relative difference 1e-14).


def _matrix_model():
    m = dm.Model("mat")
    x = m.continuous("X", shape=(4, 5), lb=-2, ub=2)
    return m, x


@pytest.mark.parametrize("ord_", [1, "inf"])
def test_matrix_one_and_inf_norms_match_numpy(ord_):
    m, x = _matrix_model()
    node = compile_to_nl_array(dm.norm(x, ord_), m).reshape(-1)[0]
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(5):
        vals, flat = _flat_point(m, rng)
        a = vals[x]
        expected = np.linalg.norm(a, 1 if ord_ == 1 else np.inf)
        assert node.eval(flat.tolist()) == pytest.approx(expected, rel=1e-14, abs=1e-14)
        checked += 1
    assert checked == 5


def test_matrix_spectral_norm_is_still_refused():
    m, x = _matrix_model()
    with pytest.raises(UnsupportedForTape, match="spectral"):
        compile_to_nl_array(dm.norm(x, 2), m)


@pytest.mark.parametrize("what", ["prod", "norminf"])
def test_reductions_past_max_depth_lower_as_a_balanced_tree(what):
    import pounce

    n = int(pounce.NlExpr.max_depth) + 2001  # past the limit, odd remainder
    m = dm.Model("deep")
    v = m.continuous("v", shape=(n,), lb=0.999, ub=1.001)
    expr = dm.prod(v) if what == "prod" else dm.norm(v, "inf")
    node = compile_to_nl_array(expr, m).reshape(-1)[0]
    x = np.random.default_rng(8).uniform(0.999, 1.001, n)
    expected = float(np.prod(x)) if what == "prod" else float(np.max(np.abs(x)))
    assert node.eval(x.tolist()) == pytest.approx(expected, rel=1e-12)
    grad = np.asarray(node.gradient(x.tolist()), dtype=float)
    if what == "prod":
        np.testing.assert_allclose(grad, expected / x, rtol=1e-11)
    else:
        assert grad[np.argmax(np.abs(x))] == pytest.approx(1.0)
        assert np.count_nonzero(grad) == 1


def test_short_reductions_keep_the_left_fold_bit_for_bit():
    """Below the depth limit the fold is unchanged, so every model that lowered
    before this change evaluates in the same order (bound-neutral by
    construction): compare against an explicit left fold."""
    m = dm.Model("short")
    v = m.continuous("v", shape=(7,), lb=0.5, ub=1.5)
    node = compile_to_nl_array(dm.prod(v), m).reshape(-1)[0]
    x = np.random.default_rng(9).uniform(0.5, 1.5, 7)
    acc = x[0]
    for t in x[1:]:
        acc = acc * t
    assert node.eval(x.tolist()) == acc


def test_empty_sums_lower_to_zero_constants():
    m = dm.Model("empty")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    big = m.continuous("Y", shape=(2, 3), lb=-1, ub=1)
    full = compile_to_nl_array(dm.sum(x[0:0]) + x[1], m)
    flat = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0], dtype=float)
    assert full.reshape(-1)[0].eval(flat.tolist()) == pytest.approx(0.2)
    per_row = compile_to_nl_array(dm.sum(big[:, 0:0], axis=1), m)
    assert per_row.shape == (2,)
    assert [n.eval(flat.tolist()) for n in per_row] == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# Matrix 1-/inf-norms now RELAX too (canonical spelling) and certify
# --------------------------------------------------------------------------- #
def _lp_oracle(c_mat, b, ord_, lb=-2.0, ub=2.0):
    """``min ||X||_ord s.t. <C, X> >= b``, X in [lb, ub]^(r x c), as an LP.

    Variables: X (r*c), U = |X| (r*c), t. Rows: X - U <= 0, -X - U <= 0, each
    column (ord 1) or row (ord inf) sum of U <= t, -<C, X> <= -b. Independent of
    discopt: scipy's HiGHS LP with the textbook epigraph of a max of abs-sums.
    """
    from scipy.optimize import linprog

    r, c = c_mat.shape
    n = r * c
    cost = np.zeros(2 * n + 1)
    cost[-1] = 1.0
    rows, rhs = [], []
    for k in range(n):
        for sgn in (1.0, -1.0):
            row = np.zeros(2 * n + 1)
            row[k], row[n + k] = sgn, -1.0
            rows.append(row)
            rhs.append(0.0)
    lines = (
        [[i * c + j for i in range(r)] for j in range(c)]
        if ord_ == 1
        else [[i * c + j for j in range(c)] for i in range(r)]
    )
    for line in lines:
        row = np.zeros(2 * n + 1)
        row[[n + k for k in line]] = 1.0
        row[-1] = -1.0
        rows.append(row)
        rhs.append(0.0)
    row = np.zeros(2 * n + 1)
    row[:n] = -c_mat.reshape(-1)
    rows.append(row)
    rhs.append(-b)
    bounds = [(lb, ub)] * n + [(0, None)] * n + [(0, None)]
    res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ord_", [1, "inf"])
def test_matrix_norm_models_certify_at_the_lp_optimum(seed, ord_):
    """Before: the matrix norm was opaque to the relaxation (no dual bound) and the
    model went to the convex single NLP, which stalled at the kink -- measured
    ``min norm(X, 1)`` s.t. ``sum(X) >= 3`` on a 2x3 X: objective 3.91 against
    the optimum 1.0, uncertified. Now it is spelled out as max-of-abs-sums and
    routed to the spatial B&B like a vector norm; the bound must never exceed the
    LP oracle's optimum."""
    rng = np.random.default_rng(seed)
    c_mat = rng.uniform(0.2, 1.5, size=(2, 3)) * rng.choice([-1.0, 1.0], size=(2, 3))
    b = 2.0
    oracle = _lp_oracle(c_mat, b, 1 if ord_ == 1 else "inf")
    m = dm.Model("matnorm")
    x = m.continuous("X", shape=(2, 3), lb=-2.0, ub=2.0)
    m.subject_to(dm.sum(c_mat * x) >= b)
    m.minimize(dm.norm(x, ord_))
    r = m.solve(time_limit=60)
    assert r.bound is not None
    assert r.bound <= oracle + 1e-6, (r.bound, oracle)
    assert r.gap_certified, (r.status, r.objective, r.bound, oracle)
    assert r.objective == pytest.approx(oracle, abs=1e-5, rel=1e-5)
