"""The spectral norm ``dm.norm(X, 2)`` of a matrix: evaluated without JAX, and
relaxed so it certifies (#1903).

``sigma_1`` has no finite formula in the entries, so the POUNCE tape cannot
express it, and before #1903 every model containing it went to the JAX evaluator
and had no dual bound (the relaxation saw only ``norm >= 0``). Now:

* **Evaluation.** It is a numpy-evaluated *hole* in the tape
  (``discopt._tape_holes``): value from the SVD, gradient ``u_1 v_1^T``, Hessian by
  second-order perturbation of the Jordan-Wielandt matrix. The evaluator folds it
  back into x-space by the chain rule.
* **Relaxation.** ``sigma_1(X) = max over unit u, v of u^T X v``, so
  ``w >= u^T X v`` is a valid linear row for *any* unit pair; the builder emits
  the box centre's singular pairs and ``w >= +-x_ij``
  (``uniform_relax._build_spectral_norm``). The aux column's interval
  (``interval_eval._matrix_norm_interval``) bounds it above by
  ``min(sigma_1(max|X|), sigma_1(C) + sigma_1(R))`` and below by the centre cut's
  minimum over the box, which is exact on a point box.
* **Routing.** ``sigma_1`` is non-differentiable where its top singular values
  coincide, which is where minimizers sit (``min ||X||_2`` s.t. ``trace X >= 2``
  is solved by ``X = I``), so it routes to the spatial B&B like the 1-/inf-norms.

Oracles are independent of discopt: box maximisation by vertex enumeration (a
convex function's maximum over a box is at a vertex), minimisation by exact duality
(``<E, X> <= ||E||_* sigma_1(X)``, the nuclear norm being the spectral norm's dual)
plus a feasible point that attains the bound.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._tape_holes import spectral_norm, spectral_norm_grad, spectral_norm_hess

pytestmark = [pytest.mark.smoke]


# --------------------------------------------------------------------------- #
# the numpy derivatives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (4, 3), (1, 5)])
def test_numpy_gradient_and_hessian_match_finite_differences(shape):
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(5):
        x = rng.standard_normal(shape)
        g = spectral_norm_grad(x)
        h = spectral_norm_hess(x).reshape(x.size, x.size)
        step = 1e-6
        for k in range(x.size):
            e = np.zeros(x.size)
            e[k] = step
            e = e.reshape(shape)
            fd = (spectral_norm(x + e) - spectral_norm(x - e)) / (2 * step)
            assert fd == pytest.approx(g.reshape(-1)[k], abs=1e-7)
            fdh = (spectral_norm_grad(x + e) - spectral_norm_grad(x - e)).reshape(-1) / (2 * step)
            np.testing.assert_allclose(fdh, h[k], atol=2e-5)
            checked += 1
        np.testing.assert_allclose(h, h.T, atol=1e-12)
    assert checked > 0


# --------------------------------------------------------------------------- #
# the evaluator
# --------------------------------------------------------------------------- #
def _nonlinear_arg_model():
    m = dm.Model("spec_eval")
    x = m.continuous("X", shape=(3, 2), lb=-2, ub=2)
    m.minimize(dm.sum(x**2) - x[0, 0] + 0.5 * dm.norm(x, 2))
    m.subject_to(dm.norm(x**2 + 0.5 * x, 2) <= 3.0)
    m.subject_to(dm.norm(dm.exp(0.3 * x), 2) ** 2 + x[1, 0] <= 9.0)
    return m


@pytest.mark.requires_jax
@pytest.mark.parametrize("dense", [True, False])
def test_tape_with_holes_matches_the_jax_evaluator(monkeypatch, dense):
    """Value, gradient, constraints, Jacobian and Lagrangian Hessian, on both the
    dense and the sparse chain-rule paths, against the JAX evaluator. The model has
    nonlinear hole arguments, so the argument-curvature term is exercised."""
    import discopt._tape_nlp_evaluator as T
    from discopt._relax.nlp_evaluator import NLPEvaluator

    monkeypatch.setattr(T, "_HOLE_DENSE_MAX", 10**9 if dense else -1)
    m = _nonlinear_arg_model()
    tape = T.TapeNLPEvaluator(m)
    jx = NLPEvaluator(m)
    assert len(tape._holes) == 3 and tape._hole_dense is dense
    rng = np.random.default_rng(1)
    lo, hi = tape.variable_bounds
    checked = 0
    for _ in range(5):
        x = rng.uniform(lo, hi)
        lam = rng.standard_normal(tape.n_constraints)
        for f in (
            lambda e: e.evaluate_objective(x),
            lambda e: e.evaluate_gradient(x),
            lambda e: e.evaluate_constraints(x),
            lambda e: e.evaluate_jacobian(x),
            lambda e: e.evaluate_lagrangian_hessian(x, 1.3, lam),
        ):
            np.testing.assert_allclose(f(tape), f(jx), rtol=1e-12, atol=1e-12)
            checked += 1
    assert checked == 25


def test_declared_structures_cover_every_nonzero():
    """A value outside the declared sparsity is dropped silently by the solver, so
    check the structure against the dense values the evaluator itself returns."""
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    m = _nonlinear_arg_model()
    ev = TapeNLPEvaluator(m)
    rng = np.random.default_rng(2)
    lo, hi = ev.variable_bounds
    rows, cols = ev.hessian_structure()
    jr, jc = ev.jacobian_structure()
    assert np.all(rows >= cols)
    checked = 0
    for _ in range(5):
        x = rng.uniform(lo, hi)
        lam = rng.standard_normal(ev.n_constraints)
        h = ev.evaluate_lagrangian_hessian(x, 1.0, lam)
        mask = np.zeros_like(h, bool)
        mask[rows, cols] = True
        mask |= mask.T
        assert np.abs(h[~mask]).max(initial=0.0) == 0.0
        assert np.abs(h).max() > 0.0
        j = ev.evaluate_jacobian(x)
        jm = np.zeros_like(j, bool)
        jm[jr, jc] = True
        assert np.abs(j[~jm]).max(initial=0.0) == 0.0
        checked += 1
    assert checked == 5


def test_point_evaluator_handles_the_hole():
    """Evaluated through the numpy hole, not the JAX fallback."""
    from discopt._point_eval import _tape_or_none, evaluate_at

    m = dm.Model("p")
    x = m.continuous("X", shape=(2, 3), lb=-2, ub=2)
    y = m.continuous("y", lb=0, ub=1)
    expr = dm.norm(x**2, 2) + y * dm.norm(x, 2)
    pt = np.random.default_rng(4).uniform(-1, 1, 7)
    a = pt[:6].reshape(2, 3)
    ref = np.linalg.norm(a**2, 2) + pt[6] * np.linalg.norm(a, 2)
    lowered = _tape_or_none(expr, m)
    assert lowered is not None and len(lowered[1]) == 2, "the hole path did not fire"
    assert float(evaluate_at(expr, m, pt)) == pytest.approx(ref, rel=1e-14)


# --------------------------------------------------------------------------- #
# the relaxation
# --------------------------------------------------------------------------- #
def test_no_row_and_no_aux_bound_cuts_the_graph():
    """Every lifted point ``(X, sigma_1(X))`` with ``X`` in the box satisfies every
    emitted row and the aux column's bounds."""
    from discopt._relax.uniform_relax import build_uniform_relaxation

    rng = np.random.default_rng(5)
    checked = rows_seen = 0
    for trial in range(60):
        shape = tuple(int(v) for v in rng.integers(1, 5, size=2))
        lo = rng.uniform(-2.0, 1.0, shape)
        hi = lo + rng.uniform(0.0, 2.0, shape)
        m = dm.Model("rel")
        x = m.continuous("X", shape=shape, lb=lo, ub=hi)
        m.minimize(dm.norm(x, 2) if trial % 2 else -dm.norm(x, 2))
        rel = build_uniform_relaxation(m)
        assert [k for k, _ in rel.coverage.values()] == ["multivar"]
        a = rel.model._A_ub
        a = np.zeros((0, rel.n_orig + rel.n_aux)) if a is None else np.asarray(a.todense())
        b = np.zeros(0) if rel.model._A_ub is None else np.asarray(rel.model._b_ub)
        rows_seen += a.shape[0]
        w_lo, w_hi = rel.model._bounds[rel.n_orig]
        for _ in range(100):
            pt = rng.uniform(lo, hi)
            sig = np.linalg.norm(pt, 2)
            assert w_lo - 1e-12 <= sig <= w_hi + 1e-12, (shape, sig, w_lo, w_hi)
            lifted = np.concatenate([pt.reshape(-1), [sig]])
            assert np.max(a @ lifted - b, initial=-np.inf) <= 1e-9 * max(1.0, sig)
            checked += 1
    assert rows_seen > 0 and checked == 6000


# --------------------------------------------------------------------------- #
# certification against independent oracles
# --------------------------------------------------------------------------- #
def _vertex_max(lo, hi):
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=lo.size):
        best = max(best, np.linalg.norm(np.where(np.reshape(bits, lo.shape), hi, lo), 2))
    return best


@pytest.mark.parametrize("seed", [0, 1])
def test_box_maximisation_certifies_at_the_best_vertex(seed):
    """The nonconvex use: maximise sigma_1 over a box (optimum at a vertex)."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-2, 0, (2, 3))
    hi = lo + rng.uniform(0.5, 2.5, (2, 3))
    oracle = _vertex_max(lo, hi)
    m = dm.Model("maxspec")
    x = m.continuous("X", shape=(2, 3), lb=lo, ub=hi)
    m.maximize(dm.norm(x, 2))
    r = m.solve(time_limit=60)
    assert r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound >= oracle - 1e-6, (r.bound, oracle)  # maximise: bound above
    assert r.objective == pytest.approx(oracle, abs=1e-5)


def _duality_lower_bound(e: np.ndarray, rhs: float) -> float:
    """Exact lower bound on ``sigma_1(X)`` given ``<E, X> >= rhs`` (``rhs > 0``).

    The nuclear norm is the spectral norm's dual, so ``<E, X> <= ||E||_* sigma_1(X)``
    for every ``X``; hence ``sigma_1(X) >= rhs / ||E||_*``. A feasible point attaining
    it proves optimality -- a rigorous oracle with no iteration. (A Kelley
    cutting-plane oracle was tried first and stalled for >3000 rounds at exactly
    these optima, which have tied singular values.)
    """
    return rhs / float(np.linalg.svd(e, compute_uv=False).sum())


def test_degenerate_minimisation_certifies():
    """``min sigma_1(X)`` s.t. ``trace X >= 2``: the optimum ``X = I`` has two equal
    singular values -- the non-differentiable case the routing change is for.
    Oracle: ``trace X = <I, X>`` and ``||I||_* = 2`` give ``sigma_1 >= 1``, attained
    by ``X = I``."""
    oracle = _duality_lower_bound(np.eye(2), 2.0)
    assert oracle == pytest.approx(1.0) and np.linalg.norm(np.eye(2), 2) == pytest.approx(1.0)
    m = dm.Model("trace")
    x = m.continuous("X", shape=(2, 2), lb=-2, ub=2)
    m.subject_to(x[0, 0] + x[1, 1] >= 2.0)
    m.minimize(dm.norm(x, 2))
    r = m.solve(time_limit=120)
    assert r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= oracle + 1e-6
    assert r.objective == pytest.approx(oracle, abs=1e-5)


def test_mixed_integer_model_certifies_against_enumeration():
    """Binaries switch entries on; the optimum ``diag(1, 1)`` is again degenerate.

    Oracle: for each binary pattern ``S``, ``sum(X) = <E_S, X> >= 2`` gives
    ``sigma_1 >= 2 / ||E_S||_*``, so the pattern costs at least that plus
    ``0.3 |S|``. The smallest such bound over all 15 nonempty patterns is 1.6, and
    ``X = I`` with the diagonal pattern attains it, so 1.6 is the optimum.
    """
    best = np.inf
    for bits in itertools.product((0, 1), repeat=4):
        on = np.array(bits, float).reshape(2, 2)
        if not on.any():
            continue  # sum(X) >= 2 is infeasible with every entry fixed at 0
        best = min(best, _duality_lower_bound(on, 2.0) + 0.3 * on.sum())
    attained = np.linalg.norm(np.eye(2), 2) + 0.3 * 2
    assert best == pytest.approx(attained) == pytest.approx(1.6)
    m = dm.Model("mixed")
    x = m.continuous("X", shape=(2, 2), lb=-2, ub=2)
    y = m.binary("y", shape=(2, 2))
    m.subject_to(dm.sum(x) >= 2.0)
    m.subject_to(x <= 2 * y)
    m.subject_to(x >= -2 * y)
    m.minimize(dm.norm(x, 2) + 0.3 * dm.sum(y))
    r = m.solve(time_limit=120)
    assert r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= best + 1e-6
    assert r.objective == pytest.approx(best, abs=1e-5)
