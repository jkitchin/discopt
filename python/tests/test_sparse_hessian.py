"""
Tests for compressed Lagrangian-Hessian evaluation (issue #95).

The dense ``jax.hessian`` of the Lagrangian is ``O(n^2)`` to assemble and
compile, which hangs large DAE/collocation solves. ``sparse_hessian`` recovers
only the structural nonzeros via colored Hessian-vector products, with
dense-row separation so arrowhead (parameter-estimation) Hessians collapse to a
small constant number of seeds.

These tests assert the recovered values match the dense Lagrangian Hessian at
the reported COO positions, that the seed count stays small for arrowhead
structure, and that the evaluator routes DAE models through the sparse path
while leaving small/dense and Gauss-Newton models on the dense path.
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.sparse_hessian import build_hessian_coloring

pytestmark = pytest.mark.unit


def _dae_estimation_model(nfe: int = 25, n_states: int = 1):
    """Radau-collocation parameter estimation with an arrowhead Hessian.

    The estimated rate constant ``k`` couples nonlinearly to every state
    collocation variable, producing one dense Hessian row/column.
    """
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("dae_est")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=nfe, ncp=3, scheme="radau")
    dae = DAEBuilder(m, cs)
    for s in range(n_states):
        dae.add_state(f"a{s}", bounds=(0, 2), initial=1.0)
    dae.add_control("u", bounds=(0, 1))
    k = m.continuous("k", lb=0.01, ub=5.0)
    dae.set_ode(lambda t, x, z, u: {f"a{s}": -k * x[f"a{s}"] + u["u"] for s in range(n_states)})
    v = dae.discretize()
    nodes = [0.155, 0.645, 1.0]
    m.minimize(
        sum(
            (v["a0"][i, j] - np.exp(-1.3 * ((i + nodes[j - 1]) / nfe))) ** 2
            for i in range(nfe)
            for j in range(1, 4)
        )
    )
    return m, k


def _general_nonlinear_model(n: int = 80):
    """A sparse but non-arrowhead nonlinear model (chained bilinear coupling)."""
    m = dm.Model("chain")
    x = m.continuous("x", shape=(n,), lb=-2, ub=2)
    m.minimize(sum(x[i] ** 2 for i in range(n)))
    for i in range(n - 1):
        m.subject_to(x[i] * x[i + 1] <= 1.0)
    return m


def _dense_values_at_coo(ev, x, obj_factor, lam):
    ev._ensure_coo_cache()
    h = np.asarray(ev.evaluate_lagrangian_hessian(x, obj_factor, lam))
    return h[ev._hess_rows, ev._hess_cols]


class TestSparseHessianValues:
    """Recovered values must equal the dense Lagrangian Hessian at COO positions."""

    @pytest.mark.parametrize("n_states", [1, 2])
    def test_dae_arrowhead_matches_dense(self, n_states):
        sys.setrecursionlimit(100000)
        m, _k = _dae_estimation_model(nfe=25, n_states=n_states)
        ev = NLPEvaluator(m)
        assert ev._use_sparse_hessian()
        n, mc = ev.n_variables, ev.n_constraints
        rng = np.random.default_rng(7)
        for _ in range(5):
            x = rng.uniform(-1, 1, n)
            lam = rng.uniform(-1, 1, mc)
            obj_factor = float(rng.uniform(0.1, 2.0))
            sparse_vals = ev.evaluate_hessian_values(x, obj_factor, lam)
            dense_vals = _dense_values_at_coo(ev, x, obj_factor, lam)
            np.testing.assert_allclose(sparse_vals, dense_vals, atol=1e-9, rtol=1e-9)

    def test_general_nonlinear_matches_dense(self):
        m = _general_nonlinear_model(n=80)
        ev = NLPEvaluator(m)
        assert ev._use_sparse_hessian()
        n, mc = ev.n_variables, ev.n_constraints
        rng = np.random.default_rng(11)
        for _ in range(5):
            x = rng.uniform(-1, 1, n)
            lam = rng.uniform(-1, 1, mc)
            obj_factor = float(rng.uniform(0.1, 2.0))
            sparse_vals = ev.evaluate_hessian_values(x, obj_factor, lam)
            dense_vals = _dense_values_at_coo(ev, x, obj_factor, lam)
            np.testing.assert_allclose(sparse_vals, dense_vals, atol=1e-9, rtol=1e-9)


class TestSeedCount:
    """Dense-row separation keeps the seed count small for arrowhead Hessians."""

    def test_arrowhead_seed_count_is_small_and_grows_slowly(self):
        sys.setrecursionlimit(100000)
        seed_counts = {}
        for nfe in (25, 75):
            m, _k = _dae_estimation_model(nfe=nfe, n_states=1)
            ev = NLPEvaluator(m)
            ev._ensure_coo_cache()
            seed, _si, _lr = build_hessian_coloring(
                ev.sparsity_pattern, ev._hess_rows, ev._hess_cols
            )
            seed_counts[nfe] = seed.shape[1]
        # A handful of seeds regardless of element count; n_vars roughly triples
        # from nfe=25 to nfe=75 but the seed count must not.
        assert seed_counts[25] <= 8
        assert seed_counts[75] <= 8


class TestRouting:
    """The evaluator enables the sparse path only when it pays off."""

    def test_small_dense_model_uses_dense_path(self):
        m = dm.Model("small")
        x = m.continuous("x", shape=(3,), lb=-10, ub=10)
        m.minimize(x[0] ** 2 + x[1] * x[2])
        m.subject_to(x[0] + x[1] + x[2] <= 10)
        ev = NLPEvaluator(m)
        assert not ev._use_sparse_hessian()
        # Dense path still returns correct values.
        rng = np.random.default_rng(3)
        x0 = rng.uniform(-1, 1, ev.n_variables)
        lam = rng.uniform(-1, 1, ev.n_constraints)
        sparse_vals = ev.evaluate_hessian_values(x0, 1.0, lam)
        dense_vals = _dense_values_at_coo(ev, x0, 1.0, lam)
        np.testing.assert_allclose(sparse_vals, dense_vals, atol=1e-9)

    def test_gauss_newton_disables_hvp_path(self):
        n = 60
        m = dm.Model("gn")
        x = m.continuous("x", shape=(n,), lb=-2, ub=2)
        A = np.eye(n)
        b = 0.3 * np.ones(n)
        m.minimize(dm.sum((A @ x - b) ** 2))
        for i in range(n - 1):
            m.subject_to(x[i] * x[i + 1] <= 1.0)
        ev = NLPEvaluator(m, gauss_newton=True)
        assert ev.is_gauss_newton
        # The compressed-HVP path is wired for the exact Lagrangian only.
        assert ev._lagrangian_hvp_fn_jit is None
        assert not ev._use_sparse_hessian()


class TestEndToEndSolve:
    """A discretized collocation NLP solves (does not hang) via the sparse path."""

    def test_dae_collocation_solve_does_not_hang(self):
        sys.setrecursionlimit(100000)
        # nfe=25 keeps this a fast plumbing smoke test: it still routes through
        # the compressed-HVP Hessian path (asserted below), but solves in a few
        # seconds even on the ipm fallback used in CI (where pounce is absent).
        # The O(n²)→O(seeds) scaling that actually defeats the dense-assembly
        # hang is proven separately, without a solve, by TestSeedCount.
        m, _k = _dae_estimation_model(nfe=25, n_states=1)
        assert NLPEvaluator(m)._use_sparse_hessian()
        try:
            import pounce  # noqa: F401

            backend = "pounce"
        except ImportError:
            backend = "ipm"
        result = m.solve(nlp_solver=backend, max_nodes=1)
        assert result.status in {"optimal", "feasible"}
        # The least-squares objective should be driven near zero.
        assert result.objective < 1e-4
