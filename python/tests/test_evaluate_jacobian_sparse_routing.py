"""Regression: ``NLPEvaluator.evaluate_jacobian`` must route large models through
the sparse coloring path instead of compiling the dense ``jax.jacfwd``.

The dense compile replicates the constraint program once per input variable, so
for a large model its XLA jaxpr explodes during MLIR lowering and aborts the
process with a native SIGBUS/SIGILL — uncatchable. Above
``_DENSE_JACOBIAN_COMPILE_LIMIT`` (m * n) the method computes the (sparse)
Jacobian via O(chromatic-number) JVPs and densifies it, preserving the dense
(m, n) return so every caller stays safe. Below the cap the faster dense path is
unchanged.

These tests pin the routing deterministically (without compiling a
multi-million-entry Jacobian, which would crash the whole pytest process):
above the cap the raw dense path must not be touched; below it, it must be.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.nlp_evaluator as NE  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402


def _small_model() -> dm.Model:
    m = dm.Model("nl")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x + y)
    m.subject_to(x * y <= 0.5)  # nonlinear -> jac_fn_jit is built
    m.subject_to(x + 2 * y <= 1.0)
    return m


def test_large_model_routes_to_sparse_and_skips_dense(monkeypatch):
    ev = NE.NLPEvaluator(_small_model())
    n = ev._n_variables
    x = np.zeros(n)

    # Force the "large model" branch for this tiny model.
    monkeypatch.setattr(NE, "_DENSE_JACOBIAN_COMPILE_LIMIT", 0)

    # Stand in a known sparse Jacobian and mark the sparse fn as available so
    # _ensure_sparse_jac_fn() short-circuits to "available".
    m_rows = ev._n_constraints
    known = sp.csc_matrix(np.arange(m_rows * n, dtype=np.float64).reshape(m_rows, n))
    ev._sparse_jac_fn = lambda _x: known  # hasattr -> True, not None -> available

    # The raw dense path must NOT be exercised above the cap.
    def _boom(_x):
        raise AssertionError("raw dense Jacobian compiled above the cap")

    monkeypatch.setattr(ev, "_evaluate_dense_jacobian", _boom)

    J = ev.evaluate_jacobian(x)
    assert isinstance(J, np.ndarray)
    assert J.shape == (m_rows, n)
    np.testing.assert_array_equal(J, known.toarray())


def test_small_model_uses_dense_path(monkeypatch):
    ev = NE.NLPEvaluator(_small_model())
    x = np.zeros(ev._n_variables)

    # Real cap (1e6) >> this model's m * n, so the dense path must be used.
    called = {"dense": False}
    real_dense = ev._evaluate_dense_jacobian

    def _spy(xx):
        called["dense"] = True
        return real_dense(xx)

    monkeypatch.setattr(ev, "_evaluate_dense_jacobian", _spy)
    # If sparse were (wrongly) chosen it would short-circuit before _spy.
    ev._sparse_jac_fn = lambda _x: (_ for _ in ()).throw(
        AssertionError("sparse path used below the cap")
    )

    J = ev.evaluate_jacobian(x)
    assert called["dense"], "small model must use the dense Jacobian path"
    assert J.shape == (ev._n_constraints, ev._n_variables)
