"""Regression: node bound-tightening must not compile a giant dense Jacobian.

``NLPEvaluator.evaluate_jacobian`` uses ``jax.jit(jax.jacfwd(...))`` — a
forward-mode *dense* Jacobian whose compiled XLA program replicates the whole
constraint system once per input variable. On a large model the resulting jaxpr
explodes during MLIR lowering and XLA aborts the **process** with a native
SIGBUS / SIGILL — not a catchable Python exception, so the ``try/except`` around
the call cannot save it.

`rsyn0810m03hfsg` (1185 vars x 1935 constraints ~ 2.3M dense entries, but only
~4.5k nonzeros — 0.2% dense) crashed in ``_tighten_node_bounds_with_status``
once presolve was fast enough to reach node bound-tightening.

Fix: above ``_MAX_DENSE_JACOBIAN_ELEMS`` (n * m), skip the dense two-point
linearity test and fall back to the Jacobian-free structural / interval
nonlinear tightening. (The downstream FBBT row loop there is O(m * n^2) in pure
Python anyway — intractable at that scale — so skipping it loses nothing usable
on large models.)

This guards the crash deterministically without needing to actually compile the
multi-million-entry Jacobian (which would crash the whole pytest process rather
than fail this test).
"""

from __future__ import annotations

import discopt.solver as S
import numpy as np


class _ExplodingEvaluator:
    """Stand-in evaluator whose dense ``evaluate_jacobian`` must never be called
    above the size cap (calling it on the real thing would crash the process)."""

    def __init__(self, n_constraints: int):
        self.n_constraints = n_constraints
        self._model = object()  # opaque; the fallback is monkeypatched below

    def evaluate_jacobian(self, x):  # pragma: no cover - must not be reached
        raise AssertionError(
            "dense evaluate_jacobian was called above the dense-Jacobian cap — "
            "the guard did not short-circuit, risking the XLA process abort"
        )

    def evaluate_constraints(self, x):  # pragma: no cover
        raise AssertionError("evaluate_constraints reached above the cap")


def test_large_model_skips_dense_jacobian_and_falls_back(monkeypatch):
    n = 2000
    m = 2000  # n * m = 4,000,000 > _MAX_DENSE_JACOBIAN_ELEMS
    assert n * m > S._MAX_DENSE_JACOBIAN_ELEMS

    sentinel_lb = np.full(n, -1.0)
    sentinel_ub = np.full(n, 2.0)

    called = {"fallback": False}

    def _fake_fallback(model, lb, ub):
        called["fallback"] = True
        return sentinel_lb, sentinel_ub, False

    monkeypatch.setattr(S, "_apply_nonlinear_tightening_with_status", _fake_fallback)

    ev = _ExplodingEvaluator(n_constraints=m)
    node_lb = np.zeros(n)
    node_ub = np.ones(n)
    cl_list = [-1e20] * m  # non-empty so we pass the "no constraints" early return
    cu_list = [1e20] * m

    lb, ub, infeasible = S._tighten_node_bounds_with_status(ev, node_lb, node_ub, cl_list, cu_list)

    assert called["fallback"], "guard did not route to the Jacobian-free fallback"
    assert infeasible is False
    np.testing.assert_array_equal(lb, sentinel_lb)
    np.testing.assert_array_equal(ub, sentinel_ub)


def test_small_model_still_attempts_jacobian_test(monkeypatch):
    """Below the cap the guard must NOT short-circuit — evaluate_jacobian is
    still consulted (here it raises, which the function's own try/except turns
    into the same fallback, proving the dense path was entered)."""
    n = 10
    m = 10  # n * m = 100, far below the cap
    assert n * m <= S._MAX_DENSE_JACOBIAN_ELEMS

    entered = {"jac": False, "fallback": False}

    class _SmallEval:
        n_constraints = m
        _model = object()

        def evaluate_jacobian(self, x):
            entered["jac"] = True
            raise RuntimeError("force fallback after the dense path is entered")

    def _fake_fallback(model, lb, ub):
        entered["fallback"] = True
        return np.zeros(n), np.ones(n), False

    monkeypatch.setattr(S, "_apply_nonlinear_tightening_with_status", _fake_fallback)
    # No structural mask for the opaque model.
    monkeypatch.setattr(S, "_cached_structural_linear_mask", lambda *a, **k: None)

    S._tighten_node_bounds_with_status(
        _SmallEval(), np.zeros(n), np.ones(n), [-1e20] * m, [1e20] * m
    )

    assert entered["jac"], "dense Jacobian path should be entered below the cap"
    assert entered["fallback"]
