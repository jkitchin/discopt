"""Evaluator wrapping McCormick relaxation closures for the discopt NLP protocol.

The McCormick relaxation closures produced by ``_jax/relaxation_compiler.py``
have signature ``(x_cv, x_cc, lb, ub) -> (cv, cc)``. The convex relaxation
NLP we want to solve is::

    minimize    cv_obj(x)        (or -cc_obj(x) when negate=True)
    subject to  cv_g(x)   <= 0   for sense in {"<=", "=="}
                -cc_g(x)  <= 0   for sense == ">="
                lb <= x <= ub

This module wraps those closures into an object implementing the discopt
NLP evaluator protocol — ``evaluate_objective``, ``evaluate_gradient``,
``evaluate_constraints``, ``evaluate_jacobian``, ``evaluate_lagrangian_hessian``,
plus structure / sparse-values methods. The same evaluator is consumed by
POUNCE (via :mod:`discopt.solvers.nlp_pounce`) and the JAX IPM.

Bounds (lb, ub) are runtime args of the jit'd callables, so a single
compiled module serves every B&B node on a given problem.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


class McCormickRelaxationEvaluator:
    """Wrap McCormick relaxation closures for the discopt NLP evaluator protocol.

    The relaxation closures are kept by identity, so a single evaluator
    instance can be reused across B&B nodes by calling :meth:`set_bounds`
    before each solve. The underlying JAX jits are keyed on shape only,
    so per-node solves hit the XLA cache after the first.
    """

    def __init__(
        self,
        obj_relax_fn: Callable,
        con_relax_fns: Optional[list[Callable]],
        con_senses: Optional[list[str]],
        n_vars: int,
        negate: bool = False,
    ) -> None:
        self._n = int(n_vars)
        self._negate = bool(negate)
        self._fns = tuple(con_relax_fns) if con_relax_fns else ()
        self._senses = tuple(con_senses) if con_senses else ()
        if (len(self._fns) > 0) != (len(self._senses) > 0):
            raise ValueError("con_relax_fns and con_senses must both be set or both empty.")
        self._m = len(self._fns)

        negate = self._negate

        def obj_scalar(x, lb, ub):
            cv, cc = obj_relax_fn(x, x, lb, ub)
            return -cc if negate else cv

        if self._m > 0:
            fns_local = self._fns
            senses_local = self._senses

            def cons_vec(x, lb, ub):
                vals = []
                for fn, sense in zip(fns_local, senses_local):
                    cv, cc = fn(x, x, lb, ub)
                    if sense == ">=":
                        vals.append(-cc)
                    else:  # "<=" and "=="
                        vals.append(cv)
                return jnp.stack(vals)

            self._cons_jit = jax.jit(cons_vec)
            self._jac_jit = jax.jit(jax.jacobian(cons_vec, argnums=0))
        else:
            self._cons_jit = None
            self._jac_jit = None

        self._obj_jit = jax.jit(obj_scalar)
        self._grad_jit = jax.jit(jax.grad(obj_scalar, argnums=0))

        cons_jit = self._cons_jit

        def lagrangian(x, obj_factor, lam, lb, ub):
            L = obj_factor * obj_scalar(x, lb, ub)
            if cons_jit is not None:
                L = L + jnp.dot(lam, cons_jit(x, lb, ub))
            return L

        self._lag_hess_jit = jax.jit(jax.hessian(lagrangian, argnums=0))

        # Bounds are set per-node via set_bounds().
        self._lb: Optional[jnp.ndarray] = None
        self._ub: Optional[jnp.ndarray] = None

    # ------------------------------------------------------------------
    # Per-node setup
    # ------------------------------------------------------------------

    def set_bounds(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """Bind the variable bounds used for the next solve."""
        self._lb = jnp.asarray(lb, dtype=jnp.float64)
        self._ub = jnp.asarray(ub, dtype=jnp.float64)

    def _require_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._lb is None or self._ub is None:
            raise RuntimeError(
                "McCormickRelaxationEvaluator: call set_bounds(lb, ub) before evaluating."
            )
        return self._lb, self._ub

    # ------------------------------------------------------------------
    # Protocol — values
    # ------------------------------------------------------------------

    def evaluate_objective(self, x: np.ndarray) -> float:
        lb, ub = self._require_bounds()
        return float(self._obj_jit(jnp.asarray(x, dtype=jnp.float64), lb, ub))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        lb, ub = self._require_bounds()
        return np.asarray(self._grad_jit(jnp.asarray(x, dtype=jnp.float64), lb, ub))

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        if self._cons_jit is None:
            return np.empty(0, dtype=np.float64)
        lb, ub = self._require_bounds()
        return np.asarray(self._cons_jit(jnp.asarray(x, dtype=jnp.float64), lb, ub))

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        if self._jac_jit is None:
            return np.empty((0, self._n), dtype=np.float64)
        lb, ub = self._require_bounds()
        return np.asarray(self._jac_jit(jnp.asarray(x, dtype=jnp.float64), lb, ub))

    def evaluate_lagrangian_hessian(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        lb, ub = self._require_bounds()
        if self._m == 0:
            lam_jax = jnp.zeros(0, dtype=jnp.float64)
        else:
            lam_jax = jnp.asarray(lambda_, dtype=jnp.float64)
        return np.asarray(
            self._lag_hess_jit(
                jnp.asarray(x, dtype=jnp.float64),
                float(obj_factor),
                lam_jax,
                lb,
                ub,
            )
        )

    # ------------------------------------------------------------------
    # Protocol — structure (dense fallback; McCormick relaxations are
    # typically dense over the variables touched by each constraint, and
    # the per-node nature of the solve makes one-shot dense evaluation
    # cheaper than running sparsity detection)
    # ------------------------------------------------------------------

    def has_sparse_structure(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Protocol — metadata
    # ------------------------------------------------------------------

    @property
    def n_variables(self) -> int:
        return self._n

    @property
    def n_constraints(self) -> int:
        return self._m

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb, ub = self._require_bounds()
        return np.asarray(lb), np.asarray(ub)
