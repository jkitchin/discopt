"""
NLP Evaluator: JIT-compiled objective, gradient, Hessian, constraint, and Jacobian.

Wraps the DAG compiler output to provide evaluation callbacks suitable for
NLP solvers (cyipopt in Phase 1, Rust Ipopt later).

All evaluate_* methods accept and return numpy arrays for compatibility
with C-based solvers.
"""

from __future__ import annotations

import sys
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/jaxminlp_benchmarks")
from jaxminlp_api.core import Constraint, Model, ObjectiveSense

from discopt._jax.dag_compiler import compile_constraint, compile_objective


class NLPEvaluator:
    """
    JAX-based NLP evaluation layer providing JIT-compiled callbacks.

    Wraps the DAG compiler output to provide objective, gradient, Hessian,
    constraint, and Jacobian evaluations suitable for NLP solvers.

    All methods return numpy arrays (not JAX arrays) for compatibility
    with cyipopt and other C-based solvers.

    Usage:
        evaluator = NLPEvaluator(model)
        obj = evaluator.evaluate_objective(x)
        grad = evaluator.evaluate_gradient(x)
        hess = evaluator.evaluate_hessian(x)
        cons = evaluator.evaluate_constraints(x)
        jac = evaluator.evaluate_jacobian(x)
    """

    def __init__(self, model: Model) -> None:
        """
        Compile model expressions into JIT-compiled evaluation functions.

        Args:
            model: A Model with objective and constraints set.
        """
        if model._objective is None:
            raise ValueError("Model has no objective set.")

        self._model = model
        self._negate = model._objective.sense == ObjectiveSense.MAXIMIZE

        # Compute variable count
        self._n_variables = sum(v.size for v in model._variables)
        self._n_constraints = len(model._constraints)

        # Compile objective
        raw_obj_fn = compile_objective(model)
        if self._negate:
            def obj_fn(x_flat: jnp.ndarray) -> jnp.ndarray:
                return -raw_obj_fn(x_flat)
        else:
            obj_fn = raw_obj_fn
        self._obj_fn = jax.jit(obj_fn)

        # Compile gradient (jax.grad of scalar objective)
        self._grad_fn = jax.jit(jax.grad(obj_fn))

        # Compile Hessian
        self._hess_fn = jax.jit(jax.hessian(obj_fn))

        # Compile constraints
        if self._n_constraints > 0:
            constraint_fns = []
            for c in model._constraints:
                if isinstance(c, Constraint):
                    constraint_fns.append(compile_constraint(c, model))
                else:
                    # Skip non-standard constraints (_IndicatorConstraint, etc.)
                    self._n_constraints -= 1

            self._constraint_fns = constraint_fns

            if len(constraint_fns) > 0:
                def stacked_constraints(x_flat: jnp.ndarray) -> jnp.ndarray:
                    return jnp.array([fn(x_flat) for fn in constraint_fns])

                self._cons_fn = jax.jit(stacked_constraints)
                self._jac_fn = jax.jit(jax.jacobian(stacked_constraints))
            else:
                self._cons_fn = None
                self._jac_fn = None
        else:
            self._constraint_fns = []
            self._cons_fn = None
            self._jac_fn = None

    def evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective at x. Returns scalar."""
        x_jax = jnp.array(x, dtype=jnp.float64)
        return float(self._obj_fn(x_jax))

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient of objective at x. Returns (n,) array."""
        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.asarray(self._grad_fn(x_jax))

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian of objective at x. Returns (n, n) array."""
        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.asarray(self._hess_fn(x_jax))

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraint bodies at x. Returns (m,) array."""
        if self._cons_fn is None:
            return np.array([], dtype=np.float64)
        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.asarray(self._cons_fn(x_jax))

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian of constraints at x. Returns (m, n) array."""
        if self._jac_fn is None:
            return np.empty((0, self._n_variables), dtype=np.float64)
        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.asarray(self._jac_fn(x_jax))

    @property
    def n_variables(self) -> int:
        """Total number of variables (flat)."""
        return self._n_variables

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return self._n_constraints

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lb, ub) arrays of shape (n,) for all variables."""
        lbs = []
        ubs = []
        for v in self._model._variables:
            lbs.append(v.lb.flatten())
            ubs.append(v.ub.flatten())
        return np.concatenate(lbs), np.concatenate(ubs)
