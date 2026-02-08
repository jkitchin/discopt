"""
Batch Relaxation Evaluator: evaluate McCormick relaxations for batches of B&B nodes.

Wraps compiled relaxation functions with jax.vmap for vectorized evaluation
across multiple subproblems in a single fused XLA kernel. This is the key
component enabling GPU-accelerated branch-and-bound.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from discopt._jax.relaxation_compiler import (
    compile_constraint_relaxation,
    compile_objective_relaxation,
    compile_relaxation,
)
from discopt.modeling.core import Constraint, Expression, Model


class BatchRelaxationEvaluator:
    """Evaluate McCormick relaxations for a batch of B&B nodes.

    Each B&B node defines variable bounds (lb, ub). This class uses jax.vmap
    to evaluate the relaxation across all nodes simultaneously, producing
    lower and upper bounds on the objective (or constraint) for each node.

    The evaluator uses point relaxations (x_cv = x_cc = midpoint of bounds)
    which is the standard approach for bound computation in spatial B&B.

    Args:
        relax_fn: A compiled relaxation function from relaxation_compiler.
            Signature: (x_cv, x_cc, lb, ub) -> (cv, cc)
        n_vars: Number of variables (length of flat variable vector).
    """

    def __init__(self, relax_fn: Callable, n_vars: int) -> None:
        self._relax_fn = relax_fn
        self._n_vars = n_vars

        # vmap over axis 0 of all four arguments (batch dimension)
        self._vmapped_fn = jax.vmap(relax_fn)
        self._jitted_vmapped_fn = jax.jit(self._vmapped_fn)

    @property
    def n_vars(self) -> int:
        return self._n_vars

    def evaluate_batch(
        self,
        lb_batch: jnp.ndarray,
        ub_batch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate relaxations for a batch of B&B nodes.

        Uses the midpoint of each node's bounds as the evaluation point
        (point relaxation: x_cv = x_cc = midpoint).

        Args:
            lb_batch: Lower bounds for each node, shape (N, n_vars).
            ub_batch: Upper bounds for each node, shape (N, n_vars).

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (N,), where:
                lower_bounds[i] = convex underestimator value for node i
                upper_bounds[i] = concave overestimator value for node i
        """
        # Point relaxation: evaluate at midpoint
        mid = 0.5 * (lb_batch + ub_batch)
        cv_batch, cc_batch = self._jitted_vmapped_fn(mid, mid, lb_batch, ub_batch)
        return cv_batch, cc_batch

    def evaluate_batch_at(
        self,
        x_cv_batch: jnp.ndarray,
        x_cc_batch: jnp.ndarray,
        lb_batch: jnp.ndarray,
        ub_batch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate relaxations at specific points for a batch of nodes.

        This is the general form where the caller provides explicit
        cv/cc relaxation values (not just midpoint).

        Args:
            x_cv_batch: Convex relaxation values, shape (N, n_vars).
            x_cc_batch: Concave relaxation values, shape (N, n_vars).
            lb_batch: Lower bounds, shape (N, n_vars).
            ub_batch: Upper bounds, shape (N, n_vars).

        Returns:
            Tuple of (cv_batch, cc_batch), each shape (N,).
        """
        cv_batch, cc_batch = self._jitted_vmapped_fn(x_cv_batch, x_cc_batch, lb_batch, ub_batch)
        return cv_batch, cc_batch


def batch_evaluator_from_objective(model: Model) -> BatchRelaxationEvaluator:
    """Create a BatchRelaxationEvaluator for the model's objective.

    Args:
        model: Model with an objective set.

    Returns:
        BatchRelaxationEvaluator ready for batch evaluation.
    """
    relax_fn = compile_objective_relaxation(model)
    n_vars = sum(v.size for v in model._variables)
    return BatchRelaxationEvaluator(relax_fn, n_vars)


def batch_evaluator_from_expression(expr: Expression, model: Model) -> BatchRelaxationEvaluator:
    """Create a BatchRelaxationEvaluator for an arbitrary expression.

    Args:
        expr: Expression to build the evaluator for.
        model: Model containing variable definitions.

    Returns:
        BatchRelaxationEvaluator ready for batch evaluation.
    """
    relax_fn = compile_relaxation(expr, model)
    n_vars = sum(v.size for v in model._variables)
    return BatchRelaxationEvaluator(relax_fn, n_vars)


def batch_evaluator_from_constraint(
    constraint: Constraint, model: Model
) -> BatchRelaxationEvaluator:
    """Create a BatchRelaxationEvaluator for a constraint body.

    Args:
        constraint: Constraint whose body to evaluate.
        model: Model containing variable definitions.

    Returns:
        BatchRelaxationEvaluator ready for batch evaluation.
    """
    relax_fn = compile_constraint_relaxation(constraint, model)
    n_vars = sum(v.size for v in model._variables)
    return BatchRelaxationEvaluator(relax_fn, n_vars)
