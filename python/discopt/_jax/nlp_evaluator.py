"""
NLP Evaluator: JIT-compiled objective, gradient, Hessian, constraint, and Jacobian.

Wraps the DAG compiler output to provide evaluation callbacks suitable for
NLP solvers (cyipopt in Phase 1, Rust Ipopt later).

All evaluate_* methods accept and return numpy arrays for compatibility
with C-based solvers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax.dag_compiler import compile_constraint, compile_objective
from discopt.modeling.core import Constraint, Model, ObjectiveSense


def validate_sparse_values(evaluator, x: np.ndarray, atol: float = 1e-8) -> bool:
    """Check that sparse COO values agree with dense evaluation.

    Computes both dense and sparse Jacobian/Hessian at x and verifies that
    the sparse values match the dense values at the declared positions.

    Returns True if validation passes, False otherwise.
    """
    import logging

    logger = logging.getLogger("discopt.sparse")
    ok = True

    # Validate Jacobian
    if evaluator.n_constraints > 0:
        jac_dense = evaluator.evaluate_jacobian(x)
        rows, cols = evaluator.jacobian_structure()
        sparse_vals = evaluator.evaluate_jacobian_values(x)
        dense_at_pos = jac_dense[rows, cols]
        if not np.allclose(sparse_vals, dense_at_pos, atol=atol):
            max_err = float(np.max(np.abs(sparse_vals - dense_at_pos)))
            logger.warning("Sparse Jacobian validation failed (max err=%.2e)", max_err)
            ok = False

    # Validate Hessian
    m = evaluator.n_constraints
    lam = np.ones(m, dtype=np.float64)
    hess_dense = evaluator.evaluate_lagrangian_hessian(x, 1.0, lam)
    rows, cols = evaluator.hessian_structure()
    sparse_vals = evaluator.evaluate_hessian_values(x, 1.0, lam)
    dense_at_pos = hess_dense[rows, cols]
    if not np.allclose(sparse_vals, dense_at_pos, atol=atol):
        max_err = float(np.max(np.abs(sparse_vals - dense_at_pos)))
        logger.warning("Sparse Hessian validation failed (max err=%.2e)", max_err)
        ok = False

    return ok


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

        # Compile Hessian (objective only)
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

        # Compile Lagrangian Hessian: obj_factor * ∇²f(x) + Σᵢ λᵢ ∇²gᵢ(x)
        cons_fn = self._cons_fn

        def lagrangian(x, obj_factor, lam):
            L = obj_factor * obj_fn(x)
            if cons_fn is not None:
                L = L + jnp.dot(lam, cons_fn(x))
            return L

        self._lagrangian_hess_fn = jax.jit(jax.hessian(lagrangian, argnums=0))

    def evaluate_lagrangian_hessian(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Evaluate Hessian of the Lagrangian at x.

        H = obj_factor * ∇²f(x) + Σᵢ λᵢ ∇²gᵢ(x)
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        lam = jnp.array(lambda_, dtype=jnp.float64)
        return np.asarray(self._lagrangian_hess_fn(x_jax, obj_factor, lam))

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

    def evaluate_sparse_jacobian(self, x: np.ndarray):
        """Evaluate Jacobian as a sparse CSC matrix using compressed JVPs.

        Uses sparsity detection and graph coloring to evaluate the Jacobian
        in O(p) JVPs where p is the chromatic number (typically 5-20).
        Falls back to dense evaluation if sparsity infrastructure is unavailable.

        Returns:
            scipy.sparse.csc_matrix of shape (m, n), or dense (m, n) ndarray
            if sparse evaluation is not applicable.
        """
        if self._cons_fn is None:
            import scipy.sparse as sp

            return sp.csc_matrix((0, self._n_variables), dtype=np.float64)

        # Lazy-initialize sparse infrastructure
        if not hasattr(self, "_sparse_jac_fn"):
            self._sparse_jac_fn = None
            self._sparse_pattern = None
            try:
                from discopt._jax.sparse_jacobian import make_sparse_jac_fn
                from discopt._jax.sparsity import (
                    compute_coloring,
                    detect_sparsity_dag,
                    make_seed_matrix,
                    should_use_sparse,
                )

                pattern = detect_sparsity_dag(self._model)
                self._sparse_pattern = pattern
                if should_use_sparse(pattern):
                    colors, n_colors = compute_coloring(pattern)
                    seed = make_seed_matrix(colors, n_colors, pattern.n_vars)
                    self._sparse_jac_fn = make_sparse_jac_fn(self._cons_fn, pattern, colors, seed)
            except Exception:
                pass

        if self._sparse_jac_fn is not None:
            return self._sparse_jac_fn(x)

        # Fallback to dense
        return self.evaluate_jacobian(x)

    @property
    def sparsity_pattern(self):
        """Lazily compute and return the sparsity pattern for this model."""
        if not hasattr(self, "_sparse_pattern") or self._sparse_pattern is None:
            try:
                from discopt._jax.sparsity import detect_sparsity_dag

                self._sparse_pattern = detect_sparsity_dag(self._model)
            except Exception:
                self._sparse_pattern = None
        return self._sparse_pattern

    def has_sparse_structure(self) -> bool:
        """True if this evaluator provides sparse COO structure.

        Uses the DAG sparsity detector and the should_use_sparse threshold
        (density < 15%, n >= 50) to decide.
        """
        if not hasattr(self, "_use_sparse"):
            self._use_sparse = False
            pattern = self.sparsity_pattern
            if pattern is not None:
                try:
                    from discopt._jax.sparsity import should_use_sparse

                    self._use_sparse = should_use_sparse(pattern)
                except Exception:
                    pass
        return self._use_sparse

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return Jacobian sparsity as COO (rows, cols) arrays.

        When sparse structure is available, returns only nonzero positions.
        Otherwise returns dense (all m*n positions).
        """
        self._ensure_coo_cache()
        return (self._jac_rows, self._jac_cols)

    def hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower-triangle Hessian sparsity as COO (rows, cols) arrays.

        When sparse structure is available, returns only nonzero lower-triangle
        positions. Otherwise returns all lower-triangle positions.
        """
        self._ensure_coo_cache()
        return (self._hess_rows, self._hess_cols)

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian values as 1-D array matching jacobian_structure order."""
        self._ensure_coo_cache()
        if self.has_sparse_structure() and self._jac_fn is not None:
            # Use sparse evaluation and extract at COO positions
            jac_sparse = self.evaluate_sparse_jacobian(x)
            import scipy.sparse as sp

            if sp.issparse(jac_sparse):
                jac_dense = jac_sparse.toarray()
            else:
                jac_dense = np.asarray(jac_sparse)
            return jac_dense[self._jac_rows, self._jac_cols].astype(np.float64)
        # Dense path: flatten full Jacobian
        jac = self.evaluate_jacobian(x)
        return jac[self._jac_rows, self._jac_cols].astype(np.float64)

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Return Lagrangian Hessian values as 1-D array matching hessian_structure order."""
        self._ensure_coo_cache()
        h = self.evaluate_lagrangian_hessian(x, obj_factor, lambda_)
        return h[self._hess_rows, self._hess_cols].astype(np.float64)

    def _ensure_coo_cache(self) -> None:
        """Lazily compute and cache COO index arrays for Jacobian and Hessian."""
        if hasattr(self, "_jac_rows"):
            return

        n = self._n_variables
        m = self._n_constraints

        if self.has_sparse_structure():
            pattern = self.sparsity_pattern
            # Jacobian: convert CSR boolean to COO indices
            jac_coo = pattern.jacobian_sparsity.tocoo()
            self._jac_rows = jac_coo.row.astype(np.intp)
            self._jac_cols = jac_coo.col.astype(np.intp)
            # Hessian: extract lower triangle from symmetric pattern
            hess_coo = pattern.hessian_sparsity.tocoo()
            lower_mask = hess_coo.row >= hess_coo.col
            self._hess_rows = hess_coo.row[lower_mask].astype(np.intp)
            self._hess_cols = hess_coo.col[lower_mask].astype(np.intp)
        else:
            # Dense fallback
            if m > 0:
                rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
                self._jac_rows = rows.flatten().astype(np.intp)
                self._jac_cols = cols.flatten().astype(np.intp)
            else:
                self._jac_rows = np.array([], dtype=np.intp)
                self._jac_cols = np.array([], dtype=np.intp)
            self._hess_rows, self._hess_cols = np.tril_indices(n)

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
