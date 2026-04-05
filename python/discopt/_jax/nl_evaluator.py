"""
NLP Evaluator for .nl files: wraps the Rust PyModelRepr with numerical derivatives.

Provides the same interface as NLPEvaluator but evaluates objective/constraints
via the Rust expression evaluator and computes gradients/Hessians/Jacobians
via finite differences.
"""

from __future__ import annotations

import numpy as np


class NLPEvaluatorFromNl:
    """
    Evaluator for models loaded from .nl files via the Rust parser.

    Uses the Rust PyModelRepr for function evaluations and finite-difference
    approximations for derivatives. This enables solving .nl models through
    cyipopt without needing JAX-compiled expression DAGs.
    """

    def __init__(self, model) -> None:
        """
        Create an NL evaluator from a Model with _nl_repr attribute.

        Args:
            model: A Model created by from_nl() with _nl_repr set.
        """
        self._model = model
        self._nl_repr = model._nl_repr
        self._n_variables = model._nl_repr.n_vars
        self._n_constraints = model._nl_repr.n_constraints
        self._negate = model._objective.sense.value == "maximize"

        # Finite difference step size
        self._eps = 1e-7

    def evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective at x. Returns scalar."""
        val = self._nl_repr.evaluate_objective(np.asarray(x, dtype=np.float64))
        return -val if self._negate else val

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient of objective at x via central finite differences."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        grad = np.empty(n, dtype=np.float64)
        eps = self._eps
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            fp = self._nl_repr.evaluate_objective(x_plus)
            fm = self._nl_repr.evaluate_objective(x_minus)
            grad[i] = (fp - fm) / (2.0 * eps)
        if self._negate:
            grad = -grad
        return grad

    def evaluate_hessian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian of objective at x via finite differences."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        hess = np.empty((n, n), dtype=np.float64)
        eps = self._eps
        self._nl_repr.evaluate_objective(x)  # warm-up / validate x
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                fpp = self._nl_repr.evaluate_objective(x_pp)
                fpm = self._nl_repr.evaluate_objective(x_pm)
                fmp = self._nl_repr.evaluate_objective(x_mp)
                fmm = self._nl_repr.evaluate_objective(x_mm)
                h = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
                hess[i, j] = h
                hess[j, i] = h
        if self._negate:
            hess = -hess
        return hess

    def evaluate_lagrangian_hessian(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Evaluate Hessian of the Lagrangian via finite differences.

        H = obj_factor * ∇²f(x) + Σᵢ λᵢ ∇²gᵢ(x)
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        hess = np.empty((n, n), dtype=np.float64)
        eps = self._eps
        sign = -1.0 if self._negate else 1.0

        def _lagrangian(xp):
            val = obj_factor * sign * self._nl_repr.evaluate_objective(xp)
            for k in range(self._n_constraints):
                val += lambda_[k] * self._nl_repr.evaluate_constraint(k, xp)
            return val

        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                h = (
                    _lagrangian(x_pp) - _lagrangian(x_pm) - _lagrangian(x_mp) + _lagrangian(x_mm)
                ) / (4.0 * eps * eps)
                hess[i, j] = h
                hess[j, i] = h
        return hess

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraint bodies at x. Returns (m,) array."""
        if self._n_constraints == 0:
            return np.array([], dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        vals = np.empty(self._n_constraints, dtype=np.float64)
        for i in range(self._n_constraints):
            vals[i] = self._nl_repr.evaluate_constraint(i, x)
        return vals

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian of constraints at x via central finite differences."""
        if self._n_constraints == 0:
            return np.empty((0, self._n_variables), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n = self._n_variables
        m = self._n_constraints
        jac = np.empty((m, n), dtype=np.float64)
        eps = self._eps
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            cp = self.evaluate_constraints(x_plus)
            cm = self.evaluate_constraints(x_minus)
            jac[:, j] = (cp - cm) / (2.0 * eps)
        return jac

    def has_sparse_structure(self) -> bool:
        """True if numeric probing detected useful sparsity."""
        self._ensure_coo_cache()
        return self._use_sparse

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return Jacobian sparsity as COO (rows, cols) arrays."""
        self._ensure_coo_cache()
        return (self._jac_rows, self._jac_cols)

    def hessian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower-triangle Hessian sparsity as COO (rows, cols) arrays."""
        self._ensure_coo_cache()
        return (self._hess_rows, self._hess_cols)

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian values as 1-D array matching jacobian_structure order."""
        self._ensure_coo_cache()
        jac = self.evaluate_jacobian(x)
        return jac[self._jac_rows, self._jac_cols].astype(np.float64)

    def evaluate_hessian_values(
        self, x: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        """Return Lagrangian Hessian values as 1-D array matching hessian_structure."""
        self._ensure_coo_cache()
        h = self.evaluate_lagrangian_hessian(x, obj_factor, lambda_)
        return h[self._hess_rows, self._hess_cols].astype(np.float64)

    def _ensure_coo_cache(self) -> None:
        """Lazily detect sparsity via numeric probing and cache COO indices."""
        if hasattr(self, "_jac_rows"):
            return

        n = self._n_variables
        m = self._n_constraints
        lb, ub = self.variable_bounds

        # Clamp bounds for sampling
        lb_safe = np.where(np.isfinite(lb), lb, -10.0)
        ub_safe = np.where(np.isfinite(ub), ub, 10.0)

        rng = np.random.RandomState(42)
        n_probes = 3
        threshold = 1e-10

        jac_nonzero = np.zeros((m, n), dtype=bool)
        hess_nonzero = np.zeros((n, n), dtype=bool)

        for _ in range(n_probes):
            x_probe = lb_safe + rng.rand(n) * (ub_safe - lb_safe)
            if m > 0:
                jac = self.evaluate_jacobian(x_probe)
                jac_nonzero |= np.abs(jac) > threshold
            lam = rng.randn(m) if m > 0 else np.array([], dtype=np.float64)
            hess = self.evaluate_lagrangian_hessian(x_probe, 1.0, lam)
            hess_nonzero |= np.abs(hess) > threshold

        # Make Hessian pattern symmetric
        hess_nonzero = hess_nonzero | hess_nonzero.T

        # Decide whether sparsity is worth exploiting
        jac_nnz = int(jac_nonzero.sum())
        hess_nnz = int(hess_nonzero.sum())
        jac_total = m * n if m > 0 else 1
        hess_total = n * n if n > 0 else 1
        jac_density = jac_nnz / jac_total
        hess_density = hess_nnz / hess_total
        self._use_sparse = n >= 50 and (jac_density < 0.15 or hess_density < 0.15)

        if self._use_sparse:
            if m > 0:
                jac_r, jac_c = np.where(jac_nonzero)
                self._jac_rows = jac_r.astype(np.intp)
                self._jac_cols = jac_c.astype(np.intp)
            else:
                self._jac_rows = np.array([], dtype=np.intp)
                self._jac_cols = np.array([], dtype=np.intp)
            hess_r, hess_c = np.where(hess_nonzero)
            lower_mask = hess_r >= hess_c
            self._hess_rows = hess_r[lower_mask].astype(np.intp)
            self._hess_cols = hess_c[lower_mask].astype(np.intp)
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
