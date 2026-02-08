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
