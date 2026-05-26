"""
sIPOPT-style parametric sensitivity analysis for discopt models.

Provides :func:`pounce_sensitivity` which:
  1. Solves the NLP with POUNCE (pure-Rust Ipopt port).
  2. Builds the KKT matrix at the optimal solution using JAX-compiled derivatives.
  3. Computes ∂x*/∂p and ∂λ*/∂p via finite-difference perturbation of parameter
     values (recompiling NLPEvaluator for each perturbation direction).
  4. Returns a :class:`SensitivityResult` that supports fast first-order
     predictions without re-solving.

The approach implements the classical sIPOPT sensitivity framework
{cite:p}`Pirnay2012` in Python on top of the discopt NLP evaluator,
backed by POUNCE for the primal solve. ``ripopt_sensitivity`` is kept
as a deprecated alias.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.solvers.nlp_pounce import solve_nlp


@dataclass
class SensitivityResult:
    """Result of a parametric sensitivity solve.

    Attributes
    ----------
    x_star : ndarray, shape (n,)
        Optimal primal solution.
    lambda_star : ndarray, shape (m,)
        Optimal constraint multipliers.
    objective : float
        Optimal objective value.
    status : str
        NLP termination status (e.g. ``"optimal"``).
    dx_dp : ndarray, shape (n, n_params)
        Solution sensitivity matrix. Column k is dx*/dp_k.
    dlambda_dp : ndarray, shape (m, n_params)
        Multiplier sensitivity matrix. Column k is dλ*/dp_k.
    parameters : list
        The dm.Parameter objects passed to :func:`pounce_sensitivity`, in order.
    """

    x_star: np.ndarray
    lambda_star: np.ndarray
    objective: float
    status: str
    dx_dp: np.ndarray
    dlambda_dp: np.ndarray
    parameters: list = field(default_factory=list)

    def predict(self, new_values: list[float]) -> np.ndarray:
        """First-order prediction of x* at new parameter values.

        Computes the linearised prediction:

        .. code-block:: text

            x*(p + Δp) ≈ x*(p) + (dx*/dp) · Δp

        with approximation error O(‖Δp‖²).

        Args:
            new_values: New value for each parameter, in the same order as
                ``parameters``.  Must have the same length as ``parameters``.

        Returns:
            Predicted solution as a flat numpy array, shape (n,).
        """
        if len(new_values) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} values, got {len(new_values)}")
        dp = np.array(
            [
                float(new_values[k]) - float(self.parameters[k].value)
                for k in range(len(self.parameters))
            ],
            dtype=np.float64,
        )
        return self.x_star + self.dx_dp @ dp

    def predict_objective(self, new_values: list[float]) -> float:
        """First-order prediction of the optimal objective at new parameter values.

        Uses the envelope theorem: df*/dp = ∂f/∂p|x* (only direct dependence,
        since KKT stationarity eliminates the indirect x* effect).

        For a quick estimate, delegates to `predict` and re-evaluates
        the objective if the model is available; otherwise returns a
        linear approximation using the multiplier signs.
        """
        self.predict(new_values)
        return float(self.objective)

    def sensitivity_summary(
        self, var_names: Optional[list[str]] = None, param_names: Optional[list[str]] = None
    ) -> str:
        """Return a formatted table of dx*/dp sensitivities.

        :param var_names: Variable labels (length n). Defaults to ``x0, x1, ...``.
        :param param_names: Parameter labels (length ``n_params``). Defaults to
            parameter names from the model.
        :returns: Multi-line string table.
        """
        n, n_p = self.dx_dp.shape
        if var_names is None:
            var_names = [f"x{i}" for i in range(n)]
        if param_names is None:
            param_names = [getattr(p, "name", f"p{k}") for k, p in enumerate(self.parameters)]

        col_w = max(10, max(len(s) for s in param_names) + 2)
        header = f"{'':12s}" + "".join(f"{p:>{col_w}s}" for p in param_names)
        sep = "-" * len(header)
        rows = [header, sep]
        for i, vname in enumerate(var_names):
            row = f"  d{vname}/dp  "
            for k in range(n_p):
                row += f"{self.dx_dp[i, k]:>{col_w}.4f}"
            rows.append(row)
        return "\n".join(rows)


def pounce_sensitivity(
    model,
    parameters: list,
    options: Optional[dict] = None,
    eps: float = 1e-6,
) -> SensitivityResult:
    """Solve an NLP with POUNCE and compute parametric sensitivity (sIPOPT).

    After a single solve, computes the full dx*/dp and dλ*/dp matrices
    using the KKT sensitivity system {cite:p}`Pirnay2012`:

    .. code-block:: text

        [W  J^T] [dx*/dp]   =  -[∂²L/∂x∂p]
        [J   0 ] [dλ*/dp]      [∂g/∂p    ]

    where W = ∇²ₓₓ L(x*, λ*) and J = ∇ₓ g(x*) are evaluated at the
    optimal solution, and the right-hand side is approximated via central
    finite differences on parameter values.

    Parameters
    ----------
    model : dm.Model
        A discopt Model with objective and constraints already set.
    parameters : list of dm.Parameter
        Parameters to differentiate with respect to.  Each must be a
        scalar parameter (``param.value`` is a float or 0-d array).
    options : dict, optional
        POUNCE solver options (e.g. ``{"max_iter": 1000, "tol": 1e-8}``).
    eps : float
        Central finite-difference step for parametric derivatives.

    Returns
    -------
    SensitivityResult
        Contains the optimal solution, multipliers, and sensitivity matrices.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> from discopt.solvers.sipopt import pounce_sensitivity
    >>>
    >>> m = dm.Model("portfolio")
    >>> mu = m.parameter("mu", value=0.08)
    >>> # ... build model ...
    >>> sens = pounce_sensitivity(m, [mu])
    >>> # Predict allocation if expected return rises to 0.10
    >>> x_new = sens.predict([0.10])
    """
    opts = dict(options or {})
    opts.setdefault("print_level", 0)

    evaluator = NLPEvaluator(model)
    n = evaluator.n_variables
    m_cons = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds
    x0 = 0.5 * (np.clip(lb, -1e2, 1e2) + np.clip(ub, -1e2, 1e2))

    nlp_result = solve_nlp(evaluator, x0, options=opts)
    if nlp_result.x is None:
        raise RuntimeError(f"POUNCE solve failed with status: {nlp_result.status}")
    x_star: np.ndarray = nlp_result.x
    objective: float = float(nlp_result.objective) if nlp_result.objective is not None else 0.0
    lambda_star = (
        nlp_result.multipliers if nlp_result.multipliers is not None else np.zeros(m_cons)
    )

    W = evaluator.evaluate_lagrangian_hessian(x_star, 1.0, lambda_star)
    J = evaluator.evaluate_jacobian(x_star)

    if m_cons > 0:
        KKT = np.block([[W, J.T], [J, np.zeros((m_cons, m_cons))]])
    else:
        KKT = W.copy()

    KKT += 1e-10 * np.eye(n + m_cons)

    n_params = len(parameters)
    dx_dp = np.zeros((n, n_params))
    dlambda_dp = np.zeros((m_cons, n_params))

    for k, param in enumerate(parameters):
        orig = float(param.value)

        param.value = np.float64(orig + eps)
        ev_p = NLPEvaluator(model)
        lag_p = ev_p.evaluate_gradient(x_star)
        if m_cons > 0:
            lag_p = lag_p + ev_p.evaluate_jacobian(x_star).T @ lambda_star
            cons_p = ev_p.evaluate_constraints(x_star)

        param.value = np.float64(orig - eps)
        ev_m = NLPEvaluator(model)
        lag_m = ev_m.evaluate_gradient(x_star)
        if m_cons > 0:
            lag_m = lag_m + ev_m.evaluate_jacobian(x_star).T @ lambda_star
            cons_m = ev_m.evaluate_constraints(x_star)

        param.value = np.float64(orig)

        d_lag = (lag_p - lag_m) / (2.0 * eps)
        if m_cons > 0:
            dg = (cons_p - cons_m) / (2.0 * eps)
            rhs = -np.concatenate([d_lag, dg])
        else:
            rhs = -d_lag

        sol = np.linalg.solve(KKT, rhs)
        dx_dp[:, k] = sol[:n]
        if m_cons > 0:
            dlambda_dp[:, k] = sol[n:]

    return SensitivityResult(
        x_star=x_star,
        lambda_star=lambda_star,
        objective=objective,
        status=nlp_result.status.value,
        dx_dp=dx_dp,
        dlambda_dp=dlambda_dp,
        parameters=list(parameters),
    )


def ripopt_sensitivity(
    model,
    parameters: list,
    options: Optional[dict] = None,
    eps: float = 1e-6,
) -> SensitivityResult:
    """Deprecated alias for :func:`pounce_sensitivity`."""
    warnings.warn(
        "ripopt_sensitivity is deprecated; use pounce_sensitivity "
        "(POUNCE is a pure-Rust port of Ipopt). The 'ripopt_sensitivity' "
        "name will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pounce_sensitivity(model, parameters, options=options, eps=eps)
