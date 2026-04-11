"""discopt.estimate -- Model-based parameter estimation.

Estimate unknown parameters from experimental data using weighted
least-squares (or maximum likelihood) formulated as an NLP and solved
with discopt's NLP solvers.

Quick Start
-----------
>>> from discopt.estimate import Experiment, ExperimentModel, estimate_parameters
>>> exp = MyExperiment()
>>> result = estimate_parameters(exp, data, initial_guess={"k": 0.5})
>>> print(result.summary())

See Also
--------
discopt.doe : Optimal design of experiments using the same Experiment interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

import discopt.modeling as dm
from discopt.modeling.core import Expression, Model, Variable

# ─────────────────────────────────────────────────────────────
# Experiment interface
# ─────────────────────────────────────────────────────────────


class ExperimentModel:
    """A discopt Model annotated with DoE / estimation metadata.

    Bundles a discopt Model with labels identifying which parts are
    unknown parameters, design inputs, measured responses, and
    measurement errors.

    Parameters
    ----------
    model : Model
        The underlying discopt optimization model.
    unknown_parameters : dict[str, Variable]
        Parameters to estimate, modeled as Variables with bounds.
    design_inputs : dict[str, Variable]
        Experimental conditions controlled by the experimenter.
    responses : dict[str, Expression]
        Model predictions corresponding to measured quantities.
        Each expression should be a scalar at a given solution point.
    measurement_error : dict[str, float]
        Standard deviation of measurement noise for each response.
        Keys must match ``responses``.

    Raises
    ------
    ValueError
        If response names and measurement_error names don't match.
    """

    def __init__(
        self,
        model: Model,
        unknown_parameters: dict[str, Variable],
        design_inputs: dict[str, Variable],
        responses: dict[str, Expression],
        measurement_error: dict[str, float],
    ):
        # Validate key consistency
        resp_keys = set(responses.keys())
        err_keys = set(measurement_error.keys())
        if resp_keys != err_keys:
            missing = resp_keys - err_keys
            extra = err_keys - resp_keys
            parts = []
            if missing:
                parts.append(f"responses missing error: {missing}")
            if extra:
                parts.append(f"error missing response: {extra}")
            raise ValueError(f"response and measurement_error keys must match. {'; '.join(parts)}")

        self.model = model
        self.unknown_parameters = unknown_parameters
        self.design_inputs = design_inputs
        self.responses = responses
        self.measurement_error = measurement_error

    @property
    def n_parameters(self) -> int:
        """Number of unknown parameters."""
        return len(self.unknown_parameters)

    @property
    def n_responses(self) -> int:
        """Number of measured responses."""
        return len(self.responses)

    @property
    def response_names(self) -> list[str]:
        """Ordered list of response names."""
        return list(self.responses.keys())

    @property
    def parameter_names(self) -> list[str]:
        """Ordered list of unknown parameter names."""
        return list(self.unknown_parameters.keys())


class Experiment:
    """Base class for model-based experiments.

    Subclass this and implement :meth:`create_model` to define your
    experiment. The returned :class:`ExperimentModel` bundles the discopt
    Model together with metadata labeling unknown parameters, design
    inputs, measured responses, and measurement errors.

    Examples
    --------
    >>> class MyExperiment(Experiment):
    ...     def create_model(self, **kwargs):
    ...         m = dm.Model("my_exp")
    ...         k = m.continuous("k", lb=0, ub=10)
    ...         # ... build model ...
    ...         return ExperimentModel(
    ...             model=m,
    ...             unknown_parameters={"k": k},
    ...             design_inputs={},
    ...             responses={"y": k * 2},
    ...             measurement_error={"y": 0.1},
    ...         )
    """

    def create_model(self, **kwargs) -> ExperimentModel:
        r"""Build the experiment model.

        Parameters
        ----------
        \*\*kwargs
            Keyword arguments, typically initial guesses for parameters
            or fixed design conditions.

        Returns
        -------
        ExperimentModel
            Annotated discopt model.
        """
        raise NotImplementedError("Subclass must implement create_model()")


# ─────────────────────────────────────────────────────────────
# Estimation result
# ─────────────────────────────────────────────────────────────


@dataclass
class EstimationResult:
    """Result of parameter estimation.

    Attributes
    ----------
    parameters : dict[str, float]
        Estimated parameter values.
    covariance : numpy.ndarray
        Parameter covariance matrix (inverse of FIM at the solution).
    fim : numpy.ndarray
        Fisher Information Matrix at the estimated parameters.
    objective : float
        Final objective value (weighted sum of squared residuals).
    solve_result : SolveResult
        The underlying discopt solve result.
    parameter_names : list[str]
        Ordered parameter names matching matrix rows/columns.
    n_observations : int
        Number of data points used in the estimation.
    """

    parameters: dict[str, float]
    covariance: np.ndarray
    fim: np.ndarray
    objective: float
    solve_result: dm.SolveResult
    parameter_names: list[str]
    n_observations: int

    @property
    def standard_errors(self) -> dict[str, float]:
        """Standard errors for each parameter (sqrt of covariance diagonal)."""
        se = np.sqrt(np.diag(self.covariance))
        return dict(zip(self.parameter_names, se))

    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        """Approximate 95% confidence intervals for each parameter.

        Uses t-distribution with ``n_observations - n_parameters``
        degrees of freedom.
        """
        from scipy.stats import t as t_dist

        n_params = len(self.parameters)
        dof = max(1, self.n_observations - n_params)
        t_val = t_dist.ppf(0.975, df=dof)
        ci = {}
        for i, name in enumerate(self.parameter_names):
            se = np.sqrt(self.covariance[i, i])
            val = self.parameters[name]
            ci[name] = (val - t_val * se, val + t_val * se)
        return ci

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Parameter correlation matrix."""
        d = np.sqrt(np.diag(self.covariance))
        # Avoid division by zero for perfectly determined parameters
        d = np.where(d > 0, d, 1.0)
        return np.asarray(self.covariance / np.outer(d, d))

    def summary(self) -> str:
        """Human-readable estimation summary."""
        lines = ["Parameter Estimation Results", "=" * 50]
        ci = self.confidence_intervals
        se = self.standard_errors
        for name in self.parameter_names:
            val = self.parameters[name]
            lo, hi = ci[name]
            lines.append(f"  {name:>12s} = {val:12.6g}  ± {se[name]:.4g}  [{lo:.4g}, {hi:.4g}]")
        lines.append(f"  {'Objective':>12s} = {self.objective:.6g}")
        lines.append(f"  {'N obs':>12s} = {self.n_observations}")
        det = np.linalg.det(self.fim)
        lines.append(f"  {'FIM det':>12s} = {det:.4g}")
        cond = np.linalg.cond(self.fim)
        lines.append(f"  {'FIM cond':>12s} = {cond:.4g}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main estimation function
# ─────────────────────────────────────────────────────────────


def estimate_parameters(
    experiment: Experiment,
    data: dict[str, Union[float, np.ndarray]],
    *,
    initial_guess: dict[str, float] | None = None,
    solver_options: dict | None = None,
) -> EstimationResult:
    """Estimate unknown parameters from experimental data.

    Builds a weighted least-squares NLP and solves it:

    .. math::

        \\min_{\\theta} \\sum_i \\left(\\frac{y_i^{obs} - y_i^{model}(\\theta)}{\\sigma_i}\\right)^2

    Parameters
    ----------
    experiment : Experiment
        Experiment definition with :meth:`~Experiment.create_model`.
    data : dict[str, float or numpy.ndarray]
        Observed values for each response. Keys must match response names
        from the ExperimentModel. Scalar or 1-D array values.
    initial_guess : dict[str, float], optional
        Starting values for unknown parameters. Passed as kwargs to
        ``experiment.create_model()``.
    solver_options : dict, optional
        Options passed to the solver.

    Returns
    -------
    EstimationResult
        Estimated parameters, covariance, FIM, and diagnostics.

    Raises
    ------
    ValueError
        If data keys don't match response names, or if the solve fails.
    """
    kwargs = dict(initial_guess or {})
    em = experiment.create_model(**kwargs)

    # Validate data keys match responses
    resp_keys = set(em.responses.keys())
    data_keys = set(data.keys())
    if not data_keys.issubset(resp_keys):
        raise ValueError(f"Data keys {data_keys - resp_keys} not in response names {resp_keys}")

    # Build weighted least-squares objective
    residual_terms = []
    n_obs = 0
    for name in em.response_names:
        if name not in data:
            continue
        y_obs = np.asarray(data[name], dtype=np.float64)
        y_model = em.responses[name]
        sigma = em.measurement_error[name]

        if y_obs.ndim == 0:
            # Scalar observation
            residual_terms.append(((y_obs.item() - y_model) / sigma) ** 2)
            n_obs += 1
        else:
            # Should not happen for scalar responses — treat as single value
            residual_terms.append(((float(y_obs.flat[0]) - y_model) / sigma) ** 2)
            n_obs += 1

    if not residual_terms:
        raise ValueError("No data matched any response names")

    em.model.minimize(dm.sum(residual_terms))

    # Solve
    solve_out = em.model.solve(**(solver_options or {}))
    # solve() returns SolveResult (not streaming iterator) by default
    result: dm.SolveResult = solve_out  # type: ignore[assignment]

    if result.status not in ("optimal", "feasible"):
        raise ValueError(f"Estimation solve failed with status: {result.status}")

    # Extract estimated parameters
    params = {}
    for name, var in em.unknown_parameters.items():
        val = result.value(var)
        params[name] = float(np.asarray(val).flat[0])

    # Compute FIM at the solution for covariance estimation
    fim = _compute_estimation_fim(em, result)

    # Covariance = FIM^{-1} (with regularization for near-singular FIM)
    try:
        covariance = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        # Regularize singular FIM
        covariance = np.linalg.pinv(fim)

    return EstimationResult(
        parameters=params,
        covariance=covariance,
        fim=fim,
        objective=result.objective or 0.0,
        solve_result=result,
        parameter_names=em.parameter_names,
        n_observations=n_obs,
    )


def _compute_estimation_fim(
    em: ExperimentModel,
    result: dm.SolveResult,
) -> np.ndarray:
    """Compute Fisher Information Matrix at the estimation solution.

    Uses JAX autodiff to compute the Jacobian dy/dθ, then:
    FIM = J^T Σ^{-1} J

    For estimation, the unknown parameters are Variables, so the Jacobian
    is computed w.r.t. the variable values (not model Parameters).
    """
    import jax
    import jax.numpy as jnp

    from discopt._jax.differentiable import _compile_parametric_node
    from discopt._jax.parametric import extract_x_flat

    model = em.model
    x_flat = extract_x_flat(result, model)

    # Compile response functions
    # These are functions of x_flat (variables include the unknown params)
    response_fns = []
    for name in em.response_names:
        fn = _compile_parametric_node(em.responses[name], model)
        response_fns.append(fn)

    # We need the Jacobian of responses w.r.t. the unknown parameter
    # variables. Since unknown_parameters are Variables in the x_flat vector,
    # we compute d(responses)/d(x_flat) and extract the relevant columns.

    # Find indices of unknown parameter variables in x_flat
    param_indices = []
    for name, var in em.unknown_parameters.items():
        offset = 0
        for v in model._variables:
            if v is var:
                for i in range(v.size):
                    param_indices.append(offset + i)
                break
            offset += v.size

    # Build a dummy p_flat (model may or may not have Parameters)
    p_parts = []
    for p in model._parameters:
        p_parts.append(np.asarray(p.value, dtype=np.float64).ravel())
    if p_parts:
        p_flat = jnp.array(np.concatenate(p_parts), dtype=jnp.float64)
    else:
        p_flat = jnp.zeros(0, dtype=jnp.float64)

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    # Full Jacobian w.r.t. x_flat
    J_full = jax.jacobian(response_vector)(x_flat)
    # shape: (n_responses, n_x_total)

    # Extract columns for unknown parameters only
    J = J_full[:, param_indices]
    # shape: (n_responses, n_unknown_params)

    # Measurement covariance (diagonal)
    sigma = np.array([em.measurement_error[name] for name in em.response_names])
    Sigma_inv = np.diag(1.0 / sigma**2)

    # FIM = J^T Σ^{-1} J
    fim = np.asarray(J.T @ Sigma_inv @ J)

    return fim
