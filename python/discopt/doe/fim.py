"""Fisher Information Matrix computation via JAX autodiff.

Computes the FIM for model-based design of experiments using exact
Jacobian computation (no finite differences). The FIM quantifies how
much information an experiment provides about unknown parameters.

Mathematical background
-----------------------
For a model with responses ``y = f(θ, d)`` and measurement error
covariance ``Σ``, the Fisher Information Matrix is:

    FIM = J^T Σ^{-1} J

where ``J`` is the sensitivity Jacobian ``∂y/∂θ`` evaluated at the
nominal parameter values and design conditions.

The FIM is used to:
- Assess parameter identifiability (rank of FIM)
- Predict parameter estimation precision (Cov(θ) ≈ FIM^{-1})
- Optimize experimental design (maximize information content)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.estimate import Experiment, ExperimentModel


@dataclass
class FIMResult:
    """Result of Fisher Information Matrix computation.

    Attributes
    ----------
    fim : numpy.ndarray
        Fisher Information Matrix, shape ``(n_params, n_params)``.
    jacobian : numpy.ndarray
        Sensitivity Jacobian ``∂y/∂θ``, shape ``(n_responses, n_params)``.
    parameter_names : list[str]
        Ordered parameter names matching FIM rows/columns.
    response_names : list[str]
        Ordered response names matching Jacobian rows.
    """

    fim: np.ndarray
    jacobian: np.ndarray
    parameter_names: list[str]
    response_names: list[str]

    @property
    def d_optimal(self) -> float:
        """D-optimality criterion: ``log(det(FIM))``."""
        det = np.linalg.det(self.fim)
        if det <= 0:
            return -np.inf
        return float(np.log(det))

    @property
    def a_optimal(self) -> float:
        """A-optimality criterion: ``trace(FIM^{-1})``."""
        try:
            return float(np.trace(np.linalg.inv(self.fim)))
        except np.linalg.LinAlgError:
            return np.inf

    @property
    def e_optimal(self) -> float:
        """E-optimality criterion: minimum eigenvalue of FIM."""
        return float(np.min(np.linalg.eigvalsh(self.fim)))

    @property
    def me_optimal(self) -> float:
        """Modified E-optimality: condition number of FIM."""
        return float(np.linalg.cond(self.fim))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics as a dictionary."""
        return {
            "log_det_fim": self.d_optimal,
            "trace_fim_inv": self.a_optimal,
            "min_eigenvalue": self.e_optimal,
            "condition_number": self.me_optimal,
        }


def compute_fim(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    prior_fim: np.ndarray | None = None,
    method: str = "autodiff",
    fd_step: float = 1e-5,
) -> FIMResult:
    """Compute the Fisher Information Matrix via JAX autodiff.

    Uses ``jax.jacobian`` to compute exact sensitivities ``∂y/∂θ``, then:

        FIM = J^T Σ^{-1} J + FIM_prior

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal values for unknown parameters. These are set on the
        corresponding variables before computing the Jacobian.
    design_values : dict[str, float], optional
        Values for design input variables. If provided, these are fixed
        before computing the Jacobian.
    prior_fim : numpy.ndarray, optional
        Prior FIM from previous experiments (for sequential DoE).
    method : str, default "autodiff"
        Sensitivity computation method: ``"autodiff"`` (exact JAX) or
        ``"finite_difference"`` (central differences, for validation).
    fd_step : float, default 1e-5
        Relative perturbation size for finite differences (only used
        when ``method="finite_difference"``).

    Returns
    -------
    FIMResult
        FIM, Jacobian, and optimality metrics.
    """

    from discopt._jax.differentiable import _compile_parametric_node
    from discopt._jax.parametric import extract_x_flat

    # Build the model at nominal parameter values
    em = experiment.create_model(**param_values)

    # Set design variable values if provided
    if design_values:
        for name, val in design_values.items():
            if name in em.design_inputs:
                var = em.design_inputs[name]
                # Fix design variable by setting lb = ub = val
                val_arr = np.asarray(val, dtype=np.float64)
                if var.shape:
                    val_arr = np.full(var.shape, val_arr)
                var.lb = val_arr
                var.ub = val_arr

    # Solve the model to get x* at nominal parameters
    em.model.minimize(
        sum((em.unknown_parameters[n] - param_values[n]) ** 2 for n in em.parameter_names)
    )
    result = em.model.solve()

    x_flat = extract_x_flat(result, em.model)

    # Compile response functions
    response_fns = []
    for name in em.response_names:
        fn = _compile_parametric_node(em.responses[name], em.model)
        response_fns.append(fn)

    # Find indices of unknown parameter variables in x_flat
    param_indices = _get_param_indices(em)

    # Build p_flat for any model Parameters (distinct from unknown_parameters)
    p_flat = _build_p_flat(em.model)

    if method == "autodiff":
        J = _compute_jacobian_autodiff(response_fns, x_flat, p_flat, param_indices)
    elif method == "finite_difference":
        J = _compute_jacobian_fd(response_fns, x_flat, p_flat, param_indices, fd_step)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'autodiff' or 'finite_difference'.")

    # Measurement covariance (diagonal)
    sigma = np.array([em.measurement_error[name] for name in em.response_names])
    Sigma_inv = np.diag(1.0 / sigma**2)

    # FIM = J^T Σ^{-1} J
    fim = np.asarray(J.T @ Sigma_inv @ J)

    if prior_fim is not None:
        fim = fim + prior_fim

    return FIMResult(
        fim=fim,
        jacobian=np.asarray(J),
        parameter_names=em.parameter_names,
        response_names=em.response_names,
    )


def check_identifiability(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    tol: float = 1e-6,
) -> "IdentifiabilityResult":
    """Check if parameters are structurally identifiable.

    Computes the FIM and checks its rank. A full-rank FIM indicates
    all parameters can be estimated independently from the data.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_values : dict[str, float], optional
        Design input values.
    tol : float, default 1e-6
        Tolerance for determining near-zero singular values.

    Returns
    -------
    IdentifiabilityResult
        Identifiability assessment.
    """
    fim_result = compute_fim(experiment, param_values, design_values)
    fim = fim_result.fim
    n_params = len(fim_result.parameter_names)

    # SVD for robust rank computation
    singular_values = np.linalg.svd(fim, compute_uv=False)
    rank = int(np.sum(singular_values > tol * singular_values[0]))

    # Identify problematic parameters (near-zero singular directions)
    _, _, Vt = np.linalg.svd(fim)
    problematic = []
    for i in range(rank, n_params):
        # The i-th right singular vector corresponds to an unidentifiable direction
        direction = Vt[i]
        # Find which parameter has the largest component
        max_idx = int(np.argmax(np.abs(direction)))
        problematic.append(fim_result.parameter_names[max_idx])

    return IdentifiabilityResult(
        is_identifiable=(rank == n_params),
        fim_rank=rank,
        n_parameters=n_params,
        problematic_parameters=problematic,
        condition_number=fim_result.me_optimal,
        fim_result=fim_result,
    )


@dataclass
class IdentifiabilityResult:
    """Result of identifiability analysis.

    Attributes
    ----------
    is_identifiable : bool
        True if all parameters are identifiable (FIM is full rank).
    fim_rank : int
        Numerical rank of the FIM.
    n_parameters : int
        Total number of unknown parameters.
    problematic_parameters : list[str]
        Parameters that are not identifiable (in unidentifiable directions).
    condition_number : float
        Condition number of the FIM.
    fim_result : FIMResult
        The underlying FIM computation result.
    """

    is_identifiable: bool
    fim_rank: int
    n_parameters: int
    problematic_parameters: list[str]
    condition_number: float
    fim_result: FIMResult


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────


def _get_param_indices(em: ExperimentModel) -> list[int]:
    """Find indices of unknown parameter variables in the flat x vector."""
    param_indices = []
    for name, var in em.unknown_parameters.items():
        offset = 0
        for v in em.model._variables:
            if v is var:
                for i in range(v.size):
                    param_indices.append(offset + i)
                break
            offset += v.size
    return param_indices


def _build_p_flat(model):
    """Build parameter flat vector for any model Parameters."""
    import jax.numpy as jnp

    p_parts = []
    for p in model._parameters:
        p_parts.append(np.asarray(p.value, dtype=np.float64).ravel())
    if p_parts:
        return jnp.array(np.concatenate(p_parts), dtype=jnp.float64)
    return jnp.zeros(0, dtype=jnp.float64)


def _compute_jacobian_autodiff(response_fns, x_flat, p_flat, param_indices):
    """Compute Jacobian via JAX autodiff."""
    import jax
    import jax.numpy as jnp

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    J_full = jax.jacobian(response_vector)(x_flat)
    return J_full[:, param_indices]


def _compute_jacobian_fd(response_fns, x_flat, p_flat, param_indices, step):
    """Compute Jacobian via central finite differences."""
    import jax.numpy as jnp

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    n_responses = len(response_fns)
    n_params = len(param_indices)
    J = np.zeros((n_responses, n_params))

    for j, idx in enumerate(param_indices):
        x_plus = x_flat.at[idx].set(x_flat[idx] + step)
        x_minus = x_flat.at[idx].set(x_flat[idx] - step)
        J[:, j] = (response_vector(x_plus) - response_vector(x_minus)) / (2 * step)

    return J
