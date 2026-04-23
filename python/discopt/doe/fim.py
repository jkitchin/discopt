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


@dataclass
class IdentifiabilityResult:
    """Minimal identifiability assessment (backwards-compatible).

    Attributes
    ----------
    is_identifiable : bool
        True if all parameters are identifiable (FIM is full rank).
    fim_rank : int
        Numerical rank of the FIM.
    n_parameters : int
        Total number of unknown parameters.
    problematic_parameters : list[str]
        Parameters with the largest component in the null directions.
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


@dataclass
class IdentifiabilityDiagnostics:
    """Full Belsley/Gutenkunst identifiability diagnostic bundle.

    Returned by :func:`diagnose_identifiability`. Superset of
    :class:`IdentifiabilityResult`; includes everything needed to apply
    the regression-diagnostic rules of Belsley, Kuh & Welsch (1980) and
    the sloppy-model spectrum of Gutenkunst et al. (2007).

    Scaling conventions
    -------------------
    - ``singular_values``, ``condition_indices``, ``variance_decomposition``,
      ``vif``: computed on the *unit-column-length* scaled Jacobian (each
      column divided by its 2-norm). This is the Belsley convention; no
      mean-centering since there is no intercept in a sensitivity Jacobian.
    - ``log_eigenvalue_spectrum``, ``normalized_log_spectrum``,
      ``standard_errors``, ``correlation_matrix``: computed on the physical
      FIM = J^T Sigma^-1 J (unscaled).

    Notes
    -----
    Yao ranking and condition indices are *not* invariant under
    reparameterization (e.g. theta -> log theta). Profile likelihood is.
    If the condition number is large, try a log-scale reparameterization
    before concluding non-identifiability.

    Attributes
    ----------
    is_identifiable : bool
        True if all parameters are identifiable (FIM is full rank).
    fim_rank : int
        Numerical rank of the FIM.
    n_parameters : int
        Total number of unknown parameters.
    condition_number : float
        Condition number of the FIM (physical, unscaled).
    fim_result : FIMResult
        Underlying FIM computation result.
    singular_values : numpy.ndarray
        Singular values of the scaled Jacobian, descending.
    condition_indices : numpy.ndarray
        Belsley condition indices eta_k = sigma_max / sigma_k.
    vif : dict[str, float]
        Variance inflation factor per parameter; ``nan`` if undefined.
    variance_decomposition : numpy.ndarray
        Belsley pi_{jk}, shape ``(n_params, n_params)``. Rows sum to 1.
    correlation_matrix : numpy.ndarray
        Parameter correlation from FIM^-1 (pseudoinverse if singular).
        Entries touching a null direction are ``nan``.
    log_eigenvalue_spectrum : numpy.ndarray
        log10 of FIM eigenvalues, sorted descending.
    normalized_log_spectrum : numpy.ndarray
        log10(lambda_k / lambda_max); the Gutenkunst sloppy-model form.
    null_space : list[dict[str, float]]
        One entry per null direction (sigma_k < tol). Each entry maps
        parameter name to the (sign-normalized) coefficient in the
        right singular vector.
    standard_errors : dict[str, float]
        sqrt(diag(FIM^-1)); ``nan`` for parameters without identifiability.
    warnings : list[str]
        Human-readable flags for problematic diagnostics.
    problematic_parameters : list[str]
        Parameters with the largest component in a null direction
        (one per null direction; for backwards compatibility with
        :class:`IdentifiabilityResult`).
    """

    is_identifiable: bool
    fim_rank: int
    n_parameters: int
    condition_number: float
    fim_result: FIMResult
    singular_values: np.ndarray
    condition_indices: np.ndarray
    vif: dict[str, float]
    variance_decomposition: np.ndarray
    correlation_matrix: np.ndarray
    log_eigenvalue_spectrum: np.ndarray
    normalized_log_spectrum: np.ndarray
    null_space: list[dict[str, float]]
    standard_errors: dict[str, float]
    warnings: list[str]
    problematic_parameters: list[str]


def diagnose_identifiability(
    experiment: Experiment,
    param_values: dict[str, float] | None = None,
    design_values: dict[str, float] | None = None,
    *,
    tol: float | None = None,
    estimation_result=None,
) -> IdentifiabilityDiagnostics:
    """Full identifiability diagnostics (Belsley + Gutenkunst).

    Computes the FIM and the scaled sensitivity Jacobian, then returns
    condition indices, variance-inflation factors, variance-decomposition
    proportions, the correlation matrix, the sloppy-model eigenvalue
    spectrum, and a null-space report.

    The function replaces :func:`check_identifiability` for new code;
    ``check_identifiability`` is kept as a thin wrapper.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float], optional
        Nominal parameter values (typically a fitted estimate). Either
        this or ``estimation_result`` must be supplied. If both are
        supplied, ``param_values`` wins.
    design_values : dict[str, float], optional
        Design input values.
    tol : float, optional
        Relative tolerance on singular values for the rank decision.
        Defaults to the LAPACK convention
        ``max(n_rows, n_params) * eps``.
    estimation_result : EstimationResult, optional
        A fit produced by :func:`discopt.estimate.estimate_parameters`.
        When supplied, its ``parameters`` dict is used as the nominal
        point. Convenience for the common pattern
        ``diagnose_identifiability(exp, estimation_result=res)``.

    Returns
    -------
    IdentifiabilityDiagnostics
        Full diagnostic bundle.
    """
    if param_values is None:
        if estimation_result is None:
            raise TypeError(
                "diagnose_identifiability requires either param_values or estimation_result"
            )
        param_values = dict(estimation_result.parameters)
    fim_result = compute_fim(experiment, param_values, design_values)
    return _diagnostics_from_fim_result(fim_result, tol=tol)


def _diagnostics_from_fim_result(
    fim_result: FIMResult,
    *,
    tol: float | None = None,
) -> IdentifiabilityDiagnostics:
    """Build diagnostics from an existing FIMResult.

    Factored out so both :func:`diagnose_identifiability` and
    :func:`check_identifiability` can use the same linear-algebra path.
    """
    fim = np.asarray(fim_result.fim, dtype=np.float64)
    jac = np.asarray(fim_result.jacobian, dtype=np.float64)
    names = list(fim_result.parameter_names)
    n_params = len(names)

    # Scaled Jacobian: unit-column-length. Columns with zero norm (a
    # parameter with no sensitivity) get a zero column; they will be
    # flagged as non-identifiable by the singular-value test below.
    col_norms = np.linalg.norm(jac, axis=0)
    safe_norms = np.where(col_norms > 0, col_norms, 1.0)
    J_s = jac / safe_norms
    J_s[:, col_norms == 0] = 0.0

    n_rows = max(J_s.shape[0], 1)
    if tol is None:
        tol = max(n_rows, n_params) * np.finfo(np.float64).eps

    # SVD of the scaled Jacobian.
    if J_s.shape[0] == 0:
        sv = np.zeros(n_params)
        Vt = np.eye(n_params)
    else:
        _, sv_raw, Vt = np.linalg.svd(J_s, full_matrices=False)
        sv = np.concatenate([sv_raw, np.zeros(n_params - sv_raw.size)])
        if Vt.shape[0] < n_params:
            # Pad Vt with orthogonal completion rows for missing directions
            extra = np.eye(n_params)[Vt.shape[0] :]
            Vt = np.vstack([Vt, extra])

    sv_max = sv[0] if sv.size and sv[0] > 0 else 0.0
    if sv_max > 0:
        rank = int(np.sum(sv > tol * sv_max))
    else:
        rank = 0

    # Condition indices: sigma_max / sigma_k (infinity for null directions).
    with np.errstate(divide="ignore"):
        condition_indices = np.where(sv > 0, sv_max / np.maximum(sv, np.finfo(float).tiny), np.inf)
    if sv_max == 0:
        condition_indices = np.full(n_params, np.inf)

    # Belsley variance-decomposition proportions.
    # phi_{j,k} = V_{j,k}^2 / sigma_k^2 ; pi_{j,k} = phi_{j,k} / sum_k phi_{j,k}
    V = Vt.T  # columns are right singular vectors
    sv_sq = np.where(sv > 0, sv**2, np.finfo(float).tiny)
    phi = (V**2) / sv_sq[np.newaxis, :]
    row_sums = phi.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    vdp = phi / row_sums

    # VIF via the inverse of the correlation matrix of the unit-column-
    # length Jacobian: VIF_j = [C^-1]_{jj}.
    if J_s.shape[0] > 0:
        C = J_s.T @ J_s  # = correlation matrix since columns are unit length
    else:
        C = np.zeros((n_params, n_params))
    try:
        C_inv = np.linalg.inv(C)
        vif_array = np.diag(C_inv)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C)
        vif_array = np.diag(C_inv)

    # Parameters whose direction is deficient → VIF is effectively infinite.
    # Detect by checking whether each parameter's axis vector lies
    # (almost) in the span of null singular vectors.
    null_indices = np.where(sv <= tol * max(sv_max, np.finfo(float).tiny))[0]
    null_directions = V[:, null_indices] if null_indices.size else np.zeros((n_params, 0))
    if null_directions.size:
        projections = np.sum(null_directions**2, axis=1)  # per parameter
        vif_array = np.where(projections > 0.01, np.inf, vif_array)
    vif = {names[j]: float(vif_array[j]) for j in range(n_params)}

    # FIM-based correlation matrix and standard errors.
    try:
        fim_inv = np.linalg.inv(fim)
        singular_fim = False
    except np.linalg.LinAlgError:
        fim_inv = np.linalg.pinv(fim)
        singular_fim = True

    diag = np.diag(fim_inv)
    se_array = np.where(diag >= 0, np.sqrt(np.clip(diag, 0.0, None)), np.nan)
    if singular_fim and null_directions.size:
        # Parameters with large null-direction projection have no
        # meaningful standard error or correlation.
        projections = np.sum(null_directions**2, axis=1)
        mask = projections > 0.01
        se_array = np.where(mask, np.nan, se_array)
    standard_errors = {names[j]: float(se_array[j]) for j in range(n_params)}

    # Correlation matrix.
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(np.clip(np.diag(fim_inv), 0.0, None))
        safe_d = np.where(d > 0, d, np.nan)
        corr = fim_inv / np.outer(safe_d, safe_d)
    if singular_fim and null_directions.size:
        projections = np.sum(null_directions**2, axis=1)
        affected = projections > 0.01
        corr[affected, :] = np.nan
        corr[:, affected] = np.nan

    # Eigenvalue spectrum of the physical FIM.
    eigvals = np.linalg.eigvalsh(fim)
    eigvals = np.sort(eigvals)[::-1]  # descending
    eig_max = eigvals[0] if eigvals.size and eigvals[0] > 0 else 0.0
    if eig_max > 0:
        clipped = np.clip(eigvals, eig_max * np.finfo(float).eps, None)
        log_spectrum = np.log10(clipped)
        normalized_log = np.log10(clipped / eig_max)
    else:
        log_spectrum = np.full(n_params, -np.inf)
        normalized_log = np.full(n_params, -np.inf)

    # Null-space report.
    null_space: list[dict[str, float]] = []
    problematic: list[str] = []
    for idx in null_indices:
        direction = V[:, idx].copy()
        # Sign normalization: largest-magnitude entry positive.
        max_mag = int(np.argmax(np.abs(direction)))
        if direction[max_mag] < 0:
            direction = -direction
        null_space.append({names[j]: float(direction[j]) for j in range(n_params)})
        problematic.append(names[max_mag])

    # Warnings.
    warnings_out: list[str] = []
    for k in range(n_params):
        eta = condition_indices[k]
        if eta > 30:
            warnings_out.append(
                f"serious collinearity: condition index eta_{k + 1} = {eta:.3g} > 30"
            )
        elif eta > 10 and not np.isinf(eta):
            warnings_out.append(f"mild collinearity: condition index eta_{k + 1} = {eta:.3g} > 10")
    for name, v in vif.items():
        if np.isfinite(v) and v > 10:
            warnings_out.append(f"VIF[{name}] = {v:.3g} > 10")
        elif np.isinf(v):
            warnings_out.append(f"VIF[{name}] is infinite (parameter lies in a null direction)")
    # Correlation warnings on finite entries only.
    for i in range(n_params):
        for j in range(i + 1, n_params):
            rho = corr[i, j]
            if np.isfinite(rho) and abs(rho) > 0.95:
                warnings_out.append(f"|rho[{names[i]},{names[j]}]| = {abs(rho):.3g} > 0.95")

    return IdentifiabilityDiagnostics(
        is_identifiable=(rank == n_params),
        fim_rank=rank,
        n_parameters=n_params,
        condition_number=fim_result.me_optimal,
        fim_result=fim_result,
        singular_values=sv,
        condition_indices=condition_indices,
        vif=vif,
        variance_decomposition=vdp,
        correlation_matrix=corr,
        log_eigenvalue_spectrum=log_spectrum,
        normalized_log_spectrum=normalized_log,
        null_space=null_space,
        standard_errors=standard_errors,
        warnings=warnings_out,
        problematic_parameters=problematic,
    )


def check_identifiability(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    tol: float = 1e-6,
) -> IdentifiabilityResult:
    """Minimal identifiability check (backwards-compatible).

    Computes the FIM and reports its rank plus a representative
    problematic parameter per null direction. For the full Belsley /
    Gutenkunst diagnostic toolkit, use :func:`diagnose_identifiability`.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_values : dict[str, float], optional
        Design input values.
    tol : float, default 1e-6
        Absolute-like tolerance (scaled by the top singular value of FIM)
        used to decide the rank. Kept for backwards compatibility.

    Returns
    -------
    IdentifiabilityResult
        Minimal identifiability assessment.
    """
    fim_result = compute_fim(experiment, param_values, design_values)
    fim = fim_result.fim
    n_params = len(fim_result.parameter_names)

    singular_values = np.linalg.svd(fim, compute_uv=False)
    if singular_values.size and singular_values[0] > 0:
        rank = int(np.sum(singular_values > tol * singular_values[0]))
    else:
        rank = 0

    _, _, Vt = np.linalg.svd(fim)
    problematic: list[str] = []
    for i in range(rank, n_params):
        direction = Vt[i]
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
