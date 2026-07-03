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
from typing import Any, Callable

import numpy as np

from discopt.estimate import Experiment, ExperimentModel

# A parameter axis whose squared projection onto the null-space basis
# exceeds this value is treated as lying *in* the null space — VIF is
# reported as infinite and the FIM-based standard error / correlations
# are masked to NaN. 1% captures "effectively unidentifiable" while
# avoiding spurious flagging from round-off in the right singular
# vectors.
_NULL_PROJECTION_THRESHOLD = 0.01


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


def _design_source_map(em: ExperimentModel) -> dict | None:
    """Classify every model variable as a parameter or a design input.

    Returns ``{id(var): ("param"|"design", name, var)}`` when the model is a
    *pure explicit response model* — it has no constraints and every variable
    is either an unknown parameter or a design input, so the solution point
    ``x*`` is fully determined by the nominal parameters and the fixed design.
    Returns ``None`` when the model has constraints or any other variable (an
    implicit state that genuinely requires a solve).
    """
    from discopt.parametric import variable_slices

    if getattr(em.model, "_constraints", None):
        return None
    src: dict[str, tuple[str, str, Any]] = {}
    for name, var in em.unknown_parameters.items():
        src[var.name] = ("param", name, var)
    for name, var in em.design_inputs.items():
        src[var.name] = ("design", name, var)
    for vname in variable_slices(em.model):
        if vname not in src:
            return None
    return src


def _direct_var_values(
    src_entry: tuple[str, str, Any],
    param_values: dict[str, float],
    design_values: dict[str, float] | None,
) -> np.ndarray | None:
    """Values for one variable at ``x*`` without solving.

    Parameters take their nominal value (clipped to the variable bounds, to
    match the box-constrained least-squares solve they replace); design inputs
    take their fixed design value. Returns ``None`` on any shape mismatch or a
    missing design value, signalling the caller to fall back to the solve.
    """
    kind, name, var = src_entry
    size = int(getattr(var, "size", 1) or 1)
    if kind == "param":
        pv = np.asarray(param_values[name], dtype=np.float64).ravel()
        if pv.size == 1:
            arr = np.full(size, float(pv[0]))
        elif pv.size == size:
            arr = pv.astype(np.float64).copy()
        else:
            return None
        clipped = np.clip(
            arr, np.asarray(var.lb, dtype=np.float64), np.asarray(var.ub, dtype=np.float64)
        )
        return np.asarray(clipped, dtype=np.float64)
    if not design_values or name not in design_values:
        return None
    dv: np.ndarray = np.asarray(design_values[name], dtype=np.float64).ravel()
    if dv.size == 1:
        return np.full(size, float(dv[0]))
    if dv.size == size:
        out: np.ndarray = dv.copy()
        return out
    return None


def _assemble_x_flat_direct(em, param_values, design_values):
    """Assemble the flat solution vector ``x*`` directly, or ``None``.

    Bypasses the QP solve for pure explicit response models (see
    :func:`_design_source_map`). The result is identical to solving
    ``min Σ(θ - θ_nom)²`` with the design fixed, but with no solver call.
    """
    from discopt.parametric import variable_slices

    src = _design_source_map(em)
    if src is None:
        return None
    parts = []
    for vname in variable_slices(em.model):
        arr = _direct_var_values(src[vname], param_values, design_values)
        if arr is None:
            return None
        parts.append(arr)
    import jax.numpy as jnp

    return jnp.array(np.concatenate(parts), dtype=jnp.float64)


def _assemble_x_flat_batch_direct(em, param_values, design_points):
    """Stack ``x*`` for many design points into one ``(B, n)`` array, or ``None``."""
    from discopt.parametric import variable_slices

    src = _design_source_map(em)
    if src is None:
        return None
    var_names = list(variable_slices(em.model))
    rows = []
    for dp in design_points:
        parts = []
        for vname in var_names:
            arr = _direct_var_values(src[vname], param_values, dp)
            if arr is None:
                return None
            parts.append(arr)
        rows.append(np.concatenate(parts))
    import jax.numpy as jnp

    return jnp.asarray(np.stack(rows, axis=0), dtype=jnp.float64)


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

    from discopt.parametric import compile_expression, extract_x_flat, flatten_params

    # Build the model at nominal parameter values
    em = experiment.create_model(**param_values)

    # Fast path: for a pure explicit response model (no constraints; every
    # variable is an unknown parameter or a design input) the solution point
    # x* is fully determined by the nominal parameters and the fixed design.
    # The QP solve below would merely reconstruct values we already know, so
    # assemble x* directly and skip it. ``x_flat`` here is identical (to solver
    # tolerance) to the solved one.
    x_flat = _assemble_x_flat_direct(em, param_values, design_values)

    if x_flat is None:
        # General path: a constrained / implicit-state model genuinely needs a
        # solve to recover x*. Fix the design, then minimise Σ(θ - θ_nom)².
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

        em.model.minimize(
            sum((em.unknown_parameters[n] - param_values[n]) ** 2 for n in em.parameter_names)
        )
        result = em.model.solve()

        x_flat = extract_x_flat(result, em.model)

    # Compile response functions
    response_fns = []
    for name in em.response_names:
        fn = compile_expression(em.responses[name], em.model)
        response_fns.append(fn)

    # Find indices of unknown parameter variables in x_flat
    param_indices = _get_param_indices(em)

    # Build p_flat for any model Parameters (distinct from unknown_parameters)
    p_flat = flatten_params(em.model)

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


def compute_fim_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_points: list[dict[str, float]],
    *,
    prior_fim: np.ndarray | None = None,
    method: str = "autodiff",
    fd_step: float = 1e-5,
) -> list[FIMResult]:
    """Compute the FIM for a batch of design points (multi-RHS).

    Fast path — for a pure explicit response model (no constraints; every
    variable an unknown parameter or design input) the per-point ``x*`` is
    assembled directly (no QP solve) and the response Jacobian is evaluated for
    the whole batch in a single ``vmap`` pass. This is the multi-RHS analogue
    of :func:`compute_fim`: the model is built and the response functions
    compiled once, then reused across every design point.

    Returns one :class:`FIMResult` per design point, in input order. Falls back
    to a per-point :func:`compute_fim` loop for any model that needs a solve or
    a non-autodiff ``method``, so the result is always identical to calling
    :func:`compute_fim` on each point.
    """
    from discopt.parametric import compile_expression, flatten_params

    if not design_points:
        return []

    em = experiment.create_model(**param_values)

    X = None
    if method == "autodiff":
        X = _assemble_x_flat_batch_direct(em, param_values, design_points)
    if X is None:
        return [
            compute_fim(
                experiment, param_values, dp, prior_fim=prior_fim, method=method, fd_step=fd_step
            )
            for dp in design_points
        ]

    import jax
    import jax.numpy as jnp

    response_fns = [compile_expression(em.responses[n], em.model) for n in em.response_names]
    param_indices = _get_param_indices(em)
    p_flat = flatten_params(em.model)

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    # One traced Jacobian, vmapped across the batch axis of x*.
    J_all = np.asarray(jax.vmap(jax.jacobian(response_vector))(X))
    J_all = J_all[:, :, param_indices]

    sigma = np.array([em.measurement_error[name] for name in em.response_names])
    Sigma_inv = np.diag(1.0 / sigma**2)

    results: list[FIMResult] = []
    for b in range(J_all.shape[0]):
        J = J_all[b]
        fim = J.T @ Sigma_inv @ J
        if prior_fim is not None:
            fim = fim + prior_fim
        results.append(
            FIMResult(
                fim=np.asarray(fim),
                jacobian=np.asarray(J),
                parameter_names=em.parameter_names,
                response_names=em.response_names,
            )
        )
    return results


def _make_direct_fim_evaluator(
    experiment: Experiment,
    param_values: dict[str, float],
    *,
    prior_fim: np.ndarray | None = None,
) -> Callable[[dict[str, float] | None], FIMResult] | None:
    """Return a reusable FIM evaluator that compiles the response Jacobian once.

    For a *pure explicit response model* (see :func:`_design_source_map`) the
    model is built and the response Jacobian JIT-compiled a *single* time; the
    returned ``evaluator(design_values) -> FIMResult`` then reuses that compiled
    Jacobian for every design point, assembling ``x*`` directly (no solve). This
    is the single-point analogue of :func:`compute_fim_batch`, intended for the
    adaptive scipy refinement loop in :mod:`discopt.doe.design`, where design
    points are chosen one at a time and the same Jacobian is evaluated many
    times.

    Returns ``None`` when the model has constraints or implicit state (the
    caller falls back to per-call :func:`compute_fim`, which solves). For every
    design point the returned FIM is identical (to float tolerance) to calling
    :func:`compute_fim` on that point — only the per-call model rebuild and JAX
    re-trace are eliminated.
    """
    import jax
    import jax.numpy as jnp

    from discopt.parametric import compile_expression, flatten_params

    em = experiment.create_model(**param_values)
    if _design_source_map(em) is None:
        return None

    response_fns = [compile_expression(em.responses[n], em.model) for n in em.response_names]
    param_indices = _get_param_indices(em)
    p_flat = flatten_params(em.model)

    def response_vector(x_flat_arg):
        return jnp.stack([fn(x_flat_arg, p_flat) for fn in response_fns])

    # Compile the Jacobian once; the JIT cache keys on x*'s (fixed) shape, so
    # every subsequent design point reuses the same compiled trace.
    jac = jax.jit(jax.jacobian(response_vector))
    sigma = np.array([em.measurement_error[name] for name in em.response_names])
    Sigma_inv = np.diag(1.0 / sigma**2)
    param_names = em.parameter_names
    response_names = em.response_names

    def evaluator(design_values: dict[str, float] | None) -> FIMResult:
        x_flat = _assemble_x_flat_direct(em, param_values, design_values)
        if x_flat is None:
            # Per-point shape/missing-design mismatch: fall back to the solve.
            return compute_fim(experiment, param_values, design_values, prior_fim=prior_fim)
        J = np.asarray(jac(x_flat))[:, param_indices]
        fim = J.T @ Sigma_inv @ J
        if prior_fim is not None:
            fim = fim + prior_fim
        return FIMResult(
            fim=np.asarray(fim),
            jacobian=np.asarray(J),
            parameter_names=param_names,
            response_names=response_names,
        )

    return evaluator


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
            # When J_s has fewer rows than columns, SVD returns only
            # rank-m right singular vectors. Complete them to an
            # orthonormal basis of R^{n_params}. A full-mode QR of V
            # (n_params × m) yields Q of shape (n_params, n_params)
            # whose first m columns match V's column space and whose
            # remaining n_params - m columns are an orthonormal basis
            # for the orthogonal complement — the true null space.
            # Using standard-basis rows directly would generally not be
            # orthogonal to the existing Vt.
            Q, _ = np.linalg.qr(Vt.T, mode="complete")
            extra = Q[:, Vt.shape[0] :].T
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
        null_projections = np.sum(null_directions**2, axis=1)  # per parameter
    else:
        null_projections = np.zeros(n_params)
    in_null = null_projections > _NULL_PROJECTION_THRESHOLD
    if null_directions.size:
        vif_array = np.where(in_null, np.inf, vif_array)
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
        se_array = np.where(in_null, np.nan, se_array)
    standard_errors = {names[j]: float(se_array[j]) for j in range(n_params)}

    # Correlation matrix.
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(np.clip(np.diag(fim_inv), 0.0, None))
        safe_d = np.where(d > 0, d, np.nan)
        corr = fim_inv / np.outer(safe_d, safe_d)
    if singular_fim and null_directions.size:
        corr[in_null, :] = np.nan
        corr[:, in_null] = np.nan

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
    from discopt.parametric import variable_slices

    slices = variable_slices(em.model)
    param_indices: list[int] = []
    for var in em.unknown_parameters.values():
        sl = slices[var.name]
        param_indices.extend(range(sl.start, sl.stop))
    return param_indices


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
