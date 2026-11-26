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
discopt.doe : Optimal design of experiments using the same Experiment
    interface, provided by the standalone ``discopt-doe`` plugin
    (``pip install discopt-doe``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Union

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
    active_bounds : dict[str, str]
        Parameters on a bound (``"lower"``, ``"upper"`` or ``"fixed"``). Their
        rows and columns of ``covariance`` are ``nan``: the covariance is the
        inverse of the FIM over the *free* parameters only.
    information : FisherInformation or None
        The full :func:`fisher` result (sensitivities, reduced gradient, ...).
    """

    parameters: dict[str, float]
    covariance: np.ndarray
    fim: np.ndarray
    objective: float
    solve_result: dm.SolveResult
    parameter_names: list[str]
    n_observations: int
    active_bounds: dict[str, str] = field(default_factory=dict)
    information: "FisherInformation | None" = None

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
    fixed_parameters: dict[str, float] | None = None,
    solver_options: dict[str, Any] | None = None,
    hessian: str = "gauss-newton",
) -> EstimationResult:
    """Estimate unknown parameters from experimental data.

    Builds a weighted least-squares NLP and solves it:

    .. math::

        \\min_{\\theta} \\sum_i \\left(\\frac{y_i^{obs} - y_i^{model}(\\theta)}{\\sigma_i}\\right)^2

    The returned ``objective`` is this sum, which equals the *deviance*
    (2 x negative log-likelihood, up to a constant) under Gaussian noise.
    Profile likelihood and chi-square-based tests use this convention.

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
    fixed_parameters : dict[str, float], optional
        Parameters to hold fixed during the estimation. After
        ``create_model``, the variable's lower and upper bounds are both
        set to the supplied value so the solver cannot move it. Used
        by ``discopt.doe.profile_likelihood`` (discopt-doe plugin) and useful
        standalone for sub-model fits and one-at-a-time sensitivity
        studies.
    solver_options : dict, optional
        Options passed to the solver. When omitted, estimation uses a local
        continuous NLP solve (``skip_convex_check=True``) rather than global
        spatial branch-and-bound, which is usually inappropriate for
        nonlinear least-squares fitting.
    hessian : {"gauss-newton", "observed"}
        Information matrix passed to :func:`fisher`: the expected (Gauss-Newton)
        ``Jᵀ W J`` or the observed ``½ ∇²`` of the fitted objective.

    Returns
    -------
    EstimationResult
        Estimated parameters, covariance, FIM, and diagnostics.

    Raises
    ------
    ValueError
        If data keys don't match response names, or if the solve fails.
    KeyError
        If a name in ``fixed_parameters`` is not an unknown parameter.
    """
    kwargs = dict(initial_guess or {})
    if fixed_parameters:
        # Fixed values take precedence over initial guesses for the same
        # parameter, so create_model starts at the fixed value.
        kwargs.update({k: float(v) for k, v in fixed_parameters.items()})
    em = experiment.create_model(**kwargs)

    if fixed_parameters:
        for name, value in fixed_parameters.items():
            if name not in em.unknown_parameters:
                raise KeyError(
                    f"{name!r} is not an unknown parameter (known: {list(em.unknown_parameters)})"
                )
            var = em.unknown_parameters[name]
            val_arr = np.asarray(float(value), dtype=np.float64)
            if var.shape:
                val_arr = np.full(var.shape, val_arr)
            var.lb = val_arr
            var.ub = val_arr

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
            # Repeated observations of the same response (e.g. accumulated
            # by sequential_doe across rounds). Each element is an
            # independent measurement and contributes its own residual.
            for val in y_obs.flat:
                residual_terms.append(((float(val) - y_model) / sigma) ** 2)
                n_obs += 1

    if not residual_terms:
        raise ValueError("No data matched any response names")

    em.model.minimize(dm.sum(residual_terms))

    # Solve. Parameter estimation is a local nonlinear least-squares workflow;
    # callers that need certified global optimization can override this.
    effective_solver_options: dict[str, Any] = {"skip_convex_check": True}
    effective_solver_options.update(solver_options or {})
    solve_out = em.model.solve(**effective_solver_options)
    # solve() returns SolveResult (not streaming iterator) by default
    result: dm.SolveResult = solve_out  # type: ignore[assignment]

    # ``local_optimal`` is what a local solver (``solver="pounce"``/``"ipopt"``)
    # reports at a verified KKT point -- the same kind of answer as this function's
    # own default local solve, so refusing it made every local-solver call raise
    # (#1617). Any other status has no point to estimate from.
    if result.status not in ("optimal", "feasible", "local_optimal"):
        raise ValueError(f"Estimation solve failed with status: {result.status}")

    # Extract estimated parameters
    params = {}
    for name, var in em.unknown_parameters.items():
        raw = np.asarray(result.value(var))
        params[name] = float(raw.flat[0])

    # Information at the solution. Per-response replication counts come from the
    # observed data so that repeated measurements (e.g. accumulated across
    # sequential_doe rounds) contribute proportionally; a response with no data
    # gets weight zero. One implementation with :func:`fisher` (#2367): sparse,
    # JAX-free, and bound-aware -- a parameter on its bound gets no Wald interval.
    n_reps = {
        name: (max(int(np.atleast_1d(np.asarray(data[name])).size), 1) if name in data else 0)
        for name in em.response_names
    }
    info = fisher(
        result,
        em.unknown_parameters,
        em.responses,
        sigma=em.measurement_error,
        model=em.model,
        hessian=hessian,
        n_reps=n_reps,
    )
    fim = info.fim
    covariance = info.covariance

    return EstimationResult(
        parameters=params,
        covariance=covariance,
        fim=fim,
        objective=result.objective or 0.0,
        solve_result=result,
        parameter_names=em.parameter_names,
        n_observations=n_obs,
        active_bounds=dict(info.active_bounds),
        information=info,
    )


# ─────────────────────────────────────────────────────────────
# Fisher information and parameter sensitivities (#2367)
# ─────────────────────────────────────────────────────────────

# Tolerance for deciding that an inequality constraint is active at the
# solution. An active inequality binds the states, so the equality-only
# implicit-function argument in :func:`fisher` no longer characterizes
# dz/dθ and the sensitivity is refused rather than silently computed from the
# equalities alone.
#
# This is the house absolute constraint tolerance (1e-6), not something
# tighter: a local NLP solve routinely stops a few times 1e-6 *inside* a
# constraint it is genuinely pressed against. Measured on the regression
# model in ``TestConstrainedModelFIM``, a binding ``z[0] <= 2`` settles at a
# residual of -4.4e-06. Judging that slack would classify a bound state as
# free and hand back exactly the wrong Jacobian, so the test errs toward
# refusing.
_ACTIVE_INEQUALITY_TOL = 1e-6

# Above this condition number the state Jacobian ∂g/∂z is solved, but the
# resulting sensitivities carry few correct digits, so the caller is warned.
_ILL_CONDITIONED_STATE_JACOBIAN = 1e12

# Below this many states the condition number of ∂g/∂z is computed exactly
# (dense SVD); above it, it is the 1-norm estimate from the sparse LU.
_DENSE_CONDITION_MAX = 500

_HESSIAN_MODES = ("gauss-newton", "observed")


@dataclass
class FisherInformation:
    """Fisher information, sensitivities and covariance at one point of a model.

    Returned by :func:`fisher`. Every matrix is ordered by
    :attr:`parameter_names` (one entry per scalar parameter, array parameters
    flattened C-order as ``name[i]``).

    Attributes
    ----------
    parameter_names : list[str]
        Scalar parameter labels.
    parameter_values : numpy.ndarray
        Parameter values at the point, shape ``(p,)``.
    jacobian : numpy.ndarray
        Total derivative ``dy/dθ`` of the (flattened) responses, shape
        ``(n_responses, p)``. It follows the states along the equality
        constraints (see :func:`fisher`).
    sensitivities : numpy.ndarray
        Total derivative ``dx/dθ`` of every model variable, shape ``(n_x, p)``,
        in the model's flat variable order. Parameter rows are the identity,
        pinned (``lb == ub``) rows are zero.
    fim : numpy.ndarray
        The information matrix, shape ``(p, p)``: ``Jᵀ W J`` for
        ``hessian="gauss-newton"``, ``objective_scale · ∇²φ(θ)`` (the reduced
        Hessian of the model's objective) for ``hessian="observed"``.
    covariance : numpy.ndarray
        ``inv(fim)`` on the *free* parameters. Rows and columns of parameters in
        :attr:`active_bounds` are ``nan``: a Wald interval from the full
        inverse is not valid for a parameter sitting on its bound.
    active_bounds : dict[str, str]
        ``{name: "lower" | "upper" | "fixed"}`` for every parameter on a bound.
    free : numpy.ndarray
        Boolean mask, shape ``(p,)``: the parameters the covariance covers.
    rank : int
        Numerical rank of the free block of the FIM. When it is below
        ``free.sum()`` the free block is singular, a warning was raised, and the
        covariance is the pseudo-inverse.
    hessian : str
        ``"gauss-newton"`` or ``"observed"``.
    reduced_gradient : numpy.ndarray or None
        ``dφ/dθ``, the gradient of the model's (minimized) objective along the
        constraints, shape ``(p,)``. At a local minimum this is the bound
        multiplier of each active parameter bound (positive at a lower bound,
        negative at an upper one) and zero for a free parameter, so it stands in
        for bound duals the NLP solver withheld. ``None`` when the model has no
        objective.
    response_values : numpy.ndarray
        The flattened responses at the point.
    state_jacobian_condition : float or None
        Condition number of ∂g/∂z (exact for small systems, the 1-norm estimate
        otherwise); ``None`` when there are no states.
    """

    parameter_names: list[str]
    parameter_values: np.ndarray
    jacobian: np.ndarray
    sensitivities: np.ndarray
    fim: np.ndarray
    covariance: np.ndarray
    active_bounds: dict[str, str]
    free: np.ndarray
    rank: int
    hessian: str
    reduced_gradient: np.ndarray | None
    response_values: np.ndarray
    state_jacobian_condition: float | None

    @property
    def standard_errors(self) -> dict[str, float]:
        """Standard error per parameter (``nan`` for a parameter on a bound)."""
        se = np.sqrt(np.diag(self.covariance))
        return dict(zip(self.parameter_names, (float(s) for s in se)))


def _flat_layout(model: Model) -> tuple[dict[int, slice], np.ndarray, np.ndarray]:
    """``({id(var): slice}, lb, ub)`` over the model's flat variable vector."""
    slices: dict[int, slice] = {}
    lbs, ubs = [], []
    offset = 0
    for v in model._variables:
        size = int(v.size)
        slices[id(v)] = slice(offset, offset + size)
        lbs.append(np.broadcast_to(np.asarray(v.lb, dtype=np.float64).ravel(), (size,)))
        ubs.append(np.broadcast_to(np.asarray(v.ub, dtype=np.float64).ravel(), (size,)))
        offset += size
    if not lbs:
        return slices, np.zeros(0), np.zeros(0)
    return slices, np.concatenate(lbs), np.concatenate(ubs)


def _point_vector(point: Any, model: Model) -> np.ndarray:
    """The flat ``x`` of a SolveResult, a ``{name: value}`` dict, or a flat array."""
    if isinstance(point, dm.SolveResult):
        if point.x is None:
            raise ValueError("The SolveResult has no solution vector (result.x is None).")
        values = point.x
    elif isinstance(point, dict):
        values = point
    else:
        x = np.asarray(point, dtype=np.float64).ravel()
        n = sum(int(v.size) for v in model._variables)
        if x.size != n:
            raise ValueError(f"point has {x.size} entries; the model has {n} variable columns")
        return x
    parts = []
    for v in model._variables:
        if v.name not in values:
            raise ValueError(f"point has no value for variable {v.name!r}")
        parts.append(
            np.broadcast_to(np.asarray(values[v.name], dtype=np.float64).ravel(), (int(v.size),))
        )
    return np.concatenate(parts) if parts else np.zeros(0)


def _normalize_responses(responses: Any) -> tuple[list[str], list[Expression]]:
    if isinstance(responses, dict):
        return list(responses.keys()), list(responses.values())
    if isinstance(responses, Expression):
        return ["y"], [responses]
    items = list(responses)
    return [f"y{i}" for i in range(len(items))], items


def _normalize_parameters(parameters: Any) -> list[Variable]:
    if isinstance(parameters, Variable):
        params = [parameters]
    elif isinstance(parameters, dict):
        params = list(parameters.values())
    else:
        params = list(parameters)
    for p in params:
        if not isinstance(p, Variable):
            raise TypeError(
                f"parameters must be Variables of the model; got {type(p).__name__}. "
                "Pass the whole Variable (an array Variable contributes one parameter "
                "per element)."
            )
    return params


def _per_response(value: Any, names: list[str], sizes: list[int], what: str) -> np.ndarray:
    """Broadcast ``value`` (scalar, flat array, or per-response dict) to the flat responses."""
    total = sum(sizes)
    if isinstance(value, dict):
        missing = set(names) - set(value)
        if missing:
            raise ValueError(f"{what} has no entry for response(s) {sorted(missing)}")
        parts = [
            np.broadcast_to(np.asarray(value[n], dtype=np.float64).ravel(), (s,))
            for n, s in zip(names, sizes)
        ]
        return np.concatenate(parts) if parts else np.zeros(0)
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(total, float(arr[0]))
    if arr.size != total:
        raise ValueError(f"{what} has {arr.size} entries; the responses flatten to {total}")
    return arr


def _response_jacobian(model: Model, exprs: list[Expression], x: np.ndarray, lb, ub):
    """``(values, sparse dy/dx, sizes)`` of the responses at ``x``.

    The POUNCE tape when it represents every response (no JAX), otherwise the
    legacy JAX compiler — the same fallback the NLP evaluator takes.
    """
    import scipy.sparse as sp

    from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_array
    from discopt._tape_nlp_evaluator import pounce_usable

    n = x.size
    if pounce_usable():
        try:
            arrays = [compile_to_nl_array(e, model).reshape(-1) for e in exprs]
        except UnsupportedForTape:
            arrays = None
        if arrays is not None:
            import pounce

            sizes = [int(a.size) for a in arrays]
            rows = [node for a in arrays for node in a.tolist()]
            if not rows:
                return np.zeros(0), sp.csr_matrix((0, n)), sizes
            prob = pounce.build_nl_problem(
                n,
                pounce.NlExpr.const_(0.0),
                constraints=rows,
                x_l=np.asarray(lb, dtype=np.float64),
                x_u=np.asarray(ub, dtype=np.float64),
            )
            r, c = prob.jacobian_structure()
            vals = np.asarray(prob.jacobian(x), dtype=np.float64)
            jac = sp.coo_matrix(
                (vals, (np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64))),
                shape=(len(rows), n),
            ).tocsr()
            values = np.asarray(prob.constraints(x), dtype=np.float64)
            return values, jac, sizes

    from discopt._optional_jax import require_jax

    require_jax(
        "discopt.estimate.fisher for a response the POUNCE tape cannot represent "
        "(dm.custom, dm.external, ...)"
    )
    import jax

    from discopt.parametric import compile_expression, flatten_params

    p_flat = flatten_params(model)
    values_parts, jac_parts, sizes = [], [], []
    for e in exprs:
        fn = compile_expression(e, model)

        def flat_fn(xa, fn=fn):
            return jax.numpy.ravel(fn(xa, p_flat))

        v = np.asarray(flat_fn(x), dtype=np.float64).ravel()
        J = np.asarray(jax.jacobian(flat_fn)(x), dtype=np.float64).reshape(v.size, n)
        values_parts.append(v)
        jac_parts.append(sp.csr_matrix(J))
        sizes.append(int(v.size))
    values = np.concatenate(values_parts) if values_parts else np.zeros(0)
    jac = sp.vstack(jac_parts, format="csr") if jac_parts else sp.csr_matrix((0, n))
    return values, jac, sizes


def _exact_hessian_evaluator(model: Model, ev: Any) -> Any:
    """``ev``, or a fresh evaluator with the exact objective Hessian if ``ev`` is Gauss-Newton.

    ``hessian="observed"`` is the *exact* curvature; a Gauss-Newton evaluator
    (``Model._gauss_newton_hessian``) would hand back ``2 JᵀJ`` for the
    objective and silently turn the observed information into the expected one.
    """
    attr = getattr(ev, "is_gauss_newton", False)
    is_gn = bool(attr() if callable(attr) else attr)
    if not is_gn:
        return ev
    # Through the one evaluator funnel (#1063), with the flag cleared for the
    # build: both backends read it at construction, and it is part of the
    # evaluator-cache fingerprint, so this returns the exact-Hessian evaluator
    # cached under the flag-off key, never the Gauss-Newton one.
    from discopt._tape_nlp_evaluator import make_evaluator

    saved = getattr(model, "_gauss_newton_hessian", False)
    model._gauss_newton_hessian = False
    try:
        exact = make_evaluator(model)
    finally:
        model._gauss_newton_hessian = saved
    attr = getattr(exact, "is_gauss_newton", False)
    if bool(attr() if callable(attr) else attr):
        raise AssertionError(
            "make_evaluator returned a Gauss-Newton evaluator with the flag cleared; "
            "the observed information would be the expected one"
        )
    return exact


def _sparse_lagrangian_hessian(ev: Any, x: np.ndarray, obj_factor: float, lam: np.ndarray):
    """The full symmetric Lagrangian Hessian as CSR, from the lower-triangle COO."""
    import scipy.sparse as sp

    n = x.size
    rows, cols = ev.hessian_structure()
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(ev.evaluate_hessian_values(x, obj_factor, lam), dtype=np.float64)
    lower = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    strict = sp.tril(lower, k=-1)
    return (lower + strict.T).tocsr()


def _condition(G_z, lu) -> float:
    """Condition number of the square sparse ``G_z`` (exact when small)."""
    from scipy.sparse.linalg import LinearOperator, onenormest

    n = G_z.shape[0]
    if n <= _DENSE_CONDITION_MAX:
        return float(np.linalg.cond(G_z.toarray()))
    inv_op = LinearOperator(
        (n, n),
        matvec=lambda b: lu.solve(np.asarray(b, dtype=np.float64).ravel()),
        rmatvec=lambda b: lu.solve(np.asarray(b, dtype=np.float64).ravel(), trans="T"),
        dtype=np.float64,
    )
    return float(onenormest(G_z) * onenormest(inv_op))


def fisher(
    point: Any,
    parameters: Any,
    responses: Any,
    sigma: Any = 1.0,
    *,
    model: Model | None = None,
    hessian: str = "gauss-newton",
    n_reps: Any = None,
    objective_scale: float = 0.5,
    bound_tol: float = 1e-6,
) -> FisherInformation:
    r"""Fisher information, ``dy/dθ`` and a bound-aware covariance at a model point.

    Works on any discopt model -- a hand-built one, a :class:`~discopt.dae.DAEBuilder`
    or :class:`~discopt.dae.MOLBuilder` transcription, or an
    :class:`ExperimentModel` -- after ``model.solve()`` (or at any feasible point
    you supply). The unknown parameters are model ``Variable``\ s; every other
    column that is not pinned at ``lb == ub`` is a *state*, which the model's
    equality constraints ``g(θ, z) = 0`` must determine. The responses follow θ
    through those states, so the derivative is the total one,

    .. math::
        \frac{dy}{d\theta} = \frac{\partial y}{\partial x}\,Z, \qquad
        Z = \frac{dx}{d\theta} = \begin{bmatrix} I \\ -G_z^{-1} G_\theta \\ 0
        \end{bmatrix}

    (parameters, states, pinned columns). ``G = ∂g/∂x`` comes from the tape's
    sparse Jacobian and ``G_z`` is factored once with a sparse LU
    (``scipy.sparse.linalg.splu``); no dense ``(m, n)`` matrix is ever formed,
    so a collocation or method-of-lines fit with ~10⁵-10⁶ rows is fine. No JAX
    is needed unless a response contains an expression the tape cannot represent.

    Parameters
    ----------
    point : SolveResult, dict, or numpy.ndarray
        Where to evaluate: a solve result (its model is used unless ``model`` is
        given), a ``{variable_name: value}`` dict, or the flat variable vector.
    parameters : Variable, list of Variable, or dict of str to Variable
        The unknown parameters θ.
    responses : Expression, list of Expression, or dict of str to Expression
        The measured predictions. Array-valued expressions are flattened.
    sigma : float, array, or dict, default 1.0
        Measurement standard deviation, broadcast to the flattened responses
        (a dict is keyed like ``responses``). Used for ``"gauss-newton"``.
    model : Model, optional
        Required when ``point`` is not a SolveResult.
    hessian : {"gauss-newton", "observed"}
        ``"gauss-newton"`` (expected information) is ``Jᵀ W J`` with
        ``W = diag(n_reps / sigma²)``. ``"observed"`` is
        ``objective_scale · ∇²φ(θ)``: the exact Hessian of the model's minimized
        objective along the constraints, ``Zᵀ ∇²ₓₓL Z`` with
        ``L = f + λᵀg`` and the multipliers from stationarity in the states,
        ``G_zᵀ λ = -∇_z f`` (so solver duals are not needed). It is exact at
        any feasible point, not only at the optimum.
    n_reps : float, array, or dict, optional
        Per-response replication counts multiplying the weights.
    objective_scale : float, default 0.5
        Factor from the objective's Hessian to the information. The default
        matches the deviance ``Σ ((y_obs - y)/σ)²`` that
        :func:`estimate_parameters` minimizes (``-2 log L`` up to a constant);
        use ``1.0`` if the objective is the negative log-likelihood itself.
    bound_tol : float, default 1e-6
        A parameter within ``bound_tol · max(1, |bound|)`` of a bound is on it.

    Returns
    -------
    FisherInformation

    Raises
    ------
    ValueError
        If an inequality constraint is active, if the equality system is not
        square in the free states, or if ``∂g/∂z`` is singular: in each case
        ``dz/dθ`` is not determined by the equalities, and a partial derivative
        would be a silently wrong covariance rather than an approximate one.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    if hessian not in _HESSIAN_MODES:
        raise ValueError(f"hessian must be one of {_HESSIAN_MODES}; got {hessian!r}")
    if model is None:
        model = getattr(point, "_model", None)
        if model is None:
            raise ValueError("model= is required when point is not a solved SolveResult")

    params = _normalize_parameters(parameters)
    resp_names, resp_exprs = _normalize_responses(responses)
    slices, lb, ub = _flat_layout(model)
    x = _point_vector(point, model)
    n_x = x.size

    # Parameter columns and names.
    param_cols: list[int] = []
    param_names: list[str] = []
    for var in params:
        sl = slices.get(id(var))
        if sl is None:
            raise ValueError(f"parameter {var.name!r} is not a variable of this model")
        param_cols.extend(range(sl.start, sl.stop))
        if var.shape == ():
            param_names.append(var.name)
        else:
            param_names.extend(f"{var.name}[{i}]" for i in range(int(var.size)))
    if len(set(param_cols)) != len(param_cols):
        raise ValueError("a parameter Variable was passed more than once")
    P = np.asarray(param_cols, dtype=np.int64)
    p = P.size

    # Every column of x is exactly one of three things, and conflating the last
    # two is a bug that refuses well-posed models:
    #
    #   1. an unknown parameter θ            -> the columns we differentiate for
    #   2. a *pinned* variable (lb == ub)    -> data, not a state
    #   3. everything else                   -> a state z, which the equalities
    #                                           must determine from θ
    #
    # Case 2 is the design-input case: a multi-experiment campaign pins each
    # block's conditions with ``Variable.fix``. d(pinned)/dθ = 0, but counting
    # them as states would make the squareness test refuse a model whose actual
    # state system is square. The classification is on the box, so any fixed
    # column is handled by the same rule.
    pinned = lb == ub
    is_param = np.zeros(n_x, dtype=bool)
    is_param[P] = True
    S = np.flatnonzero(~is_param & ~pinned)

    # Bound activity of the parameters.
    theta = x[P]
    lo, hi = lb[P], ub[P]
    active_bounds: dict[str, str] = {}
    free = np.ones(p, dtype=bool)
    for i, name in enumerate(param_names):
        if lo[i] == hi[i]:
            active_bounds[name] = "fixed"
        elif np.isfinite(lo[i]) and theta[i] - lo[i] <= bound_tol * max(1.0, abs(lo[i])):
            active_bounds[name] = "lower"
        elif np.isfinite(hi[i]) and hi[i] - theta[i] <= bound_tol * max(1.0, abs(hi[i])):
            active_bounds[name] = "upper"
        else:
            continue
        free[i] = False

    # Constraint rows: classify equalities, refuse an active inequality.
    from discopt.solver import _make_evaluator

    has_objective = model._objective is not None
    ev = _make_evaluator(model) if has_objective else None
    if ev is None and (hessian == "observed" or model._constraints):
        raise ValueError(
            "fisher needs the model's objective to build its evaluator (and, for "
            "hessian='observed', to differentiate); set one with model.minimize()."
        )

    eq_rows: list[int] = []
    if ev is not None and ev.n_constraints:
        g = np.asarray(ev.evaluate_constraints(x), dtype=np.float64)
        active: list[str] = []
        for start, stop, con in ev.constraint_row_map():
            if con.sense == "==":
                eq_rows.extend(range(start, stop))
                continue
            residual = g[start:stop] - float(con.rhs)
            if con.sense == "<=":
                is_active = residual >= -_ACTIVE_INEQUALITY_TOL
            elif con.sense == ">=":
                is_active = residual <= _ACTIVE_INEQUALITY_TOL
            else:
                raise ValueError(
                    f"Constraint {con.name!r} has unrecognized sense {con.sense!r}; "
                    "cannot classify it for the parameter sensitivities."
                )
            if bool(np.any(is_active)):
                active.append(str(con.name))
        if active:
            raise ValueError(
                "Cannot compute exact parameter sensitivities: the inequality "
                f"constraint(s) {sorted(active)} are active at this point. An active "
                "inequality binds the state variables, so dz/dθ is not determined "
                "by the equality constraints alone and the Fisher Information "
                "Matrix would be wrong. Reformulate the active inequality as an "
                "equality, or relax it so it is slack at the solution."
            )

    # Z = dx/dθ.
    Z = np.zeros((n_x, p), dtype=np.float64)
    Z[P, np.arange(p)] = 1.0
    lu = None
    condition: float | None = None
    G = None
    if eq_rows:
        assert ev is not None
        G_all = ev.evaluate_sparse_jacobian(x)
        G = sp.csr_matrix(G_all)[np.asarray(eq_rows, dtype=np.int64)]
        n_eq = G.shape[0]
        if n_eq != S.size:
            raise ValueError(
                "Cannot compute exact parameter sensitivities: the model has "
                f"{n_eq} equality constraint row(s) but {S.size} free state "
                "variable(s), so the states are not determined by the constraints "
                "given the parameters. The Fisher Information Matrix needs a "
                "square, solvable state system; add the missing equations, or pin "
                f"the variables no equation determines. ({int(pinned.sum())} "
                "column(s) are pinned at lb == ub and were counted as data rather "
                "than states.)"
            )
        Gc = G.tocsc()
        G_z = Gc[:, S]
        G_theta = Gc[:, P].toarray()
        try:
            lu = splu(G_z.tocsc())
        except RuntimeError as exc:
            raise ValueError(
                "Cannot compute exact parameter sensitivities: the state Jacobian "
                "dg/dz is singular at this point, so dz/dtheta is not defined. The "
                "model does not determine its states uniquely from the parameters "
                "here."
            ) from exc
        dz = lu.solve(-G_theta) if p else np.zeros((S.size, 0))
        if not np.all(np.isfinite(dz)):
            raise ValueError(
                "Cannot compute exact parameter sensitivities: solving dg/dz for "
                "dz/dtheta produced non-finite values, so dg/dz is numerically "
                "singular at this point."
            )
        Z[S] = dz
        condition = _condition(G_z, lu)
        if condition > _ILL_CONDITIONED_STATE_JACOBIAN:
            warnings.warn(
                "State Jacobian dg/dz is ill-conditioned at this point (condition "
                f"number {condition:.3e}). The parameter sensitivities, and the "
                "covariance and standard errors derived from them, may carry few "
                "correct digits.",
                RuntimeWarning,
                stacklevel=2,
            )
    # With no equality rows (no constraints, or only inactive inequalities)
    # nothing locally determines the states, so their rows of Z stay zero: the
    # partial derivative is the total one.

    # Responses and their total Jacobian.
    y, J_y, sizes = _response_jacobian(model, resp_exprs, x, lb, ub)
    J = np.asarray(J_y @ Z, dtype=np.float64)

    # Multipliers from stationarity in the states, and the reduced gradient.
    reduced_gradient: np.ndarray | None = None
    lam_full: np.ndarray | None = None
    if ev is not None:
        grad = np.asarray(ev.evaluate_gradient(x), dtype=np.float64)
        lam_full = np.zeros(int(ev.n_constraints), dtype=np.float64)
        if lu is not None:
            lam_eq = lu.solve(-grad[S], trans="T")
            lam_full[np.asarray(eq_rows, dtype=np.int64)] = lam_eq
        reduced_gradient = np.asarray(Z.T @ grad, dtype=np.float64)

    if hessian == "gauss-newton":
        weights = 1.0 / _per_response(sigma, resp_names, sizes, "sigma") ** 2
        if n_reps is not None:
            weights = weights * _per_response(n_reps, resp_names, sizes, "n_reps")
        fim = np.asarray(J.T @ (weights[:, None] * J), dtype=np.float64)
    else:
        assert ev is not None and lam_full is not None
        ev_exact = _exact_hessian_evaluator(model, ev)
        H = _sparse_lagrangian_hessian(ev_exact, x, 1.0, lam_full)
        fim = float(objective_scale) * np.asarray(Z.T @ (H @ Z), dtype=np.float64)
        fim = 0.5 * (fim + fim.T)

    covariance = np.full((p, p), np.nan, dtype=np.float64)
    idx = np.flatnonzero(free)
    rank = 0
    if idx.size:
        F_ff = fim[np.ix_(idx, idx)]
        rank = int(np.linalg.matrix_rank(F_ff, hermitian=True))
        if rank < idx.size:
            warnings.warn(
                f"The Fisher information of the {idx.size} free parameter(s) is "
                f"singular (numerical rank {rank}); some parameter combination is "
                "not identifiable from these responses. The covariance is the "
                "pseudo-inverse and its standard errors are not meaningful along "
                "the unidentified directions.",
                RuntimeWarning,
                stacklevel=2,
            )
            cov_ff = np.linalg.pinv(F_ff, hermitian=True)
        else:
            cov_ff = np.linalg.inv(F_ff)
        covariance[np.ix_(idx, idx)] = cov_ff

    return FisherInformation(
        parameter_names=param_names,
        parameter_values=np.asarray(theta, dtype=np.float64).copy(),
        jacobian=J,
        sensitivities=Z,
        fim=fim,
        covariance=covariance,
        active_bounds=active_bounds,
        free=free,
        rank=rank,
        hessian=hessian,
        reduced_gradient=reduced_gradient,
        response_values=y,
        state_jacobian_condition=condition,
    )
