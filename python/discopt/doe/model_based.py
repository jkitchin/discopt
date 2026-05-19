"""Model-based active-learning optimization.

When to use this module
-----------------------

Use :func:`model_based_optimize_round` when:

* you already have a **mechanistic / parametric model** of the response
  -- a kinetic rate law, a heat-transfer correlation, a competitive
  isotherm -- expressed as a :class:`discopt.estimate.Experiment`;
* you trust the functional form enough that you want to extrapolate, or
* runs are expensive enough that you cannot afford the data a GP needs
  to learn the shape on its own.

Contrast with :func:`discopt.doe.optimize_round` -- which uses an
*empirical* surrogate (GP, response-surface polynomial, LPR) that knows
nothing about the underlying physics. The empirical version is the right
default when you have no model in hand; the model-based version is the
right tool when you do.

Contrast with :func:`discopt.doe.optimal_experiment` -- which uses a
parametric model too, but optimizes the experimental design for
**parameter precision** (D-/A-/E-optimality on the FIM), not for the
best *response*. This module is the response-optimization counterpart
that reuses the same FIM machinery for uncertainty quantification.

How one round works
-------------------

A single call to :func:`model_based_optimize_round`:

1. opens the workbook and reads the completed runs;
2. refits the experiment's unknown parameters by maximum likelihood
   (Gauss-Newton via :func:`scipy.optimize.least_squares` with a
   JAX-computed Jacobian);
3. computes the parameter covariance ``Σ_θ ≈ FIM⁻¹`` at the fitted
   point;
4. draws a pool of ``n_candidates`` Sobol points inside the input box;
5. for each candidate ``d``, predicts the response mean
   ``μ(d) = f(d; θ̂)`` and the linearized predictive variance

       σ²(d) = ∇θ f(d; θ̂) · Σ_θ · ∇θ f(d; θ̂)ᵀ + σ²_meas

   -- the first term is *parameter uncertainty* (shrinks as more data
   comes in), the second is *measurement noise* (constant, set by the
   workbook's declared σ);
6. scores candidates with the chosen acquisition function (same EI/UCB
   used by :func:`optimize_round`);
7. greedily picks the top ``batch_size``, using mean-imputation fantasy
   refits to diversify within a batch;
8. appends the batch to the workbook as pending runs.

The acquisition functions consume only ``(mean, std)`` and do not care
how those came to be -- so EI on a GP and EI on a mechanistic model
look identical at the boundary.

Tradeoffs vs. an empirical surrogate
------------------------------------

============================  =======================  =======================
Property                      Empirical (GP / RSM)     Model-based (this)
============================  =======================  =======================
Functional form               learned from data        supplied by user
Extrapolation                 unsafe                   safe *iff* model right
Sample efficiency             ~ 5-20 d points          ~ 1-2 d points
Per-round cost                surrogate fit            param-estimation NLP
Categorical inputs            tricky                   natural (model decides)
Wrong-model risk              none                     systematic bias
============================  =======================  =======================

If you cannot decide -- run both in parallel; the agreement (or lack
of it) is itself a diagnostic.

Quick start
-----------

>>> from discopt.doe import model_based_optimize_round, OptimizationCriterion
>>> from my_kinetics import MichaelisMenten   # an Experiment subclass
>>> result = model_based_optimize_round(
...     workbook="reactor.xlsx",
...     experiment=MichaelisMenten(),
...     initial_guess={"Vmax": 1.0, "Km": 0.5},
...     criterion=OptimizationCriterion.MAXIMIZE,
...     acquisition="expected_improvement",
...     batch_size=4,
... )
>>> print(result.parameters)
>>> print(result.next_designs)

References
----------

* Asprey, Macchietto (2000). *Statistical tools for optimal dynamic
  model building.* C&CE 24:1261-1267. -- model-based DoE foundations.
* Franceschini, Macchietto (2008). *Model-based design of experiments
  for parameter precision: state of the art.* Chem.Eng.Sci. 63:4846. --
  survey including the response-optimization variant used here.
* Jones, Schonlau, Welch (1998). *Efficient Global Optimization of
  Expensive Black-Box Functions.* -- the EI machinery this module reuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from discopt.doe.acquisition import resolve_acquisition
from discopt.doe.optimize import OptimizationCriterion, _sample_candidates
from discopt.doe.workbook import Workbook
from discopt.estimate import Experiment


@dataclass
class ModelBasedRoundResult:
    """Outcome of one model-based active-learning round.

    Same shape as :class:`~discopt.doe.OptimizationRoundResult` plus
    fitted-parameter diagnostics.
    """

    next_designs: list[dict[str, float]]
    new_run_ids: list[int]
    incumbent_x: dict[str, float] | None
    incumbent_y: float | None
    acquisition_scores: list[float]
    surrogate_mode: str | None
    n_completed: int
    workbook_path: str
    parameters: dict[str, float] = field(default_factory=dict)
    parameter_se: dict[str, float] = field(default_factory=dict)
    fim_log_det: float | None = None
    log: list[str] = field(default_factory=list)


class ParametricSurrogate:
    """Mechanistic-model surrogate for active-learning optimization.

    Implements the :class:`~discopt.doe.surrogate.Surrogate` protocol on
    top of a :class:`~discopt.estimate.Experiment`. ``fit(X, y)`` refits
    the unknown parameters by maximum likelihood; ``predict(X)`` returns
    ``(μ, σ)`` where σ comes from linearizing the response in the
    parameters around the fit and adding measurement noise in
    quadrature.

    The class attribute ``_is_discopt_surrogate = True`` tells
    :func:`~discopt.doe.surrogate.coerce_surrogate` not to re-wrap this
    in the sklearn adapter.

    Parameters
    ----------
    experiment : Experiment
        Parametric model.
    input_names : sequence of str
        Names of the design-input columns in the X matrix, in column
        order. Each name must appear in ``experiment.design_inputs``.
    response_name : str
        Key in ``experiment.responses`` whose value is the predicted
        scalar response.
    initial_guess : dict[str, float], optional
        Starting values for the parameter NLS. Defaults to 1.0 for any
        parameter not provided.
    measurement_noise_var : float
        ``σ²_meas`` -- variance of the measurement noise. Added in
        quadrature to the linearized predictive variance.
    regularize : float, default 1e-12
        Ridge added to the FIM before inversion to keep the covariance
        well-defined in the early rounds.
    """

    _is_discopt_surrogate = True

    def __init__(
        self,
        experiment: Experiment,
        *,
        input_names: Sequence[str],
        response_name: str,
        initial_guess: dict[str, float] | None = None,
        measurement_noise_var: float = 1.0,
        regularize: float = 1e-12,
    ) -> None:
        self.experiment = experiment
        self.input_names = list(input_names)
        self.response_name = response_name
        self.initial_guess = dict(initial_guess or {})
        self.measurement_noise_var = float(measurement_noise_var)
        self.regularize = float(regularize)
        self.mode = "parametric"

        self.parameters_: dict[str, float] | None = None
        self.parameter_names_: list[str] = []
        self.covariance_: np.ndarray | None = None
        self.fim_: np.ndarray | None = None

        self._compile()

    # ------------------------------------------------------------------
    # JIT compilation of f(design, theta) and its parameter Jacobian
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        import jax
        import jax.numpy as jnp

        from discopt._jax.differentiable import _compile_parametric_node

        em = self.experiment.create_model(**self.initial_guess)
        if self.response_name not in em.responses:
            raise ValueError(
                f"experiment has no response {self.response_name!r}; "
                f"available: {list(em.responses)}"
            )
        missing = [n for n in self.input_names if n not in em.design_inputs]
        if missing:
            raise ValueError(
                f"workbook inputs {missing!r} not found in experiment.design_inputs "
                f"{list(em.design_inputs)!r}"
            )
        self.parameter_names_ = list(em.unknown_parameters.keys())

        # Flat offsets for every variable in the model.
        offsets: dict[int, int] = {}
        offset = 0
        for v in em.model._variables:
            offsets[id(v)] = offset
            offset += v.size
        n_x = offset

        design_idx = [offsets[id(em.design_inputs[n])] for n in self.input_names]
        param_idx = [offsets[id(em.unknown_parameters[n])] for n in self.parameter_names_]

        # p_flat for any model Parameters (distinct from unknown parameters).
        p_parts: list[np.ndarray] = []
        for p in em.model._parameters:
            p_parts.append(np.asarray(p.value, dtype=np.float64).ravel())
        if p_parts:
            p_flat_const = jnp.array(np.concatenate(p_parts), dtype=jnp.float64)
        else:
            p_flat_const = jnp.zeros(0, dtype=jnp.float64)

        response_fn = _compile_parametric_node(em.responses[self.response_name], em.model)
        d_idx = jnp.asarray(design_idx, dtype=jnp.int32)
        p_idx = jnp.asarray(param_idx, dtype=jnp.int32)

        def y_one(d_vec, theta_vec):
            x_flat = jnp.zeros(n_x, dtype=jnp.float64)
            x_flat = x_flat.at[d_idx].set(d_vec)
            x_flat = x_flat.at[p_idx].set(theta_vec)
            return response_fn(x_flat, p_flat_const)

        self._y_one = jax.jit(y_one)
        self._y_batch = jax.jit(jax.vmap(y_one, in_axes=(0, None)))
        self._jac_batch = jax.jit(jax.vmap(jax.jacrev(y_one, argnums=1), in_axes=(0, None)))

        def residuals(theta_vec, D, Y):
            return jax.vmap(y_one, in_axes=(0, None))(D, theta_vec) - Y

        self._residuals = jax.jit(residuals)
        self._residuals_jac = jax.jit(jax.jacrev(residuals, argnums=0))

        self._n_params = len(param_idx)

    # ------------------------------------------------------------------
    # Surrogate protocol
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParametricSurrogate":
        """Refit the unknown parameters by nonlinear least squares.

        Computes the FIM and parameter covariance at the fitted point.
        """
        import jax.numpy as jnp
        from scipy.optimize import least_squares

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[1] != len(self.input_names):
            raise ValueError(f"X has {X.shape[1]} columns but {len(self.input_names)} input names")

        theta0 = np.array(
            [float(self.initial_guess.get(n, 1.0)) for n in self.parameter_names_],
            dtype=float,
        )
        # Warm-start from the previous fit when available.
        if self.parameters_ is not None:
            theta0 = np.array([self.parameters_[n] for n in self.parameter_names_], dtype=float)

        D = jnp.asarray(X, dtype=jnp.float64)
        Y = jnp.asarray(y, dtype=jnp.float64)

        def f_resid(theta: np.ndarray) -> np.ndarray:
            return np.asarray(self._residuals(jnp.asarray(theta), D, Y), dtype=float)

        def f_jac(theta: np.ndarray) -> np.ndarray:
            return np.asarray(self._residuals_jac(jnp.asarray(theta), D, Y), dtype=float)

        res = least_squares(f_resid, theta0, jac=f_jac, method="lm")
        theta_hat = np.asarray(res.x, dtype=float)

        J = np.asarray(self._jac_batch(D, jnp.asarray(theta_hat)), dtype=float)
        sigma2 = max(self.measurement_noise_var, 1e-30)
        fim = (J.T @ J) / sigma2 + self.regularize * np.eye(self._n_params)
        try:
            cov = np.linalg.inv(fim)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(fim)

        self.parameters_ = {n: float(theta_hat[i]) for i, n in enumerate(self.parameter_names_)}
        self.fim_ = fim
        self.covariance_ = cov
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior response mean and std at the candidate designs."""
        import jax.numpy as jnp

        if self.parameters_ is None or self.covariance_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        theta = jnp.asarray([self.parameters_[n] for n in self.parameter_names_], dtype=jnp.float64)
        D = jnp.asarray(X, dtype=jnp.float64)
        mu = np.asarray(self._y_batch(D, theta), dtype=float).ravel()
        J = np.asarray(self._jac_batch(D, theta), dtype=float)
        # Diag of J Σ J^T per row.
        var_param = np.einsum("ki,ij,kj->k", J, self.covariance_, J)
        var_total = var_param + self.measurement_noise_var
        std = np.sqrt(np.clip(var_total, 0.0, None))
        return mu, std


def model_based_optimize_round(
    workbook: str | Path | Workbook,
    *,
    experiment: Experiment | None = None,
    initial_guess: dict[str, float] | None = None,
    response: str | None = None,
    input_names: Sequence[str] | None = None,
    criterion: OptimizationCriterion | str = OptimizationCriterion.MAXIMIZE,
    acquisition: str | Callable = "expected_improvement",
    batch_size: int = 1,
    n_candidates: int = 2048,
    candidate_sampler: str = "sobol",
    bounds: Sequence[tuple[float, float]] | None = None,
    seed: int | None = None,
    acquisition_kwargs: dict[str, Any] | None = None,
    measurement_noise_override: float | None = None,
) -> ModelBasedRoundResult:
    """Propose the next batch of experiments using a parametric model.

    Parameters
    ----------
    workbook : path or Workbook
        Workbook containing the completed runs. The next batch is appended.
    experiment : Experiment, optional
        Parametric model. When omitted, the experiment is rebuilt from
        the workbook (for built-in templates and ``module_callable``).
    initial_guess : dict[str, float], optional
        Starting parameter values. Defaults to the
        ``param_initial_guess`` stored in the workbook.
    response : str, optional
        Response column name. Defaults to the workbook's stored value.
    input_names : sequence of str, optional
        Design-input column order. Defaults to the workbook's input_specs.
    criterion : OptimizationCriterion or str
        ``"maximize"`` or ``"minimize"``.
    acquisition : str or callable
        Looked up via :func:`~discopt.doe.acquisition.resolve_acquisition`.
    batch_size : int, default 1
        Number of new experiments to recommend.
    n_candidates : int, default 2048
        Size of the Sobol candidate pool.
    candidate_sampler : ``"sobol"`` or ``"uniform"``
        Sampler used to draw candidates inside the bounding box.
    bounds : sequence of (lo, hi), optional
        Per-input box. Defaults to the workbook's input_specs.
    seed : int, optional
        Reproducible candidate sampling.
    acquisition_kwargs : dict, optional
        Extra kwargs forwarded to the acquisition function.
    measurement_noise_override : float, optional
        Use this σ instead of the workbook's stored measurement_error.
    """
    crit = OptimizationCriterion(criterion) if isinstance(criterion, str) else criterion
    direction = crit.direction
    acq_fn = resolve_acquisition(acquisition)
    acq_kwargs = dict(acquisition_kwargs or {})

    wb = workbook if isinstance(workbook, Workbook) else Workbook.open(Path(workbook))

    if experiment is None:
        experiment, _ = wb.rebuild_experiment()
    if initial_guess is None:
        initial_guess = wb.param_initial_guess()

    specs = wb.input_specs()
    names = list(input_names) if input_names is not None else [s.name for s in specs]
    response_name = response or wb.response_name()
    sigma = (
        wb.measurement_error() if measurement_noise_override is None else measurement_noise_override
    )

    if bounds is None:
        bounds_arr = np.array([(s.lb, s.ub) for s in specs], dtype=float)
    else:
        bounds_arr = np.asarray(list(bounds), dtype=float)
    if bounds_arr.shape != (len(names), 2):
        raise ValueError(f"bounds shape {bounds_arr.shape} does not match {len(names)} input(s)")

    completed = wb.completed_runs()
    if not completed:
        raise ValueError(
            "no completed runs in workbook -- fill in at least one response "
            "before calling model_based_optimize_round"
        )

    X = np.array([[float(r[n]) for n in names] for r in completed], dtype=float)
    y = np.array([float(r[response_name]) for r in completed], dtype=float)

    s = ParametricSurrogate(
        experiment,
        input_names=names,
        response_name=response_name,
        initial_guess=initial_guess,
        measurement_noise_var=float(sigma) ** 2,
    )
    s.fit(X, y)

    rng = np.random.default_rng(seed)
    candidates = _sample_candidates(bounds_arr, n_candidates, candidate_sampler, rng)

    incumbent_idx = int(np.argmax(direction * y))
    incumbent_y = float(y[incumbent_idx])
    incumbent_x = {n: float(X[incumbent_idx, i]) for i, n in enumerate(names)}

    chosen_idx: list[int] = []
    chosen_scores: list[float] = []
    X_fantasy = X.copy()
    y_fantasy = y.copy()
    incumbent_for_acq = incumbent_y

    for _ in range(int(batch_size)):
        kw = dict(acq_kwargs)
        kw["y_best"] = incumbent_for_acq
        kw["direction"] = direction
        try:
            scores = acq_fn(s, candidates, **kw)
        except TypeError:
            scores = acq_fn(s, candidates, direction=direction)
        scores = np.asarray(scores, dtype=float).ravel()
        if chosen_idx:
            scores[chosen_idx] = -np.inf
        pick = int(np.argmax(scores))
        chosen_idx.append(pick)
        chosen_scores.append(float(scores[pick]))

        mu_pick, _ = s.predict(candidates[pick : pick + 1])
        X_fantasy = np.vstack([X_fantasy, candidates[pick : pick + 1]])
        y_fantasy = np.concatenate([y_fantasy, mu_pick])
        if direction * float(mu_pick[0]) > direction * incumbent_for_acq:
            incumbent_for_acq = float(mu_pick[0])
        s.fit(X_fantasy, y_fantasy)

    next_designs = [{n: float(candidates[i, j]) for j, n in enumerate(names)} for i in chosen_idx]
    batch_idx = wb.next_batch_index()
    new_run_ids = wb.append_runs(batch_idx, next_designs)

    assert s.covariance_ is not None and s.fim_ is not None and s.parameters_ is not None
    parameter_se = {
        n: float(np.sqrt(max(s.covariance_[i, i], 0.0))) for i, n in enumerate(s.parameter_names_)
    }
    sign, logdet = np.linalg.slogdet(s.fim_)
    fim_log_det = float(logdet) if sign > 0 else float("-inf")

    wb.log(
        "model_based_optimize",
        {
            "criterion": crit.value,
            "acquisition": acquisition if isinstance(acquisition, str) else acq_fn.__name__,
            "batch_size": int(batch_size),
            "n_completed": len(completed),
            "fim_log_det": fim_log_det,
        },
    )
    wb.save()

    return ModelBasedRoundResult(
        next_designs=next_designs,
        new_run_ids=new_run_ids,
        incumbent_x=incumbent_x,
        incumbent_y=incumbent_y,
        acquisition_scores=chosen_scores,
        surrogate_mode="parametric",
        n_completed=len(completed),
        workbook_path=str(wb.path),
        parameters=dict(s.parameters_),
        parameter_se=parameter_se,
        fim_log_det=fim_log_det,
    )


__all__ = [
    "ModelBasedRoundResult",
    "ParametricSurrogate",
    "model_based_optimize_round",
]
