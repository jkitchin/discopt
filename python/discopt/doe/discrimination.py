"""Pre-experiment design criteria for **model discrimination**.

Given several candidate ``Experiment`` instances (one per hypothesised
model structure), find the design that best separates them. Five
criteria are exposed via the :class:`DiscriminationCriterion` enum:

- ``HR`` — Hunter-Reiner (1965), squared difference of point predictions.
- ``BF`` — Buzzi-Ferraris-Forzatti (1984) multiresponse, normalized
  by measurement and prediction-variance covariances. **Default.**
- ``JR`` — Jensen-Rényi divergence on per-model Gaussian predictives
  (Olofsson, Deisenroth & Misener 2019). Symmetric, M-model-friendly.
- ``MI`` — Mutual information :math:`I(M; y \\mid d)` between model
  index and future data, estimated by nested Monte Carlo on the
  Gaussian predictives (Lindley 1956; Foster et al. 2019).
- ``DT`` — DT-compound (Atkinson, Bogacka & Bogacki 1998), a weighted
  blend of D-optimal precision and discrimination.

All criteria operate in **prediction space** -- only :math:`\\hat y_i(d)`
and the prediction covariance :math:`V_i = J_i \\, \\mathrm{FIM}_i^{-1}
J_i^\\top` cross model boundaries. Candidate models therefore may have
**different parameter sets** without any name alignment.

Usage
-----

>>> from discopt.doe import discriminate_design, DiscriminationCriterion
>>> result = discriminate_design(
...     experiments={"arrhenius": ArrheniusExp(), "eyring": EyringExp()},
...     param_estimates={"arrhenius": {"A": 1e3, "Ea": 50e3},
...                      "eyring":    {"dH": 50e3, "dS": 0.0}},
...     design_bounds={"T": (300.0, 700.0)},
...     criterion=DiscriminationCriterion.BF,
... )
>>> result.design                                    # {"T": 480.0}
>>> result.criterion_value                           # scalar
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from discopt.doe.design import DesignCriterion
from discopt.doe.fim import FIMResult
from discopt.estimate import Experiment

_SINGULAR_SENTINEL = 1e30
_LOG_2PI = float(np.log(2 * np.pi))


class DiscriminationCriterion(str, Enum):
    """Criteria for model-discrimination experimental design."""

    HR = "hunter_reiner"
    BF = "buzzi_ferraris"
    JR = "jensen_renyi"
    MI = "mutual_information"
    DT = "dt_compound"


@dataclass
class DiscriminationDesignResult:
    """Result of :func:`discriminate_design` /
    :func:`discriminate_compound`.

    Attributes
    ----------
    design : dict[str, float]
        Optimal design point.
    criterion : DiscriminationCriterion
        Criterion that was optimised.
    criterion_value : float
        Criterion value at the optimal design (after sign normalisation
        for maximisation).
    fim_results : dict[str, FIMResult]
        Per-model FIMResult evaluated at the optimal design.
    predicted_responses : dict[str, dict[str, float]]
        Per-model predicted response means at the optimal design.
    prediction_covariances : dict[str, numpy.ndarray]
        Per-model ``V_i = J_i FIM_i^{-1} J_i^T`` at the optimal design,
        indexed by model name; arrays of shape ``(n_responses, n_responses)``.
    pairwise_divergence : numpy.ndarray or None
        ``(M, M)`` array of pairwise contributions to the criterion
        (e.g. squared differences for HR, weighted norms for BF).
        ``None`` for criteria where this concept does not apply
        (currently MI).
    model_names : list[str]
        Ordered names of candidate models.
    warnings : list[str]
        Human-readable flags surfaced during optimisation.
    """

    design: dict[str, float]
    criterion: DiscriminationCriterion
    criterion_value: float
    fim_results: dict[str, FIMResult]
    predicted_responses: dict[str, dict[str, float]]
    prediction_covariances: dict[str, np.ndarray]
    pairwise_divergence: np.ndarray | None
    model_names: list[str]
    warnings: list[str]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def discriminate_design(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    *,
    criterion: DiscriminationCriterion = DiscriminationCriterion.BF,
    model_priors: dict[str, float] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    mi_samples: int = 2000,
    seed: int | None = None,
) -> DiscriminationDesignResult:
    """Find the design that best separates the candidate models.

    Parameters
    ----------
    experiments : dict[str, Experiment]
        Candidate models keyed by user-chosen names.
    param_estimates : dict[str, dict[str, float]]
        Nominal parameter values per model. Keys must match
        ``experiments`` exactly. Each model may have its own parameter
        names; names need not be aligned across models.
    design_bounds : dict[str, tuple[float, float]]
        Lower / upper bounds for each design input variable. Keys must
        be a subset of the design inputs of every candidate model.
    criterion : DiscriminationCriterion, default BF
        Which discrimination criterion to optimise.
    model_priors : dict[str, float], optional
        Prior probability per model (used by HR, BF, JR, MI). Defaults
        to a uniform prior.
    n_starts : int, default 10
        Multi-start sample count.
    local_refine : bool, default True
        If True, refine the best multi-start point with L-BFGS-B.
    mi_samples : int, default 2000
        Outer sample count for the MI nested Monte Carlo estimator.
    seed : int, optional
        RNG seed for reproducibility (used by multi-start sampling and
        the MI estimator).

    Returns
    -------
    DiscriminationDesignResult
    """
    _validate_inputs(experiments, param_estimates, design_bounds)
    model_names = list(experiments.keys())
    weights = _normalise_priors(model_priors, model_names)

    rng = np.random.default_rng(seed)
    rng_seed = int(rng.integers(0, 2**31 - 1))

    def objective(design: dict[str, float]) -> float:
        """Return *negative* criterion value for minimisation."""
        try:
            preds = _predict_all_models(experiments, param_estimates, design)
            value, _ = _evaluate_criterion(criterion, preds, weights, mi_samples, rng_seed)
        except Exception:
            return _SINGULAR_SENTINEL
        if not np.isfinite(value):
            return _SINGULAR_SENTINEL
        return -value  # all discrimination criteria are maximised

    best_design = _optimize_over_design(
        objective, design_bounds, n_starts=n_starts, local_refine=local_refine, seed=rng_seed
    )
    if best_design is None:
        raise RuntimeError("No feasible design found for discrimination")

    # Final evaluation at the optimum to populate the result.
    preds = _predict_all_models(experiments, param_estimates, best_design)
    crit_value, pairwise = _evaluate_criterion(criterion, preds, weights, mi_samples, rng_seed)

    return DiscriminationDesignResult(
        design=best_design,
        criterion=DiscriminationCriterion(criterion),
        criterion_value=float(crit_value),
        fim_results={name: preds[name].fim_result for name in model_names},
        predicted_responses={
            name: dict(zip(preds[name].response_names, preds[name].y_hat)) for name in model_names
        },
        prediction_covariances={name: preds[name].V for name in model_names},
        pairwise_divergence=pairwise,
        model_names=model_names,
        warnings=[],
    )


def discriminate_compound(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    *,
    discrimination_weight: float = 0.5,
    precision_criterion: str = DesignCriterion.D_OPTIMAL,
    discrimination_criterion: DiscriminationCriterion = DiscriminationCriterion.BF,
    precision_model: str | None = None,
    model_priors: dict[str, float] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    mi_samples: int = 2000,
    seed: int | None = None,
) -> DiscriminationDesignResult:
    r"""DT-compound design balancing precision and discrimination.

    Optimises ``(1 - λ) * Φ_precision(M_p) + λ * Φ_discrimination``
    where ``Φ_precision`` is one of the standard FIM criteria
    evaluated against the ``precision_model`` and ``Φ_discrimination``
    is the requested discrimination criterion.

    Parameters
    ----------
    discrimination_weight : float, default 0.5
        ``λ ∈ [0, 1]``. ``λ = 0`` collapses to pure D-optimal for
        ``precision_model``; ``λ = 1`` collapses to the requested
        ``discrimination_criterion``.
    precision_criterion : DesignCriterion, default D_OPTIMAL
        FIM criterion used for the precision term.
    discrimination_criterion : DiscriminationCriterion, default BF
        Discrimination objective `Φ_discrimination`.
    precision_model : str, optional
        Which model anchors the precision objective. Defaults to the
        lexicographically first key in ``experiments`` and a warning
        is added to ``result.warnings``.
    \*\*kwargs : dict
        Additional parameters; see :func:`discriminate_design`.

    Returns
    -------
    DiscriminationDesignResult
        ``criterion`` is set to :attr:`DiscriminationCriterion.DT` and
        ``criterion_value`` is the compound objective at the optimum.
    """
    if not 0.0 <= discrimination_weight <= 1.0:
        raise ValueError("discrimination_weight must be in [0, 1]")
    _validate_inputs(experiments, param_estimates, design_bounds)
    model_names = list(experiments.keys())
    weights = _normalise_priors(model_priors, model_names)

    warnings_out: list[str] = []
    if precision_model is None:
        precision_model = sorted(experiments.keys())[0]
        warnings_out.append(f"precision_model not specified; defaulting to {precision_model!r}")
    if precision_model not in experiments:
        raise KeyError(f"precision_model {precision_model!r} not in experiments")

    rng = np.random.default_rng(seed)
    rng_seed = int(rng.integers(0, 2**31 - 1))
    lam = float(discrimination_weight)

    def objective(design: dict[str, float]) -> float:
        try:
            preds = _predict_all_models(experiments, param_estimates, design)
            disc_value, _ = _evaluate_criterion(
                discrimination_criterion, preds, weights, mi_samples, rng_seed
            )
            prec_value = _precision_value(preds[precision_model].fim_result, precision_criterion)
        except Exception:
            return _SINGULAR_SENTINEL
        if not (np.isfinite(disc_value) and np.isfinite(prec_value)):
            return _SINGULAR_SENTINEL
        return -((1.0 - lam) * prec_value + lam * disc_value)

    best_design = _optimize_over_design(
        objective, design_bounds, n_starts=n_starts, local_refine=local_refine, seed=rng_seed
    )
    if best_design is None:
        raise RuntimeError("No feasible design found for compound discrimination")

    preds = _predict_all_models(experiments, param_estimates, best_design)
    disc_value, pairwise = _evaluate_criterion(
        discrimination_criterion, preds, weights, mi_samples, rng_seed
    )
    prec_value = _precision_value(preds[precision_model].fim_result, precision_criterion)
    compound_value = (1.0 - lam) * prec_value + lam * disc_value

    return DiscriminationDesignResult(
        design=best_design,
        criterion=DiscriminationCriterion.DT,
        criterion_value=float(compound_value),
        fim_results={name: preds[name].fim_result for name in model_names},
        predicted_responses={
            name: dict(zip(preds[name].response_names, preds[name].y_hat)) for name in model_names
        },
        prediction_covariances={name: preds[name].V for name in model_names},
        pairwise_divergence=pairwise,
        model_names=model_names,
        warnings=warnings_out,
    )


# ─────────────────────────────────────────────────────────────────────
# Internal: prediction + covariance per model
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _ModelPrediction:
    """Bundled per-model evaluation at a single design point."""

    y_hat: np.ndarray  # shape (R,)
    V: np.ndarray  # prediction covariance, shape (R, R)
    Sigma_y: np.ndarray  # measurement noise diag, shape (R, R)
    response_names: list[str]
    fim_result: FIMResult


def _predict_all_models(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_values: dict[str, float],
) -> dict[str, _ModelPrediction]:
    """Compute (y_hat, V, Sigma_y, FIM, J) for every model at one design."""
    return {
        name: _predict_with_covariance(experiments[name], param_estimates[name], design_values)
        for name in experiments
    }


def _predict_with_covariance(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float],
) -> _ModelPrediction:
    """Evaluate y_hat, the FIM, and the prediction covariance V at one design.

    Mirrors the compile/solve pipeline used inside
    :func:`discopt.doe.fim.compute_fim`, with the addition that we also
    read the predicted response values from the same solve, avoiding a
    second model build per design point.
    """
    from discopt._jax.differentiable import _compile_parametric_node
    from discopt._jax.parametric import extract_x_flat
    from discopt.doe.fim import _build_p_flat, _compute_jacobian_autodiff, _get_param_indices

    em = experiment.create_model(**param_values)

    # Fix design variables (same recipe as compute_fim).
    for name, val in design_values.items():
        if name not in em.design_inputs:
            continue
        var = em.design_inputs[name]
        val_arr = np.asarray(float(val), dtype=np.float64)
        if var.shape:
            val_arr = np.full(var.shape, val_arr)
        var.lb = val_arr
        var.ub = val_arr

    # Trivial dummy objective to pin parameters at their nominal values.
    em.model.minimize(
        sum((em.unknown_parameters[n] - param_values[n]) ** 2 for n in em.parameter_names)
    )
    result = em.model.solve()

    x_flat = extract_x_flat(result, em.model)

    # Compile response functions and compute predicted means.
    response_fns = [_compile_parametric_node(em.responses[n], em.model) for n in em.response_names]
    p_flat = _build_p_flat(em.model)
    y_hat = np.array([float(np.asarray(fn(x_flat, p_flat)).flat[0]) for fn in response_fns])

    # Jacobian of responses w.r.t. unknown parameters.
    param_indices = _get_param_indices(em)
    J = np.asarray(_compute_jacobian_autodiff(response_fns, x_flat, p_flat, param_indices))

    sigma = np.array([em.measurement_error[n] for n in em.response_names], dtype=np.float64)
    Sigma_y = np.diag(sigma**2)
    Sigma_inv = np.diag(1.0 / sigma**2)
    fim = J.T @ Sigma_inv @ J
    V = J @ np.linalg.pinv(fim) @ J.T

    fim_result = FIMResult(
        fim=np.asarray(fim),
        jacobian=J,
        parameter_names=em.parameter_names,
        response_names=em.response_names,
    )

    return _ModelPrediction(
        y_hat=y_hat,
        V=np.asarray(V),
        Sigma_y=Sigma_y,
        response_names=em.response_names,
        fim_result=fim_result,
    )


# ─────────────────────────────────────────────────────────────────────
# Internal: criterion dispatch
# ─────────────────────────────────────────────────────────────────────


def _evaluate_criterion(
    criterion: DiscriminationCriterion,
    preds: dict[str, _ModelPrediction],
    weights: dict[str, float],
    mi_samples: int,
    seed: int,
) -> tuple[float, np.ndarray | None]:
    """Dispatch to the requested criterion and return (value, pairwise)."""
    crit = DiscriminationCriterion(criterion)
    if crit is DiscriminationCriterion.HR:
        return _criterion_hunter_reiner(preds, weights)
    if crit is DiscriminationCriterion.BF:
        return _criterion_buzzi_ferraris(preds, weights)
    if crit is DiscriminationCriterion.JR:
        return _criterion_jensen_renyi(preds, weights)
    if crit is DiscriminationCriterion.MI:
        value = _criterion_mutual_information(preds, weights, mi_samples, seed)
        return value, None
    if crit is DiscriminationCriterion.DT:
        raise ValueError(
            "DT-compound is not a standalone criterion; call "
            "discriminate_compound() instead of discriminate_design(..., criterion=DT)."
        )
    raise ValueError(f"Unknown discrimination criterion: {criterion!r}")


def _criterion_hunter_reiner(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray]:
    """``Σ w_i w_j ||ŷ_i − ŷ_j||²``."""
    names = list(preds.keys())
    M = len(names)
    pw = np.zeros((M, M))
    for i, j in combinations(range(M), 2):
        diff = preds[names[i]].y_hat - preds[names[j]].y_hat
        contrib = float(diff @ diff)
        pw[i, j] = pw[j, i] = contrib
    total = sum(
        weights[names[i]] * weights[names[j]] * pw[i, j] for i, j in combinations(range(M), 2)
    )
    return float(total), pw


def _criterion_buzzi_ferraris(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray]:
    """``Σ w_i w_j (ŷ_i − ŷ_j)^T (Σ_y + V_i + V_j)^{-1} (ŷ_i − ŷ_j)``."""
    names = list(preds.keys())
    M = len(names)
    pw = np.zeros((M, M))
    for i, j in combinations(range(M), 2):
        ni, nj = names[i], names[j]
        # Common Σ_y: averaging is well-defined since both come from the same
        # ExperimentModel.measurement_error mapping; if response_names align.
        if preds[ni].response_names != preds[nj].response_names:
            raise ValueError(
                f"Models {ni!r} and {nj!r} have different response names; "
                "discrimination requires the same response namespace."
            )
        diff = preds[ni].y_hat - preds[nj].y_hat
        cov = preds[ni].Sigma_y + preds[ni].V + preds[nj].V
        contrib = float(diff @ np.linalg.solve(cov, diff))
        pw[i, j] = pw[j, i] = contrib
    total = sum(
        weights[names[i]] * weights[names[j]] * pw[i, j] for i, j in combinations(range(M), 2)
    )
    return float(total), pw


def _criterion_jensen_renyi(
    preds: dict[str, _ModelPrediction], weights: dict[str, float]
) -> tuple[float, np.ndarray | None]:
    """Jensen-Rényi divergence with α=2 on the Gaussian predictives.

    For Gaussian components ``p_i = N(μ_i, S_i)`` with ``S_i = Σ_y + V_i``
    and prior weights ``w_i``, the α=2 Rényi entropy is

        H_2(p_i) = -log ∫ p_i^2 = (n/2) log(4π) + (1/2) log det S_i

    (uses the identity ``∫ N(x;μ,S)² dx = (4π)^(-n/2) (det S)^(-1/2)``)
    and the mixture α=2 entropy is

        H_2(Σ w_i p_i) = -log Σ_{i,j} w_i w_j N(μ_i; μ_j, S_i + S_j).

    JR returns ``H_2(mixture) − Σ w_i H_2(p_i)``; equals 0 when all
    components are identical.
    """
    names = list(preds.keys())
    M = len(names)
    n = preds[names[0]].y_hat.shape[0]
    means = np.stack([preds[name].y_hat for name in names])
    covs = [preds[name].Sigma_y + preds[name].V for name in names]
    log_weights = np.array([np.log(weights[name]) for name in names])

    # Per-component α=2 entropy.
    H_components = np.array([0.5 * (n * np.log(4 * np.pi) + _logdet(covs[i])) for i in range(M)])

    # Mixture α=2 entropy via the closed-form ∫ p_i p_j = N(μ_i; μ_j, S_i + S_j).
    log_int = np.full((M, M), -np.inf)
    for i in range(M):
        for j in range(M):
            log_int[i, j] = _log_gaussian(means[i], means[j], covs[i] + covs[j])
    log_w = log_weights[:, None] + log_weights[None, :]
    H_mix = -_logsumexp((log_w + log_int).ravel())

    jr = H_mix - float(np.sum(np.exp(log_weights) * H_components))
    return float(jr), None


def _criterion_mutual_information(
    preds: dict[str, _ModelPrediction],
    weights: dict[str, float],
    mi_samples: int,
    seed: int,
) -> float:
    """``I(M; y | d)`` via nested Monte Carlo on Gaussian predictives."""
    names = list(preds.keys())
    M = len(names)
    n = preds[names[0]].y_hat.shape[0]
    means = [preds[name].y_hat for name in names]
    covs = [preds[name].Sigma_y + preds[name].V for name in names]
    w = np.array([weights[name] for name in names])
    log_w = np.log(w)

    rng = np.random.default_rng(seed)
    chol = [np.linalg.cholesky(_pd(C)) for C in covs]

    # Sample (M, y) ~ Σ w_i p_i. Vectorized: pick component indices, then draw.
    component = rng.choice(M, size=mi_samples, p=w)
    z = rng.standard_normal((mi_samples, n))
    samples = np.empty((mi_samples, n))
    for k in range(M):
        idx = np.where(component == k)[0]
        if idx.size:
            samples[idx] = means[k] + z[idx] @ chol[k].T

    # log p(y_n | M=k) for each (n, k).
    log_p = np.empty((mi_samples, M))
    for k in range(M):
        log_p[:, k] = _log_gaussian_batch(samples, means[k], covs[k])

    # H(y) ≈ -mean_n log Σ_k w_k p(y_n | M=k).
    log_marginal = _logsumexp_axis(log_p + log_w, axis=1)
    H_y = -float(np.mean(log_marginal))

    # H(y | M) = Σ w_k H(y | M=k); closed-form Gaussian entropy.
    H_yM = float(sum(w[k] * (0.5 * (n * (1.0 + _LOG_2PI) + _logdet(covs[k]))) for k in range(M)))

    return H_y - H_yM


def _precision_value(fim_result: FIMResult, criterion: str) -> float:
    """Return the precision-criterion value (with maximisation sign)."""
    if criterion == DesignCriterion.D_OPTIMAL:
        return fim_result.d_optimal  # already maximisation (log det)
    if criterion == DesignCriterion.A_OPTIMAL:
        return -fim_result.a_optimal  # trace of FIM^-1; minimise
    if criterion == DesignCriterion.E_OPTIMAL:
        return fim_result.e_optimal  # max min-eig
    if criterion == DesignCriterion.ME_OPTIMAL:
        return -fim_result.me_optimal  # condition number; minimise
    raise ValueError(f"Unknown precision_criterion: {criterion!r}")


# ─────────────────────────────────────────────────────────────────────
# Internal: validation, optimisation, math utilities
# ─────────────────────────────────────────────────────────────────────


def _validate_inputs(
    experiments: dict[str, Experiment],
    param_estimates: dict[str, dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
) -> None:
    if len(experiments) < 2:
        raise ValueError(
            f"Need at least 2 candidate models, got {len(experiments)}: {list(experiments)}"
        )
    if set(experiments.keys()) != set(param_estimates.keys()):
        raise ValueError(
            "experiments and param_estimates must share keys; "
            f"experiments={set(experiments)}, param_estimates={set(param_estimates)}"
        )
    if not design_bounds:
        raise ValueError("design_bounds must be non-empty")


def _normalise_priors(priors: dict[str, float] | None, model_names: list[str]) -> dict[str, float]:
    if priors is None:
        w = 1.0 / len(model_names)
        return {name: w for name in model_names}
    if set(priors) != set(model_names):
        raise ValueError(f"model_priors keys {set(priors)} != experiments keys {set(model_names)}")
    total = sum(priors.values())
    if total <= 0:
        raise ValueError("model_priors must sum to a positive value")
    return {name: priors[name] / total for name in model_names}


def _optimize_over_design(
    objective: Callable[[dict[str, float]], float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_starts: int,
    local_refine: bool,
    seed: int,
) -> dict[str, float] | None:
    """Multi-start + optional L-BFGS-B over a scalar minimisation objective."""
    design_names = list(design_bounds.keys())
    rng = np.random.default_rng(seed)

    candidates: list[dict[str, float]] = []
    for _ in range(n_starts):
        candidates.append({n: float(rng.uniform(*design_bounds[n])) for n in design_names})
    for n in design_names:
        lo, hi = design_bounds[n]
        for val in (lo, hi):
            point = {nn: 0.5 * (design_bounds[nn][0] + design_bounds[nn][1]) for nn in design_names}
            point[n] = float(val)
            candidates.append(point)

    best_design: dict[str, float] | None = None
    best_value = np.inf
    for cand in candidates:
        val = objective(cand)
        if val < best_value:
            best_value = val
            best_design = cand

    if best_design is None or not np.isfinite(best_value):
        return None

    if local_refine:
        bounds = [design_bounds[n] for n in design_names]
        x0 = np.array([best_design[n] for n in design_names], dtype=float)

        def _wrapped(x: np.ndarray) -> float:
            return objective({n: float(v) for n, v in zip(design_names, x)})

        try:
            res = minimize(_wrapped, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_value:
                best_value = float(res.fun)
                best_design = {n: float(v) for n, v in zip(design_names, res.x)}
        except Exception:
            pass

    return best_design


def _logdet(M: np.ndarray) -> float:
    """``log det`` via SVD; safe for symmetric PD matrices."""
    sign, val = np.linalg.slogdet(M)
    if sign <= 0:
        # Regularise.
        eps = max(1e-12, 1e-10 * float(np.trace(M)) / max(M.shape[0], 1))
        sign, val = np.linalg.slogdet(M + eps * np.eye(M.shape[0]))
    return float(val)


def _log_gaussian(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """``log N(x; μ, Σ)`` for a single sample (regularises Σ if needed)."""
    n = x.size
    diff = x - mu
    cov_safe = _pd(cov)
    L = np.linalg.cholesky(cov_safe)
    sol = np.linalg.solve(L, diff)
    quad = float(sol @ sol)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (n * _LOG_2PI + log_det + quad)


def _log_gaussian_batch(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``log N(X; μ, Σ)`` for a batch ``X`` of shape ``(N, n)``."""
    n = X.shape[1]
    cov_safe = _pd(cov)
    L = np.linalg.cholesky(cov_safe)
    diff = (X - mu).T  # (n, N)
    sol = np.linalg.solve(L, diff)
    quad = np.sum(sol**2, axis=0)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return np.asarray(-0.5 * (n * _LOG_2PI + log_det + quad))


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _logsumexp_axis(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.asarray(
        (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)
    )


def _pd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return ``M`` regularised to be symmetric positive-definite."""
    M = 0.5 * (M + M.T)
    diag_floor = max(eps, eps * float(np.trace(np.abs(M))) / max(M.shape[0], 1))
    return np.asarray(M + diag_floor * np.eye(M.shape[0]))


__all__ = [
    "DiscriminationCriterion",
    "DiscriminationDesignResult",
    "discriminate_compound",
    "discriminate_design",
]
