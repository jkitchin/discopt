"""Post-experiment model selection helpers.

Derived from :class:`discopt.estimate.EstimationResult`, whose
``.objective`` equals the **deviance** ``−2 · log L(θ̂)`` up to a
constant under Gaussian noise. This makes every metric in this module
a one-line derivation:

- ``log L̂ = −0.5 · objective``
- ``AIC = 2·p + objective``
- ``BIC = p · log(n) + objective``
- ``LRT G² = objective_nested − objective_full`` on ``df = p_full − p_nested``
- Vuong ``z`` requires the per-observation log-likelihoods; these are
  rebuilt from the residuals via each candidate model's responses.

Usage
-----

>>> from discopt.doe import model_selection, likelihood_ratio_test
>>> res = model_selection({"arrh": est_a, "eyr": est_e}, method="aic")
>>> res.best_model, res.weights

References
----------
Akaike (1973); Schwarz (1978); Hurvich & Tsai (1989) AICc;
Wilks (1938) LRT; Vuong (1989) non-nested LRT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import chi2, norm

from discopt.estimate import EstimationResult, Experiment

SelectionMethod = Literal["aic", "aicc", "bic", "lrt", "vuong"]


@dataclass
class ModelSelectionResult:
    """Bundle of scores / weights / p-values from a selection test.

    Attributes
    ----------
    method : {"aic", "aicc", "bic", "lrt", "vuong"}
    scores : dict[str, float]
        Per-model score (lower is better for AIC/BIC).
    weights : dict[str, float] or None
        Softmax weights of ``-0.5 * Δscore`` (None for LRT / Vuong).
    best_model : str
        Name of the top-ranked model.
    p_value : float or None
        Null-hypothesis p-value for LRT / Vuong tests.
    nested_pair : (str, str) or None
        ``(nested, full)`` names for LRT only.
    z_statistic : float or None
        Vuong test statistic only.
    warnings : list[str]
    """

    method: SelectionMethod
    scores: dict[str, float]
    weights: dict[str, float] | None
    best_model: str
    p_value: float | None = None
    nested_pair: tuple[str, str] | None = None
    z_statistic: float | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def model_selection(
    estimation_results: dict[str, EstimationResult],
    *,
    method: Literal["aic", "aicc", "bic"] = "aic",
) -> ModelSelectionResult:
    """Rank candidate fits by AIC, AICc, or BIC.

    All candidates must have been fit to the same data (same
    ``n_observations``). The deviance convention of
    :func:`discopt.estimate.estimate_parameters` makes these one-liners.

    Parameters
    ----------
    estimation_results : dict[str, EstimationResult]
        Per-model fitted results keyed by user-chosen names.
    method : {"aic", "aicc", "bic"}, default "aic"

    Returns
    -------
    ModelSelectionResult
        ``scores``, softmax ``weights``, and the ``best_model``.
    """
    if len(estimation_results) < 2:
        raise ValueError(f"Need at least 2 candidate fits; got {list(estimation_results)}")
    n_obs = {name: res.n_observations for name, res in estimation_results.items()}
    if len(set(n_obs.values())) > 1:
        raise ValueError(
            f"Candidates must share n_observations; got {n_obs}. "
            "AIC/BIC comparisons require the same data."
        )
    n = next(iter(n_obs.values()))

    warnings_out: list[str] = []
    scores: dict[str, float] = {}
    for name, res in estimation_results.items():
        p = len(res.parameter_names)
        dev = float(res.objective)
        if method == "aic":
            scores[name] = 2.0 * p + dev
        elif method == "aicc":
            # Hurvich & Tsai small-sample correction.
            if n - p - 1 <= 0:
                warnings_out.append(
                    f"model {name!r}: n - p - 1 = {n - p - 1} <= 0; "
                    "AICc undefined, falling back to AIC."
                )
                scores[name] = 2.0 * p + dev
            else:
                scores[name] = 2.0 * p + dev + (2.0 * p * (p + 1)) / (n - p - 1)
        elif method == "bic":
            scores[name] = p * np.log(n) + dev
        else:
            raise ValueError(f"Unknown method: {method!r}")

    best_model = min(scores, key=lambda name: scores[name])
    # Softmax weights on -Δscore/2.
    min_score = min(scores.values())
    unnormalised = {name: np.exp(-0.5 * (s - min_score)) for name, s in scores.items()}
    total = sum(unnormalised.values())
    weights = {name: float(v / total) for name, v in unnormalised.items()}

    return ModelSelectionResult(
        method=method,
        scores=scores,
        weights=weights,
        best_model=best_model,
        warnings=warnings_out,
    )


def likelihood_ratio_test(
    nested: EstimationResult,
    full: EstimationResult,
    *,
    nested_name: str = "nested",
    full_name: str = "full",
    alpha: float = 0.05,
) -> ModelSelectionResult:
    """Likelihood-ratio test on a nested pair (Wilks 1938).

    Returns the ``G² = objective_nested − objective_full`` statistic
    and its :math:`\\chi^2_{df}` p-value with ``df = p_full − p_nested``.
    The full model is declared "best" when the null is rejected at
    level ``alpha``; otherwise the parsimony principle keeps the nested
    model as "best".

    Parameters
    ----------
    nested, full : EstimationResult
        Fitted results. ``nested.parameter_names`` must be a subset of
        ``full.parameter_names`` (nested relationship) and both must
        share ``n_observations``.
    nested_name, full_name : str
        Labels to use in the result's ``scores`` and ``best_model``.

    Returns
    -------
    ModelSelectionResult
        ``scores`` are the two deviances; ``p_value`` is the χ² tail
        probability; ``nested_pair = (nested_name, full_name)``.
    """
    p_nested = len(nested.parameter_names)
    p_full = len(full.parameter_names)
    if p_full <= p_nested:
        raise ValueError(
            f"Full model must have more parameters than nested ({p_full} vs {p_nested})"
        )
    if not set(nested.parameter_names).issubset(full.parameter_names):
        raise ValueError(
            f"Nested parameter_names {nested.parameter_names} must be a "
            f"subset of full parameter_names {full.parameter_names}"
        )
    if nested.n_observations != full.n_observations:
        raise ValueError(
            f"Mismatched n_observations: nested={nested.n_observations}, full={full.n_observations}"
        )

    G2 = float(nested.objective - full.objective)
    df = p_full - p_nested
    warnings_out: list[str] = []
    if G2 < 0:
        warnings_out.append(
            f"G² = {G2:.3g} < 0; full model fit is worse than nested. "
            "Likely a convergence issue; treat p-value with suspicion."
        )
        G2 = max(G2, 0.0)
    p_value = float(chi2.sf(G2, df=df))
    best = full_name if p_value < alpha else nested_name

    return ModelSelectionResult(
        method="lrt",
        scores={nested_name: float(nested.objective), full_name: float(full.objective)},
        weights=None,
        best_model=best,
        p_value=p_value,
        nested_pair=(nested_name, full_name),
        warnings=warnings_out,
    )


def vuong_test(
    res_a: EstimationResult,
    res_b: EstimationResult,
    data: dict,
    experiments: dict[str, Experiment],
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    alpha: float = 0.05,
) -> ModelSelectionResult:
    """Vuong (1989) likelihood-ratio test for non-nested models.

    Computes per-observation log-likelihoods :math:`\\ell^A_n,
    \\ell^B_n` under Gaussian noise, forms the mean difference
    :math:`\\bar m`, and reports the z-statistic
    :math:`z = \\sqrt{N}\\,\\bar m / s_m`. ``|z| < z_{1-\\alpha/2}`` is
    read as "statistically indistinguishable"; otherwise the sign of
    :math:`\\bar m` picks the winner.

    Parameters
    ----------
    res_a, res_b : EstimationResult
        Fits to be compared.
    data : dict
        Observed data used for both fits (same keys and values that
        were passed to :func:`estimate_parameters`).
    experiments : dict[str, Experiment]
        Mapping containing at least the candidate experiment for each
        result. Keys must include ``name_a`` and ``name_b`` (if these
        are None, the two keys present in ``experiments`` are used in
        the order given).
    name_a, name_b : str, optional
        Labels for each model. Default to the first two keys of
        ``experiments`` in iteration order.
    alpha : float, default 0.05
        Two-sided significance level for the "indistinguishable" region.

    Returns
    -------
    ModelSelectionResult
        ``scores`` gives each model's summed log-likelihood;
        ``p_value`` is the two-sided p; ``z_statistic`` is the Vuong
        ``z``. ``best_model = "indistinguishable"`` inside the
        acceptance region.
    """
    if len(experiments) < 2:
        raise ValueError(
            f"experiments dict must contain both candidate models; got {list(experiments)}"
        )
    keys = list(experiments.keys())
    if name_a is None:
        name_a = keys[0]
    if name_b is None:
        name_b = keys[1]
    if name_a not in experiments or name_b not in experiments:
        raise KeyError(f"{name_a!r} and {name_b!r} must both be in experiments")
    if res_a.n_observations != res_b.n_observations:
        raise ValueError(
            f"Mismatched n_observations: {name_a}={res_a.n_observations}, "
            f"{name_b}={res_b.n_observations}"
        )

    ll_a = _per_obs_loglik(experiments[name_a], res_a, data)
    ll_b = _per_obs_loglik(experiments[name_b], res_b, data)

    m = ll_a - ll_b
    N = m.size
    if N < 2:
        raise ValueError(f"Vuong needs N >= 2 observations; got N={N}")
    m_bar = float(np.mean(m))
    s_m = float(np.std(m, ddof=1))
    warnings_out: list[str] = []
    if s_m <= 0:
        warnings_out.append(
            "Vuong statistic has zero variance; models give identical "
            "per-observation log-likelihoods."
        )
        z = 0.0
    else:
        z = float(np.sqrt(N) * m_bar / s_m)

    p_value = float(2.0 * norm.sf(abs(z)))
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    if abs(z) < z_crit:
        best = "indistinguishable"
    else:
        best = name_a if z > 0 else name_b

    return ModelSelectionResult(
        method="vuong",
        scores={name_a: float(ll_a.sum()), name_b: float(ll_b.sum())},
        weights=None,
        best_model=best,
        p_value=p_value,
        nested_pair=None,
        z_statistic=z,
        warnings=warnings_out,
    )


# ─────────────────────────────────────────────────────────────────────
# Internal: per-observation log-likelihood reconstruction
# ─────────────────────────────────────────────────────────────────────


def _per_obs_loglik(experiment: Experiment, result: EstimationResult, data: dict) -> np.ndarray:
    """Return the per-observation log-likelihoods used to form the fit.

    Rebuilds the experiment at the fitted parameters, evaluates each
    response, and applies the Gaussian log-pdf ``-0.5 * [log(2πσ²) +
    ((y − ŷ)/σ)²]`` per observation.
    """
    from discopt._jax.differentiable import _compile_parametric_node
    from discopt._jax.parametric import extract_x_flat
    from discopt.doe.fim import _build_p_flat

    em = experiment.create_model(**result.parameters)
    # Pin every parameter at the fitted value so the solve is trivial.
    for name, val in result.parameters.items():
        var = em.unknown_parameters[name]
        val_arr = np.asarray(float(val), dtype=np.float64)
        if var.shape:
            val_arr = np.full(var.shape, val_arr)
        var.lb = val_arr
        var.ub = val_arr

    em.model.minimize(
        sum((em.unknown_parameters[n] - result.parameters[n]) ** 2 for n in em.parameter_names)
    )
    solve = em.model.solve()
    x_flat = extract_x_flat(solve, em.model)
    p_flat = _build_p_flat(em.model)

    y_hat: list[float] = []
    sigma: list[float] = []
    y_obs: list[float] = []
    for name in em.response_names:
        if name not in data:
            continue
        fn = _compile_parametric_node(em.responses[name], em.model)
        y_hat.append(float(np.asarray(fn(x_flat, p_flat)).flat[0]))
        sigma.append(float(em.measurement_error[name]))
        y_obs.append(float(np.asarray(data[name]).flat[0]))

    y_hat_arr = np.array(y_hat)
    sigma_arr = np.array(sigma)
    y_obs_arr = np.array(y_obs)
    resid = (y_obs_arr - y_hat_arr) / sigma_arr
    return np.asarray(-0.5 * (np.log(2.0 * np.pi * sigma_arr**2) + resid**2))


__all__ = [
    "ModelSelectionResult",
    "likelihood_ratio_test",
    "model_selection",
    "vuong_test",
]
