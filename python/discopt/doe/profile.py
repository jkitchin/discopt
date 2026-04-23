"""Profile likelihood confidence intervals and identifiability.

Implements the Raue et al. (2009) profile-likelihood algorithm on top
of :func:`discopt.estimate.estimate_parameters`. For each parameter to
profile, we step outward from the global estimate, re-solve the
estimation problem with the parameter fixed at each step, and record
the resulting objective.

Likelihood convention
---------------------
``estimate_parameters`` minimizes

    D(theta) = sum_i ((y_i - yhat_i(theta)) / sigma_i)^2

which is the *deviance*, i.e. ``2 * negative log-likelihood`` (up to a
constant) under Gaussian noise. The profile-likelihood confidence
region therefore uses the deviance form of the likelihood-ratio test,

    D(theta_i = c) - D(theta_hat) <= chi2_{1, 1-alpha},

with no factor of 1/2. All thresholds in this module use this
convention, matching ``result.objective`` directly.

References
----------
Raue, A., Kreutz, C., Maiwald, T., et al. Structural and practical
identifiability analysis of partially observed dynamical models by
exploiting the profile likelihood. *Bioinformatics* 25, 1923-1929
(2009).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import chi2

from discopt.estimate import EstimationResult, Experiment, estimate_parameters

ProfileShape = Literal["bounded", "one_sided_lower", "one_sided_upper", "flat"]


@dataclass
class ProfileLikelihoodResult:
    """Result of a single-parameter profile-likelihood scan.

    Attributes
    ----------
    parameter : str
        Name of the profiled parameter.
    theta_values : numpy.ndarray
        Parameter values visited, sorted ascending.
    neg_log_lik : numpy.ndarray
        Deviance ``D(theta)`` at each ``theta`` value. Despite the name
        (kept for user-facing familiarity) these are deviance values,
        ``2 * NLL``, matching ``estimate_parameters.objective``.
    theta_hat : float
        Maximum-likelihood estimate of the profiled parameter.
    objective_hat : float
        Deviance at ``theta_hat``.
    ci_lower : float or None
        Lower confidence-interval bound (linear interpolation between
        straddling grid points). ``None`` if the profile never crosses
        the threshold on the lower side.
    ci_upper : float or None
        Upper confidence-interval bound. ``None`` if unbounded above.
    confidence_level : float
        Nominal confidence level used to set the threshold.
    threshold : float
        The deviance threshold ``D(theta_hat) + chi2.ppf(alpha, 1)``.
    shape : str
        One of ``"bounded"``, ``"one_sided_lower"``,
        ``"one_sided_upper"``, ``"flat"``.
    warnings : list[str]
        Human-readable flags (non-monotone arm, NLP retries, etc.).
    """

    parameter: str
    theta_values: np.ndarray
    neg_log_lik: np.ndarray
    theta_hat: float
    objective_hat: float
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float
    threshold: float
    shape: ProfileShape
    warnings: list[str]


def profile_likelihood(
    experiment: Experiment,
    data: dict,
    parameter_name: str,
    *,
    confidence_level: float = 0.95,
    max_steps: int = 40,
    target_delta_loglik: float = 0.2,
    initial_estimate: EstimationResult | None = None,
    initial_step: float | None = None,
) -> ProfileLikelihoodResult:
    """Compute the profile likelihood for one parameter.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition; must expose ``parameter_name`` in its
        ``unknown_parameters``.
    data : dict
        Observed response values (same format as
        :func:`estimate_parameters`).
    parameter_name : str
        Parameter to profile.
    confidence_level : float, default 0.95
        Nominal coverage level.
    max_steps : int, default 40
        Maximum number of profile points per direction.
    target_delta_loglik : float, default 0.2
        Target deviance increment per step. The step size is adjusted
        after each solve to approximately achieve this (Raue 2009 Sec 2.3).
    initial_estimate : EstimationResult, optional
        Pre-computed global fit. If omitted, one is computed.
    initial_step : float, optional
        Override the starting step size. Defaults to a curvature-based
        estimate from the FIM diagonal.

    Returns
    -------
    ProfileLikelihoodResult
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    if initial_estimate is None:
        initial_estimate = estimate_parameters(experiment, data)
    if parameter_name not in initial_estimate.parameters:
        raise KeyError(
            f"{parameter_name!r} not in estimated parameters ({list(initial_estimate.parameters)})"
        )

    theta_hat = initial_estimate.parameters[parameter_name]
    objective_hat = float(initial_estimate.objective)
    threshold_offset = float(chi2.ppf(confidence_level, df=1))
    threshold = objective_hat + threshold_offset

    em = experiment.create_model(**initial_estimate.parameters)
    var = em.unknown_parameters.get(parameter_name)
    lb = float(var.lb) if var is not None else -np.inf
    ub = float(var.ub) if var is not None else np.inf

    # Curvature-based initial step size from FIM diagonal.
    if initial_step is None:
        idx = initial_estimate.parameter_names.index(parameter_name)
        fim_diag = float(initial_estimate.fim[idx, idx])
        # D(theta_hat + h) ~ D(theta_hat) + 2 * fim_diag * h^2
        # (factor 2 because D is deviance, not NLL)
        if fim_diag > 0:
            initial_step = np.sqrt(target_delta_loglik / (2.0 * fim_diag))
        else:
            initial_step = 1e-2 * max(abs(theta_hat), 1.0)

    warnings_out: list[str] = []
    lower_points, lower_obj = _profile_direction(
        experiment,
        data,
        parameter_name,
        theta_hat,
        objective_hat,
        direction=-1,
        bound=lb,
        step0=initial_step,
        threshold=threshold,
        max_steps=max_steps,
        target=target_delta_loglik,
        other_init=dict(initial_estimate.parameters),
        warnings_out=warnings_out,
    )
    upper_points, upper_obj = _profile_direction(
        experiment,
        data,
        parameter_name,
        theta_hat,
        objective_hat,
        direction=+1,
        bound=ub,
        step0=initial_step,
        threshold=threshold,
        max_steps=max_steps,
        target=target_delta_loglik,
        other_init=dict(initial_estimate.parameters),
        warnings_out=warnings_out,
    )

    # Stitch arms together (lower arm reversed so theta ascends).
    theta_vals = np.concatenate(
        [np.asarray(lower_points[::-1]), [theta_hat], np.asarray(upper_points)]
    )
    obj_vals = np.concatenate([np.asarray(lower_obj[::-1]), [objective_hat], np.asarray(upper_obj)])

    ci_lower = _interp_crossing(lower_points, lower_obj, threshold, theta_hat, objective_hat)
    ci_upper = _interp_crossing(upper_points, upper_obj, threshold, theta_hat, objective_hat)

    crossed_lower = _crossed(lower_obj, threshold)
    crossed_upper = _crossed(upper_obj, threshold)
    hit_lb = bool(lower_points) and abs(lower_points[-1] - lb) < 1e-12
    hit_ub = bool(upper_points) and abs(upper_points[-1] - ub) < 1e-12

    max_delta = float(np.max(obj_vals - objective_hat)) if obj_vals.size else 0.0
    if max_delta < 0.1 * threshold_offset:
        shape: ProfileShape = "flat"
        if not crossed_lower and not crossed_upper:
            warnings_out.append(
                "profile is flat: |D(theta) - D(theta_hat)| stays below "
                f"{0.1 * threshold_offset:.3g}; parameter is not "
                "practically identifiable"
            )
    elif crossed_lower and crossed_upper:
        shape = "bounded"
    elif crossed_upper and not crossed_lower:
        shape = "one_sided_upper"
        if hit_lb:
            warnings_out.append(
                f"lower arm hit parameter bound lb={lb:.6g} without crossing the threshold"
            )
        else:
            warnings_out.append("lower arm exhausted max_steps without crossing the threshold")
    elif crossed_lower and not crossed_upper:
        shape = "one_sided_lower"
        if hit_ub:
            warnings_out.append(
                f"upper arm hit parameter bound ub={ub:.6g} without crossing the threshold"
            )
        else:
            warnings_out.append("upper arm exhausted max_steps without crossing the threshold")
    else:
        shape = "flat"
        warnings_out.append(
            "neither arm crossed the threshold; parameter may be practically non-identifiable"
        )

    for arm_name, arm_obj in (("lower", lower_obj), ("upper", upper_obj)):
        if _non_monotone(arm_obj):
            warnings_out.append(
                f"{arm_name} arm is non-monotone; consider multi-start on the full problem"
            )

    return ProfileLikelihoodResult(
        parameter=parameter_name,
        theta_values=theta_vals,
        neg_log_lik=obj_vals,
        theta_hat=theta_hat,
        objective_hat=objective_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        threshold=threshold,
        shape=shape,
        warnings=warnings_out,
    )


def profile_all(
    experiment: Experiment,
    data: dict,
    *,
    confidence_level: float = 0.95,
    initial_estimate: EstimationResult | None = None,
    **kwargs,
) -> dict[str, ProfileLikelihoodResult]:
    """Run :func:`profile_likelihood` for every unknown parameter."""
    if initial_estimate is None:
        initial_estimate = estimate_parameters(experiment, data)
    return {
        name: profile_likelihood(
            experiment,
            data,
            name,
            confidence_level=confidence_level,
            initial_estimate=initial_estimate,
            **kwargs,
        )
        for name in initial_estimate.parameter_names
    }


def _profile_direction(
    experiment: Experiment,
    data: dict,
    name: str,
    theta_hat: float,
    objective_hat: float,
    *,
    direction: int,
    bound: float,
    step0: float,
    threshold: float,
    max_steps: int,
    target: float,
    other_init: dict[str, float],
    warnings_out: list[str],
) -> tuple[list[float], list[float]]:
    """Walk out in one direction, re-solving at each step.

    Returns (theta_values, objective_values) in step order (starting
    from the point *after* ``theta_hat``).
    """
    theta_vals: list[float] = []
    obj_vals: list[float] = []
    current_theta = theta_hat
    current_obj = objective_hat
    step = step0
    warm = dict(other_init)

    for _ in range(max_steps):
        proposal = current_theta + direction * step
        # Clip to bound.
        if direction > 0 and proposal >= bound:
            proposal = bound
        if direction < 0 and proposal <= bound:
            proposal = bound

        try:
            warm_no_fixed = {k: v for k, v in warm.items() if k != name}
            res = estimate_parameters(
                experiment,
                data,
                initial_guess=warm_no_fixed,
                fixed_parameters={name: proposal},
            )
            new_obj = float(res.objective)
        except Exception as exc:  # pragma: no cover - solver failures
            # One retry with halved step.
            step = step / 2.0
            if step < 1e-12 * max(abs(theta_hat), 1.0):
                warnings_out.append(
                    f"profile direction {direction:+d} abandoned after NLP "
                    f"failure near theta={proposal:.6g}: {exc}"
                )
                break
            continue

        theta_vals.append(proposal)
        obj_vals.append(new_obj)
        warm = dict(res.parameters)

        delta = new_obj - current_obj
        current_theta = proposal
        current_obj = new_obj

        # Stopping conditions.
        if new_obj > threshold:
            break
        if abs(proposal - bound) < 1e-12:
            break

        # Adaptive step (Raue 2009 Sec 2.3): step *= target / delta, clipped.
        if delta > 0:
            factor = target / delta
            step = step * max(0.1, min(10.0, factor))
        else:
            step = step * 10.0
        # Safety ceiling: never step further than the remaining distance
        # to the bound, and never below a tiny fraction of the estimate.
        if np.isfinite(bound):
            step = min(step, abs(bound - current_theta))
        step = max(step, 1e-12 * max(abs(theta_hat), 1.0))

    return theta_vals, obj_vals


def _interp_crossing(
    theta_vals: list[float],
    obj_vals: list[float],
    threshold: float,
    theta_hat: float,
    objective_hat: float,
) -> float | None:
    """Linearly interpolate the theta at which the objective crosses the threshold.

    The arm's "previous" point before its first entry is ``(theta_hat,
    objective_hat)``; beyond that, consecutive arm points are paired.
    Returns ``None`` if no crossing occurred.
    """
    if not theta_vals or not obj_vals:
        return None

    def lerp(t0: float, t1: float, o0: float, o1: float) -> float:
        if o1 == o0:
            return float(t1)
        frac = (threshold - o0) / (o1 - o0)
        return float(t0 + frac * (t1 - t0))

    prev_t = theta_hat
    prev_o = objective_hat
    for t, o in zip(theta_vals, obj_vals):
        if o >= threshold:
            return lerp(prev_t, t, prev_o, o)
        prev_t, prev_o = t, o
    return None


def _crossed(obj_vals: list[float], threshold: float) -> bool:
    return bool(obj_vals) and max(obj_vals) >= threshold


def _non_monotone(obj_vals: list[float]) -> bool:
    """Detect a non-monotone arm (later points lower than earlier ones by a
    meaningful margin).

    We tolerate small numerical oscillations: a drop of more than
    ``1e-3 * max(obj)`` relative to a previous point counts.
    """
    if len(obj_vals) < 2:
        return False
    arr = np.asarray(obj_vals, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drops = running_max - arr
    scale = max(arr.max(), 1.0)
    return bool(np.any(drops > 1e-3 * scale))
