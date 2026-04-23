"""Sequential model-discrimination loop.

Alternates between fitting all candidate models and designing the
experiment that best separates the survivors, in the spirit of
:func:`discopt.doe.sequential.sequential_doe` but with a discrimination
criterion and an AIC-weight stopping rule.

Usage
-----

>>> from discopt.doe import sequential_discrimination, DiscriminationCriterion
>>> rounds = sequential_discrimination(
...     experiments={"first": FirstOrderExp(), "second": SecondOrderExp()},
...     initial_data={"C_0": 0.95, "C_1": 0.80, ...},
...     design_bounds={"t": (0.0, 10.0), "C0": (1.0, 5.0)},
...     n_rounds=5,
...     run_experiment=my_simulator,
...     criterion=DiscriminationCriterion.BF,
... )
>>> rounds[-1].selection.weights   # per-model AIC weights at the final round
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Union

import numpy as np

from discopt.doe.discrimination import (
    DiscriminationCriterion,
    DiscriminationDesignResult,
    discriminate_design,
)
from discopt.doe.selection import ModelSelectionResult, model_selection
from discopt.estimate import EstimationResult, Experiment, estimate_parameters


@dataclass
class DiscriminationRound:
    """One round of :func:`sequential_discrimination`.

    Attributes
    ----------
    round : int
        Zero-indexed round number.
    estimation_results : dict[str, EstimationResult]
        Fit for each candidate model on data accumulated up to and
        including this round.
    selection : ModelSelectionResult
        AIC / BIC weights after this round's fit.
    design : DiscriminationDesignResult or None
        Recommended next design. ``None`` when the loop stopped because
        ``stop_when_dominant`` was met.
    collected_data : dict or None
        New data gathered at this round's design (present only when
        ``run_experiment`` was provided).
    """

    round: int
    estimation_results: dict[str, EstimationResult]
    selection: ModelSelectionResult
    design: DiscriminationDesignResult | None = None
    collected_data: dict | None = None


def sequential_discrimination(
    experiments: dict[str, Experiment],
    initial_data: dict[str, Union[float, np.ndarray]],
    design_bounds: dict[str, tuple[float, float]],
    *,
    n_rounds: int = 5,
    run_experiment: Callable[[dict[str, float]], dict] | None = None,
    criterion: DiscriminationCriterion = DiscriminationCriterion.BF,
    selection_method: Literal["aic", "bic"] = "aic",
    stop_when_dominant: float = 0.95,
    initial_guesses: dict[str, dict[str, float]] | None = None,
    callback: Callable[[DiscriminationRound], None] | None = None,
    mi_samples: int = 2000,
    n_starts: int = 10,
    local_refine: bool = True,
    seed: int | None = None,
) -> list[DiscriminationRound]:
    """Run the sequential discrimination loop.

    At each round:

    1. Fit every candidate model with
       :func:`discopt.estimate.estimate_parameters` on the data
       accumulated so far.
    2. Score the fits with :func:`discopt.doe.selection.model_selection`
       using ``selection_method``.
    3. If ``max(weights) >= stop_when_dominant``, stop and return.
    4. Otherwise call :func:`discriminate_design` to pick the next
       design; if ``run_experiment`` is supplied, run it, accumulate
       the returned data, and loop. If not, the final round carries
       the recommended design and the caller is expected to run it and
       re-enter the loop with the updated data.

    Parameters
    ----------
    experiments : dict[str, Experiment]
        Candidate models keyed by user-chosen names.
    initial_data : dict
        Starting dataset used for the first round's fits.
    design_bounds : dict[str, tuple[float, float]]
        Bounds on the design variables.
    n_rounds : int, default 5
        Maximum number of rounds.
    run_experiment : callable, optional
        ``design -> new_data`` simulator / lab runner. When ``None``,
        the loop returns after the first design is proposed.
    criterion : DiscriminationCriterion, default BF
    selection_method : {"aic", "bic"}, default "aic"
    stop_when_dominant : float, default 0.95
        Early-stop threshold on the maximum selection weight.
    initial_guesses : dict[str, dict[str, float]], optional
        Per-model initial-guess dicts for the first round's fits.
    callback : callable, optional
        ``DiscriminationRound -> None`` invoked after every round.
    mi_samples : int, default 2000
        Forwarded to :func:`discriminate_design` for MI criterion.
    n_starts, local_refine, seed
        Forwarded to :func:`discriminate_design`.

    Returns
    -------
    list[DiscriminationRound]
    """
    if len(experiments) < 2:
        raise ValueError(f"Need at least 2 candidate models; got {list(experiments)}")
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1; got {n_rounds}")

    initial_guesses = dict(initial_guesses or {})
    rng = np.random.default_rng(seed)

    # Copy initial data so we can accumulate without mutating the caller's dict.
    accumulated: dict[str, Union[float, np.ndarray]] = dict(initial_data)
    rounds: list[DiscriminationRound] = []

    for k in range(n_rounds):
        estimation_results = {
            name: estimate_parameters(exp, accumulated, initial_guess=initial_guesses.get(name))
            for name, exp in experiments.items()
        }
        selection = model_selection(estimation_results, method=selection_method)
        top_weight = max(selection.weights.values()) if selection.weights else 0.0

        if top_weight >= stop_when_dominant or k == n_rounds - 1:
            # Stop. No design for this round unless the caller explicitly
            # wants the "final recommendation" semantics; we omit it for
            # clarity. If the stop came from max_weight, design is None;
            # if from n_rounds exhaustion and there is still a run_experiment
            # callback, we still leave design = None since there's no next
            # iteration to consume it.
            round_record = DiscriminationRound(
                round=k,
                estimation_results=estimation_results,
                selection=selection,
                design=None,
                collected_data=None,
            )
            rounds.append(round_record)
            if callback is not None:
                callback(round_record)
            break

        # Design next experiment using the current parameter estimates.
        param_estimates = {name: dict(res.parameters) for name, res in estimation_results.items()}
        round_seed = int(rng.integers(0, 2**31 - 1))
        design = discriminate_design(
            experiments=experiments,
            param_estimates=param_estimates,
            design_bounds=design_bounds,
            criterion=criterion,
            model_priors=selection.weights,  # posterior-like weighting
            n_starts=n_starts,
            local_refine=local_refine,
            mi_samples=mi_samples,
            seed=round_seed,
        )

        new_data = None
        if run_experiment is not None:
            new_data = run_experiment(design.design)
            accumulated = _accumulate_data(accumulated, new_data)
            # Update initial guesses with the fitted values so the next
            # round starts from the current best estimates.
            for name, res in estimation_results.items():
                initial_guesses[name] = dict(res.parameters)

        round_record = DiscriminationRound(
            round=k,
            estimation_results=estimation_results,
            selection=selection,
            design=design,
            collected_data=new_data,
        )
        rounds.append(round_record)
        if callback is not None:
            callback(round_record)

        if run_experiment is None:
            # Caller-driven mode: hand back the design and let them run.
            break

    return rounds


def _accumulate_data(
    existing: dict[str, Union[float, np.ndarray]],
    new: dict[str, Union[float, np.ndarray]],
) -> dict[str, Union[float, np.ndarray]]:
    """Merge new observations into the accumulated dataset.

    Keys that collide are concatenated along axis 0 after being
    promoted to arrays; fresh keys are added as-is. This mirrors the
    conventions of :func:`sequential_doe`.
    """
    merged: dict[str, Union[float, np.ndarray]] = dict(existing)
    for key, val in new.items():
        if key in merged:
            prev = np.atleast_1d(merged[key])
            cur = np.atleast_1d(val)
            merged[key] = np.concatenate([prev, cur])
        else:
            merged[key] = val
    return merged


__all__ = [
    "DiscriminationRound",
    "sequential_discrimination",
]
