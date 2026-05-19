"""Active-learning loop for response optimization.

When to use this module
-----------------------

Use :func:`optimize_round` when the question is **"what input gives the
best response?"** -- minimize cost, maximize yield, find the operating
point that lands on a target -- and experiments are expensive enough
that you cannot afford a full factorial or response-surface design.

This is *active learning*: every new run is chosen to be the most
informative for the optimization objective, not to fill out a fixed
matrix.

Contrast with the rest of :mod:`discopt.doe`:

* :func:`~discopt.doe.optimal_experiment` -- design for **parameter
  estimation** (maximize Fisher information). Different objective:
  precise model parameters, not best response.
* :func:`~discopt.doe.factorial_2level_design` -- design for
  **factor screening** ("does this factor matter at all?"). Use that
  first if you don't yet know which factors to optimize over.
* :func:`~discopt.doe.batch_optimal_experiment` -- batch FIM design.
  Same parameter-estimation objective, batch interface.

A good full sequence is **screen → optimize**: run a 2-level factorial
to drop irrelevant factors, then call :func:`optimize_round` on the
surviving ones.

How one round works
-------------------

A single call to :func:`optimize_round`:

1. opens the workbook and reads every row that has a non-empty
   response value (the *completed* runs);
2. (optionally) standardizes inputs to zero mean + unit variance;
3. fits the chosen :class:`~discopt.doe.surrogate.Surrogate` to the
   completed runs;
4. draws a pool of ``n_candidates`` Sobol points inside the input
   bounds;
5. scores each candidate with the acquisition function;
6. greedily picks the top-``batch_size`` -- after each pick, the
   chosen point is added back into the surrogate with its predicted
   mean as a "fantasy" response, so the next pick sees lower
   uncertainty there and the batch diversifies;
7. appends the batch to the workbook as new pending runs and returns
   an :class:`OptimizationRoundResult`.

You then run the experiments, fill in the response column of the
workbook, and call :func:`optimize_round` again. There is no fixed
budget -- stop when the incumbent stops improving, or when you've
spent the runs you can afford.

Surrogate choice (convenience vs. control)
------------------------------------------

The ``surrogate`` argument accepts three escalating levels of
specificity. They all funnel through
:func:`~discopt.doe.surrogate.coerce_surrogate`:

* **String preset** -- ``"gp"`` (default-ish; Matern(5/2) + white
  noise) or ``"response-surface"`` (degree-2 polynomial + Bayesian
  ridge). Zero knobs. Good for first contact.
* **Any scikit-learn-compatible estimator** -- wrapped automatically
  in :class:`~discopt.doe.surrogate._SklearnUQAdapter`, which probes
  for ``predict(X, return_std=True)`` (sklearn GP / BayesianRidge /
  ARDRegression), then ``predict(X, return_interval=True)``
  (`pycse.sklearn.lpr.LinearLPR`), then falls back to a residual
  bootstrap. This is the right path when you want a specific GP
  kernel or LPR-style local prediction without writing any adapter
  code.
* **Custom object** implementing the
  :class:`~discopt.doe.surrogate.Surrogate` protocol (``fit`` +
  ``predict`` returning ``(mean, std)``). Opt in by setting the class
  attribute ``_is_discopt_surrogate = True`` so the router does not
  re-wrap it. Full escape hatch for bespoke Bayesian models.

The acquisition function only sees ``(mean, std)`` arrays. It never
looks inside the surrogate. That is the abstraction boundary --
swap kernels, swap libraries, none of the optimization code cares.

Acquisition choice
------------------

* ``"expected_improvement"`` (alias ``"ei"``) -- the textbook
  Bayesian-optimization choice. Balances exploitation and exploration
  through the surrogate's uncertainty. Default.
* ``"ucb"`` / ``"lcb"`` / ``"confidence_bound"`` -- ``μ ± κ σ`` with
  the sign tied to direction. Tune the explore/exploit balance via
  ``acquisition_kwargs={"kappa": ...}``. Larger κ explores more.
* ``"steepest_ascent"`` -- ignores σ entirely; picks the point with
  the best predicted mean in the optimization direction. Use with a
  response-surface surrogate to reproduce classical Box-Wilson
  behavior. Not recommended with a GP -- without uncertainty you give
  up the main reason to fit a GP in the first place.

Categorical and mixed-input factors
-----------------------------------

The driver currently expects every input column in the workbook to be
numeric (it builds an X matrix). If you have categorical factors,
encode them upstream as 0/1 indicator columns or a small integer
code, and add the encoded columns to the workbook's ``input_specs``.
Future versions may auto-encode at the adapter boundary.

Quick start
-----------

>>> from discopt.doe import optimize_round, OptimizationCriterion
>>> result = optimize_round(
...     workbook="opt.xlsx",
...     criterion=OptimizationCriterion.MAXIMIZE,
...     surrogate="gp",                 # or a sklearn estimator, or a Surrogate
...     acquisition="expected_improvement",
...     batch_size=4,
... )
>>> print(result.next_designs)
>>> print(f"incumbent so far: y={result.incumbent_y:.3f} at {result.incumbent_x}")

References
----------

* Jones, Schonlau, Welch (1998). *Efficient Global Optimization of
  Expensive Black-Box Functions.* JoGO 13:455-492. -- EI + Kriging.
* Box, Wilson (1951). *On the Experimental Attainment of Optimum
  Conditions.* JRSS-B 13:1-45. -- steepest-ascent / response-surface
  methodology.
* Snoek, Larochelle, Adams (2012). *Practical Bayesian Optimization
  of Machine Learning Algorithms.* NeurIPS. -- modern BO recipes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from discopt.doe.acquisition import resolve_acquisition
from discopt.doe.surrogate import Surrogate, coerce_surrogate
from discopt.doe.workbook import Workbook


class OptimizationCriterion(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    @property
    def direction(self) -> int:
        return 1 if self is OptimizationCriterion.MAXIMIZE else -1


@dataclass
class OptimizationRoundResult:
    """Outcome of one active-learning round."""

    next_designs: list[dict[str, float]]
    new_run_ids: list[int]
    incumbent_x: dict[str, float] | None
    incumbent_y: float | None
    acquisition_scores: list[float]
    surrogate_mode: str | None
    n_completed: int
    workbook_path: str
    log: list[str] = field(default_factory=list)


def optimize_round(
    workbook: str | Path | Workbook,
    *,
    criterion: OptimizationCriterion | str = OptimizationCriterion.MAXIMIZE,
    surrogate: object = "gp",
    acquisition: str | Callable = "expected_improvement",
    batch_size: int = 1,
    n_candidates: int = 2048,
    candidate_sampler: str = "sobol",
    bounds: Sequence[tuple[float, float]] | None = None,
    input_names: Sequence[str] | None = None,
    standardize_inputs: bool = True,
    seed: int | None = None,
    acquisition_kwargs: dict[str, Any] | None = None,
) -> OptimizationRoundResult:
    """Propose the next batch of experiments using an active-learning surrogate.

    Parameters
    ----------
    workbook : path or Workbook
        Workbook containing the completed runs. The next batch is appended.
    criterion : OptimizationCriterion or str
        ``"maximize"`` or ``"minimize"``.
    surrogate : str, sklearn estimator, or Surrogate
        Passed through :func:`~discopt.doe.surrogate.coerce_surrogate`.
    acquisition : str or callable
        Looked up via :func:`~discopt.doe.acquisition.resolve_acquisition`.
    batch_size : int, default 1
        Number of new experiments to recommend.
    n_candidates : int, default 2048
        Size of the candidate pool sampled inside the bounding box.
    candidate_sampler : ``"sobol"`` or ``"uniform"``
        Sobol gives better space-filling for the same budget.
    bounds : sequence of (lo, hi), optional
        Per-input box. Defaults to the input_specs stored in the workbook.
    input_names : sequence of str, optional
        Names of the input columns. Defaults to the workbook's input_specs
        in the original order.
    standardize_inputs : bool, default True
        If True, the surrogate sees inputs standardized to zero mean +
        unit variance (computed on the completed runs). Acquisition is
        evaluated in the standardized space; recommended points are
        unstandardized before being written back.
    seed : int, optional
        Reproducible candidate sampling.
    acquisition_kwargs : dict, optional
        Extra kwargs forwarded to the acquisition function (e.g.
        ``{"xi": 0.01}`` for EI, ``{"kappa": 2.5}`` for UCB).
    """
    crit = OptimizationCriterion(criterion) if isinstance(criterion, str) else criterion
    direction = crit.direction
    acq_fn = resolve_acquisition(acquisition)
    acq_kwargs = dict(acquisition_kwargs or {})

    wb = workbook if isinstance(workbook, Workbook) else Workbook.open(Path(workbook))

    specs = wb.input_specs()
    names = list(input_names) if input_names is not None else [s.name for s in specs]
    if bounds is None:
        bounds_arr = np.array([(s.lb, s.ub) for s in specs], dtype=float)
    else:
        bounds_arr = np.asarray(list(bounds), dtype=float)
    if bounds_arr.shape != (len(names), 2):
        raise ValueError(f"bounds shape {bounds_arr.shape} does not match {len(names)} input(s)")

    response = wb.response_name()
    completed = wb.completed_runs()
    if not completed:
        raise ValueError(
            "no completed runs in workbook -- fill in at least one response "
            "column before calling optimize_round"
        )

    X_raw = np.array([[float(r[n]) for n in names] for r in completed], dtype=float)
    y = np.array([float(r[response]) for r in completed], dtype=float)

    if standardize_inputs:
        mu_x = X_raw.mean(axis=0)
        sd_x = X_raw.std(axis=0, ddof=0)
        sd_x = np.where(sd_x > 0.0, sd_x, 1.0)
        X = (X_raw - mu_x) / sd_x
    else:
        mu_x = np.zeros(len(names))
        sd_x = np.ones(len(names))
        X = X_raw

    s = coerce_surrogate(surrogate)
    s.fit(X, y)

    rng = np.random.default_rng(seed)
    candidates_raw = _sample_candidates(bounds_arr, n_candidates, candidate_sampler, rng)
    candidates = (candidates_raw - mu_x) / sd_x if standardize_inputs else candidates_raw

    incumbent_idx = int(np.argmax(direction * y))
    incumbent_y = float(y[incumbent_idx])
    incumbent_x = {n: float(X_raw[incumbent_idx, i]) for i, n in enumerate(names)}

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

        # Mean-imputation: pretend the chosen point's response is the
        # surrogate's mean. Re-fit so the next pick sees lower
        # uncertainty there and diversifies.
        mu_pick, _ = s.predict(candidates[pick : pick + 1])
        X_fantasy = np.vstack([X_fantasy, candidates[pick : pick + 1]])
        y_fantasy = np.concatenate([y_fantasy, mu_pick])
        if direction * float(mu_pick[0]) > direction * incumbent_for_acq:
            incumbent_for_acq = float(mu_pick[0])
        s = coerce_surrogate(surrogate)
        s.fit(X_fantasy, y_fantasy)

    next_designs = [
        {n: float(candidates_raw[i, j]) for j, n in enumerate(names)} for i in chosen_idx
    ]
    batch_idx = wb.next_batch_index()
    new_run_ids = wb.append_runs(batch_idx, next_designs)
    wb.log(
        "optimize",
        {
            "criterion": crit.value,
            "acquisition": acquisition if isinstance(acquisition, str) else acq_fn.__name__,
            "batch_size": int(batch_size),
            "n_completed": len(completed),
            "surrogate_mode": getattr(s, "mode", None),
        },
    )
    wb.save()

    return OptimizationRoundResult(
        next_designs=next_designs,
        new_run_ids=new_run_ids,
        incumbent_x=incumbent_x,
        incumbent_y=incumbent_y,
        acquisition_scores=chosen_scores,
        surrogate_mode=getattr(s, "mode", None),
        n_completed=len(completed),
        workbook_path=str(wb.path),
    )


# ──────────────────────────────────────────────────────────────────
# Candidate sampling
# ──────────────────────────────────────────────────────────────────


def _sample_candidates(
    bounds: np.ndarray, n: int, sampler: str, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` points uniformly in the box defined by ``bounds``."""
    d = bounds.shape[0]
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    if sampler == "uniform":
        u = rng.uniform(size=(n, d))
    elif sampler == "sobol":
        try:
            from scipy.stats import qmc

            m = int(math.ceil(math.log2(max(n, 2))))
            engine = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
            u = engine.random_base2(m=m)[:n]
        except ImportError:
            u = rng.uniform(size=(n, d))
    else:
        raise ValueError(f"unknown candidate_sampler {sampler!r}")
    return lo + (hi - lo) * u


_ = Surrogate  # ensure protocol import is exported for downstream users


__all__ = [
    "OptimizationCriterion",
    "OptimizationRoundResult",
    "optimize_round",
]
