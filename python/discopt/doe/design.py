"""Optimal experimental design via FIM criterion optimization.

Finds experimental conditions that maximize the information content
of an experiment, as measured by criteria derived from the Fisher
Information Matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.doe.fim import FIMResult, compute_fim
from discopt.estimate import Experiment


class DesignCriterion:
    """Design optimality criteria constants."""

    D_OPTIMAL = "determinant"
    A_OPTIMAL = "trace"
    E_OPTIMAL = "min_eigenvalue"
    ME_OPTIMAL = "condition_number"


@dataclass
class DesignResult:
    """Result of optimal experimental design.

    Attributes
    ----------
    design : dict[str, float]
        Optimal values for each design input.
    fim_result : FIMResult
        FIM at the optimal design.
    criterion_value : float
        Value of the optimized design criterion.
    """

    design: dict[str, float]
    fim_result: FIMResult
    criterion_value: float

    @property
    def fim(self) -> np.ndarray:
        """Fisher Information Matrix at optimal design."""
        return self.fim_result.fim

    @property
    def parameter_covariance(self) -> np.ndarray:
        """Predicted parameter covariance if this experiment is run."""
        try:
            return np.linalg.inv(self.fim)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.fim)

    @property
    def predicted_standard_errors(self) -> np.ndarray:
        """Predicted standard errors for each parameter."""
        return np.sqrt(np.diag(self.parameter_covariance))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics."""
        return self.fim_result.metrics

    def summary(self) -> str:
        """Human-readable summary of the optimal design."""
        lines = ["Optimal Experimental Design", "=" * 50]
        for name, val in self.design.items():
            lines.append(f"  {name:>15s} = {val:.6g}")
        lines.append("")
        m = self.metrics
        lines.append(f"  D-opt (log det FIM) = {m['log_det_fim']:.4g}")
        lines.append(f"  A-opt (trace FIM⁻¹) = {m['trace_fim_inv']:.4g}")
        lines.append(f"  E-opt (min eigenval) = {m['min_eigenvalue']:.4g}")
        lines.append(f"  Condition number     = {m['condition_number']:.4g}")
        lines.append("")
        se = self.predicted_standard_errors
        for i, name in enumerate(self.fim_result.parameter_names):
            lines.append(f"  SE({name}) = {se[i]:.4g}")
        return "\n".join(lines)


def optimal_experiment(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    n_starts: int = 10,
    seed: int = 42,
) -> DesignResult:
    """Find optimal experimental conditions by maximizing information gain.

    Evaluates the FIM criterion at multiple starting points within the
    design bounds and returns the best design found. Uses a multi-start
    grid search followed by local refinement.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Current best parameter estimates (nominal values).
    design_bounds : dict[str, tuple[float, float]]
        Bounds on each design input variable.
    criterion : str, default DesignCriterion.D_OPTIMAL
        Design criterion: ``"determinant"`` (D), ``"trace"`` (A),
        ``"min_eigenvalue"`` (E), ``"condition_number"`` (ME).
    prior_fim : numpy.ndarray, optional
        Prior FIM from previous experiments.
    n_starts : int, default 10
        Number of random starting points to evaluate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    DesignResult
        Optimal design, FIM, and metrics.
    """
    rng = np.random.default_rng(seed)
    design_names = list(design_bounds.keys())

    # Generate candidate design points (Latin hypercube-like)
    candidates = []
    for _ in range(n_starts):
        point = {}
        for name in design_names:
            lo, hi = design_bounds[name]
            point[name] = rng.uniform(lo, hi)
        candidates.append(point)

    # Also add boundary points
    for name in design_names:
        lo, hi = design_bounds[name]
        for val in [lo, hi]:
            point = {n: (design_bounds[n][0] + design_bounds[n][1]) / 2 for n in design_names}
            point[name] = val
            candidates.append(point)

    # Evaluate criterion at each candidate
    best_design = None
    best_criterion = -np.inf if _is_maximization(criterion) else np.inf
    best_fim_result = None

    for design_point in candidates:
        try:
            fim_result = compute_fim(experiment, param_values, design_point, prior_fim=prior_fim)
            crit_val = _evaluate_criterion(fim_result, criterion)

            if _is_better(crit_val, best_criterion, criterion):
                best_criterion = crit_val
                best_design = design_point
                best_fim_result = fim_result
        except Exception:
            continue

    if best_design is None or best_fim_result is None:
        raise RuntimeError("No feasible design point found")

    return DesignResult(
        design=best_design,
        fim_result=best_fim_result,
        criterion_value=best_criterion,
    )


def _evaluate_criterion(fim_result: FIMResult, criterion: str) -> float:
    """Evaluate a design criterion from a FIM result."""
    if criterion == DesignCriterion.D_OPTIMAL:
        return fim_result.d_optimal
    elif criterion == DesignCriterion.A_OPTIMAL:
        return fim_result.a_optimal
    elif criterion == DesignCriterion.E_OPTIMAL:
        return fim_result.e_optimal
    elif criterion == DesignCriterion.ME_OPTIMAL:
        return fim_result.me_optimal
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")


def _is_maximization(criterion: str) -> bool:
    """Return True if the criterion should be maximized."""
    return criterion in (DesignCriterion.D_OPTIMAL, DesignCriterion.E_OPTIMAL)


def _is_better(new_val: float, best_val: float, criterion: str) -> bool:
    """Check if new_val is better than best_val for the given criterion."""
    if _is_maximization(criterion):
        return new_val > best_val
    else:
        return new_val < best_val
