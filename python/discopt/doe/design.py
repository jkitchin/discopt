"""Optimal experimental design via FIM criterion optimization.

Finds experimental conditions that maximize the information content
of an experiment, as measured by criteria derived from the Fisher
Information Matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from discopt.doe.fim import FIMResult, compute_fim, compute_fim_batch
from discopt.estimate import Experiment

_SINGULAR_SENTINEL = 1e12

# A constraint is a callable that takes a design dict and returns a scalar.
# Equality constraints are enforced as ``fn(design) == 0`` and inequality
# constraints as ``fn(design) >= 0`` (scipy SLSQP convention).
DesignConstraint = Callable[[dict[str, float]], float]


def sum_constraint(variables: Sequence[str], total: float = 1.0) -> DesignConstraint:
    """Return an equality constraint that enforces ``sum(variables) == total``.

    Useful for mixture designs where component fractions or volumes must
    sum to a fixed amount.

    Parameters
    ----------
    variables : sequence of str
        Design variable names that participate in the sum.
    total : float, default ``1.0``
        Required sum value.

    Returns
    -------
    Callable[[dict[str, float]], float]
        Function ``g(design)`` returning ``sum(design[v] for v in variables) - total``.
        Pass to ``optimal_experiment(..., equality_constraints=[g])``.
    """
    names = tuple(variables)
    target = float(total)

    def _g(design: dict[str, float]) -> float:
        return float(sum(design[n] for n in names) - target)

    return _g


def sample_simplex(
    variables: Sequence[str],
    total: float,
    rng: np.random.Generator,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Draw a Dirichlet sample on the simplex ``sum(variables) == total``.

    When ``bounds`` is supplied, samples that fall outside the per-axis
    bounds are clipped and rescaled; for tight bounds this may bias the
    sample, but it always returns a feasible point on the equality
    surface (up to one rescaling pass).
    """
    names = list(variables)
    q = len(names)
    weights = rng.dirichlet(np.ones(q))
    point = {n: float(total * w) for n, w in zip(names, weights)}
    if bounds is None:
        return point
    # Clip to bounds, then rescale the free mass back to ``total``.
    clipped = {n: float(np.clip(point[n], bounds[n][0], bounds[n][1])) for n in names}
    s = sum(clipped.values())
    if s > 0 and abs(s - total) > 1e-12:
        scale = total / s
        clipped = {n: clipped[n] * scale for n in names}
    return clipped


def project_to_simplex(
    point: dict[str, float],
    variables: Sequence[str],
    total: float = 1.0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Project ``point`` onto ``sum(variables) == total`` with non-negativity.

    Uses the standard Euclidean projection onto the probability simplex
    (Wang & Carreira-Perpiñán, 2013), scaled by ``total``. Variables
    outside ``variables`` are returned unchanged. If ``bounds`` is given,
    each projected component is then clipped to its box; this may push
    the sum slightly off ``total`` if the bounds are tight, but provides
    a reasonable warm start for SLSQP.
    """
    names = list(variables)
    v = np.array([point[n] for n in names], dtype=float)
    # Project v onto {w : w >= 0, sum(w) = total}.
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    rho = np.where(u - cssv / (np.arange(len(u)) + 1) > 0)[0]
    if len(rho) == 0:
        rho_idx = 0
    else:
        rho_idx = int(rho[-1])
    theta = cssv[rho_idx] / (rho_idx + 1)
    w = np.maximum(v - theta, 0.0)

    projected: dict[str, float] = dict(point)
    for n, val in zip(names, w):
        projected[n] = float(val)
    if bounds is not None:
        for n in names:
            lo, hi = bounds[n]
            projected[n] = float(np.clip(projected[n], lo, hi))
    return projected


class DesignCriterion:
    """Design optimality criteria constants."""

    D_OPTIMAL = "determinant"
    A_OPTIMAL = "trace"
    E_OPTIMAL = "min_eigenvalue"
    ME_OPTIMAL = "condition_number"


class BatchStrategy:
    """Strategies for batch / parallel experimental design."""

    GREEDY = "greedy"
    JOINT = "joint"
    PENALIZED = "penalized"


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


@dataclass
class BatchDesignResult:
    """Result of joint / batch optimal experimental design.

    Attributes
    ----------
    designs : list[dict[str, float]]
        Optimal values for each of the ``N`` designs in the batch.
    fim_results : list[FIMResult]
        Per-experiment FIMs, each computed without the prior or other
        batch members folded in (so ``sum(r.fim for r in fim_results)
        + prior_fim == joint_fim``).
    joint_fim : numpy.ndarray
        Sum of per-experiment FIMs plus ``prior_fim`` (if supplied).
        This is the FIM that the chosen criterion was evaluated on.
    criterion_value : float
        Value of the optimized design criterion on ``joint_fim``.
    strategy : str
        Name of the batch strategy used (see :class:`BatchStrategy`).
    per_round_criterion : list[float] or None
        For greedy / penalized strategies, the criterion value after
        each successive pick. ``None`` for joint strategy.
    """

    designs: list[dict[str, float]]
    fim_results: list[FIMResult]
    joint_fim: np.ndarray
    criterion_value: float
    strategy: str
    per_round_criterion: list[float] | None = None

    @property
    def n_experiments(self) -> int:
        return len(self.designs)

    @property
    def parameter_covariance(self) -> np.ndarray:
        """Predicted parameter covariance after running the full batch."""
        try:
            return np.linalg.inv(self.joint_fim)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.joint_fim)

    @property
    def predicted_standard_errors(self) -> np.ndarray:
        """Predicted standard errors for each parameter."""
        return np.sqrt(np.diag(self.parameter_covariance))

    @property
    def metrics(self) -> dict[str, float]:
        """All optimality metrics evaluated on the joint FIM."""
        return _metrics_from_fim(self.joint_fim)

    @property
    def parameter_names(self) -> list[str]:
        return self.fim_results[0].parameter_names if self.fim_results else []

    def to_design_result(self) -> DesignResult:
        """Return a single-experiment ``DesignResult`` view (requires N=1)."""
        if self.n_experiments != 1:
            raise ValueError(
                f"to_design_result requires n_experiments == 1, got {self.n_experiments}"
            )
        return DesignResult(
            design=self.designs[0],
            fim_result=self.fim_results[0],
            criterion_value=self.criterion_value,
        )

    def summary(self) -> str:
        """Human-readable summary of the batch design."""
        lines = [
            f"Batch Optimal Design (N={self.n_experiments}, strategy={self.strategy})",
            "=" * 60,
        ]
        for i, design in enumerate(self.designs):
            lines.append(f"  Experiment {i + 1}:")
            for name, val in design.items():
                lines.append(f"    {name:>15s} = {val:.6g}")
        lines.append("")
        m = self.metrics
        lines.append(f"  D-opt (log det joint FIM) = {m['log_det_fim']:.4g}")
        lines.append(f"  A-opt (trace joint FIM⁻¹) = {m['trace_fim_inv']:.4g}")
        lines.append(f"  E-opt (min eigenval)      = {m['min_eigenvalue']:.4g}")
        lines.append(f"  Condition number          = {m['condition_number']:.4g}")
        lines.append("")
        se = self.predicted_standard_errors
        for i, name in enumerate(self.parameter_names):
            lines.append(f"  SE({name}) = {se[i]:.4g}")
        return "\n".join(lines)


def _metrics_from_fim(fim: np.ndarray) -> dict[str, float]:
    """All optimality metrics computed from a raw FIM matrix."""
    det = float(np.linalg.det(fim))
    log_det = float(np.log(det)) if det > 0 else float("-inf")
    try:
        tr_inv = float(np.trace(np.linalg.inv(fim)))
    except np.linalg.LinAlgError:
        tr_inv = float("inf")
    return {
        "log_det_fim": log_det,
        "trace_fim_inv": tr_inv,
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(fim))),
        "condition_number": float(np.linalg.cond(fim)),
    }


def _criterion_from_fim(fim: np.ndarray, criterion: str) -> float:
    """Evaluate a design criterion directly on a FIM matrix."""
    metrics = _metrics_from_fim(fim)
    if criterion == DesignCriterion.D_OPTIMAL:
        return metrics["log_det_fim"]
    elif criterion == DesignCriterion.A_OPTIMAL:
        return metrics["trace_fim_inv"]
    elif criterion == DesignCriterion.E_OPTIMAL:
        return metrics["min_eigenvalue"]
    elif criterion == DesignCriterion.ME_OPTIMAL:
        return metrics["condition_number"]
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")


def optimal_experiment(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
    prior_fim: np.ndarray | None = None,
    equality_constraints: Sequence[DesignConstraint] | None = None,
    inequality_constraints: Sequence[DesignConstraint] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    seed: int = 42,
) -> DesignResult:
    """Find optimal experimental conditions by maximizing information gain.

    Evaluates the FIM criterion at multiple starting points within the
    design bounds and refines the best candidate with a bounded local
    solver. When constraints are supplied, the refiner switches to SLSQP.

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
    equality_constraints : sequence of callable, optional
        Constraints ``g(design) == 0``. For mixture designs use
        :func:`sum_constraint`.
    inequality_constraints : sequence of callable, optional
        Constraints ``h(design) >= 0`` (scipy SLSQP convention).
    feasible_projection : callable, optional
        Maps a raw candidate to a constraint-feasible candidate. Used to
        seed the multi-start with feasible points. For mixture designs
        pass a partial of :func:`project_to_simplex`.
    n_starts : int, default 10
        Number of random starting points to evaluate.
    local_refine : bool, default True
        If True, refine the best multi-start candidate with
        scipy.optimize.minimize (L-BFGS-B without constraints, SLSQP
        when ``equality_constraints`` or ``inequality_constraints`` are
        supplied).
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    DesignResult
        Optimal design, FIM, and metrics.
    """
    design_names = list(design_bounds.keys())
    eq = list(equality_constraints) if equality_constraints else []
    ineq = list(inequality_constraints) if inequality_constraints else []
    candidates = _multi_start_candidates(
        design_bounds, n_starts, seed, projection=feasible_projection
    )

    best_design, best_criterion, best_fim_result = _scan_candidates(
        experiment, param_values, candidates, criterion, prior_fim
    )

    if best_design is None or best_fim_result is None:
        raise RuntimeError("No feasible design point found")

    if local_refine:
        refined = _refine_single_design(
            experiment,
            param_values,
            best_design,
            design_names,
            design_bounds,
            criterion,
            prior_fim,
            equality_constraints=eq,
            inequality_constraints=ineq,
        )
        if refined is not None and _is_better(refined[1], best_criterion, criterion):
            best_design, best_criterion, best_fim_result = refined

    return DesignResult(
        design=best_design,
        fim_result=best_fim_result,
        criterion_value=best_criterion,
    )


def _multi_start_candidates(
    design_bounds: dict[str, tuple[float, float]],
    n_starts: int,
    seed: int,
    projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    """Generate random interior + boundary design candidates.

    When ``projection`` is supplied, each candidate is passed through it
    before being returned (used to seed multi-start with feasible points
    for problems with equality constraints, e.g. mixture designs).
    """
    rng = np.random.default_rng(seed)
    design_names = list(design_bounds.keys())

    candidates: list[dict[str, float]] = []
    for _ in range(n_starts):
        candidates.append({name: rng.uniform(*design_bounds[name]) for name in design_names})

    for name in design_names:
        lo, hi = design_bounds[name]
        for val in (lo, hi):
            point = {n: (design_bounds[n][0] + design_bounds[n][1]) / 2 for n in design_names}
            point[name] = val
            candidates.append(point)

    if projection is not None:
        candidates = [projection(c) for c in candidates]
    return candidates


def _scan_candidates(
    experiment: Experiment,
    param_values: dict[str, float],
    candidates: list[dict[str, float]],
    criterion: str,
    prior_fim: np.ndarray | None,
) -> tuple[dict[str, float] | None, float, FIMResult | None]:
    """Evaluate each candidate and return the best."""
    best_design: dict[str, float] | None = None
    best_criterion = -np.inf if _is_maximization(criterion) else np.inf
    best_fim_result: FIMResult | None = None

    # Multi-RHS: evaluate every candidate's FIM in one batched pass (no
    # per-candidate solve / Jacobian re-trace for explicit response models).
    fim_results: list[FIMResult | None]
    try:
        fim_results = list(
            compute_fim_batch(experiment, param_values, candidates, prior_fim=prior_fim)
        )
    except Exception:
        fim_results = [None] * len(candidates)

    for design_point, fim_result in zip(candidates, fim_results):
        if fim_result is None:
            continue
        try:
            crit_val = _evaluate_criterion(fim_result, criterion)
        except Exception:
            continue

        if _is_better(crit_val, best_criterion, criterion):
            best_criterion = crit_val
            best_design = design_point
            best_fim_result = fim_result

    return best_design, best_criterion, best_fim_result


def _refine_single_design(
    experiment: Experiment,
    param_values: dict[str, float],
    seed_design: dict[str, float],
    design_names: list[str],
    design_bounds: dict[str, tuple[float, float]],
    criterion: str,
    prior_fim: np.ndarray | None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
) -> tuple[dict[str, float], float, FIMResult] | None:
    """Local refinement of a single design via scipy L-BFGS-B or SLSQP."""
    maximize = _is_maximization(criterion)
    bounds = [design_bounds[n] for n in design_names]
    x0 = np.array([seed_design[n] for n in design_names], dtype=float)

    def to_design(x: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(design_names, x)}

    def objective(x: np.ndarray) -> float:
        design = to_design(x)
        try:
            fim_result = compute_fim(experiment, param_values, design, prior_fim=prior_fim)
            crit = _evaluate_criterion(fim_result, criterion)
        except Exception:
            return _SINGULAR_SENTINEL
        if not np.isfinite(crit):
            return _SINGULAR_SENTINEL
        return -crit if maximize else crit

    has_constraints = bool(equality_constraints) or bool(inequality_constraints)
    if has_constraints:
        scipy_cons = [
            {"type": "eq", "fun": (lambda x, g=g: float(g(to_design(x))))}
            for g in equality_constraints
        ] + [
            {"type": "ineq", "fun": (lambda x, h=h: float(h(to_design(x))))}
            for h in inequality_constraints
        ]
        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=scipy_cons)
        except Exception:
            return None
    else:
        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            return None

    if not res.success and not np.isfinite(res.fun):
        return None

    design = to_design(np.asarray(res.x))
    try:
        fim_result = compute_fim(experiment, param_values, design, prior_fim=prior_fim)
    except Exception:
        return None
    crit_val = _evaluate_criterion(fim_result, criterion)
    if not np.isfinite(crit_val):
        return None
    return design, crit_val, fim_result


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
    return new_val < best_val


def batch_optimal_experiment(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str = DesignCriterion.D_OPTIMAL,
    strategy: str = BatchStrategy.GREEDY,
    prior_fim: np.ndarray | None = None,
    equality_constraints: Sequence[DesignConstraint] | None = None,
    inequality_constraints: Sequence[DesignConstraint] | None = None,
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int = 10,
    local_refine: bool = True,
    min_distance: float | None = None,
    seed: int = 42,
) -> BatchDesignResult:
    """Design a batch of ``N`` experiments to run in parallel.

    Experiments are independent, so their Fisher information adds:
    ``FIM_total = Σ_i FIM(d_i) + FIM_prior``. Strategies differ in how
    they search the joint design space.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_bounds : dict[str, tuple[float, float]]
        Bounds on each design input variable.
    n_experiments : int
        Number of experiments in the batch (must be ``>= 1``).
    criterion : str, default ``DesignCriterion.D_OPTIMAL``
        Design criterion evaluated on the joint FIM.
    strategy : str, default ``BatchStrategy.GREEDY``
        Batch selection strategy. One of ``"greedy"``, ``"joint"``,
        ``"penalized"``.
    prior_fim : numpy.ndarray, optional
        Prior FIM from previously collected data.
    n_starts : int, default 10
        Multi-start budget used by each internal single-design search
        (greedy / penalized) or the joint search.
    local_refine : bool, default True
        If True, apply scipy L-BFGS-B refinement at the relevant stage
        (single-design refinement for greedy / penalized, joint-vector
        refinement for joint).
    min_distance : float, optional
        Minimum normalised distance between selected designs. Only used
        by ``"penalized"``; ignored otherwise.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    BatchDesignResult
    """
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be >= 1, got {n_experiments}")

    eq = list(equality_constraints) if equality_constraints else []
    ineq = list(inequality_constraints) if inequality_constraints else []

    if strategy == BatchStrategy.GREEDY:
        return _greedy_batch(
            experiment,
            param_values,
            design_bounds,
            n_experiments,
            criterion=criterion,
            prior_fim=prior_fim,
            equality_constraints=eq,
            inequality_constraints=ineq,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            seed=seed,
        )
    elif strategy == BatchStrategy.JOINT:
        return _joint_batch(
            experiment,
            param_values,
            design_bounds,
            n_experiments,
            criterion=criterion,
            prior_fim=prior_fim,
            equality_constraints=eq,
            inequality_constraints=ineq,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            seed=seed,
        )
    elif strategy == BatchStrategy.PENALIZED:
        return _penalized_batch(
            experiment,
            param_values,
            design_bounds,
            n_experiments,
            criterion=criterion,
            prior_fim=prior_fim,
            equality_constraints=eq,
            inequality_constraints=ineq,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            min_distance=min_distance,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown batch strategy: {strategy!r}")


def _greedy_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str,
    prior_fim: np.ndarray | None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int,
    local_refine: bool,
    seed: int,
) -> BatchDesignResult:
    """Greedy batch: pick one design at a time, folding each FIM into the prior."""
    running_prior = prior_fim.copy() if prior_fim is not None else None
    designs: list[dict[str, float]] = []
    fim_results: list[FIMResult] = []
    per_round: list[float] = []

    for i in range(n_experiments):
        picked = optimal_experiment(
            experiment,
            param_values,
            design_bounds,
            criterion=criterion,
            prior_fim=running_prior,
            equality_constraints=equality_constraints or None,
            inequality_constraints=inequality_constraints or None,
            feasible_projection=feasible_projection,
            n_starts=n_starts,
            local_refine=local_refine,
            seed=seed + i,
        )
        per_fim = compute_fim(experiment, param_values, picked.design, prior_fim=None)
        designs.append(picked.design)
        fim_results.append(per_fim)
        running_prior = per_fim.fim.copy() if running_prior is None else running_prior + per_fim.fim
        per_round.append(_criterion_from_fim(running_prior, criterion))

    assert running_prior is not None  # n_experiments >= 1
    return BatchDesignResult(
        designs=designs,
        fim_results=fim_results,
        joint_fim=running_prior,
        criterion_value=per_round[-1],
        strategy=BatchStrategy.GREEDY,
        per_round_criterion=per_round,
    )


def _joint_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str,
    prior_fim: np.ndarray | None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int,
    local_refine: bool,
    seed: int,
) -> BatchDesignResult:
    """Galvanina-style joint batch: optimize N design vectors simultaneously."""
    design_names = list(design_bounds.keys())
    d = len(design_names)
    maximize = _is_maximization(criterion)
    flat_bounds = [design_bounds[n] for n in design_names] * n_experiments
    lows = np.array([design_bounds[n][0] for n in design_names])
    highs = np.array([design_bounds[n][1] for n in design_names])

    def unpack(z: np.ndarray) -> list[dict[str, float]]:
        stacks = z.reshape(n_experiments, d)
        return [
            {name: float(stacks[i, j]) for j, name in enumerate(design_names)}
            for i in range(n_experiments)
        ]

    def joint_fim_and_pieces(
        designs: list[dict[str, float]],
    ) -> tuple[np.ndarray, list[FIMResult]] | None:
        if not designs:
            return None
        pieces: list[FIMResult] = []
        try:
            first = compute_fim(experiment, param_values, designs[0], prior_fim=None)
        except Exception:
            return None
        pieces.append(first)
        total: np.ndarray = first.fim.copy()
        for design in designs[1:]:
            try:
                piece = compute_fim(experiment, param_values, design, prior_fim=None)
            except Exception:
                return None
            pieces.append(piece)
            total = total + piece.fim
        if prior_fim is not None:
            total = total + prior_fim
        return total, pieces

    def objective(z: np.ndarray) -> float:
        designs = unpack(z)
        result = joint_fim_and_pieces(designs)
        if result is None:
            return _SINGULAR_SENTINEL
        fim, _ = result
        crit = _criterion_from_fim(fim, criterion)
        if not np.isfinite(crit):
            return _SINGULAR_SENTINEL
        return -crit if maximize else crit

    rng = np.random.default_rng(seed)

    def project_stack(z: np.ndarray) -> np.ndarray:
        if feasible_projection is None:
            return z
        stacks = z.reshape(n_experiments, d)
        out = np.empty_like(z)
        for i in range(n_experiments):
            point = {name: float(stacks[i, j]) for j, name in enumerate(design_names)}
            projected = feasible_projection(point)
            for j, name in enumerate(design_names):
                out[i * d + j] = projected[name]
        return out

    # Multi-start: pure random stacks + a few structured ones.
    starts: list[np.ndarray] = []
    for _ in range(n_starts):
        z = rng.uniform(np.tile(lows, n_experiments), np.tile(highs, n_experiments))
        starts.append(project_stack(z))
    # Structured: one stack with each row at a distinct quantile of each bound.
    if n_experiments >= 2:
        quantiles = np.linspace(0.0, 1.0, n_experiments)
        structured = np.concatenate([lows + q * (highs - lows) for q in quantiles])
        starts.append(project_stack(structured))

    best_z: np.ndarray | None = None
    best_val = _SINGULAR_SENTINEL
    for z0 in starts:
        val = objective(z0)
        if val < best_val:
            best_val = val
            best_z = z0.copy()

    if best_z is None or not np.isfinite(best_val) or best_val >= _SINGULAR_SENTINEL:
        raise RuntimeError("joint batch: no feasible starting point found")

    final_z: np.ndarray = best_z
    if local_refine:
        has_constraints = bool(equality_constraints) or bool(inequality_constraints)
        if has_constraints:

            def slot_dict(z: np.ndarray, i: int) -> dict[str, float]:
                return {name: float(z[i * d + j]) for j, name in enumerate(design_names)}

            scipy_cons = []
            for i in range(n_experiments):
                for g in equality_constraints:
                    scipy_cons.append(
                        {"type": "eq", "fun": (lambda z, ii=i, gg=g: float(gg(slot_dict(z, ii))))}
                    )
                for h in inequality_constraints:
                    scipy_cons.append(
                        {"type": "ineq", "fun": (lambda z, ii=i, hh=h: float(hh(slot_dict(z, ii))))}
                    )
            try:
                refined = minimize(
                    objective,
                    final_z,
                    method="SLSQP",
                    bounds=flat_bounds,
                    constraints=scipy_cons,
                )
                if np.isfinite(refined.fun) and refined.fun < best_val:
                    final_z = np.asarray(refined.x)
                    best_val = float(refined.fun)
            except Exception:
                pass
        else:
            try:
                refined = minimize(objective, final_z, method="L-BFGS-B", bounds=flat_bounds)
                if np.isfinite(refined.fun) and refined.fun < best_val:
                    final_z = np.asarray(refined.x)
                    best_val = float(refined.fun)
            except Exception:
                pass

    designs = unpack(final_z)
    final = joint_fim_and_pieces(designs)
    if final is None:
        raise RuntimeError("joint batch: final FIM evaluation failed")
    joint_fim, pieces = final
    criterion_value = _criterion_from_fim(joint_fim, criterion)

    return BatchDesignResult(
        designs=designs,
        fim_results=pieces,
        joint_fim=joint_fim,
        criterion_value=criterion_value,
        strategy=BatchStrategy.JOINT,
        per_round_criterion=None,
    )


def _penalized_batch(
    experiment: Experiment,
    param_values: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
    n_experiments: int,
    *,
    criterion: str,
    prior_fim: np.ndarray | None,
    equality_constraints: Sequence[DesignConstraint] = (),
    inequality_constraints: Sequence[DesignConstraint] = (),
    feasible_projection: Callable[[dict[str, float]], dict[str, float]] | None = None,
    n_starts: int,
    local_refine: bool,
    min_distance: float | None,
    seed: int,
) -> BatchDesignResult:
    """Distance-penalized greedy: pick one at a time, reject picks too close."""
    min_dist = float(min_distance) if min_distance is not None else 0.0
    design_names = list(design_bounds.keys())

    running_prior = prior_fim.copy() if prior_fim is not None else None
    designs: list[dict[str, float]] = []
    fim_results: list[FIMResult] = []
    per_round: list[float] = []

    for i in range(n_experiments):
        candidates = _multi_start_candidates(
            design_bounds, n_starts, seed + i, projection=feasible_projection
        )
        filtered = _filter_by_min_distance(candidates, designs, design_bounds, min_dist)
        if not filtered:
            # Retry with a bigger pool before giving up.
            candidates = _multi_start_candidates(
                design_bounds, n_starts * 5, seed + 1000 + i, projection=feasible_projection
            )
            filtered = _filter_by_min_distance(candidates, designs, design_bounds, min_dist)
        if not filtered:
            raise RuntimeError(
                f"penalized batch: no candidate respects min_distance={min_dist} "
                f"from the {len(designs)} already-selected designs."
            )

        best, best_crit, best_fim = _scan_candidates(
            experiment, param_values, filtered, criterion, running_prior
        )

        if best is None or best_fim is None:
            raise RuntimeError("penalized batch: no feasible candidate found")

        if local_refine:
            refined = _refine_single_design(
                experiment,
                param_values,
                best,
                design_names,
                design_bounds,
                criterion,
                running_prior,
                equality_constraints=equality_constraints,
                inequality_constraints=inequality_constraints,
            )
            if (
                refined is not None
                and _min_normalized_distance(refined[0], designs, design_bounds) >= min_dist
                and _is_better(refined[1], best_crit, criterion)
            ):
                best, best_crit, best_fim = refined

        per_fim = compute_fim(experiment, param_values, best, prior_fim=None)
        designs.append(best)
        fim_results.append(per_fim)
        running_prior = per_fim.fim.copy() if running_prior is None else running_prior + per_fim.fim
        per_round.append(_criterion_from_fim(running_prior, criterion))

    assert running_prior is not None
    return BatchDesignResult(
        designs=designs,
        fim_results=fim_results,
        joint_fim=running_prior,
        criterion_value=per_round[-1],
        strategy=BatchStrategy.PENALIZED,
        per_round_criterion=per_round,
    )


def _normalized_distance(
    a: dict[str, float],
    b: dict[str, float],
    design_bounds: dict[str, tuple[float, float]],
) -> float:
    """Euclidean distance between two designs after normalising each axis to [0, 1]."""
    total = 0.0
    for name, (lo, hi) in design_bounds.items():
        scale = hi - lo
        if scale <= 0:
            continue
        total += ((a[name] - b[name]) / scale) ** 2
    return float(np.sqrt(total))


def _min_normalized_distance(
    candidate: dict[str, float],
    existing: list[dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
) -> float:
    """Minimum normalised distance from candidate to any existing design."""
    if not existing:
        return float("inf")
    return min(_normalized_distance(candidate, e, design_bounds) for e in existing)


def _filter_by_min_distance(
    candidates: list[dict[str, float]],
    existing: list[dict[str, float]],
    design_bounds: dict[str, tuple[float, float]],
    min_dist: float,
) -> list[dict[str, float]]:
    """Drop candidates within min_dist (normalised) of any existing design."""
    if min_dist <= 0.0 or not existing:
        return list(candidates)
    return [
        c for c in candidates if _min_normalized_distance(c, existing, design_bounds) >= min_dist
    ]
