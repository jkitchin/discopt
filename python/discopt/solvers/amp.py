"""
Adaptive Multivariate Partitioning (AMP) global MINLP solver.

Implements the algorithm from:
  - CP 2016: "Tightening McCormick Relaxations via Dynamic Multivariate
    Partitioning", Nagarajan et al.
  - JOGO 2018: "An Adaptive, Multivariate Partitioning Algorithm for Global
    Optimization", Nagarajan et al.

Algorithm loop (per iteration k):
  1. Solve MILP relaxation → lower bound LB_k
  2. Fix continuous variables' interval assignments from MILP solution,
     solve NLP subproblem → upper bound UB_k
  3. Check gap: if ``(UB_k - LB_k) / abs(UB_k) ≤ rel_gap`` → CERTIFIED OPTIMAL
  4. Refine partitions adaptively around the MILP solution point
  5. Repeat until gap closed, max_iter reached, or time_limit exceeded

The MILP relaxation is built by build_milp_relaxation() in milp_relaxation.py.
Soundness guarantee: LB_k ≤ global_opt ≤ UB_k at every iteration k.
"""

from __future__ import annotations

import itertools
import logging
import time
from functools import lru_cache
from importlib.util import find_spec
from typing import Any, Callable, Optional

import numpy as np

from discopt._jax.milp_relaxation import _normalize_convhull_formulation
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.nonlinear_bound_tightening import (
    is_effectively_finite,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import (
    Model,
    ObjectiveSense,
    SolveResult,
    VarType,
)

logger = logging.getLogger(__name__)
_DEFAULT_MAX_OA_CUTS = 128
_SMALL_INT_FALLBACK_MAX_ASSIGNMENTS = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _has_cyipopt() -> bool:
    """Return True when cyipopt is importable in the active environment."""
    return find_spec("cyipopt") is not None


def _build_x_dict(x_flat: np.ndarray, model: Model) -> dict:
    """Convert flat solution vector to {var_name: array} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        result[v.name] = x_flat[offset : offset + v.size].reshape(v.shape)
        offset += v.size
    return result


def _extract_orig_solution(x_milp: np.ndarray, n_orig: int) -> np.ndarray:
    """Extract original variable values from MILP solution (drop aux vars)."""
    return x_milp[:n_orig]


def _snapshot_variable_bounds(model: Model) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    """Capture model variable bounds so temporary overrides can be restored."""
    saved_bounds: list[tuple[Any, np.ndarray, np.ndarray]] = []
    for var in model._variables:
        saved_bounds.append(
            (
                var,
                np.array(var.lb, dtype=np.float64, copy=True),
                np.array(var.ub, dtype=np.float64, copy=True),
            )
        )
    return saved_bounds


def _restore_variable_bounds(saved_bounds: list[tuple[Any, np.ndarray, np.ndarray]]) -> None:
    """Restore variable bounds previously returned by _snapshot_variable_bounds()."""
    for var, orig_lb, orig_ub in saved_bounds:
        var.lb = orig_lb
        var.ub = orig_ub


def _apply_flat_bounds_to_model(model: Model, lb: np.ndarray, ub: np.ndarray) -> None:
    """Apply flat bound arrays to model variables in-place."""
    offset = 0
    for var in model._variables:
        size = var.size
        var.lb = np.asarray(lb[offset : offset + size], dtype=np.float64).reshape(var.shape).copy()
        var.ub = np.asarray(ub[offset : offset + size], dtype=np.float64).reshape(var.shape).copy()
        offset += size


def _default_nlp_start(flat_lb: np.ndarray, flat_ub: np.ndarray) -> np.ndarray:
    """Build a neutral NLP start point that behaves sensibly on semi-infinite domains."""
    x0 = np.zeros_like(flat_lb, dtype=np.float64)
    finite_lb = np.vectorize(is_effectively_finite)(flat_lb)
    finite_ub = np.vectorize(is_effectively_finite)(flat_ub)

    both = finite_lb & finite_ub
    x0[both] = 0.5 * (flat_lb[both] + flat_ub[both])

    only_lb = finite_lb & ~finite_ub
    x0[only_lb] = np.maximum(flat_lb[only_lb], 0.0)

    only_ub = ~finite_lb & finite_ub
    x0[only_ub] = np.minimum(flat_ub[only_ub], 0.0)

    return x0


def _continuous_recovery_starts(
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    initial_point: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Candidate starts for pure-continuous incumbent recovery."""
    starts: list[np.ndarray] = []

    if initial_point is not None:
        starts.append(np.asarray(initial_point, dtype=np.float64).copy())

    lb_clip = np.clip(flat_lb, -1e8, 1e8)
    ub_clip = np.clip(flat_ub, -1e8, 1e8)
    midpoint = 0.5 * (lb_clip + ub_clip)
    fully_unbounded = (flat_lb <= -1e15) & (flat_ub >= 1e15)
    midpoint = np.where(fully_unbounded, 0.5, midpoint)
    midpoint = np.clip(
        midpoint,
        np.maximum(flat_lb, -10.0),
        np.minimum(flat_ub, 10.0),
    )
    starts.append(midpoint)
    starts.append(np.clip(np.zeros_like(flat_lb), flat_lb, flat_ub))
    starts.append(np.clip(np.ones_like(flat_lb), flat_lb, flat_ub))

    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for start in starts:
        key = tuple(float(v) for v in np.asarray(start, dtype=np.float64).ravel())
        if key not in seen:
            seen.add(key)
            unique.append(np.asarray(start, dtype=np.float64))
    return unique


def _dedupe_candidate_points(points: list[np.ndarray]) -> list[np.ndarray]:
    """Return candidate points with duplicates removed in insertion order."""
    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for point in points:
        arr = np.asarray(point, dtype=np.float64)
        key = tuple(float(v) for v in arr.ravel())
        if key not in seen:
            seen.add(key)
            unique.append(arr)
    return unique


def _normalize_initial_point(
    initial_point: Optional[np.ndarray],
    n_orig: int,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[np.ndarray]:
    """Validate and clip an optional AMP initial point."""
    if initial_point is None:
        return None

    initial_point_arr = np.asarray(initial_point, dtype=np.float64).reshape(-1)
    if initial_point_arr.size != n_orig:
        raise ValueError(
            f"AMP initial_point has length {initial_point_arr.size}; expected {n_orig}"
        )
    if not np.all(np.isfinite(initial_point_arr)):
        raise ValueError("AMP initial_point must contain only finite values")
    return np.clip(initial_point_arr, flat_lb, flat_ub)


def _remaining_wall_time(deadline: Optional[float]) -> Optional[float]:
    """Return seconds remaining until a deadline, or None when uncapped."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.perf_counter())


def _solve_nlp_subproblem(
    evaluator,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    nlp_solver: str = "ipm",
    time_limit: Optional[float] = None,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Solve the NLP relaxation with given bounds.

    Returns (x_opt, obj_val) or (None, None) on failure.
    """
    if time_limit is not None and time_limit <= 0.0:
        return None, None
    try:
        lb_clip = np.clip(lb, -1e8, 1e8)
        ub_clip = np.clip(ub, -1e8, 1e8)
        x0_clipped = np.clip(x0, lb_clip, ub_clip)
        local_deadline = time.perf_counter() + time_limit if time_limit is not None else None
        model = evaluator._model
        saved_bounds = _snapshot_variable_bounds(model)
        _apply_flat_bounds_to_model(model, lb, ub)
        try:
            solver_sequence = [nlp_solver]
            if nlp_solver == "ipm" and _has_cyipopt():
                # The pure-JAX IPM is less robust on the tightly fixed integer
                # subproblems used in AMP's local incumbent search. Retry with
                # Ipopt before giving up so feasible incumbents are not missed.
                solver_sequence.append("ipopt")

            result = None
            for solver_name in solver_sequence:
                remaining = _remaining_wall_time(local_deadline)
                if remaining is not None and remaining <= 0.0:
                    break
                options: dict[str, float | int] = {"print_level": 0, "max_iter": 300}
                if remaining is not None:
                    options["max_wall_time"] = max(remaining, 0.05)
                if solver_name == "ipm" and hasattr(evaluator, "_obj_fn"):
                    from discopt._jax.ipm import solve_nlp_ipm

                    trial = solve_nlp_ipm(
                        evaluator,
                        x0_clipped,
                        options=options,
                    )
                else:
                    from discopt.solvers.nlp_ipopt import solve_nlp

                    if remaining is not None:
                        options["max_cpu_time"] = max(remaining, 0.05)
                    trial = solve_nlp(
                        evaluator,
                        x0_clipped,
                        options=options,
                    )
                result = trial
                from discopt.solvers import SolveStatus

                if trial.status == SolveStatus.OPTIMAL:
                    break
        finally:
            _restore_variable_bounds(saved_bounds)

        from discopt.solvers import SolveStatus

        if result is not None and result.status == SolveStatus.OPTIMAL:
            obj = float(evaluator.evaluate_objective(result.x))
            return result.x, obj
    except Exception as e:
        logger.debug("AMP NLP subproblem failed: %s", e)
    return None, None


def _recover_pure_continuous_solution(
    model: Model,
    evaluator,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    *,
    nlp_solver: str,
    t_start: float,
    time_limit: float,
    initial_point: Optional[np.ndarray] = None,
) -> Optional[SolveResult]:
    """Try an AMP-local NLP solve to recover a feasible incumbent.

    This is a last-resort incumbent recovery path for pure continuous models
    where AMP failed to produce any incumbent from its relaxation loop. The
    NLP solve is treated as a local feasibility heuristic within AMP, not as
    a global solution certificate, so successful recovery is reported as
    ``"feasible"``.
    """
    remaining = max(0.0, time_limit - (time.perf_counter() - t_start))
    if remaining <= 0.0:
        return None

    solver_sequence = [nlp_solver]
    if nlp_solver == "ipm" and _has_cyipopt():
        solver_sequence = ["ipopt", "ipm"]

    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None
    deadline = t_start + time_limit

    for x0 in _continuous_recovery_starts(flat_lb, flat_ub, initial_point):
        remaining_opt = _remaining_wall_time(deadline)
        if remaining_opt is not None and remaining_opt <= 0.0:
            break
        for solver_name in solver_sequence:
            remaining_opt = _remaining_wall_time(deadline)
            if remaining_opt is not None and remaining_opt <= 0.0:
                break
            recovered_x, recovered_obj = _solve_nlp_subproblem(
                evaluator,
                x0,
                flat_lb,
                flat_ub,
                solver_name,
                time_limit=remaining_opt,
            )
            if recovered_x is None or recovered_obj is None:
                continue
            if best_obj is None or recovered_obj < best_obj:
                best_x = recovered_x
                best_obj = recovered_obj

    if best_x is None or best_obj is None:
        return None

    maximize = model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE
    obj_val = -best_obj if maximize else best_obj
    return SolveResult(
        status="feasible",
        objective=obj_val,
        bound=None,
        gap=None,
        x=_build_x_dict(best_x, model),
        wall_time=time.perf_counter() - t_start,
        gap_certified=False,
    )


def _check_integer_feasible(x: np.ndarray, model: Model, int_tol: float = 1e-5) -> bool:
    """Return True if all integer/binary variables satisfy integrality."""
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            for i in range(v.size):
                val = float(x[offset + i])
                if abs(val - round(val)) > int_tol:
                    return False
        offset += v.size
    return True


def _integer_rounding_candidates(
    x: np.ndarray,
    model: Model,
    max_candidates: int = 64,
) -> list[np.ndarray]:
    """Generate nearest-first integer rounding candidates within variable bounds."""
    base = np.asarray(x, dtype=np.float64).copy()
    integer_entries: list[tuple[int, list[int]]] = []
    full_domain_product = 1

    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            v_lb = np.asarray(v.lb, dtype=np.float64).ravel()
            v_ub = np.asarray(v.ub, dtype=np.float64).ravel()
            for i in range(v.size):
                idx = offset + i
                lb_i = float(v_lb[i])
                ub_i = float(v_ub[i])
                clipped = float(np.clip(base[idx], lb_i, ub_i))
                lo_i = int(np.ceil(lb_i - 1e-9))
                hi_i = int(np.floor(ub_i + 1e-9))

                if lo_i <= hi_i:
                    domain_size = hi_i - lo_i + 1
                    full_domain_product *= max(1, domain_size)
                else:
                    domain_size = max_candidates + 1
                    full_domain_product = max_candidates + 1

                if lo_i <= hi_i and full_domain_product <= max_candidates:
                    center = min(max(int(round(clipped)), lo_i), hi_i)
                    options = list(range(lo_i, hi_i + 1))
                    options.sort(
                        key=lambda value: (abs(value - clipped), abs(value - center), value)
                    )
                else:
                    center = int(round(clipped))
                    if lo_i <= hi_i:
                        center = min(max(center, lo_i), hi_i)

                    options = []
                    for raw in (
                        center,
                        int(np.floor(clipped)),
                        int(np.ceil(clipped)),
                    ):
                        if lo_i <= hi_i:
                            cand_int = min(max(raw, lo_i), hi_i)
                        else:
                            cand_int = raw
                        if cand_int not in options:
                            options.append(cand_int)

                    neighbor_radius = 2
                    for delta in range(1, neighbor_radius + 1):
                        for raw in (center - delta, center + delta):
                            if lo_i <= hi_i:
                                cand_int = min(max(raw, lo_i), hi_i)
                            else:
                                cand_int = raw
                            if cand_int not in options:
                                options.append(cand_int)

                integer_entries.append((idx, options))
        offset += v.size

    if not integer_entries:
        return [base]

    total_candidates = 1
    for _, options in integer_entries:
        total_candidates *= max(1, len(options))

    candidates: list[np.ndarray] = []
    if total_candidates <= max_candidates:
        option_lists = [options for _, options in integer_entries]
        for values in itertools.product(*option_lists):
            cand = base.copy()
            for (idx, _), value in zip(integer_entries, values):
                cand[idx] = float(value)
            candidates.append(cand)
    else:
        nearest = base.copy()
        for idx, options in integer_entries:
            nearest[idx] = float(options[0])
        candidates.append(nearest)
        for idx, options in integer_entries:
            for value in options[1:]:
                cand = nearest.copy()
                cand[idx] = float(value)
                candidates.append(cand)

    deduped: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for cand in candidates:
        key = tuple(float(v) for v in cand)
        if key not in seen:
            seen.add(key)
            deduped.append(cand)
    return deduped[:max_candidates]


def _round_integers(x: np.ndarray, model: Model) -> np.ndarray:
    """Round integer/binary variables to the nearest candidate."""
    return _integer_rounding_candidates(x, model)[0]


def _build_fixed_integer_bounds(
    x: np.ndarray,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fix integer and binary variables to the provided candidate values."""
    nlp_lb = flat_lb.copy()
    nlp_ub = flat_ub.copy()

    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            v_lb = np.asarray(v.lb, dtype=np.float64).ravel()
            v_ub = np.asarray(v.ub, dtype=np.float64).ravel()
            for k in range(v.size):
                idx = offset + k
                val = float(np.clip(x[idx], v_lb[k], v_ub[k]))
                rounded = int(round(val))
                lo_i = int(np.ceil(v_lb[k] - 1e-9))
                hi_i = int(np.floor(v_ub[k] + 1e-9))
                if lo_i <= hi_i:
                    rounded = min(max(rounded, lo_i), hi_i)
                nlp_lb[idx] = rounded
                nlp_ub[idx] = rounded
        offset += v.size

    return nlp_lb, nlp_ub


def _select_best_nlp_candidate(
    candidates: list[np.ndarray],
    model: Model,
    evaluator,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
    nlp_solver: str,
    deadline: Optional[float] = None,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Return the best feasible NLP candidate from a prioritized candidate list."""
    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None

    for x0_nlp in candidates:
        remaining = _remaining_wall_time(deadline)
        if remaining is not None and remaining <= 0.0:
            break
        nlp_lb, nlp_ub = _build_fixed_integer_bounds(x0_nlp, model, flat_lb, flat_ub)
        cand_x, cand_obj = _solve_nlp_subproblem(
            evaluator,
            x0_nlp,
            nlp_lb,
            nlp_ub,
            nlp_solver,
            time_limit=remaining,
        )
        if cand_x is None or cand_obj is None:
            continue
        if not _check_integer_feasible(cand_x, model):
            continue
        if not _check_constraints_with_evaluator(
            evaluator,
            cand_x,
            constraint_lb,
            constraint_ub,
        ):
            continue

        cand_obj_min = float(cand_obj)
        if best_obj is None or cand_obj_min < best_obj:
            best_x = cand_x
            best_obj = cand_obj_min

    return best_x, best_obj


def _solve_best_nlp_candidate(
    x0: np.ndarray,
    model: Model,
    evaluator,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
    nlp_solver: str,
    incumbent: Optional[np.ndarray] = None,
    initial_point: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Improve the incumbent from Alpine-ordered local NLP starts."""
    if all(v.var_type == VarType.CONTINUOUS for v in model._variables):
        starts: list[np.ndarray] = []
        for seed in (incumbent, initial_point, x0):
            if seed is not None:
                starts.append(np.clip(np.asarray(seed, dtype=np.float64), flat_lb, flat_ub))
        starts.extend(_continuous_recovery_starts(flat_lb, flat_ub))
        candidates = _dedupe_candidate_points(starts)
    else:
        candidates = []
        for seed in (incumbent, initial_point, x0, _default_nlp_start(flat_lb, flat_ub)):
            if seed is not None:
                candidates.extend(_integer_rounding_candidates(seed, model))
        candidates = _dedupe_candidate_points(candidates)

    return _select_best_nlp_candidate(
        candidates,
        model,
        evaluator,
        flat_lb,
        flat_ub,
        constraint_lb,
        constraint_ub,
        nlp_solver,
        deadline=deadline,
    )


def _small_integer_domain_size(model: Model, max_assignments: int) -> Optional[int]:
    """Return the exact integer-domain size when it is finite and small enough."""
    total = 1
    has_integer = False

    for var in model._variables:
        if var.var_type not in (VarType.BINARY, VarType.INTEGER):
            continue
        has_integer = True
        for lb_i, ub_i in zip(
            np.asarray(var.lb, dtype=np.float64).ravel(),
            np.asarray(var.ub, dtype=np.float64).ravel(),
        ):
            if not (is_effectively_finite(float(lb_i)) and is_effectively_finite(float(ub_i))):
                return None
            lo_i = int(np.ceil(float(lb_i) - 1e-9))
            hi_i = int(np.floor(float(ub_i) + 1e-9))
            if lo_i > hi_i:
                return 0
            total *= hi_i - lo_i + 1
            if total > max_assignments:
                return None

    return total if has_integer else None


def _solve_small_integer_domain_fallback(
    model: Model,
    evaluator,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
    nlp_solver: str,
    max_assignments: int = _SMALL_INT_FALLBACK_MAX_ASSIGNMENTS,
    deadline: Optional[float] = None,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Enumerate a small finite integer domain directly when the MILP relaxation fails."""
    domain_size = _small_integer_domain_size(model, max_assignments)
    if domain_size is None or domain_size == 0:
        return None, None

    base_x0 = _default_nlp_start(flat_lb, flat_ub)
    candidates = _integer_rounding_candidates(base_x0, model, max_candidates=max_assignments)
    if len(candidates) < domain_size:
        return None, None

    return _select_best_nlp_candidate(
        candidates,
        model,
        evaluator,
        flat_lb,
        flat_ub,
        constraint_lb,
        constraint_ub,
        nlp_solver,
        deadline=deadline,
    )


def _solve_milp_with_oa_recovery(
    model: Model,
    terms,
    disc_state,
    incumbent: Optional[np.ndarray],
    oa_cuts: Optional[list],
    time_limit: Optional[float],
    gap_tolerance: float,
    convhull_formulation: str,
    convhull_ebd: bool,
    convhull_ebd_encoding: str,
    bound_override: Optional[tuple[np.ndarray, np.ndarray]] = None,
):
    """Retry MILP solves after dropping the oldest half of OA cuts on infeasibility."""
    from discopt._jax.milp_relaxation import build_milp_relaxation

    active_oa_cuts = list(oa_cuts or [])
    max_retries = max(1, len(active_oa_cuts).bit_length() + 1)
    milp_result = None
    varmap = None
    mip_solve_count = 0

    for _retry in range(max_retries):
        milp_model, varmap = build_milp_relaxation(
            model,
            terms,
            disc_state,
            incumbent,
            oa_cuts=active_oa_cuts,
            convhull_formulation=convhull_formulation,
            convhull_ebd=convhull_ebd,
            convhull_ebd_encoding=convhull_ebd_encoding,
            bound_override=bound_override,
        )
        milp_result = milp_model.solve(
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )
        mip_solve_count += 1
        if milp_result.status != "infeasible" or not active_oa_cuts:
            return milp_result, varmap, active_oa_cuts, mip_solve_count

        drop_count = max(1, len(active_oa_cuts) // 2)
        logger.info(
            "AMP: MILP infeasible with %d OA cuts; dropping %d oldest cuts and retrying",
            len(active_oa_cuts),
            drop_count,
        )
        active_oa_cuts = active_oa_cuts[drop_count:]

    assert milp_result is not None
    assert varmap is not None
    return milp_result, varmap, active_oa_cuts, mip_solve_count


def _check_constraints(x: np.ndarray, model: Model, tol: float = 1e-4) -> bool:
    """Return True if all constraints are satisfied at x."""
    try:
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        evaluator = NLPEvaluator(model)
        if evaluator.n_constraints == 0:
            return True
        g = np.array(evaluator.evaluate_constraints(x))
        lb_g, ub_g = _infer_constraint_bounds(model)
        lb_g = np.asarray(lb_g, dtype=np.float64)
        ub_g = np.asarray(ub_g, dtype=np.float64)
        return bool(np.all(g >= lb_g - tol) and np.all(g <= ub_g + tol))
    except Exception as err:
        logger.warning("AMP: constraint evaluation failed; rejecting point: %s", err)
        return False


def _check_constraints_with_evaluator(
    evaluator,
    x: np.ndarray,
    lb_g: np.ndarray,
    ub_g: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """Return True if all constraints are satisfied at x."""
    try:
        if evaluator.n_constraints == 0:
            return True
        g = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
        return bool(np.all(g >= lb_g - tol) and np.all(g <= ub_g + tol))
    except Exception as err:
        logger.warning("AMP: constraint evaluation failed; rejecting point: %s", err)
        return False


def _default_milp_time_limit(
    remaining: float,
    iteration: int,
    max_iter: int,
) -> float:
    """Allocate a bounded MILP budget from the remaining AMP wall time."""
    iter_budget = remaining / max(1, max_iter - iteration + 1)
    return min(iter_budget * 3, remaining * 0.8, 60.0)


def _normalize_partition_method(
    partition_method: str,
    disc_var_pick: int | str | None,
) -> str:
    """Resolve public AMP aliases to the internal partition-selection strategy."""
    if disc_var_pick is None:
        return partition_method

    if isinstance(disc_var_pick, str):
        aliases = {
            "all": "max_cover",
            "max_cover": "max_cover",
            "min_vertex_cover": "min_vertex_cover",
            "auto": "auto",
            "adaptive": "adaptive_vertex_cover",
            "adaptive_vertex_cover": "adaptive_vertex_cover",
        }
        if disc_var_pick not in aliases:
            raise ValueError(
                f"Unsupported disc_var_pick string: {disc_var_pick!r}. "
                "Choose from 'all', 'max_cover', 'min_vertex_cover', "
                "'auto', or 'adaptive_vertex_cover'."
            )
        return aliases[disc_var_pick]

    if disc_var_pick == 0:
        return "max_cover"
    if disc_var_pick == 1:
        return "min_vertex_cover"
    if disc_var_pick == 2:
        return "auto"
    if disc_var_pick == 3:
        return "adaptive_vertex_cover"

    raise ValueError(
        f"Unsupported disc_var_pick integer: {disc_var_pick!r}. Choose from 0, 1, 2, or 3."
    )


def _default_obbt_time_limit_per_lp(
    remaining: float,
    n_orig: int,
) -> float:
    """Allocate a bounded per-LP budget for OBBT presolve."""
    if not np.isfinite(remaining) or remaining <= 0.0:
        return 0.0
    obbt_budget = min(10.0, 0.1 * remaining)
    return max(0.05, obbt_budget / max(1, 2 * n_orig))


def _compute_relative_gap(
    abs_gap: Optional[float],
    upper_bound: float,
) -> Optional[float]:
    """Return a relative gap, or None when the upper bound is numerically zero."""
    if (
        abs_gap is None
        or abs_gap < 0.0
        or not np.isfinite(upper_bound)
        or abs(upper_bound) <= 1e-10
    ):
        return None
    return abs(abs_gap) / abs(upper_bound)


def _prune_oa_cuts(oa_cuts: list, max_cuts: int = _DEFAULT_MAX_OA_CUTS) -> None:
    """Keep only the most recent OA cuts to cap MILP growth."""
    overflow = len(oa_cuts) - max_cuts
    if overflow > 0:
        del oa_cuts[:overflow]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_amp(
    model: Model,
    rel_gap: float = 1e-3,
    abs_tol: float = 1e-6,
    time_limit: float = 3600.0,
    max_iter: int = 50,
    n_init_partitions: int = 2,
    partition_method: str = "auto",
    nlp_solver: str = "ipm",
    iteration_callback: Optional[Callable] = None,
    milp_time_limit: Optional[float] = None,
    milp_gap_tolerance: Optional[float] = None,
    apply_partitioning: bool = True,
    disc_var_pick: int | str | None = None,
    partition_scaling_factor: float = 10.0,
    disc_add_partition_method: str = "adaptive",
    disc_abs_width_tol: float = 1e-3,
    convhull_formulation: str = "disaggregated",
    convhull_ebd: bool = False,
    convhull_ebd_encoding: str = "gray",
    presolve_bt: bool = True,
    initial_point: Optional[np.ndarray] = None,
    use_start_as_incumbent: bool = False,
    skip_convex_check: bool = False,
) -> SolveResult:
    """Solve MINLP globally using Adaptive Multivariate Partitioning (AMP).

    Parameters
    ----------
    model : Model
        A validated discopt Model.
    rel_gap : float
        Relative gap tolerance: terminate when ``(UB-LB)/abs(UB) ≤ rel_gap``.
    abs_tol : float
        Absolute gap tolerance: terminate when UB-LB ≤ abs_tol.
    time_limit : float
        Wall-clock limit in seconds.
    max_iter : int
        Maximum number of AMP iterations.
    n_init_partitions : int
        Number of initial uniform intervals per partition variable.
    partition_method : str
        Variable selection: ``"auto"``, ``"max_cover"``, or ``"min_vertex_cover"``.
    nlp_solver : str
        NLP backend for upper-bound subproblems: ``"ipm"`` or ``"ipopt"``.
    iteration_callback : callable, optional
        Called each iteration with dict: {"iteration", "lower_bound", "upper_bound"}.
    milp_time_limit : float, optional
        Per-MILP-call time limit (defaults to remaining time).
    milp_gap_tolerance : float
        MILP solver gap tolerance (default 1e-4).
    apply_partitioning : bool
        If False, solve a single relaxation/NLP pass without adaptive refinement.
    disc_var_pick : int | str, optional
        Alpine-style alias for partition selection:
        0/``"all"`` → max_cover, 1 → min_vertex_cover, 2 → auto,
        3 → adaptive weighted cover. This intentionally collapses Alpine's
        separate `disc_var_pick_algo` and `disc_var_pick` options into one
        user-facing control.
    partition_scaling_factor : float
        Width scaling used by adaptive partition refinement.
    disc_add_partition_method : str
        Refinement update rule: ``"adaptive"`` or ``"uniform"``.
    disc_abs_width_tol : float
        Absolute partition-width convergence tolerance.
    convhull_formulation : str
        Piecewise bilinear formulation: ``"disaggregated"``, ``"sos2"``,
        ``"facet"``, or ``"lambda"`` (alias for ``"sos2"``).
    convhull_ebd : bool
        Replace SOS2 interval binaries with an embedded logarithmic encoding.
    convhull_ebd_encoding : str
        Embedded encoding scheme for the SOS2 formulation. ``"gray"`` is the
        only option that stays SOS2-compatible for arbitrary partition counts;
        ``"binary"`` is only valid for two partitions.
    presolve_bt : bool
        Run LP-based OBBT before the AMP loop to tighten variable bounds.
    initial_point : ndarray, optional
        Validated model start point used by AMP's local incumbent-improvement
        phase. Candidate local NLP starts are tried as incumbent, model start,
        MILP point, then safe fallback starts.
    use_start_as_incumbent : bool
        If True, accept a feasible initial point as the first incumbent before
        the AMP bounding loop starts, matching Alpine's warm-start policy.
    skip_convex_check : bool
        If True, force AMP even when the model is detected as a pure
        continuous convex problem.

    Returns
    -------
    SolveResult
        With gap_certified=True if termination is by gap criterion. When AMP
        has a valid incumbent but no certificate, it returns ``"feasible"``
        together with the incumbent and any trustworthy bound information,
        even if the wall-clock limit ended the proof search.
    """
    t_start = time.perf_counter()

    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.discretization import (
        add_adaptive_partition,
        add_uniform_partition,
        check_partition_convergence,
        initialize_partitions,
    )
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt._jax.partition_selection import pick_partition_vars
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    assert model._objective is not None
    maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
    pure_continuous = all(v.var_type == VarType.CONTINUOUS for v in model._variables)
    part_lbs: list[float] = []
    part_ubs: list[float] = []

    if partition_scaling_factor <= 1.0:
        raise ValueError("partition_scaling_factor must be > 1.0")
    if disc_add_partition_method not in {"adaptive", "uniform"}:
        raise ValueError("disc_add_partition_method must be 'adaptive' or 'uniform'")

    partition_mode = _normalize_partition_method(partition_method, disc_var_pick)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)
    if convhull_ebd and convhull_mode != "sos2":
        raise ValueError("convhull_ebd requires convhull_formulation='sos2' or the 'lambda' alias.")

    n_orig = sum(v.size for v in model._variables)
    flat_lb, flat_ub = flat_variable_bounds(model)
    initial_point_arr = _normalize_initial_point(initial_point, n_orig, flat_lb, flat_ub)

    if pure_continuous and not skip_convex_check:
        try:
            from discopt._jax.convexity import classify_model as _classify_convexity
            from discopt.solver import _solve_continuous

            is_convex, _ = _classify_convexity(model, use_certificate=True)
            if is_convex:
                logger.info(
                    "AMP: convex NLP detected; solving with single NLP "
                    "(global optimality guaranteed; partitioning skipped)"
                )
                result = _solve_continuous(
                    model,
                    time_limit,
                    ipopt_options=None,
                    t_start=t_start,
                    nlp_solver=nlp_solver,
                    initial_point=initial_point_arr,
                )
                result.convex_fast_path = True
                return result
        except Exception as exc:
            logger.debug("AMP: convex delegation check failed: %s", exc)

    def _from_minimization_space(value: float) -> float:
        return -float(value) if maximize else float(value)

    tightened_lb, tightened_ub, nonlinear_bt_stats = tighten_nonlinear_bounds(
        model, flat_lb, flat_ub
    )
    if nonlinear_bt_stats.infeasible:
        logger.info(
            "AMP: nonlinear bound tightening proved infeasibility: %s",
            nonlinear_bt_stats.infeasibility_reason,
        )
        return SolveResult(
            status="infeasible",
            wall_time=time.perf_counter() - t_start,
            gap_certified=True,
        )
    if nonlinear_bt_stats.n_tightened > 0:
        flat_lb = tightened_lb
        flat_ub = tightened_ub
        logger.info(
            "AMP: nonlinear bound tightening adjusted %d bounds via %s",
            nonlinear_bt_stats.n_tightened,
            ", ".join(nonlinear_bt_stats.applied_rules),
        )
    evaluator = NLPEvaluator(model)
    constraint_lb, constraint_ub = _infer_constraint_bounds(model)
    deadline = t_start + time_limit
    oa_convexity = classify_oa_cut_convexity(model)
    if evaluator.n_constraints > 0 and not all(oa_convexity.constraint_mask):
        logger.warning(
            "AMP: generating OA cuts only for %d of %d constraints classified convex",
            sum(1 for is_convex in oa_convexity.constraint_mask if is_convex),
            len(oa_convexity.constraint_mask),
        )

    # ── Classify nonlinear terms ─────────────────────────────────────────────
    terms = classify_nonlinear_terms(model)
    logger.info(
        "AMP: %d bilinear, %d trilinear, %d multilinear, %d monomial, %d general_nl terms",
        len(terms.bilinear),
        len(terms.trilinear),
        len(terms.multilinear),
        len(terms.monomial),
        len(terms.general_nl),
    )

    # Tighten the initial McCormick domain before selecting partition bounds.
    if presolve_bt:
        remaining = max(0.0, time_limit - (time.perf_counter() - t_start))
        obbt_time_limit = _default_obbt_time_limit_per_lp(remaining, n_orig)
        if obbt_time_limit > 0.0:
            try:
                from discopt._jax.obbt import run_obbt
            except ImportError as err:
                logger.warning("AMP: OBBT presolve unavailable; continuing without it: %s", err)
            else:
                obbt_result = run_obbt(
                    model,
                    lb=flat_lb.copy(),
                    ub=flat_ub.copy(),
                    time_limit_per_lp=obbt_time_limit,
                )
                if obbt_result.n_tightened > 0:
                    flat_lb = obbt_result.tightened_lb
                    flat_ub = obbt_result.tightened_ub
                    logger.info(
                        "AMP: OBBT tightened %d bounds in %.3fs before partitioning",
                        obbt_result.n_tightened,
                        obbt_result.total_lp_time,
                    )
        else:
            logger.info("AMP: skipping OBBT presolve because no wall-clock budget remains")

    if initial_point_arr is not None:
        initial_point_arr = np.clip(initial_point_arr, flat_lb, flat_ub)

    # ── Select partition variables ───────────────────────────────────────────
    if apply_partitioning:
        part_vars = pick_partition_vars(terms, method=partition_mode)
    else:
        part_vars = []

    # If no bilinear/multilinear terms, still partition monomial variables
    # to add tangent cuts at more points and tighten the lower bound.
    if apply_partitioning and not part_vars and terms.monomial:
        part_vars = sorted(set(var_idx for var_idx, _ in terms.monomial))
        logger.info("AMP: no bilinear terms; partitioning %d monomial vars", len(part_vars))
    elif not apply_partitioning:
        logger.info("AMP: partitioning disabled; running a single fixed relaxation pass")
    else:
        logger.info("AMP: partitioning %d variables via %s", len(part_vars), partition_mode)

    # ── Initialize partitions ────────────────────────────────────────────────
    if part_vars:
        part_lbs = [float(flat_lb[i]) for i in part_vars]
        part_ubs = [float(flat_ub[i]) for i in part_vars]
        disc_state = initialize_partitions(
            part_vars,
            lb=part_lbs,
            ub=part_ubs,
            n_init=n_init_partitions,
            scaling_factor=partition_scaling_factor,
            abs_width_tol=disc_abs_width_tol,
        )
    else:
        from discopt._jax.discretization import DiscretizationState

        disc_state = DiscretizationState(
            scaling_factor=partition_scaling_factor,
            abs_width_tol=disc_abs_width_tol,
        )

    LB = -np.inf
    UB = np.inf
    incumbent = None
    if (
        use_start_as_incumbent
        and initial_point_arr is not None
        and _check_integer_feasible(initial_point_arr, model)
        and _check_constraints_with_evaluator(
            evaluator,
            initial_point_arr,
            constraint_lb,
            constraint_ub,
        )
    ):
        initial_obj = float(evaluator.evaluate_objective(initial_point_arr))
        if np.isfinite(initial_obj):
            UB = initial_obj
            incumbent = initial_point_arr.copy()
            logger.info("AMP: accepted feasible initial point as incumbent")
        else:
            logger.info("AMP: feasible initial point has non-finite objective; not using incumbent")
    gap_certified = False
    oa_cuts: list = []  # accumulated OA linearizations from NLP incumbents
    mip_count = 0
    termination_reason = "iteration_limit"

    for iteration in range(1, max_iter + 1):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            logger.info("AMP: time limit reached at iteration %d", iteration)
            termination_reason = "time_limit"
            break

        remaining = time_limit - elapsed
        milp_tl = (
            milp_time_limit
            if milp_time_limit is not None
            else _default_milp_time_limit(remaining, iteration, max_iter)
        )

        # ── Step 1: Solve MILP relaxation → lower bound ──────────────────────
        # MILP gap tolerance: no tighter than needed for overall convergence.
        _milp_gap_tol = (
            milp_gap_tolerance if milp_gap_tolerance is not None else min(rel_gap / 2, 1e-3)
        )

        try:
            milp_result, varmap, active_oa_cuts, iter_mip_count = _solve_milp_with_oa_recovery(
                model=model,
                terms=terms,
                disc_state=disc_state,
                incumbent=incumbent,
                oa_cuts=oa_cuts,
                time_limit=milp_tl,
                gap_tolerance=_milp_gap_tol,
                convhull_formulation=convhull_mode,
                convhull_ebd=convhull_ebd,
                convhull_ebd_encoding=convhull_ebd_encoding,
                bound_override=(flat_lb, flat_ub),
            )
            mip_count += iter_mip_count
            oa_cuts = active_oa_cuts
        except Exception as e:
            logger.warning("AMP: MILP build/solve failed at iteration %d: %s", iteration, e)
            termination_reason = "error"
            break

        if milp_result.status in ("infeasible", "error"):
            logger.info(
                "AMP: MILP infeasible/error at iteration %d, status=%s",
                iteration,
                milp_result.status,
            )
            termination_reason = "error" if milp_result.status == "error" else "infeasible"
            if LB == -np.inf:
                # Problem may be infeasible
                if iteration == 1 and incumbent is None:
                    if pure_continuous:
                        recovered = _recover_pure_continuous_solution(
                            model,
                            evaluator,
                            flat_lb,
                            flat_ub,
                            nlp_solver=nlp_solver,
                            t_start=t_start,
                            time_limit=time_limit,
                            initial_point=initial_point_arr,
                        )
                        if recovered is not None:
                            recovered.mip_count = mip_count
                            return recovered
                    fallback_x, fallback_obj = _solve_small_integer_domain_fallback(
                        model,
                        evaluator,
                        flat_lb,
                        flat_ub,
                        constraint_lb,
                        constraint_ub,
                        nlp_solver,
                        deadline=deadline,
                    )
                    if fallback_x is not None and fallback_obj is not None:
                        return SolveResult(
                            status="feasible",
                            objective=_from_minimization_space(fallback_obj),
                            bound=None,
                            gap=None,
                            x=_build_x_dict(fallback_x, model),
                            wall_time=time.perf_counter() - t_start,
                            mip_count=mip_count,
                            gap_certified=False,
                        )
                    if milp_result.status == "infeasible":
                        return SolveResult(
                            status="infeasible",
                            wall_time=time.perf_counter() - t_start,
                            mip_count=mip_count,
                        )
            break

        if milp_result.objective is not None:
            new_lb = float(milp_result.objective)
            # Soundness: LB must be non-decreasing
            LB = max(LB, new_lb)

        logger.debug("AMP iter %d: LB=%.6g, UB=%.6g", iteration, LB, UB)

        # ── Step 2: NLP upper-bound subproblem ───────────────────────────────
        # Use MILP solution point as initial point for NLP
        if milp_result.x is not None:
            x0 = _extract_orig_solution(milp_result.x, n_orig)
            x0 = np.clip(x0, flat_lb, flat_ub)
        else:
            x0 = _default_nlp_start(flat_lb, flat_ub)

        x_nlp, obj_nlp_min = _solve_best_nlp_candidate(
            x0,
            model,
            evaluator,
            flat_lb,
            flat_ub,
            constraint_lb,
            constraint_ub,
            nlp_solver,
            incumbent=incumbent,
            initial_point=initial_point_arr,
            deadline=deadline,
        )

        if x_nlp is not None and obj_nlp_min is not None:
            # Verify feasibility and update UB in the canonical minimization space.
            if obj_nlp_min < UB:
                UB = obj_nlp_min
                incumbent = x_nlp.copy()
                logger.debug(
                    "AMP iter %d: new incumbent objective=%.6g",
                    iteration,
                    _from_minimization_space(UB),
                )

                # Accumulate OA tangent cuts at this NLP solution to tighten
                # the next MILP relaxation.  Uses existing OA infrastructure
                # from cutting_planes.py which handles all constraint senses.
                try:
                    from discopt._jax.cutting_planes import (
                        generate_oa_cuts_from_evaluator,
                    )
                    from discopt.modeling.core import Constraint

                    _x_orig = x_nlp[:n_orig]
                    if evaluator.n_constraints > 0:
                        _senses = [c.sense for c in model._constraints if isinstance(c, Constraint)]
                        cuts = generate_oa_cuts_from_evaluator(
                            evaluator,
                            _x_orig,
                            constraint_senses=_senses,
                            convex_mask=oa_convexity.constraint_mask,
                        )
                        for cut in cuts:
                            if np.linalg.norm(cut.coeffs) < 1e-12:
                                continue
                            if cut.sense == ">=":
                                # Convert to <= form for milp_relaxation.py
                                oa_cuts.append((-cut.coeffs, -cut.rhs))
                            elif cut.sense == "==":
                                oa_cuts.append((cut.coeffs, cut.rhs))
                                oa_cuts.append((-cut.coeffs, -cut.rhs))
                            else:
                                oa_cuts.append((cut.coeffs, cut.rhs))
                        _prune_oa_cuts(oa_cuts)
                except Exception as _oa_err:
                    logger.debug("AMP: OA cut computation failed: %s", _oa_err)

        # ── Step 3: Gap check ────────────────────────────────────────────────
        if iteration_callback is not None:
            if maximize:
                callback_lb = _from_minimization_space(UB) if UB < np.inf else -np.inf
                callback_ub = _from_minimization_space(LB) if LB > -np.inf else np.inf
            else:
                callback_lb = LB
                callback_ub = UB
            iteration_callback(
                {
                    "iteration": iteration,
                    "lower_bound": callback_lb,
                    "upper_bound": callback_ub,
                }
            )

        if UB < np.inf and LB > -np.inf:
            raw_abs_gap = UB - LB
            if maximize:
                display_lb = _from_minimization_space(UB)
                display_ub = _from_minimization_space(LB)
            else:
                display_lb = LB
                display_ub = UB
            if raw_abs_gap < -abs_tol:
                logger.warning(
                    "AMP iter %d: invalid bound ordering LB=%.6g, UB=%.6g; "
                    "skipping gap certification",
                    iteration,
                    display_lb,
                    display_ub,
                )
            else:
                abs_gap = max(0.0, raw_abs_gap)
                rel_g = _compute_relative_gap(abs_gap, UB)
                if rel_g is None:
                    logger.info(
                        "AMP iter %d: LB=%.6g, UB=%.6g, abs_gap=%.6g (relative gap undefined)",
                        iteration,
                        display_lb,
                        display_ub,
                        abs_gap,
                    )
                else:
                    logger.info(
                        "AMP iter %d: LB=%.6g, UB=%.6g, gap=%.4g%%",
                        iteration,
                        display_lb,
                        display_ub,
                        100 * rel_g,
                    )
                if abs_gap <= abs_tol or (rel_g is not None and rel_g <= rel_gap):
                    gap_certified = True
                    if rel_g is None:
                        logger.info(
                            "AMP: gap certified at iteration %d by absolute tolerance",
                            iteration,
                        )
                    else:
                        logger.info(
                            "AMP: gap certified at iteration %d (gap=%.4g%%)",
                            iteration,
                            100 * rel_g,
                        )
                    break

        # ── Step 4: Adaptive partition refinement ────────────────────────────
        if not part_vars:
            # No partition variables → single iteration
            if UB < np.inf:
                gap_certified = False  # no lower bound from partitioning
            break

        if (
            partition_mode == "adaptive_vertex_cover"
            and incumbent is not None
            and milp_result.x is not None
        ):
            distances = {
                i: abs(float(incumbent[i]) - float(x0[i])) for i in terms.partition_candidates
            }
            adaptive_vars = pick_partition_vars(
                terms,
                method="adaptive_vertex_cover",
                distance=distances,
            )
            if adaptive_vars and set(adaptive_vars) != set(part_vars):
                logger.info(
                    "AMP: updating adaptive partition set from %d to %d variables",
                    len(part_vars),
                    len(adaptive_vars),
                )
                new_vars = [i for i in adaptive_vars if i not in disc_state.partitions]
                if new_vars:
                    new_lbs = [float(flat_lb[i]) for i in new_vars]
                    new_ubs = [float(flat_ub[i]) for i in new_vars]
                    init_state = initialize_partitions(
                        new_vars,
                        lb=new_lbs,
                        ub=new_ubs,
                        n_init=n_init_partitions,
                        scaling_factor=partition_scaling_factor,
                        abs_width_tol=disc_abs_width_tol,
                    )
                    disc_state.partitions.update(init_state.partitions)
                part_vars = adaptive_vars
                part_lbs = [float(flat_lb[i]) for i in part_vars]
                part_ubs = [float(flat_ub[i]) for i in part_vars]

        # Use MILP solution (original vars) as the refinement point
        refine_solution: dict[int, float] = {}
        if milp_result.x is not None:
            x_orig = milp_result.x[:n_orig]
            for i in part_vars:
                refine_solution[i] = float(x_orig[i])

        if disc_add_partition_method == "uniform":
            disc_state = add_uniform_partition(
                disc_state,
                refine_solution,
                part_vars,
                part_lbs,
                part_ubs,
            )
        else:
            disc_state = add_adaptive_partition(
                disc_state,
                refine_solution,
                part_vars,
                part_lbs,
                part_ubs,
            )

        # Check partition convergence
        if check_partition_convergence(disc_state):
            logger.info("AMP: partition convergence at iteration %d", iteration)
            gap_certified = False
            break

    # ── Build final result ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start

    if UB >= np.inf and LB == -np.inf:
        if termination_reason == "infeasible":
            status = "infeasible"
        elif termination_reason == "time_limit":
            status = "time_limit"
        elif termination_reason == "error":
            status = "error"
        else:
            status = "iteration_limit"
        return SolveResult(
            status=status,
            wall_time=elapsed,
            mip_count=mip_count,
            gap_certified=False,
        )

    if incumbent is not None:
        raw_abs_gap_final = UB - LB if LB > -np.inf else None
        bound_is_trustworthy = raw_abs_gap_final is None or raw_abs_gap_final >= -abs_tol
        if not bound_is_trustworthy and raw_abs_gap_final is not None:
            logger.warning(
                "AMP: final bound ordering invalid (LB=%.6g, UB=%.6g); omitting bound and gap",
                _from_minimization_space(LB),
                _from_minimization_space(UB),
            )

        abs_gap_final = (
            None
            if raw_abs_gap_final is None or not bound_is_trustworthy
            else max(0.0, raw_abs_gap_final)
        )
        rel_gap_final = _compute_relative_gap(abs_gap_final, UB)
        status = "optimal" if gap_certified else "feasible"

        return SolveResult(
            status=status,
            objective=_from_minimization_space(UB),
            bound=(_from_minimization_space(LB) if LB > -np.inf and bound_is_trustworthy else None),
            gap=float(rel_gap_final) if rel_gap_final is not None else None,
            x=_build_x_dict(incumbent, model),
            wall_time=elapsed,
            mip_count=mip_count,
            gap_certified=gap_certified,
        )

    # No feasible solution found
    if pure_continuous:
        recovered = _recover_pure_continuous_solution(
            model,
            evaluator,
            flat_lb,
            flat_ub,
            nlp_solver=nlp_solver,
            t_start=t_start,
            time_limit=time_limit,
            initial_point=initial_point_arr,
        )
        if recovered is not None:
            recovered.mip_count = mip_count
            return recovered

    if termination_reason == "time_limit" or elapsed >= time_limit:
        status = "time_limit"
    elif termination_reason == "error":
        status = "error"
    elif termination_reason == "infeasible":
        status = "infeasible"
    else:
        status = "iteration_limit"

    return SolveResult(
        status=status,
        objective=None,
        bound=_from_minimization_space(LB) if LB > -np.inf else None,
        gap=None,
        x=None,
        wall_time=elapsed,
        mip_count=mip_count,
        gap_certified=False,
    )
