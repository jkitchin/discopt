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
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from typing import Any, Callable, Optional, cast

import numpy as np

from discopt._jax.milp_relaxation import _normalize_convhull_formulation
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.nonlinear_bound_tightening import (
    is_effectively_finite,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    ObjectiveSense,
    SolveResult,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
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


def _repair_inverted_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap numerically inverted intervals to their midpoint."""
    bad = lb > ub
    if not np.any(bad):
        return lb, ub

    repaired_lb = np.array(lb, dtype=np.float64, copy=True)
    repaired_ub = np.array(ub, dtype=np.float64, copy=True)
    midpoint = 0.5 * (repaired_lb[bad] + repaired_ub[bad])
    repaired_lb[bad] = midpoint
    repaired_ub[bad] = midpoint
    return repaired_lb, repaired_ub


@dataclass
class _AmpCutoffState:
    """Mutable cutoff-tightening state carried by AMP OA cut generation."""

    flat_lb: np.ndarray
    flat_ub: np.ndarray
    part_vars: list[int]
    part_lbs: list[float]
    part_ubs: list[float]
    cutoff_obbt_done: bool = False
    last_obbt_iter: int = 0

    def bounds_tuple(self) -> tuple[np.ndarray, np.ndarray, list[int], list[float], list[float]]:
        return self.flat_lb, self.flat_ub, self.part_vars, self.part_lbs, self.part_ubs


def _refresh_partitions_for_bounds(
    model: Model,
    disc_state,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    part_vars: list[int],
    disc_abs_width_tol: float,
    n_init_partitions: int,
) -> tuple[list[int], list[float], list[float]]:
    """Apply tightened bounds and keep AMP partition endpoints consistent."""
    _apply_flat_bounds_to_model(model, flat_lb, flat_ub)
    part_vars = [i for i in part_vars if float(flat_ub[i]) - float(flat_lb[i]) > disc_abs_width_tol]
    part_lbs = [float(flat_lb[i]) for i in part_vars]
    part_ubs = [float(flat_ub[i]) for i in part_vars]
    active_part_vars = set(part_vars)
    for stale_idx in list(disc_state.partitions):
        if stale_idx not in active_part_vars:
            del disc_state.partitions[stale_idx]
    for k_pv, v_idx in enumerate(part_vars):
        pts = disc_state.partitions.get(v_idx)
        new_lo = float(part_lbs[k_pv])
        new_hi = float(part_ubs[k_pv])
        if pts is None or len(pts) < 2:
            disc_state.partitions[v_idx] = np.linspace(new_lo, new_hi, n_init_partitions + 1)
            continue
        clipped = np.clip(pts, new_lo, new_hi)
        merged = np.unique(np.concatenate([[new_lo], clipped, [new_hi]]))
        disc_state.partitions[v_idx] = np.sort(merged)
    return part_vars, part_lbs, part_ubs


def _scalar_constant(expr: Expression) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    value = np.asarray(expr.value, dtype=np.float64)
    if value.shape != ():
        return None
    return float(value)


def _flat_var_index(expr: Expression, model: Model) -> Optional[int]:
    if isinstance(expr, Variable):
        if expr.size != 1:
            return None
        return sum(existing.size for existing in model._variables[: expr._index])
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base = expr.base
        base_offset = sum(existing.size for existing in model._variables[: base._index])
        idx = expr.index
        if base.shape == ():
            return base_offset
        if not isinstance(idx, tuple):
            idx = (idx,)
        return base_offset + int(np.ravel_multi_index(idx, base.shape))
    return None


def _flatten_objective_power_terms(
    expr: Expression,
    model: Model,
    scale: float,
    groups: dict[int, dict[int, float]],
) -> Optional[float]:
    """Collect separable monomial objective terms, returning the constant offset."""
    const = _scalar_constant(expr)
    if const is not None:
        return scale * const

    flat_idx = _flat_var_index(expr, model)
    if flat_idx is not None:
        terms = groups.setdefault(flat_idx, {})
        terms[1] = terms.get(1, 0.0) + scale
        return 0.0

    if isinstance(expr, SumExpression):
        return _flatten_objective_power_terms(expr.operand, model, scale, groups)

    if isinstance(expr, SumOverExpression):
        total = 0.0
        for term in expr.terms:
            term_const = _flatten_objective_power_terms(term, model, scale, groups)
            if term_const is None:
                return None
            total += term_const
        return total

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _flatten_objective_power_terms(expr.operand, model, -scale, groups)

    if not isinstance(expr, BinaryOp):
        return None

    if expr.op in {"+", "-"}:
        left_const = _flatten_objective_power_terms(expr.left, model, scale, groups)
        right_scale = scale if expr.op == "+" else -scale
        right_const = _flatten_objective_power_terms(expr.right, model, right_scale, groups)
        if left_const is None or right_const is None:
            return None
        return left_const + right_const

    if expr.op == "*":
        left_const = _scalar_constant(expr.left)
        if left_const is not None:
            return _flatten_objective_power_terms(expr.right, model, scale * left_const, groups)
        right_const = _scalar_constant(expr.right)
        if right_const is not None:
            return _flatten_objective_power_terms(expr.left, model, scale * right_const, groups)
        return None

    if expr.op == "/":
        denom = _scalar_constant(expr.right)
        if denom is None or abs(denom) <= 1e-14:
            return None
        return _flatten_objective_power_terms(expr.left, model, scale / denom, groups)

    if expr.op == "**":
        exponent = _scalar_constant(expr.right)
        if exponent is None or abs(exponent - round(exponent)) > 1e-12:
            return None
        exponent_int = int(round(exponent))
        if exponent_int <= 0:
            return None
        flat_idx = _flat_var_index(expr.left, model)
        if flat_idx is None:
            return None
        terms = groups.setdefault(flat_idx, {})
        terms[exponent_int] = terms.get(exponent_int, 0.0) + scale
        return 0.0

    return None


def _polynomial_value(terms: dict[int, float], x: float) -> float:
    return float(sum(coeff * (x**degree) for degree, coeff in terms.items()))


def _univariate_polynomial_minimum(terms: dict[int, float], lb: float, ub: float) -> float:
    points = [lb, ub]
    if lb <= 0.0 <= ub:
        points.append(0.0)
    max_degree = max((degree for degree in terms if degree > 0), default=0)
    derivative_coeffs = [terms.get(degree, 0.0) * degree for degree in range(max_degree, 0, -1)]
    if any(abs(coeff) > 1e-14 for coeff in derivative_coeffs):
        for root in np.roots(derivative_coeffs):
            if abs(float(np.imag(root))) <= 1e-10:
                real_root = float(np.real(root))
                if lb <= real_root <= ub:
                    points.append(real_root)
    return min(_polynomial_value(terms, point) for point in points)


def _tighten_simple_power_group(
    terms: dict[int, float],
    rhs: float,
    lb: float,
    ub: float,
) -> tuple[float, float]:
    """Intersect simple monomial level sets with the current interval."""
    nonzero_terms = {degree: coeff for degree, coeff in terms.items() if abs(coeff) > 1e-14}
    if not np.isfinite(rhs) or not nonzero_terms:
        return lb, ub
    if len(nonzero_terms) != 1:
        return lb, ub

    degree, coeff = next(iter(nonzero_terms.items()))
    if degree == 1:
        bound = rhs / coeff
        if coeff > 0.0:
            return lb, min(ub, bound)
        return max(lb, bound), ub

    if degree % 2 == 1:
        signed_root = float(np.sign(rhs / coeff) * (abs(rhs / coeff) ** (1.0 / degree)))
        if coeff > 0.0:
            return lb, min(ub, signed_root)
        return max(lb, signed_root), ub

    if coeff <= 0.0 or rhs < 0.0:
        return lb, ub

    radius = float((rhs / coeff) ** (1.0 / degree))
    return max(lb, -radius), min(ub, radius)


def _tighten_bounds_with_objective_cutoff(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a separable objective cutoff to infer a finite box before cutoff OBBT."""
    if (
        model._objective is None
        or model._objective.sense != ObjectiveSense.MINIMIZE
        or not np.isfinite(cutoff)
    ):
        return flat_lb, flat_ub

    groups: dict[int, dict[int, float]] = {}
    constant = _flatten_objective_power_terms(model._objective.expression, model, 1.0, groups)
    if constant is None or not groups:
        return flat_lb, flat_ub

    tightened_lb = np.asarray(flat_lb, dtype=np.float64).copy()
    tightened_ub = np.asarray(flat_ub, dtype=np.float64).copy()
    cutoff = float(cutoff) + 1e-8 * max(1.0, abs(float(cutoff)))

    group_min: dict[int, float] = {}
    for flat_idx, terms in groups.items():
        lb_i = float(tightened_lb[flat_idx])
        ub_i = float(tightened_ub[flat_idx])
        if not (np.isfinite(lb_i) and np.isfinite(ub_i)):
            return flat_lb, flat_ub
        group_min[flat_idx] = _univariate_polynomial_minimum(terms, lb_i, ub_i)

    total_min = float(constant) + float(sum(group_min.values()))
    if total_min > cutoff + 1e-8:
        return flat_lb, flat_ub

    for flat_idx, terms in groups.items():
        other_min = total_min - group_min[flat_idx]
        rhs = cutoff - other_min
        new_lb, new_ub = _tighten_simple_power_group(
            terms,
            rhs,
            float(tightened_lb[flat_idx]),
            float(tightened_ub[flat_idx]),
        )
        if new_lb <= new_ub + 1e-10:
            tightened_lb[flat_idx] = max(tightened_lb[flat_idx], new_lb)
            tightened_ub[flat_idx] = min(tightened_ub[flat_idx], new_ub)

    return tightened_lb, tightened_ub


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
        from discopt.solver import _BoundOverrideEvaluator

        backend_evaluator = _BoundOverrideEvaluator(evaluator, lb, ub)
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
                    backend_evaluator,
                    x0_clipped,
                    options=options,
                )
            else:
                from discopt.solvers.nlp_ipopt import solve_nlp

                if remaining is not None:
                    options["max_cpu_time"] = max(remaining, 0.05)
                trial = solve_nlp(
                    cast(Any, backend_evaluator),
                    x0_clipped,
                    options=options,
                )
            result = trial
            from discopt.solvers import SolveStatus

            if trial.status == SolveStatus.OPTIMAL:
                break

        from discopt.solvers import SolveStatus

        if result is not None and result.status == SolveStatus.OPTIMAL:
            x_opt = np.asarray(result.x, dtype=np.float64)
            if not np.all(np.isfinite(x_opt)):
                logger.debug("AMP NLP subproblem returned a non-finite solution; rejecting it")
                return None, None
            obj = float(evaluator.evaluate_objective(x_opt))
            if not np.isfinite(obj):
                logger.debug("AMP NLP subproblem returned a non-finite objective; rejecting it")
                return None, None
            return x_opt, obj
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


def _validate_partition_scaling_factor(
    value: Any,
    option_name: str = "partition_scaling_factor",
) -> float:
    """Return a numeric AMP partition scaling factor."""
    try:
        factor = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{option_name} must be a finite number > 1.0") from err
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError(f"{option_name} must be a finite number > 1.0")
    return factor


def _normalize_partition_var_indices(
    selected: Any,
    n_orig: int,
    *,
    source: str,
) -> list[int]:
    """Validate flat variable indices returned by a custom AMP selection hook."""
    if selected is None:
        raise ValueError(f"{source} callable must return an iterable of flat variable indices")
    try:
        raw_indices = list(selected)
    except TypeError as err:
        raise ValueError(
            f"{source} callable must return an iterable of flat variable indices"
        ) from err

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_idx in raw_indices:
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, (int, np.integer)):
            raise ValueError(f"{source} callable returned a non-integer index: {raw_idx!r}")
        idx = int(raw_idx)
        if idx < 0 or idx >= n_orig:
            raise ValueError(
                f"{source} callable returned index {idx}, outside valid range [0, {n_orig})"
            )
        if idx not in seen:
            seen.add(idx)
            normalized.append(idx)
    return normalized


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Return a variable's start index in the flattened model vector."""
    offset = 0
    for existing in model._variables[: var._index]:
        offset += existing.size
    return offset


def _flat_index_from_expr(expr: Expression, model: Model) -> int | None:
    """Return the flat index for a scalar variable reference, if any."""
    if isinstance(expr, Variable):
        if expr.size == 1:
            return _compute_var_offset(expr, model)
        return None
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base_off = _compute_var_offset(expr.base, model)
        idx = expr.index
        if isinstance(idx, int):
            return base_off + idx
        if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
            return base_off + idx[0]
    return None


def _collect_product_factor_indices(expr: Expression, model: Model) -> list[int] | None:
    """Return flat variable factors for a pure product tree, ignoring constants."""
    indices: list[int] = []

    def _visit(node: Expression) -> bool:
        if isinstance(node, BinaryOp) and node.op == "*":
            return _visit(node.left) and _visit(node.right)
        flat_idx = _flat_index_from_expr(node, model)
        if flat_idx is not None:
            indices.append(flat_idx)
            return True
        return isinstance(node, Constant)

    if _visit(expr) and len(indices) >= 2:
        return indices
    return None


def _square_monomial_vars_in_expr(expr: Expression, model: Model) -> set[int]:
    """Collect variables that appear as square monomial terms in an expression."""
    square_vars: set[int] = set()

    def _visit(node: Expression) -> None:
        if isinstance(node, (Constant, Variable)):
            return
        if isinstance(node, IndexExpression):
            if not isinstance(node.base, Variable):
                _visit(node.base)
            return
        if isinstance(node, BinaryOp):
            if node.op == "**":
                flat_idx = _flat_index_from_expr(node.left, model)
                if flat_idx is not None and isinstance(node.right, Constant):
                    exp_val = float(node.right.value)
                    if exp_val == 2.0:
                        square_vars.add(flat_idx)
                        return
            if node.op == "*":
                factors = _collect_product_factor_indices(node, model)
                if factors is not None:
                    counts = {idx: factors.count(idx) for idx in set(factors)}
                    if len(counts) == 1 and next(iter(counts.values())) == 2:
                        square_vars.add(next(iter(counts)))
                        return
            _visit(node.left)
            _visit(node.right)
            return
        if isinstance(node, UnaryOp):
            _visit(node.operand)
            return
        if isinstance(node, FunctionCall):
            for arg in node.args:
                _visit(arg)
            return
        if isinstance(node, MatMulExpression):
            _visit(node.left)
            _visit(node.right)
            return
        if isinstance(node, SumExpression):
            _visit(node.operand)
            return
        if isinstance(node, SumOverExpression):
            for term in node.terms:
                _visit(term)

    _visit(expr)
    return square_vars


def _expr_variable_indices(
    expr: Expression,
    model: Model,
    cache: Optional[dict[int, frozenset[int]]] = None,
) -> set[int]:
    """Collect flat variable indices referenced by an expression."""
    cache_key = id(expr)
    if cache is not None and cache_key in cache:
        return set(cache[cache_key])

    result: set[int]
    flat_idx = _flat_var_index(expr, model)
    if flat_idx is not None:
        result = {flat_idx}
    elif isinstance(expr, Variable):
        start = _compute_var_offset(expr, model)
        result = set(range(start, start + expr.size))
    elif isinstance(expr, (Constant, IndexExpression)):
        result = set()
    elif isinstance(expr, BinaryOp):
        left = _expr_variable_indices(expr.left, model, cache)
        right = _expr_variable_indices(expr.right, model, cache)
        result = left | right
    elif isinstance(expr, UnaryOp):
        result = _expr_variable_indices(expr.operand, model, cache)
    elif isinstance(expr, FunctionCall):
        indices: set[int] = set()
        for arg in expr.args:
            indices.update(_expr_variable_indices(arg, model, cache))
        result = indices
    elif isinstance(expr, MatMulExpression):
        left = _expr_variable_indices(expr.left, model, cache)
        right = _expr_variable_indices(expr.right, model, cache)
        result = left | right
    elif isinstance(expr, SumExpression):
        result = _expr_variable_indices(expr.operand, model, cache)
    elif isinstance(expr, SumOverExpression):
        indices = set()
        for term in expr.terms:
            indices.update(_expr_variable_indices(term, model, cache))
        result = indices
    else:
        result = set()

    if cache is not None:
        cache[cache_key] = frozenset(result)
    return result


def _expr_has_function(
    expr: Expression,
    names: set[str],
    cache: Optional[dict[tuple[int, tuple[str, ...]], bool]] = None,
) -> bool:
    """Return true when an expression tree calls any named function."""
    cache_key = (id(expr), tuple(sorted(names)))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if isinstance(expr, FunctionCall):
        result = expr.func_name in names or any(
            _expr_has_function(arg, names, cache) for arg in expr.args
        )
    elif isinstance(expr, BinaryOp):
        result = _expr_has_function(expr.left, names, cache) or _expr_has_function(
            expr.right, names, cache
        )
    elif isinstance(expr, UnaryOp):
        result = _expr_has_function(expr.operand, names, cache)
    elif isinstance(expr, MatMulExpression):
        result = _expr_has_function(expr.left, names, cache) or _expr_has_function(
            expr.right, names, cache
        )
    elif isinstance(expr, SumExpression):
        result = _expr_has_function(expr.operand, names, cache)
    elif isinstance(expr, SumOverExpression):
        result = any(_expr_has_function(term, names, cache) for term in expr.terms)
    elif isinstance(expr, IndexExpression) and not isinstance(expr.base, Variable):
        result = _expr_has_function(expr.base, names, cache)
    else:
        result = False

    if cache is not None:
        cache[cache_key] = result
    return result


def _expr_all_vars_fixed(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    tol: float = 1e-9,
    var_index_cache: Optional[dict[int, frozenset[int]]] = None,
) -> bool:
    """Return true when all variables used by an expression have fixed bounds."""
    indices = _expr_variable_indices(expr, model, var_index_cache)
    if not indices:
        return False
    return all(float(flat_ub[idx]) - float(flat_lb[idx]) <= tol for idx in indices)


def _direct_oa_skip_reasons(
    model: Model,
    convex_mask: list[bool],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> list[str | None]:
    """Explain why each constraint row is excluded from direct tangent OA."""
    from discopt._jax.convexity.rules import Curvature, classify_expr
    from discopt._jax.cutting_planes import _quadratic_polynomial

    reasons: list[str | None] = []
    var_index_cache: dict[int, frozenset[int]] = {}
    function_cache: dict[tuple[int, tuple[str, ...]], bool] = {}
    for idx, is_convex in enumerate(convex_mask):
        if is_convex:
            reasons.append(None)
            continue

        constraint = model._constraints[idx]
        if not isinstance(constraint, Constraint):
            reasons.append("unsupported_constraint_type")
            continue

        expr = constraint.body
        if _expr_all_vars_fixed(expr, model, flat_lb, flat_ub, var_index_cache=var_index_cache):
            reasons.append("fixed_nonlinear_row")
            continue
        if constraint.sense == "==":
            reasons.append("nonaffine_equality")
            continue

        curvature = classify_expr(expr, model)
        if (constraint.sense == "<=" and curvature == Curvature.CONCAVE) or (
            constraint.sense == ">=" and curvature == Curvature.CONVEX
        ):
            reasons.append("opposite_curvature_for_direct_oa")
            continue

        if _quadratic_polynomial(expr, model) is not None:
            reasons.append("nonconvex_quadratic_alpha_bb_candidate")
            continue
        if _expr_has_function(expr, {"sin", "cos", "tan"}, function_cache):
            reasons.append("trigonometric_not_certified_convex")
            continue

        reasons.append("not_certified_convex")
    return reasons


def _equality_square_monomial_partition_candidates(model: Model, terms: Any) -> list[int]:
    """Select coupled square monomials that benefit from AMP refinement.

    Weymouth constraints have the form ``f^2 = C * (p_in^2 - p_out^2)``:
    multiple square monomials coupled by one equality.  Partitioning those
    variables lets the monomial tangent cuts adapt around the MILP point without
    making every monomial in every model a partition candidate.

    Sphere/ball constraints such as ``x^2 + y^2 + z^2 <= r`` need the same
    treatment: the tangent under-estimators for each square must tighten
    together, otherwise an unpartitioned square can leave too much room in the
    convex quadratic upper-bound constraint.
    """
    known_squares = {
        int(var_idx) for var_idx, exp in getattr(terms, "monomial", []) if int(exp) == 2
    }
    if not known_squares:
        return []

    candidates: set[int] = set()
    for constraint in model._constraints:
        if constraint.sense not in {"==", "<="}:
            continue
        square_vars = _square_monomial_vars_in_expr(constraint.body, model) & known_squares
        if len(square_vars) >= 2:
            candidates.update(square_vars)
    return sorted(candidates)


def _merge_partition_vars(selected: list[int], extra: list[int]) -> list[int]:
    """Append extra partition variables without reordering the selected prefix."""
    merged: list[int] = []
    seen: set[int] = set()
    for idx in itertools.chain(selected, extra):
        if idx not in seen:
            seen.add(idx)
            merged.append(idx)
    return merged


def _select_partition_vars_with_hook(
    terms,
    *,
    method: str,
    disc_var_pick_hook: Optional[Callable[[dict[str, Any]], Any]],
    pick_partition_vars: Callable[..., list[int]],
    n_orig: int,
    context: dict[str, Any],
) -> list[int]:
    """Select AMP partition variables through either a built-in method or user hook."""
    if disc_var_pick_hook is None:
        return pick_partition_vars(terms, method=method, distance=context.get("distance"))

    def builtin_pick(
        builtin_method: str = method,
        distance: Optional[dict[int, float]] = None,
    ) -> list[int]:
        return pick_partition_vars(terms, method=builtin_method, distance=distance)

    context = dict(context)
    context.setdefault("partition_candidates", list(getattr(terms, "partition_candidates", [])))
    context.setdefault("builtin_pick_partition_vars", builtin_pick)
    return _normalize_partition_var_indices(
        disc_var_pick_hook(context),
        n_orig,
        source="disc_var_pick",
    )


def _apply_partition_scaling_update(
    partition_scaling_factor_update: Optional[Callable[[dict[str, Any]], Any]],
    *,
    current_scaling_factor: float,
    context: dict[str, Any],
) -> float:
    """Apply an optional user hook that updates AMP's adaptive refinement width."""
    if partition_scaling_factor_update is None:
        return current_scaling_factor

    hook_context = dict(context)
    hook_context["current_scaling_factor"] = current_scaling_factor
    updated = partition_scaling_factor_update(hook_context)
    if updated is None:
        return current_scaling_factor
    return _validate_partition_scaling_factor(updated, "partition_scaling_factor_update")


def _apply_partition_refinement_hook(
    disc_add_partition_hook: Callable[[dict[str, Any]], Any],
    context: dict[str, Any],
) -> Any:
    """Run a custom AMP partition-refinement hook and validate its state result."""
    result = disc_add_partition_hook(dict(context))
    if result is None:
        result = context.get("disc_state")
    if not (
        hasattr(result, "partitions")
        and hasattr(result, "scaling_factor")
        and hasattr(result, "abs_width_tol")
    ):
        raise ValueError(
            "disc_add_partition_method callable must return a DiscretizationState "
            "or mutate context['disc_state'] in place"
        )
    return result


def _normalize_partition_method(
    partition_method: str,
    disc_var_pick: int | str | Callable[[dict[str, Any]], Any] | None,
) -> str:
    """Resolve public AMP aliases to the internal partition-selection strategy."""
    if disc_var_pick is None:
        return partition_method
    if callable(disc_var_pick):
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


def _normalize_presolve_bt_algo(presolve_bt_algo: int | str) -> str:
    """Resolve Alpine-style presolve OBBT mode aliases."""
    if isinstance(presolve_bt_algo, str):
        normalized = presolve_bt_algo.strip().lower().replace("-", "_")
        aliases = {
            "1": "lp",
            "lp": "lp",
            "obbt": "lp",
            "lp_obbt": "lp",
            "linear": "lp",
            "2": "incumbent_partitioned",
            "tmc": "incumbent_partitioned",
            "partitioned": "incumbent_partitioned",
            "partitioned_obbt": "incumbent_partitioned",
            "incumbent": "incumbent_partitioned",
            "incumbent_partitioned": "incumbent_partitioned",
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(
            f"Unsupported presolve_bt_algo: {presolve_bt_algo!r}. "
            "Choose 1/'lp' or 2/'incumbent_partitioned'."
        )

    if presolve_bt_algo == 1:
        return "lp"
    if presolve_bt_algo == 2:
        return "incumbent_partitioned"

    raise ValueError(
        f"Unsupported presolve_bt_algo: {presolve_bt_algo!r}. "
        "Choose 1/'lp' or 2/'incumbent_partitioned'."
    )


def _resolve_presolve_bt_time_limits(
    remaining: float,
    n_orig: int,
    presolve_bt_time_limit: Optional[float],
    presolve_bt_mip_time_limit: Optional[float],
) -> tuple[float, float]:
    """Return total presolve and per-subproblem OBBT budgets.

    ``presolve_bt_time_limit`` caps the whole presolve OBBT pass.  When it is
    omitted, AMP preserves the historical default: at most 10 seconds and at
    most 10% of remaining wall time.  ``presolve_bt_mip_time_limit`` caps each
    LP/MILP subproblem inside that total budget.
    """
    if presolve_bt_time_limit is not None and presolve_bt_time_limit < 0.0:
        raise ValueError("presolve_bt_time_limit must be non-negative")
    if presolve_bt_mip_time_limit is not None and presolve_bt_mip_time_limit < 0.0:
        raise ValueError("presolve_bt_mip_time_limit must be non-negative")
    if not np.isfinite(remaining) or remaining <= 0.0:
        return 0.0, 0.0

    if presolve_bt_time_limit is None:
        total_budget = min(10.0, 0.1 * remaining)
        per_subproblem = _default_obbt_time_limit_per_lp(remaining, n_orig)
    else:
        total_budget = min(float(presolve_bt_time_limit), remaining)
        per_subproblem = total_budget / max(1, 2 * n_orig)

    if presolve_bt_mip_time_limit is not None:
        per_subproblem = min(per_subproblem, float(presolve_bt_mip_time_limit))

    return max(0.0, total_budget), max(0.0, per_subproblem)


def _append_upper_bound_constraint(
    A_ub,
    b_ub: Optional[np.ndarray],
    row: np.ndarray,
    rhs: float,
):
    """Append ``row @ x <= rhs`` to dense or sparse inequality data."""
    import scipy.sparse as sp

    row_csr = sp.csr_matrix(np.asarray(row, dtype=np.float64).reshape(1, -1))
    rhs_arr = np.array([float(rhs)], dtype=np.float64)
    if A_ub is None or b_ub is None:
        return row_csr, rhs_arr
    return (
        sp.vstack([sp.csr_matrix(A_ub), row_csr], format="csr"),
        np.concatenate([np.asarray(b_ub, dtype=np.float64), rhs_arr]),
    )


def _presolve_incumbent_from_initial_point(
    initial_point: Optional[np.ndarray],
    model: Model,
    evaluator,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Return a feasible initial point and objective for incumbent-seeded presolve."""
    if initial_point is None:
        return None, None
    if not _check_integer_feasible(initial_point, model):
        return None, None
    if not _check_constraints_with_evaluator(
        evaluator,
        initial_point,
        constraint_lb,
        constraint_ub,
    ):
        return None, None

    obj = float(evaluator.evaluate_objective(initial_point))
    if not np.isfinite(obj):
        return None, None
    return np.asarray(initial_point, dtype=np.float64).copy(), obj


def _run_partitioned_obbt(
    model: Model,
    terms,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    incumbent: np.ndarray,
    incumbent_obj: float,
    *,
    partition_mode: str,
    n_init_partitions: int,
    partition_scaling_factor: float,
    disc_abs_width_tol: float,
    convhull_formulation: str,
    convhull_ebd: bool,
    convhull_ebd_encoding: str,
    total_time_limit: float,
    time_limit_per_mip: float,
    gap_tolerance: float,
    partition_scaling_factor_update: Optional[Callable[[dict[str, Any]], Any]] = None,
    disc_var_pick_hook: Optional[Callable[[dict[str, Any]], Any]] = None,
    disc_add_partition_hook: Optional[Callable[[dict[str, Any]], Any]] = None,
    min_width: float = 1e-6,
):
    """Run incumbent-seeded OBBT on AMP's partition-aware MILP relaxation."""
    from discopt._jax.discretization import (
        DiscretizationState,
        add_adaptive_partition,
        initialize_partitions,
    )
    from discopt._jax.milp_relaxation import MilpRelaxationModel, build_milp_relaxation
    from discopt._jax.obbt import ObbtResult
    from discopt._jax.partition_selection import pick_partition_vars

    n_orig = len(flat_lb)
    part_vars = _select_partition_vars_with_hook(
        terms,
        method=partition_mode,
        disc_var_pick_hook=disc_var_pick_hook,
        pick_partition_vars=pick_partition_vars,
        n_orig=n_orig,
        context={
            "stage": "presolve_obbt_selection",
            "model": model,
            "terms": terms,
            "flat_lb": flat_lb.copy(),
            "flat_ub": flat_ub.copy(),
            "partition_mode": partition_mode,
            "partition_scaling_factor": partition_scaling_factor,
            "incumbent": incumbent.copy(),
            "incumbent_objective": incumbent_obj,
        },
    )
    if disc_var_pick_hook is None:
        part_vars = _merge_partition_vars(
            part_vars,
            _equality_square_monomial_partition_candidates(model, terms),
        )
    if not part_vars and terms.monomial:
        part_vars = sorted(set(var_idx for var_idx, _ in terms.monomial))

    if part_vars:
        part_lbs = [float(flat_lb[i]) for i in part_vars]
        part_ubs = [float(flat_ub[i]) for i in part_vars]
        base_state = initialize_partitions(
            part_vars,
            lb=part_lbs,
            ub=part_ubs,
            n_init=n_init_partitions,
            scaling_factor=partition_scaling_factor,
            abs_width_tol=disc_abs_width_tol,
        )
        solution = {i: float(incumbent[i]) for i in part_vars}
        refinement_context = {
            "stage": "presolve_obbt_refinement",
            "model": model,
            "terms": terms,
            "disc_state": base_state,
            "solution": solution,
            "var_indices": list(part_vars),
            "lb": list(part_lbs),
            "ub": list(part_ubs),
            "flat_lb": flat_lb.copy(),
            "flat_ub": flat_ub.copy(),
            "partition_mode": partition_mode,
            "partition_scaling_factor": partition_scaling_factor,
            "incumbent": incumbent.copy(),
            "incumbent_objective": incumbent_obj,
        }
        partition_scaling_factor = _apply_partition_scaling_update(
            partition_scaling_factor_update,
            current_scaling_factor=partition_scaling_factor,
            context=refinement_context,
        )
        base_state.scaling_factor = partition_scaling_factor
        refinement_context["partition_scaling_factor"] = partition_scaling_factor
        refinement_context["disc_state"] = base_state
        if disc_add_partition_hook is None:
            disc_state = add_adaptive_partition(
                base_state,
                solution,
                part_vars,
                part_lbs,
                part_ubs,
            )
        else:
            disc_state = _apply_partition_refinement_hook(
                disc_add_partition_hook,
                refinement_context,
            )
    else:
        disc_state = DiscretizationState(
            scaling_factor=partition_scaling_factor,
            abs_width_tol=disc_abs_width_tol,
        )

    base_relaxation, _varmap = build_milp_relaxation(
        model,
        terms,
        disc_state,
        incumbent=incumbent,
        convhull_formulation=convhull_formulation,
        convhull_ebd=convhull_ebd,
        convhull_ebd_encoding=convhull_ebd_encoding,
        bound_override=(flat_lb.copy(), flat_ub.copy()),
    )

    A_ub = base_relaxation._A_ub
    b_ub = base_relaxation._b_ub
    if base_relaxation._objective_bound_valid:
        relaxation_incumbent_obj = float(incumbent_obj)
        if model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
            relaxation_incumbent_obj = -relaxation_incumbent_obj
        cutoff_rhs = relaxation_incumbent_obj - base_relaxation._obj_offset
        cutoff_rhs += 1e-8 * max(1.0, abs(relaxation_incumbent_obj))
        A_ub, b_ub = _append_upper_bound_constraint(
            A_ub,
            b_ub,
            base_relaxation._c,
            cutoff_rhs,
        )
    else:
        logger.info(
            "AMP: partitioned OBBT objective cutoff skipped because relaxation objective "
            "is not linearizable"
        )

    tightened_lb = np.asarray(flat_lb, dtype=np.float64).copy()
    tightened_ub = np.asarray(flat_ub, dtype=np.float64).copy()
    bounds_list = list(base_relaxation._bounds)
    for i in range(n_orig):
        bounds_list[i] = (float(tightened_lb[i]), float(tightened_ub[i]))

    candidates = [
        i
        for i in range(n_orig)
        if (
            tightened_ub[i] - tightened_lb[i] > min_width
            and is_effectively_finite(float(tightened_lb[i]))
            and is_effectively_finite(float(tightened_ub[i]))
        )
    ]

    deadline = time.perf_counter() + total_time_limit
    n_mip_solves = 0
    n_tightened = 0
    total_solve_time = 0.0

    def solve_bound_objective(c: np.ndarray, remaining: float):
        nonlocal n_mip_solves, total_solve_time
        subproblem_limit = min(time_limit_per_mip, remaining)
        if subproblem_limit <= 0.0:
            return None
        subproblem = MilpRelaxationModel(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds_list,
            obj_offset=0.0,
            integrality=base_relaxation._integrality,
            objective_bound_valid=True,
        )
        start = time.perf_counter()
        result = subproblem.solve(
            time_limit=subproblem_limit,
            gap_tolerance=gap_tolerance,
        )
        total_solve_time += time.perf_counter() - start
        n_mip_solves += 1
        return result

    for var_idx in candidates:
        remaining = _remaining_wall_time(deadline)
        if remaining is not None and remaining <= 0.0:
            break
        assert remaining is not None

        c = np.zeros(len(bounds_list), dtype=np.float64)
        c[var_idx] = 1.0
        result = solve_bound_objective(c, remaining)
        if result is not None and result.status == "optimal" and result.objective is not None:
            new_lb = float(result.objective)
            if new_lb > tightened_lb[var_idx] + 1e-8 and new_lb <= tightened_ub[var_idx] + 1e-8:
                tightened_lb[var_idx] = new_lb
                bounds_list[var_idx] = (float(tightened_lb[var_idx]), float(tightened_ub[var_idx]))
                n_tightened += 1

        remaining = _remaining_wall_time(deadline)
        if remaining is not None and remaining <= 0.0:
            break
        assert remaining is not None

        c[var_idx] = -1.0
        result = solve_bound_objective(c, remaining)
        if result is not None and result.status == "optimal" and result.objective is not None:
            new_ub = -float(result.objective)
            if new_ub < tightened_ub[var_idx] - 1e-8 and new_ub >= tightened_lb[var_idx] - 1e-8:
                tightened_ub[var_idx] = new_ub
                bounds_list[var_idx] = (float(tightened_lb[var_idx]), float(tightened_ub[var_idx]))
                n_tightened += 1

    return ObbtResult(
        tightened_lb=tightened_lb,
        tightened_ub=tightened_ub,
        n_lp_solves=n_mip_solves,
        n_tightened=n_tightened,
        total_lp_time=total_solve_time,
    )


def _run_amp_presolve_bound_tightening(
    model: Model,
    terms,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    *,
    presolve_bt_algo: int | str,
    remaining: float,
    incumbent: Optional[np.ndarray],
    incumbent_obj: Optional[float],
    n_init_partitions: int,
    partition_mode: str,
    partition_scaling_factor: float,
    disc_abs_width_tol: float,
    convhull_formulation: str,
    convhull_ebd: bool,
    convhull_ebd_encoding: str,
    milp_gap_tolerance: Optional[float],
    presolve_bt_time_limit: Optional[float],
    presolve_bt_mip_time_limit: Optional[float],
    partition_scaling_factor_update: Optional[Callable[[dict[str, Any]], Any]] = None,
    disc_var_pick_hook: Optional[Callable[[dict[str, Any]], Any]] = None,
    disc_add_partition_hook: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Run the configured AMP presolve OBBT mode and return tightened bounds."""
    from discopt._jax.obbt import run_obbt

    algo = _normalize_presolve_bt_algo(presolve_bt_algo)
    n_orig = len(flat_lb)
    total_budget, subproblem_budget = _resolve_presolve_bt_time_limits(
        remaining,
        n_orig,
        presolve_bt_time_limit,
        presolve_bt_mip_time_limit,
    )
    if total_budget <= 0.0 or subproblem_budget <= 0.0:
        logger.info("AMP: skipping OBBT presolve because no wall-clock budget remains")
        return flat_lb, flat_ub, None

    deadline = time.perf_counter() + total_budget

    if algo == "incumbent_partitioned" and (incumbent is None or incumbent_obj is None):
        logger.info("AMP: no feasible incumbent for partitioned OBBT; falling back to LP OBBT")
        algo = "lp"

    if algo == "lp":
        lp_budget = _remaining_wall_time(deadline)
        result = run_obbt(
            model,
            lb=flat_lb.copy(),
            ub=flat_ub.copy(),
            time_limit_per_lp=subproblem_budget,
            total_time_limit=lp_budget,
        )
        return result.tightened_lb, result.tightened_ub, result

    assert incumbent is not None
    assert incumbent_obj is not None
    partitioned_budget = _remaining_wall_time(deadline)
    if partitioned_budget is None or partitioned_budget <= 0.0:
        logger.info("AMP: skipping partitioned OBBT because no wall-clock budget remains")
        return flat_lb, flat_ub, None
    try:
        result = _run_partitioned_obbt(
            model,
            terms,
            flat_lb,
            flat_ub,
            incumbent,
            float(incumbent_obj),
            partition_mode=partition_mode,
            n_init_partitions=n_init_partitions,
            partition_scaling_factor=partition_scaling_factor,
            disc_abs_width_tol=disc_abs_width_tol,
            convhull_formulation=convhull_formulation,
            convhull_ebd=convhull_ebd,
            convhull_ebd_encoding=convhull_ebd_encoding,
            total_time_limit=partitioned_budget,
            time_limit_per_mip=subproblem_budget,
            gap_tolerance=milp_gap_tolerance if milp_gap_tolerance is not None else 1e-4,
            partition_scaling_factor_update=partition_scaling_factor_update,
            disc_var_pick_hook=disc_var_pick_hook,
            disc_add_partition_hook=disc_add_partition_hook,
        )
    except Exception as err:
        logger.warning(
            "AMP: partitioned OBBT presolve failed; falling back to LP OBBT: %s",
            err,
        )
        lp_budget = _remaining_wall_time(deadline)
        result = run_obbt(
            model,
            lb=flat_lb.copy(),
            ub=flat_ub.copy(),
            time_limit_per_lp=subproblem_budget,
            total_time_limit=lp_budget,
        )
    return result.tightened_lb, result.tightened_ub, result


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


def _run_cutoff_obbt(
    *,
    model: Model,
    terms,
    disc_state,
    oa_cuts: list,
    convhull_mode: str,
    UB: float,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    part_vars: list,
    part_lbs: list,
    part_ubs: list,
    n_orig: int,
    obbt_time_limit: float,
    partition_scaling_factor: float,
    disc_abs_width_tol: float,
    n_init_partitions: int,
    deadline: float,
    iteration: int,
    from_min_space: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    """Run cutoff-OBBT against the current MILP relaxation and apply tightenings.

    Uses the current ``disc_state`` and accumulated ``oa_cuts`` so the LP
    relaxation reflects the latest envelope strength.  Returns possibly
    updated (flat_lb, flat_ub, part_vars, part_lbs, part_ubs).
    """
    remaining_wall_time = _remaining_wall_time(deadline)
    if remaining_wall_time is not None and remaining_wall_time <= 0.0:
        return flat_lb, flat_ub, part_vars, part_lbs, part_ubs

    try:
        from discopt._jax.milp_relaxation import build_milp_relaxation
        from discopt._jax.obbt import run_obbt_on_relaxation

        cutoff_relax, _ = build_milp_relaxation(
            model,
            terms,
            disc_state,
            None,
            oa_cuts=oa_cuts if oa_cuts else None,
            convhull_formulation=convhull_mode,
        )
        candidates = sorted(set(part_vars) | set(terms.partition_candidates))
        if not candidates:
            return flat_lb, flat_ub, part_vars, part_lbs, part_ubs
        remaining_wall_time = _remaining_wall_time(deadline)
        if remaining_wall_time is not None and remaining_wall_time <= 0.0:
            return flat_lb, flat_ub, part_vars, part_lbs, part_ubs
        remaining = obbt_time_limit
        if remaining_wall_time is not None:
            remaining = min(remaining, remaining_wall_time)
        if remaining <= 0.0:
            return flat_lb, flat_ub, part_vars, part_lbs, part_ubs
        per_lp = min(
            remaining,
            max(0.05, min(1.0, obbt_time_limit / max(1, 2 * len(candidates)))),
        )
        result = run_obbt_on_relaxation(
            cutoff_relax,
            n_orig=n_orig,
            candidate_idxs=candidates,
            time_limit_per_lp=per_lp,
            incumbent_cutoff=float(UB),
            deadline=time.perf_counter() + remaining,
        )
        if result.n_tightened == 0:
            logger.info(
                "AMP cutoff OBBT @ iter %d: no tightening (%d LPs, %.2fs, cutoff=%.6g)",
                iteration,
                result.n_lp_solves,
                result.total_lp_time,
                from_min_space(UB),
            )
            return flat_lb, flat_ub, part_vars, part_lbs, part_ubs

        tight_lb = np.maximum(flat_lb, result.tightened_lb)
        tight_ub = np.minimum(flat_ub, result.tightened_ub)
        tight_lb, tight_ub = _repair_inverted_bounds(tight_lb, tight_ub)
        _apply_flat_bounds_to_model(model, tight_lb, tight_ub)
        flat_lb, flat_ub = tight_lb, tight_ub
        part_vars, part_lbs, part_ubs = _refresh_partitions_for_bounds(
            model,
            disc_state,
            flat_lb,
            flat_ub,
            part_vars,
            disc_abs_width_tol,
            n_init_partitions,
        )
        logger.info(
            "AMP cutoff OBBT @ iter %d: %d/%d bounds tightened in %d LPs (%.2fs, cutoff=%.6g)",
            iteration,
            result.n_tightened,
            len(candidates),
            result.n_lp_solves,
            result.total_lp_time,
            from_min_space(UB),
        )
        return flat_lb, flat_ub, part_vars, part_lbs, part_ubs
    except Exception as exc:
        logger.debug("AMP cutoff OBBT skipped: %s", exc)
        return flat_lb, flat_ub, part_vars, part_lbs, part_ubs


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def _solve_amp_impl(
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
    disc_var_pick: int | str | Callable[[dict[str, Any]], Any] | None = None,
    partition_scaling_factor: float = 10.0,
    partition_scaling_factor_update: Optional[Callable[[dict[str, Any]], Any]] = None,
    disc_add_partition_method: str | Callable[[dict[str, Any]], Any] = "adaptive",
    disc_abs_width_tol: float = 1e-3,
    convhull_formulation: str = "disaggregated",
    convhull_ebd: bool = False,
    convhull_ebd_encoding: str = "gray",
    presolve_bt: bool = True,
    presolve_bt_algo: int | str = 1,
    presolve_bt_time_limit: Optional[float] = None,
    presolve_bt_mip_time_limit: Optional[float] = None,
    initial_point: Optional[np.ndarray] = None,
    use_start_as_incumbent: bool = False,
    skip_convex_check: bool = False,
    obbt_at_root: bool = False,
    obbt_with_cutoff: bool = False,
    alphabb_cutoff_obbt: bool = True,
    obbt_time_limit: float = 30.0,
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
    disc_var_pick : int, str, or callable, optional
        Alpine-style alias for partition selection:
        0/``"all"`` → max_cover, 1 → min_vertex_cover, 2 → auto,
        3 → adaptive weighted cover. This intentionally collapses Alpine's
        separate `disc_var_pick_algo` and `disc_var_pick` options into one
        user-facing control. A callable may also be supplied; it receives a
        context dict and must return flat variable indices to partition.
    partition_scaling_factor : float
        Width scaling used by adaptive partition refinement.
    partition_scaling_factor_update : callable, optional
        Hook called before each partition-refinement step. It receives the same
        context dict shape as the partition hooks plus ``current_scaling_factor``
        and may return a new finite factor greater than 1. Returning ``None``
        keeps the current factor.
    disc_add_partition_method : str or callable
        Refinement update rule: ``"adaptive"`` or ``"uniform"``. A callable
        may also be supplied; it receives a context dict and must return a
        ``DiscretizationState`` or mutate ``context["disc_state"]`` in place.
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
    presolve_bt_algo : int or str
        Bound-tightening algorithm. ``1``/``"lp"`` keeps the cheap LP OBBT
        baseline. ``2``/``"incumbent_partitioned"`` uses a feasible incumbent
        to seed adaptive partitions and solves partition-aware OBBT MILPs,
        falling back to LP OBBT when no feasible incumbent is available.
    presolve_bt_time_limit : float, optional
        Wall-clock cap for the whole presolve OBBT pass. If omitted, AMP keeps
        its historical default of at most 10 seconds and at most 10% of the
        remaining global ``time_limit``.
    presolve_bt_mip_time_limit : float, optional
        Per-LP/MILP cap for each OBBT subproblem inside the presolve pass.
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
    obbt_at_root : bool, default False
        Run optimization-based bound tightening on the LP relaxation of the
        initial MILP envelope before iteration 1.  Tightens variable bounds
        used by every subsequent McCormick / piecewise envelope.  Default is
        off: without an incumbent cutoff, root OBBT can sometimes redirect
        adaptive partition refinement toward a worse fixed point.
    obbt_with_cutoff : bool, default False
        After the first feasible incumbent is found, re-run OBBT with the
        cutoff ``c^T x <= UB`` to exclude regions that cannot improve on the
        incumbent.  This is the standard form used by BARON / Couenne / Alpine.
        Off by default because OBBT can perturb the LP-optimal vertex enough
        to redirect adaptive partition refinement toward a worse fixed point
        on problems where variable bounds are already reasonably tight; turn
        on for problems with loose initial variable bounds.
    alphabb_cutoff_obbt : bool, default True
        Allow one cutoff OBBT pass before alpha-BB OA generation when a
        nonconvex OA row starts from effectively unbounded variable bounds.
        This is independent of the broader ``obbt_with_cutoff`` option and is
        used to create finite boxes for alpha-BB cuts on loose MINLPTests-style
        instances. Set False to disable this alpha-BB prerequisite pass.
    obbt_time_limit : float, default 30.0
        Total wall-clock budget for OBBT calls (in seconds).  Per-LP budget is
        ``min(1.0, obbt_time_limit / max(1, 2*n_candidates))``.

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

    partition_scaling_factor = _validate_partition_scaling_factor(partition_scaling_factor)
    disc_var_pick_hook = disc_var_pick if callable(disc_var_pick) else None
    disc_add_partition_hook = (
        disc_add_partition_method if callable(disc_add_partition_method) else None
    )
    if disc_add_partition_hook is None and disc_add_partition_method not in {"adaptive", "uniform"}:
        raise ValueError("disc_add_partition_method must be 'adaptive' or 'uniform'")

    partition_mode = _normalize_partition_method(partition_method, disc_var_pick)
    _normalize_presolve_bt_algo(presolve_bt_algo)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)
    if convhull_ebd and convhull_mode != "sos2":
        raise ValueError("convhull_ebd requires convhull_formulation='sos2' or the 'lambda' alias.")
    _resolve_presolve_bt_time_limits(
        remaining=time_limit,
        n_orig=max(1, sum(v.size for v in model._variables)),
        presolve_bt_time_limit=presolve_bt_time_limit,
        presolve_bt_mip_time_limit=presolve_bt_mip_time_limit,
    )

    n_orig = sum(v.size for v in model._variables)
    flat_lb, flat_ub = flat_variable_bounds(model)
    initial_point_arr = _normalize_initial_point(initial_point, n_orig, flat_lb, flat_ub)

    def _finish(result: SolveResult) -> SolveResult:
        return result

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
                return _finish(result)
        except Exception as exc:
            logger.debug("AMP: convex delegation check failed: %s", exc)

    def _from_minimization_space(value: float) -> float:
        return -float(value) if maximize else float(value)

    int_offsets: list[int] = []
    int_sizes: list[int] = []
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            int_offsets.append(offset)
            int_sizes.append(v.size)
        offset += v.size

    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    flat_lb, flat_ub, root_infeasible, root_changed = tighten_root_bounds_with_fbbt(
        model,
        flat_lb,
        flat_ub,
        int_offsets,
        int_sizes,
    )
    if root_infeasible:
        return _finish(
            SolveResult(
                status="infeasible",
                wall_time=time.perf_counter() - t_start,
                mip_count=0,
                gap_certified=True,
            )
        )
    if root_changed:
        logger.info("AMP: root FBBT tightened variable bounds before relaxation")

    tightened_lb, tightened_ub, nonlinear_bt_stats = tighten_nonlinear_bounds(
        model, flat_lb, flat_ub
    )
    if nonlinear_bt_stats.infeasible:
        logger.info(
            "AMP: nonlinear bound tightening proved infeasibility: %s",
            nonlinear_bt_stats.infeasibility_reason,
        )
        return _finish(
            SolveResult(
                status="infeasible",
                wall_time=time.perf_counter() - t_start,
                mip_count=0,
                gap_certified=True,
            )
        )
    if nonlinear_bt_stats.n_tightened > 0:
        flat_lb = tightened_lb
        flat_ub = tightened_ub
        logger.info(
            "AMP: nonlinear bound tightening adjusted %d bounds via %s",
            nonlinear_bt_stats.n_tightened,
            ", ".join(nonlinear_bt_stats.applied_rules),
        )
    if root_changed or nonlinear_bt_stats.n_tightened > 0:
        _apply_flat_bounds_to_model(model, flat_lb, flat_ub)
    evaluator = NLPEvaluator(model)
    constraint_lb, constraint_ub = _infer_constraint_bounds(model)
    deadline = t_start + time_limit
    presolve_incumbent, presolve_incumbent_obj = _presolve_incumbent_from_initial_point(
        initial_point_arr,
        model,
        evaluator,
        constraint_lb,
        constraint_ub,
    )
    oa_convexity = classify_oa_cut_convexity(model, use_certificate=True)
    direct_oa_skip_reasons = _direct_oa_skip_reasons(
        model,
        oa_convexity.constraint_mask,
        flat_lb,
        flat_ub,
    )
    direct_oa_skipped_rows = [
        (idx, reason) for idx, reason in enumerate(direct_oa_skip_reasons) if reason is not None
    ]
    if evaluator.n_constraints > 0 and direct_oa_skipped_rows:
        logger.warning(
            "AMP: direct OA skips %d of %d constraint rows not certified convex: rows=%s",
            len(direct_oa_skipped_rows),
            len(oa_convexity.constraint_mask),
            direct_oa_skipped_rows,
        )
        logger.info(
            "AMP: direct OA enabled for %d of %d constraint rows certified convex",
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
        try:
            tightened_lb, tightened_ub, obbt_result = _run_amp_presolve_bound_tightening(
                model,
                terms,
                flat_lb,
                flat_ub,
                presolve_bt_algo=presolve_bt_algo,
                remaining=remaining,
                incumbent=presolve_incumbent,
                incumbent_obj=presolve_incumbent_obj,
                n_init_partitions=n_init_partitions,
                partition_mode=partition_mode,
                partition_scaling_factor=partition_scaling_factor,
                disc_abs_width_tol=disc_abs_width_tol,
                convhull_formulation=convhull_mode,
                convhull_ebd=convhull_ebd,
                convhull_ebd_encoding=convhull_ebd_encoding,
                milp_gap_tolerance=milp_gap_tolerance,
                presolve_bt_time_limit=presolve_bt_time_limit,
                presolve_bt_mip_time_limit=presolve_bt_mip_time_limit,
                partition_scaling_factor_update=partition_scaling_factor_update,
                disc_var_pick_hook=disc_var_pick_hook,
                disc_add_partition_hook=disc_add_partition_hook,
            )
        except ImportError as err:
            logger.warning("AMP: OBBT presolve unavailable; continuing without it: %s", err)
        else:
            if obbt_result is not None and obbt_result.n_tightened > 0:
                flat_lb = tightened_lb
                flat_ub = tightened_ub
                logger.info(
                    "AMP: OBBT tightened %d bounds in %.3fs before partitioning",
                    obbt_result.n_tightened,
                    obbt_result.total_lp_time,
                )

    if initial_point_arr is not None:
        initial_point_arr = np.clip(initial_point_arr, flat_lb, flat_ub)

    # ── Select partition variables ───────────────────────────────────────────
    square_monomial_part_vars: list[int] = []
    if apply_partitioning:
        part_vars = _select_partition_vars_with_hook(
            terms,
            method=partition_mode,
            disc_var_pick_hook=disc_var_pick_hook,
            pick_partition_vars=pick_partition_vars,
            n_orig=n_orig,
            context={
                "stage": "initial_selection",
                "model": model,
                "terms": terms,
                "flat_lb": flat_lb.copy(),
                "flat_ub": flat_ub.copy(),
                "iteration": 0,
                "partition_mode": partition_mode,
                "partition_scaling_factor": partition_scaling_factor,
                "incumbent": None,
                "milp_result": None,
                "distance": None,
            },
        )
        if disc_var_pick_hook is None:
            square_monomial_part_vars = _equality_square_monomial_partition_candidates(
                model,
                terms,
            )
            part_vars = _merge_partition_vars(part_vars, square_monomial_part_vars)
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
        partition_label = "custom hook" if disc_var_pick_hook is not None else partition_mode
        logger.info("AMP: partitioning %d variables via %s", len(part_vars), partition_label)

    # ── Root OBBT on the LP relaxation of the initial MILP envelope ─────────
    # Tightens variable bounds used by every downstream McCormick / piecewise
    # envelope.  Uses an empty-partition relaxation (cheapest LP) and only
    # tightens variables whose width is large enough to plausibly matter.
    if obbt_at_root and apply_partitioning:
        try:
            from discopt._jax.discretization import DiscretizationState
            from discopt._jax.milp_relaxation import build_milp_relaxation
            from discopt._jax.obbt import run_obbt_on_relaxation

            _empty_disc = DiscretizationState(
                scaling_factor=partition_scaling_factor,
                abs_width_tol=disc_abs_width_tol,
            )
            _root_relax, _ = build_milp_relaxation(
                model,
                terms,
                _empty_disc,
                None,
                oa_cuts=None,
                convhull_formulation=convhull_mode,
            )
            # Tighten only the variables that appear in nonlinear terms — these
            # drive every envelope's tightness.  Tightening linear-only vars
            # rarely helps the relaxation and just costs LPs.
            _candidates: list[int] = sorted(set(part_vars) | set(terms.partition_candidates))
            _per_lp = max(0.05, min(1.0, obbt_time_limit / max(1, 2 * len(_candidates))))
            _obbt = run_obbt_on_relaxation(
                _root_relax,
                n_orig=n_orig,
                candidate_idxs=_candidates,
                time_limit_per_lp=_per_lp,
                deadline=min(deadline, t_start + obbt_time_limit),
            )
            if _obbt.n_tightened > 0:
                # Apply tightened bounds to the model and to flat_lb / flat_ub.
                tight_lb = np.maximum(flat_lb, _obbt.tightened_lb)
                tight_ub = np.minimum(flat_ub, _obbt.tightened_ub)
                # Soundness: don't cross — a numerical glitch could in principle
                # tighten lb past ub by a hair; clamp to avoid infeasibility.
                tight_lb, tight_ub = _repair_inverted_bounds(tight_lb, tight_ub)
                _apply_flat_bounds_to_model(model, tight_lb, tight_ub)
                flat_lb, flat_ub = tight_lb, tight_ub
                logger.info(
                    "AMP root OBBT: %d/%d bounds tightened in %d LPs (%.2fs)",
                    _obbt.n_tightened,
                    len(_candidates),
                    _obbt.n_lp_solves,
                    _obbt.total_lp_time,
                )
            else:
                logger.info(
                    "AMP root OBBT: no tighter bounds found (%d LPs, %.2fs)",
                    _obbt.n_lp_solves,
                    _obbt.total_lp_time,
                )
        except Exception as _obbt_err:
            logger.debug("AMP root OBBT skipped: %s", _obbt_err)

    # OBBT may have collapsed some candidate widths to ~0 (e.g. demand flows
    # whose bound is implied by an equality).  Drop those from part_vars so
    # initialize_partitions never sees a degenerate interval — partition
    # refinement on a width-0 variable cannot tighten anything.
    if part_vars:
        part_vars = [
            i for i in part_vars if float(flat_ub[i]) - float(flat_lb[i]) > disc_abs_width_tol
        ]

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
    cutoff_obbt_done = False
    last_obbt_iter = 0
    _obbt_period = 5

    def _append_linearized_cuts(cuts) -> int:
        appended = 0
        for cut in cuts:
            if np.linalg.norm(cut.coeffs) < 1e-12:
                continue
            if cut.sense == ">=":
                # Convert to <= form for milp_relaxation.py
                oa_cuts.append((-cut.coeffs, -cut.rhs))
            elif cut.sense == "==":
                oa_cuts.append((cut.coeffs, cut.rhs))
                oa_cuts.append((-cut.coeffs, -cut.rhs))
                appended += 2
                continue
            else:
                oa_cuts.append((cut.coeffs, cut.rhs))
            appended += 1
        return appended

    def _add_incumbent_oa_cuts_after_cutoff_tightening(
        x_incumbent: np.ndarray,
        iteration_idx: int,
        state: _AmpCutoffState,
    ) -> tuple[int, _AmpCutoffState]:
        """Append direct OA, run cutoff tightening, then append alpha-BB OA cuts."""
        if evaluator.n_constraints <= 0:
            return 0, state

        appended = 0
        try:
            from discopt._jax.cutting_planes import (
                generate_alphabb_quadratic_oa_cuts_from_evaluator,
                generate_oa_cuts_from_evaluator_report,
            )

            _x_orig = x_incumbent[:n_orig]
            _senses = [c.sense for c in model._constraints if isinstance(c, Constraint)]
            direct_report = generate_oa_cuts_from_evaluator_report(
                evaluator,
                _x_orig,
                constraint_senses=_senses,
                convex_mask=oa_convexity.constraint_mask,
                skip_reasons=direct_oa_skip_reasons,
            )
            if direct_report.skipped:
                logger.debug(
                    "AMP iter %d: direct OA skipped rows: %s",
                    iteration_idx,
                    list(direct_report.skipped),
                )
            appended += _append_linearized_cuts(direct_report.cuts)

            has_nonconvex_oa_row = not all(oa_convexity.constraint_mask)
            had_effectively_unbounded = any(
                not is_effectively_finite(float(value))
                for value in np.concatenate([state.flat_lb, state.flat_ub])
            )
            if has_nonconvex_oa_row and UB < np.inf:
                cutoff_lb, cutoff_ub = _tighten_bounds_with_objective_cutoff(
                    model,
                    state.flat_lb,
                    state.flat_ub,
                    UB,
                )
                if np.any(cutoff_lb > state.flat_lb + 1e-10) or np.any(
                    cutoff_ub < state.flat_ub - 1e-10
                ):
                    state.flat_lb = np.maximum(state.flat_lb, cutoff_lb)
                    state.flat_ub = np.minimum(state.flat_ub, cutoff_ub)
                    state.flat_lb, state.flat_ub = _repair_inverted_bounds(
                        state.flat_lb,
                        state.flat_ub,
                    )
                    _apply_flat_bounds_to_model(model, state.flat_lb, state.flat_ub)
                    try:
                        from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

                        fbbt_lb, fbbt_ub, fbbt_infeasible, _fbbt_changed = (
                            tighten_root_bounds_with_fbbt(
                                model,
                                state.flat_lb,
                                state.flat_ub,
                                int_offsets,
                                int_sizes,
                            )
                        )
                        if not fbbt_infeasible:
                            state.flat_lb = np.maximum(state.flat_lb, fbbt_lb)
                            state.flat_ub = np.minimum(state.flat_ub, fbbt_ub)
                            state.flat_lb, state.flat_ub = _repair_inverted_bounds(
                                state.flat_lb,
                                state.flat_ub,
                            )
                    except Exception as _cutoff_fbbt_err:
                        logger.debug(
                            "AMP objective cutoff FBBT skipped: %s",
                            _cutoff_fbbt_err,
                        )
                    state.part_vars, state.part_lbs, state.part_ubs = (
                        _refresh_partitions_for_bounds(
                            model,
                            disc_state,
                            state.flat_lb,
                            state.flat_ub,
                            state.part_vars,
                            disc_abs_width_tol,
                            n_init_partitions,
                        )
                    )
                    logger.info(
                        "AMP objective cutoff tightened bounds before alpha-BB OA generation"
                    )

            remaining_for_cutoff_obbt = _remaining_wall_time(deadline)
            if (
                UB < np.inf
                and not state.cutoff_obbt_done
                and remaining_for_cutoff_obbt is not None
                and remaining_for_cutoff_obbt > 0.0
                and (
                    obbt_with_cutoff
                    or (alphabb_cutoff_obbt and has_nonconvex_oa_row and had_effectively_unbounded)
                )
            ):
                state.cutoff_obbt_done = True
                (
                    state.flat_lb,
                    state.flat_ub,
                    state.part_vars,
                    state.part_lbs,
                    state.part_ubs,
                ) = _run_cutoff_obbt(
                    model=model,
                    terms=terms,
                    disc_state=disc_state,
                    oa_cuts=oa_cuts,
                    convhull_mode=convhull_mode,
                    UB=UB,
                    flat_lb=state.flat_lb,
                    flat_ub=state.flat_ub,
                    part_vars=state.part_vars,
                    part_lbs=state.part_lbs,
                    part_ubs=state.part_ubs,
                    n_orig=n_orig,
                    obbt_time_limit=obbt_time_limit,
                    partition_scaling_factor=partition_scaling_factor,
                    disc_abs_width_tol=disc_abs_width_tol,
                    n_init_partitions=n_init_partitions,
                    deadline=deadline,
                    iteration=iteration_idx,
                    from_min_space=_from_minimization_space,
                )
                state.last_obbt_iter = iteration_idx

            try:
                alphabb_cuts = generate_alphabb_quadratic_oa_cuts_from_evaluator(
                    evaluator,
                    _x_orig,
                    state.flat_lb,
                    state.flat_ub,
                    constraint_senses=_senses,
                    convex_mask=oa_convexity.constraint_mask,
                )
                appended += _append_linearized_cuts(alphabb_cuts)
            except Exception as _alphabb_oa_err:
                logger.debug(
                    "AMP: alpha-BB OA cut computation failed: %s",
                    _alphabb_oa_err,
                )

            _prune_oa_cuts(oa_cuts)
        except Exception as _oa_err:
            logger.debug("AMP: OA cut computation failed: %s", _oa_err)
        return appended, state

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
                            recovered_x = None
                            if recovered.x is not None:
                                try:
                                    recovered_x = np.concatenate(
                                        [
                                            np.asarray(recovered.x[var.name], dtype=np.float64)
                                            .reshape(-1)
                                            .copy()
                                            for var in model._variables
                                        ]
                                    )
                                except Exception:
                                    recovered_x = None
                            if recovered_x is not None and recovered.objective is not None:
                                incumbent = recovered_x
                                UB = (
                                    -float(recovered.objective)
                                    if maximize
                                    else float(recovered.objective)
                                )
                                n_before = len(oa_cuts)
                                cutoff_state = _AmpCutoffState(
                                    flat_lb,
                                    flat_ub,
                                    part_vars,
                                    part_lbs,
                                    part_ubs,
                                    cutoff_obbt_done,
                                    last_obbt_iter,
                                )
                                _, cutoff_state = _add_incumbent_oa_cuts_after_cutoff_tightening(
                                    incumbent,
                                    iteration,
                                    cutoff_state,
                                )
                                flat_lb, flat_ub, part_vars, part_lbs, part_ubs = (
                                    cutoff_state.bounds_tuple()
                                )
                                cutoff_obbt_done = cutoff_state.cutoff_obbt_done
                                last_obbt_iter = cutoff_state.last_obbt_iter
                                if len(oa_cuts) > n_before:
                                    termination_reason = "iteration_limit"
                                    logger.info(
                                        "AMP: continuing after pure-continuous recovery "
                                        "with %d OA cuts",
                                        len(oa_cuts) - n_before,
                                    )
                                    continue
                            recovered.mip_count = mip_count
                            return _finish(recovered)
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
                        return _finish(
                            SolveResult(
                                status="feasible",
                                objective=_from_minimization_space(fallback_obj),
                                bound=None,
                                gap=None,
                                x=_build_x_dict(fallback_x, model),
                                wall_time=time.perf_counter() - t_start,
                                mip_count=mip_count,
                                gap_certified=False,
                            )
                        )
                    if milp_result.status == "infeasible":
                        return _finish(
                            SolveResult(
                                status="infeasible",
                                wall_time=time.perf_counter() - t_start,
                                mip_count=mip_count,
                            )
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

                # Accumulate OA tangent cuts at this NLP solution.  Direct
                # convex OA cuts are appended first, then cutoff tightening runs
                # before alpha-BB tries to build cuts that require finite boxes.
                cutoff_state = _AmpCutoffState(
                    flat_lb,
                    flat_ub,
                    part_vars,
                    part_lbs,
                    part_ubs,
                    cutoff_obbt_done,
                    last_obbt_iter,
                )
                _, cutoff_state = _add_incumbent_oa_cuts_after_cutoff_tightening(
                    x_nlp,
                    iteration,
                    cutoff_state,
                )
                flat_lb, flat_ub, part_vars, part_lbs, part_ubs = cutoff_state.bounds_tuple()
                cutoff_obbt_done = cutoff_state.cutoff_obbt_done
                last_obbt_iter = cutoff_state.last_obbt_iter

        # ── Periodic cutoff OBBT ─────────────────────────────────────────────
        # After the first incumbent OBBT, re-run periodically as partitioning
        # refines: tighter envelopes can produce strictly better bounds even
        # without a new incumbent.
        if (
            obbt_with_cutoff
            and cutoff_obbt_done
            and UB < np.inf
            and iteration - last_obbt_iter >= _obbt_period
        ):
            flat_lb, flat_ub, part_vars, part_lbs, part_ubs = _run_cutoff_obbt(
                model=model,
                terms=terms,
                disc_state=disc_state,
                oa_cuts=oa_cuts,
                convhull_mode=convhull_mode,
                UB=UB,
                flat_lb=flat_lb,
                flat_ub=flat_ub,
                part_vars=part_vars,
                part_lbs=part_lbs,
                part_ubs=part_ubs,
                n_orig=n_orig,
                obbt_time_limit=obbt_time_limit,
                partition_scaling_factor=partition_scaling_factor,
                disc_abs_width_tol=disc_abs_width_tol,
                n_init_partitions=n_init_partitions,
                deadline=deadline,
                iteration=iteration,
                from_min_space=_from_minimization_space,
            )
            last_obbt_iter = iteration

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
            (partition_mode == "adaptive_vertex_cover" or disc_var_pick_hook is not None)
            and incumbent is not None
            and milp_result.x is not None
        ):
            distance_candidates = _merge_partition_vars(
                list(terms.partition_candidates),
                square_monomial_part_vars,
            )
            distances = {i: abs(float(incumbent[i]) - float(x0[i])) for i in distance_candidates}
            adaptive_method = (
                "adaptive_vertex_cover" if disc_var_pick_hook is None else partition_mode
            )
            adaptive_vars = _select_partition_vars_with_hook(
                terms,
                method=adaptive_method,
                disc_var_pick_hook=disc_var_pick_hook,
                pick_partition_vars=pick_partition_vars,
                n_orig=n_orig,
                context={
                    "stage": "iteration_selection",
                    "model": model,
                    "terms": terms,
                    "flat_lb": flat_lb.copy(),
                    "flat_ub": flat_ub.copy(),
                    "iteration": iteration,
                    "partition_mode": partition_mode,
                    "partition_scaling_factor": partition_scaling_factor,
                    "incumbent": incumbent.copy(),
                    "milp_solution": x0.copy(),
                    "milp_result": milp_result,
                    "distance": distances,
                    "part_vars": list(part_vars),
                },
            )
            if disc_var_pick_hook is None:
                adaptive_vars = _merge_partition_vars(adaptive_vars, square_monomial_part_vars)
            if adaptive_vars or disc_var_pick_hook is not None:
                should_update_adaptive_vars = set(adaptive_vars) != set(part_vars)
            else:
                should_update_adaptive_vars = False

            if should_update_adaptive_vars:
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

        if not part_vars:
            break

        # Use MILP solution (original vars) as the refinement point
        refine_solution: dict[int, float] = {}
        if milp_result.x is not None:
            x_orig = milp_result.x[:n_orig]
            for i in part_vars:
                refine_solution[i] = float(x_orig[i])

        refinement_context = {
            "stage": "refinement",
            "model": model,
            "terms": terms,
            "disc_state": disc_state,
            "solution": dict(refine_solution),
            "var_indices": list(part_vars),
            "lb": list(part_lbs),
            "ub": list(part_ubs),
            "flat_lb": flat_lb.copy(),
            "flat_ub": flat_ub.copy(),
            "iteration": iteration,
            "partition_mode": partition_mode,
            "partition_scaling_factor": partition_scaling_factor,
            "incumbent": None if incumbent is None else incumbent.copy(),
            "milp_solution": x0.copy(),
            "milp_result": milp_result,
            "lower_bound": LB,
            "upper_bound": UB,
        }
        partition_scaling_factor = _apply_partition_scaling_update(
            partition_scaling_factor_update,
            current_scaling_factor=partition_scaling_factor,
            context=refinement_context,
        )
        disc_state.scaling_factor = partition_scaling_factor
        refinement_context["partition_scaling_factor"] = partition_scaling_factor
        refinement_context["disc_state"] = disc_state

        if disc_add_partition_hook is not None:
            disc_state = _apply_partition_refinement_hook(
                disc_add_partition_hook,
                refinement_context,
            )
        elif disc_add_partition_method == "uniform":
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
        return _finish(
            SolveResult(
                status=status,
                wall_time=elapsed,
                mip_count=mip_count,
                gap_certified=False,
            )
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

        return _finish(
            SolveResult(
                status=status,
                objective=_from_minimization_space(UB),
                bound=(
                    _from_minimization_space(LB) if LB > -np.inf and bound_is_trustworthy else None
                ),
                gap=float(rel_gap_final) if rel_gap_final is not None else None,
                x=_build_x_dict(incumbent, model),
                wall_time=elapsed,
                mip_count=mip_count,
                gap_certified=gap_certified,
            )
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
            return _finish(recovered)

    if termination_reason == "time_limit" or elapsed >= time_limit:
        status = "time_limit"
    elif termination_reason == "error":
        status = "error"
    elif termination_reason == "infeasible":
        status = "infeasible"
    else:
        status = "iteration_limit"

    return _finish(
        SolveResult(
            status=status,
            objective=None,
            bound=_from_minimization_space(LB) if LB > -np.inf else None,
            gap=None,
            x=None,
            wall_time=elapsed,
            mip_count=mip_count,
            gap_certified=False,
        )
    )


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
    disc_var_pick: int | str | Callable[[dict[str, Any]], Any] | None = None,
    partition_scaling_factor: float = 10.0,
    partition_scaling_factor_update: Optional[Callable[[dict[str, Any]], Any]] = None,
    disc_add_partition_method: str | Callable[[dict[str, Any]], Any] = "adaptive",
    disc_abs_width_tol: float = 1e-3,
    convhull_formulation: str = "disaggregated",
    convhull_ebd: bool = False,
    convhull_ebd_encoding: str = "gray",
    presolve_bt: bool = True,
    presolve_bt_algo: int | str = 1,
    presolve_bt_time_limit: Optional[float] = None,
    presolve_bt_mip_time_limit: Optional[float] = None,
    initial_point: Optional[np.ndarray] = None,
    use_start_as_incumbent: bool = False,
    skip_convex_check: bool = False,
    obbt_at_root: bool = False,
    obbt_with_cutoff: bool = False,
    alphabb_cutoff_obbt: bool = True,
    obbt_time_limit: float = 30.0,
) -> SolveResult:
    saved_bounds = _snapshot_variable_bounds(model)
    try:
        return _solve_amp_impl(
            model,
            rel_gap=rel_gap,
            abs_tol=abs_tol,
            time_limit=time_limit,
            max_iter=max_iter,
            n_init_partitions=n_init_partitions,
            partition_method=partition_method,
            nlp_solver=nlp_solver,
            iteration_callback=iteration_callback,
            milp_time_limit=milp_time_limit,
            milp_gap_tolerance=milp_gap_tolerance,
            apply_partitioning=apply_partitioning,
            disc_var_pick=disc_var_pick,
            partition_scaling_factor=partition_scaling_factor,
            partition_scaling_factor_update=partition_scaling_factor_update,
            disc_add_partition_method=disc_add_partition_method,
            disc_abs_width_tol=disc_abs_width_tol,
            convhull_formulation=convhull_formulation,
            convhull_ebd=convhull_ebd,
            convhull_ebd_encoding=convhull_ebd_encoding,
            presolve_bt=presolve_bt,
            presolve_bt_algo=presolve_bt_algo,
            presolve_bt_time_limit=presolve_bt_time_limit,
            presolve_bt_mip_time_limit=presolve_bt_mip_time_limit,
            initial_point=initial_point,
            use_start_as_incumbent=use_start_as_incumbent,
            skip_convex_check=skip_convex_check,
            obbt_at_root=obbt_at_root,
            obbt_with_cutoff=obbt_with_cutoff,
            alphabb_cutoff_obbt=alphabb_cutoff_obbt,
            obbt_time_limit=obbt_time_limit,
        )
    finally:
        _restore_variable_bounds(saved_bounds)


solve_amp.__doc__ = _solve_amp_impl.__doc__
