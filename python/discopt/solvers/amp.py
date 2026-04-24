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
  3. Check gap: if (UB_k - LB_k) / |UB_k| ≤ rel_gap → CERTIFIED OPTIMAL
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
from discopt.modeling.core import Model, ObjectiveSense, SolveResult, VarType

logger = logging.getLogger(__name__)
_DEFAULT_MAX_OA_CUTS = 128


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
        solver_options: dict[str, float | int] = {"print_level": 0, "max_iter": 300}
        if time_limit is not None:
            solver_options["max_wall_time"] = max(time_limit, 0.05)

        prefer_ipopt = nlp_solver == "ipm" and time_limit is not None and _has_cyipopt()
        model = evaluator._model
        saved_bounds = _snapshot_variable_bounds(model)
        _apply_flat_bounds_to_model(model, lb, ub)

        try:
            if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn") and not prefer_ipopt:
                from discopt._jax.ipm import solve_nlp_ipm

                result = solve_nlp_ipm(evaluator, x0_clipped, options=solver_options)
            else:
                from discopt.solvers.nlp_ipopt import solve_nlp

                if time_limit is not None:
                    solver_options["max_cpu_time"] = max(time_limit, 0.05)
                result = solve_nlp(evaluator, x0_clipped, options=solver_options)
        finally:
            _restore_variable_bounds(saved_bounds)

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL:
            obj = float(evaluator.evaluate_objective(result.x))
            return result.x, obj
    except Exception as e:
        logger.debug("AMP NLP subproblem failed: %s", e)
    return None, None


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

                options: list[int] = []
                for raw in (
                    int(round(clipped)),
                    int(np.floor(clipped)),
                    int(np.ceil(clipped)),
                ):
                    if lo_i <= hi_i:
                        cand_int = min(max(raw, lo_i), hi_i)
                    else:
                        cand_int = int(round(clipped))
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
    return deduped


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
                rounded = round(val)
                nlp_lb[idx] = rounded
                nlp_ub[idx] = rounded
        offset += v.size

    return nlp_lb, nlp_ub


def _solve_best_nlp_candidate(
    x0: np.ndarray,
    model: Model,
    evaluator,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    constraint_lb: np.ndarray,
    constraint_ub: np.ndarray,
    nlp_solver: str,
    deadline: Optional[float] = None,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Return the best feasible NLP candidate across the integer-rounding set."""
    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None

    for x0_nlp in _integer_rounding_candidates(x0, model):
        remaining = _remaining_wall_time(deadline)
        if remaining is not None and remaining <= 0.0:
            break
        nlp_lb, nlp_ub = _build_fixed_integer_bounds(x0_nlp, model, flat_lb, flat_ub)
        if remaining is None:
            cand_x, cand_obj = _solve_nlp_subproblem(
                evaluator,
                x0_nlp,
                nlp_lb,
                nlp_ub,
                nlp_solver,
            )
        else:
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


def _solve_milp_with_oa_recovery(
    model: Model,
    terms,
    disc_state,
    incumbent: Optional[np.ndarray],
    oa_cuts: Optional[list],
    time_limit: Optional[float],
    gap_tolerance: float,
    convhull_formulation: str,
):
    """Retry MILP solves after dropping the oldest half of OA cuts on infeasibility."""
    from discopt._jax.milp_relaxation import build_milp_relaxation

    active_oa_cuts = list(oa_cuts or [])
    max_retries = max(1, len(active_oa_cuts).bit_length() + 1)
    milp_result = None
    varmap = None

    for _retry in range(max_retries):
        milp_model, varmap = build_milp_relaxation(
            model,
            terms,
            disc_state,
            incumbent,
            oa_cuts=active_oa_cuts,
            convhull_formulation=convhull_formulation,
        )
        milp_result = milp_model.solve(
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )
        if milp_result.status != "infeasible" or not active_oa_cuts:
            return milp_result, varmap, active_oa_cuts

        drop_count = max(1, len(active_oa_cuts) // 2)
        logger.info(
            "AMP: MILP infeasible with %d OA cuts; dropping %d oldest cuts and retrying",
            len(active_oa_cuts),
            drop_count,
        )
        active_oa_cuts = active_oa_cuts[drop_count:]

    assert milp_result is not None
    assert varmap is not None
    return milp_result, varmap, active_oa_cuts


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


def _compute_relative_gap(
    abs_gap: Optional[float],
    upper_bound: float,
) -> Optional[float]:
    """Return a relative gap, or None when the upper bound is numerically zero."""
    if abs_gap is None or not np.isfinite(upper_bound) or abs(upper_bound) <= 1e-10:
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
) -> SolveResult:
    """Solve MINLP globally using Adaptive Multivariate Partitioning (AMP).

    Parameters
    ----------
    model : Model
        A validated discopt Model.
    rel_gap : float
        Relative gap tolerance: terminate when (UB-LB)/|UB| ≤ rel_gap.
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

    Returns
    -------
    SolveResult
        With gap_certified=True if termination is by gap criterion.
    """
    t_start = time.perf_counter()

    from discopt._jax.discretization import (
        add_adaptive_partition,
        add_uniform_partition,
        check_partition_convergence,
        initialize_partitions,
    )
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt._jax.partition_selection import pick_partition_vars
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    assert model._objective is not None
    maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
    part_lbs: list[float] = []
    part_ubs: list[float] = []

    if partition_scaling_factor <= 1.0:
        raise ValueError("partition_scaling_factor must be > 1.0")
    if disc_add_partition_method not in {"adaptive", "uniform"}:
        raise ValueError("disc_add_partition_method must be 'adaptive' or 'uniform'")

    partition_mode = _normalize_partition_method(partition_method, disc_var_pick)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)

    def _to_minimization_space(value: float) -> float:
        return -float(value) if maximize else float(value)

    def _from_minimization_space(value: float) -> float:
        return -float(value) if maximize else float(value)

    n_orig = sum(v.size for v in model._variables)
    flat_lb, flat_ub = flat_variable_bounds(model)
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
        "AMP: %d bilinear, %d trilinear, %d monomial, %d general_nl terms",
        len(terms.bilinear),
        len(terms.trilinear),
        len(terms.monomial),
        len(terms.general_nl),
    )

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
    gap_certified = False
    oa_cuts: list = []  # accumulated OA linearizations from NLP incumbents

    for iteration in range(1, max_iter + 1):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            logger.info("AMP: time limit reached at iteration %d", iteration)
            break

        remaining = time_limit - elapsed
        milp_tl = (
            milp_time_limit
            if milp_time_limit is not None
            else _default_milp_time_limit(remaining, iteration, max_iter)
        )

        # ── Step 1: Solve MILP relaxation → lower bound ──────────────────────
        # MILP gap tolerance: no tighter than needed for overall convergence.
        if milp_gap_tolerance is not None:
            _milp_gap_tol = milp_gap_tolerance
        else:
            _milp_gap_tol = min(rel_gap / 2, 1e-3)

        try:
            milp_result, varmap, active_oa_cuts = _solve_milp_with_oa_recovery(
                model=model,
                terms=terms,
                disc_state=disc_state,
                incumbent=incumbent,
                oa_cuts=oa_cuts,
                time_limit=milp_tl,
                gap_tolerance=_milp_gap_tol,
                convhull_formulation=convhull_mode,
            )
            oa_cuts = active_oa_cuts
        except Exception as e:
            logger.warning("AMP: MILP build/solve failed at iteration %d: %s", iteration, e)
            break

        if milp_result.status in ("infeasible", "error"):
            logger.info(
                "AMP: MILP infeasible/error at iteration %d, status=%s",
                iteration,
                milp_result.status,
            )
            if LB == -np.inf:
                # Problem may be infeasible
                if iteration == 1:
                    return SolveResult(
                        status="infeasible",
                        wall_time=time.perf_counter() - t_start,
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
            x0 = 0.5 * (flat_lb + flat_ub)

        x_nlp, obj_nlp_min = _solve_best_nlp_candidate(
            x0,
            model,
            evaluator,
            flat_lb,
            flat_ub,
            constraint_lb,
            constraint_ub,
            nlp_solver,
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
            abs_gap = UB - LB
            rel_g = _compute_relative_gap(abs_gap, UB)
            if maximize:
                display_lb = _from_minimization_space(UB)
                display_ub = _from_minimization_space(LB)
            else:
                display_lb = LB
                display_ub = UB
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
        return SolveResult(
            status="infeasible",
            wall_time=elapsed,
        )

    if incumbent is not None:
        abs_gap_final = UB - LB if LB > -np.inf else None
        rel_gap_final = _compute_relative_gap(abs_gap_final, UB)

        if gap_certified:
            status = "optimal"
        elif elapsed >= time_limit:
            status = "time_limit"
        else:
            status = "feasible"

        return SolveResult(
            status=status,
            objective=_from_minimization_space(UB),
            bound=_from_minimization_space(LB) if LB > -np.inf else None,
            gap=float(rel_gap_final) if rel_gap_final is not None else None,
            x=_build_x_dict(incumbent, model),
            wall_time=elapsed,
            gap_certified=gap_certified,
        )

    # No feasible solution found
    if elapsed >= time_limit:
        status = "time_limit"
    else:
        status = "infeasible"

    return SolveResult(
        status=status,
        objective=None,
        bound=_from_minimization_space(LB) if LB > -np.inf else None,
        gap=None,
        x=None,
        wall_time=elapsed,
        gap_certified=False,
    )
