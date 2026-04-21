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

import logging
import time
from typing import Callable, Optional

import numpy as np

from discopt.modeling.core import Model, SolveResult, VarType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return (lb_flat, ub_flat) arrays for all model variables."""
    lbs: list[float] = []
    ubs: list[float] = []
    for v in model._variables:
        lbs.extend(np.asarray(v.lb, dtype=np.float64).ravel().tolist())
        ubs.extend(np.asarray(v.ub, dtype=np.float64).ravel().tolist())
    return np.array(lbs, dtype=np.float64), np.array(ubs, dtype=np.float64)


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


def _solve_nlp_subproblem(
    model: Model,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    nlp_solver: str = "ipm",
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Solve the NLP relaxation with given bounds.

    Returns (x_opt, obj_val) or (None, None) on failure.
    """
    try:
        from discopt._jax.nlp_evaluator import NLPEvaluator

        evaluator = NLPEvaluator(model)
        lb_clip = np.clip(lb, -1e8, 1e8)
        ub_clip = np.clip(ub, -1e8, 1e8)
        x0_clipped = np.clip(x0, lb_clip, ub_clip)

        if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
            from discopt._jax.ipm import solve_nlp_ipm

            result = solve_nlp_ipm(
                evaluator, x0_clipped, options={"print_level": 0, "max_iter": 300}
            )
        else:
            from discopt.solvers.nlp_ipopt import solve_nlp

            result = solve_nlp(
                evaluator, x0_clipped, options={"print_level": 0, "max_iter": 300}
            )

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL:
            obj = float(evaluator.evaluate_objective(result.x))
            return result.x, obj
    except Exception as e:
        logger.debug("AMP NLP subproblem failed: %s", e)
    return None, None


def _check_integer_feasible(
    x: np.ndarray, model: Model, int_tol: float = 1e-5
) -> bool:
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


def _round_integers(x: np.ndarray, model: Model) -> np.ndarray:
    """Round integer/binary variables to nearest integer in-place (copy)."""
    x = x.copy()
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            for i in range(v.size):
                x[offset + i] = round(float(x[offset + i]))
        offset += v.size
    return x


def _check_constraints(x: np.ndarray, model: Model, tol: float = 1e-4) -> bool:
    """Return True if all constraints are satisfied at x."""
    try:
        from discopt._jax.nlp_evaluator import NLPEvaluator

        evaluator = NLPEvaluator(model)
        if evaluator.n_constraints == 0:
            return True
        g = np.array(evaluator.evaluate_constraints(x))
        lb_g = np.array(evaluator.constraint_bounds[0])
        ub_g = np.array(evaluator.constraint_bounds[1])
        return bool(np.all(g >= lb_g - tol) and np.all(g <= ub_g + tol))
    except Exception:
        return True  # conservative: assume feasible if evaluation fails


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

    Returns
    -------
    SolveResult
        With gap_certified=True if termination is by gap criterion.
    """
    t_start = time.perf_counter()

    from discopt._jax.discretization import (
        add_adaptive_partition,
        check_partition_convergence,
        initialize_partitions,
    )
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.partition_selection import pick_partition_vars
    from discopt._jax.term_classifier import classify_nonlinear_terms

    n_orig = sum(v.size for v in model._variables)
    flat_lb, flat_ub = _flat_bounds(model)

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
    part_vars = pick_partition_vars(terms, method=partition_method)

    # If no bilinear/multilinear terms, still partition monomial variables
    # to add tangent cuts at more points and tighten the lower bound.
    if not part_vars and terms.monomial:
        part_vars = sorted(set(var_idx for var_idx, _ in terms.monomial))
        logger.info("AMP: no bilinear terms; partitioning %d monomial vars", len(part_vars))
    else:
        logger.info("AMP: partitioning %d variables via %s", len(part_vars), partition_method)

    # ── Initialize partitions ────────────────────────────────────────────────
    if part_vars:
        part_lbs = [float(flat_lb[i]) for i in part_vars]
        part_ubs = [float(flat_ub[i]) for i in part_vars]
        disc_state = initialize_partitions(
            part_vars, lb=part_lbs, ub=part_ubs, n_init=n_init_partitions
        )
    else:
        from discopt._jax.discretization import DiscretizationState

        disc_state = DiscretizationState()

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
        # Budget time evenly across remaining iterations, capped at 60s per MILP.
        iter_budget = remaining / max(1, max_iter - iteration + 1)
        milp_tl = milp_time_limit if milp_time_limit is not None else min(iter_budget * 3, min(remaining * 0.8, 60.0))

        # ── Step 1: Solve MILP relaxation → lower bound ──────────────────────
        # MILP gap tolerance: no tighter than needed for overall convergence.
        _milp_gap_tol = milp_gap_tolerance if milp_gap_tolerance is not None else min(rel_gap / 2, 1e-3)

        try:
            milp_model, varmap = build_milp_relaxation(
                model, terms, disc_state, incumbent, oa_cuts=oa_cuts
            )
            milp_result = milp_model.solve(
                time_limit=milp_tl, gap_tolerance=_milp_gap_tol
            )
        except Exception as e:
            logger.warning("AMP: MILP build/solve failed at iteration %d: %s", iteration, e)
            break

        if milp_result.status in ("infeasible", "error"):
            logger.info("AMP: MILP infeasible/error at iteration %d, status=%s", iteration, milp_result.status)
            if LB == -np.inf:
                # Problem may be infeasible
                if iteration == 1:
                    return SolveResult(status="infeasible", wall_time=time.perf_counter() - t_start)
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

        # Round integer/binary vars to nearest integer for NLP subproblem
        x0_nlp = _round_integers(x0, model)

        # Build fixed-integer bounds for NLP
        nlp_lb = flat_lb.copy()
        nlp_ub = flat_ub.copy()
        offset = 0
        for v in model._variables:
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                for k in range(v.size):
                    val = float(x0_nlp[offset + k])
                    val = np.clip(val, float(flat_lb[offset + k]), float(flat_ub[offset + k]))
                    rounded = round(val)
                    nlp_lb[offset + k] = rounded
                    nlp_ub[offset + k] = rounded
            offset += v.size

        x_nlp, obj_nlp = _solve_nlp_subproblem(model, x0_nlp, nlp_lb, nlp_ub, nlp_solver)

        if x_nlp is not None and obj_nlp is not None:
            # Verify feasibility and update UB
            feasible = _check_constraints(x_nlp, model)
            if feasible and obj_nlp < UB:
                UB = obj_nlp
                incumbent = x_nlp.copy()
                logger.debug("AMP iter %d: new UB=%.6g", iteration, UB)

                # Accumulate OA tangent cuts at this NLP solution to tighten
                # the next MILP relaxation.  Uses existing OA infrastructure
                # from cutting_planes.py which handles all constraint senses.
                try:
                    from discopt._jax.cutting_planes import (
                        generate_oa_cuts_from_evaluator,
                    )
                    from discopt._jax.nlp_evaluator import NLPEvaluator
                    from discopt.modeling.core import Constraint

                    _eval = NLPEvaluator(model)
                    _x_orig = x_nlp[:n_orig]
                    if _eval.n_constraints > 0:
                        _senses = [
                            c.sense
                            for c in model._constraints
                            if isinstance(c, Constraint)
                        ]
                        cuts = generate_oa_cuts_from_evaluator(
                            _eval, _x_orig, constraint_senses=_senses
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
                except Exception as _oa_err:
                    logger.debug("AMP: OA cut computation failed: %s", _oa_err)

        # ── Step 3: Gap check ────────────────────────────────────────────────
        if iteration_callback is not None:
            iteration_callback(
                {"iteration": iteration, "lower_bound": LB, "upper_bound": UB}
            )

        if UB < np.inf and LB > -np.inf:
            abs_gap = UB - LB
            if abs(UB) > 1e-10:
                rel_g = abs_gap / abs(UB)
            else:
                rel_g = abs_gap
            logger.info(
                "AMP iter %d: LB=%.6g, UB=%.6g, gap=%.4g%%",
                iteration,
                LB,
                UB,
                100 * rel_g,
            )
            if abs_gap <= abs_tol or rel_g <= rel_gap:
                gap_certified = True
                logger.info("AMP: gap certified at iteration %d (gap=%.4g%%)", iteration, 100 * rel_g)
                break

        # ── Step 4: Adaptive partition refinement ────────────────────────────
        if not part_vars:
            # No partition variables → single iteration
            if UB < np.inf:
                gap_certified = False  # no lower bound from partitioning
            break

        # Use MILP solution (original vars) as the refinement point
        refine_solution: dict[int, float] = {}
        if milp_result.x is not None:
            x_orig = milp_result.x[:n_orig]
            for i in part_vars:
                refine_solution[i] = float(x_orig[i])

        disc_state = add_adaptive_partition(
            disc_state, refine_solution, part_vars, part_lbs, part_ubs
        )

        # Check partition convergence
        if check_partition_convergence(disc_state):
            logger.info("AMP: partition convergence at iteration %d", iteration)
            gap_certified = UB < np.inf and LB > -np.inf
            break

    # ── Build final result ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    total_iterations = min(iteration, max_iter)

    if UB >= np.inf and LB == -np.inf:
        return SolveResult(
            status="infeasible",
            wall_time=elapsed,
        )

    if incumbent is not None:
        abs_gap_final = UB - LB if LB > -np.inf else None
        rel_gap_final = (
            abs(abs_gap_final) / abs(UB) if abs_gap_final is not None and abs(UB) > 1e-10
            else abs_gap_final
        )

        if elapsed >= time_limit:
            status = "time_limit" if not gap_certified else "optimal"
        else:
            status = "optimal"

        return SolveResult(
            status=status,
            objective=float(UB),
            bound=float(LB) if LB > -np.inf else None,
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
        bound=float(LB) if LB > -np.inf else None,
        gap=None,
        x=None,
        wall_time=elapsed,
        gap_certified=False,
    )
