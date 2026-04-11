"""
NLP solver wrapper using ripopt (Rust interior-point method).

Maps NLPEvaluator callbacks to the ripopt Rust solver via PyO3 bindings.
Provides the same solve_nlp interface as nlp_ipopt.py.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from discopt.solvers import NLPResult, SolveStatus

# Map ripopt status strings to SolveStatus enum
_RIPOPT_STATUS_MAP: dict[str, SolveStatus] = {
    "optimal": SolveStatus.OPTIMAL,
    "infeasible": SolveStatus.INFEASIBLE,
    "local_infeasibility": SolveStatus.INFEASIBLE,
    "max_iterations": SolveStatus.ITERATION_LIMIT,
    "numerical_error": SolveStatus.ERROR,
    "unbounded": SolveStatus.UNBOUNDED,
    "restoration_failed": SolveStatus.ERROR,
    "internal_error": SolveStatus.ERROR,
}


def solve_nlp(
    evaluator,
    x0: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]] = None,
    options: Optional[dict] = None,
) -> NLPResult:
    """
    Solve an NLP using ripopt (Rust interior-point method).

    Args:
        evaluator: NLPEvaluator (or compatible) providing evaluation callbacks.
        x0: Initial point (n,).
        constraint_bounds: List of (cl, cu) per constraint. None to infer from model.
        options: Solver options dict (max_iter, tol, print_level, etc.).

    Returns:
        NLPResult with solution.
    """
    from discopt._rust import solve_ripopt

    opts = dict(options) if options else {}
    opts.setdefault("print_level", 0)

    m = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds

    # Convert large-magnitude bounds to ±inf. CUTEst and other sources use
    # finite sentinels (e.g. ±1e20) for "no bound". Ipopt handles this via
    # nlp_lower_bound_inf/nlp_upper_bound_inf; we replicate that here so
    # ripopt's barrier method doesn't create terms for non-existent bounds.
    bound_inf = 1e19
    lb = np.where(lb <= -bound_inf, -np.inf, lb).astype(np.float64)
    ub = np.where(ub >= bound_inf, np.inf, ub).astype(np.float64)

    # Constraint bounds
    if constraint_bounds is not None:
        g_l = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        g_u = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0 and hasattr(evaluator, "_model"):
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        g_l, g_u = _infer_constraint_bounds(evaluator._model)
    else:
        g_l = np.empty(0, dtype=np.float64)
        g_u = np.empty(0, dtype=np.float64)

    # Also convert constraint bound sentinels
    if len(g_l) > 0:
        g_l = np.where(g_l <= -bound_inf, -np.inf, g_l).astype(np.float64)
        g_u = np.where(g_u >= bound_inf, np.inf, g_u).astype(np.float64)

    t0 = time.perf_counter()
    result = solve_ripopt(
        evaluator,
        x0.astype(np.float64),
        lb,
        ub,
        g_l,
        g_u,
        opts,
    )
    wall_time = time.perf_counter() - t0

    status_str = result["status"]
    status = _RIPOPT_STATUS_MAP.get(status_str, SolveStatus.ERROR)

    multipliers = result.get("constraint_multipliers", None)
    if multipliers is not None and len(multipliers) == 0:
        multipliers = None

    return NLPResult(
        status=status,
        x=np.asarray(result["x"]),
        objective=float(result["objective"]),
        multipliers=np.asarray(multipliers) if multipliers is not None else None,
        iterations=int(result.get("iterations", 0)),
        wall_time=wall_time,
    )
