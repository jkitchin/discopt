"""POUNCE MILP solver: matrix-form MILP via the self-hosted branch & bound.

POUNCE has no *matrix* MILP solver — "POUNCE MILP" is the discopt Rust-tree
B&B with POUNCE LP relaxations (Phase 1). This adapter exposes that B&B behind
the same signature/``MILPResult`` contract as
:func:`discopt.solvers.milp_highs.solve_milp` by building a ``Model`` from the
matrix data, running ``_solve_milp_bb(prefer_pounce=True)``, and mapping the
result back. It lets the matrix-form MILP consumers (the OA / GDP-LOA masters
and ``milp_relaxation``) run with **only POUNCE installed** (no HiGHS).
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus

# SolveResult status string -> (MILPResult SolveStatus, keep_incumbent_x).
# "feasible"/"node_limit" carry a real incumbent that consumers (OA/GDP) use
# via the ITERATION_LIMIT+x path, but are *not* certified optimal.
_STATUS_MAP: dict[str, tuple[SolveStatus, bool]] = {
    "optimal": (SolveStatus.OPTIMAL, True),
    "feasible": (SolveStatus.ITERATION_LIMIT, True),
    "node_limit": (SolveStatus.ITERATION_LIMIT, True),
    "iteration_limit": (SolveStatus.ITERATION_LIMIT, True),
    "time_limit": (SolveStatus.TIME_LIMIT, True),
    "infeasible": (SolveStatus.INFEASIBLE, False),
    "unbounded": (SolveStatus.UNBOUNDED, False),
    "error": (SolveStatus.ERROR, False),
}

_INF = 1e20


def _linear_expr(coeffs: np.ndarray, xs: list):
    """Build ``sum_j coeffs[j] * xs[j]`` over non-zero coefficients."""
    expr = None
    for j, a in enumerate(coeffs):
        if a != 0.0:
            term = float(a) * xs[j]
            expr = term if expr is None else expr + term
    if expr is None:
        # All-zero row/objective: a valid zero expression.
        return 0.0 * xs[0]
    return expr


def _add_rows(model, A, b, xs, sense: str) -> None:
    if A is None or b is None:
        return
    A = (
        np.asarray(A.todense(), dtype=np.float64)
        if sp.issparse(A)
        else np.asarray(A, dtype=np.float64)
    )
    b = np.asarray(b, dtype=np.float64).ravel()
    for i in range(A.shape[0]):
        lhs = _linear_expr(A[i], xs)
        rhs = float(b[i])
        if sense == "<=":
            model.subject_to(lhs <= rhs)
        else:
            model.subject_to(lhs == rhs)


def solve_milp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
) -> MILPResult:
    """Solve ``min c^T x`` over the MILP via the self-hosted B&B (POUNCE).

    Same signature/semantics as :func:`discopt.solvers.milp_highs.solve_milp`.
    ``bounds`` default to ``(0, +inf)`` per variable when ``None``;
    ``integrality[j] == 1`` marks variable ``j`` integer (all continuous when
    ``None``).
    """
    import discopt.modeling as dm
    from discopt.solver import _solve_milp_bb

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    if bounds is not None:
        lbs = [float(lo) for lo, _ in bounds]
        ubs = [float(hi) for _, hi in bounds]
    else:
        lbs = [0.0] * n
        ubs = [float("inf")] * n
    is_int = (
        np.zeros(n, dtype=bool) if integrality is None else (np.asarray(integrality).ravel() == 1)
    )

    m = dm.Model("milp_pounce")
    xs = []
    for j in range(n):
        lo = lbs[j] if np.isfinite(lbs[j]) else -_INF
        hi = ubs[j] if np.isfinite(ubs[j]) else _INF
        if is_int[j]:
            xs.append(m.integer(f"x{j}", lb=lo, ub=hi))
        else:
            xs.append(m.continuous(f"x{j}", lb=lo, ub=hi))

    m.minimize(_linear_expr(c_arr, xs))
    _add_rows(m, A_ub, b_ub, xs, "<=")
    _add_rows(m, A_eq, b_eq, xs, "==")

    t0 = time.perf_counter()
    res = _solve_milp_bb(
        m,
        time_limit if time_limit is not None else 1e9,
        gap_tolerance,
        16,  # batch_size
        "best_first",  # strategy
        100_000,  # max_nodes
        t0,
        prefer_pounce=True,
    )
    wall_time = time.perf_counter() - t0

    status, keep_x = _STATUS_MAP.get(res.status, (SolveStatus.ERROR, False))
    x = None
    if keep_x and res.x is not None:
        x = np.array(
            [float(np.asarray(res.x[f"x{j}"]).ravel()[0]) for j in range(n)], dtype=np.float64
        )

    return MILPResult(
        status=status,
        x=x,
        objective=res.objective if x is not None else None,
        gap=res.gap,
        node_count=int(res.node_count),
        wall_time=wall_time,
    )
