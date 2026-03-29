"""HiGHS MILP solver wrapper."""

from __future__ import annotations

from typing import Optional, Union

import highspy
import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus
from discopt.solvers.lp_highs import _build_constraint_matrix

# Mapping from HiGHS model status to our SolveStatus enum.
_STATUS_MAP = {
    highspy.HighsModelStatus.kOptimal: SolveStatus.OPTIMAL,
    highspy.HighsModelStatus.kInfeasible: SolveStatus.INFEASIBLE,
    highspy.HighsModelStatus.kUnbounded: SolveStatus.UNBOUNDED,
    highspy.HighsModelStatus.kUnboundedOrInfeasible: SolveStatus.UNBOUNDED,
    highspy.HighsModelStatus.kIterationLimit: SolveStatus.ITERATION_LIMIT,
    highspy.HighsModelStatus.kTimeLimit: SolveStatus.TIME_LIMIT,
}


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
    """Solve a mixed-integer linear program using HiGHS.

    Minimizes  c^T x  subject to:
        A_ub @ x <= b_ub   (if provided)
        A_eq @ x == b_eq   (if provided)
        bounds[i][0] <= x[i] <= bounds[i][1]  (if provided)
        x[i] integer where integrality[i] == 1

    Parameters
    ----------
    c : np.ndarray
        Objective coefficients, shape ``(n,)``.
    A_ub, b_ub : optional
        Inequality constraints.
    A_eq, b_eq : optional
        Equality constraints.
    bounds : optional
        Per-variable ``(lower, upper)`` bounds.
    integrality : np.ndarray, optional
        Array of length ``n``: 0 for continuous, 1 for integer.
        If None, all variables are continuous (degenerates to LP).
    time_limit : float, optional
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative MIP gap tolerance (default 1e-4).

    Returns
    -------
    MILPResult
    """
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)

    # ---- variable bounds -----------------------------------------------------
    inf = highspy.kHighsInf
    if bounds is not None:
        col_lower = np.array([lb for lb, _ in bounds], dtype=np.float64)
        col_upper = np.array([ub for _, ub in bounds], dtype=np.float64)
    else:
        col_lower = np.zeros(n, dtype=np.float64)
        col_upper = np.full(n, inf, dtype=np.float64)

    # ---- build constraint matrix ---------------------------------------------
    row_lower, row_upper, csc, m = _build_constraint_matrix(A_ub, b_ub, A_eq, b_eq, n)

    # ---- build HiGHS model ---------------------------------------------------
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.offset_ = 0.0
    lp.col_cost_ = c_arr
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper

    if m > 0:
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = n
        lp.a_matrix_.num_row_ = m
        lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = csc.data.astype(np.float64)

    # ---- integrality ---------------------------------------------------------
    if integrality is not None:
        int_arr = np.asarray(integrality, dtype=np.int32)
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
            for v in (int_arr == 1).tolist()
        ]

    # ---- create solver -------------------------------------------------------
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", float(gap_tolerance))

    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))

    h.passModel(lp)
    h.run()

    model_status = h.getModelStatus()
    status = _STATUS_MAP.get(model_status, SolveStatus.ERROR)

    _, wall_time = h.getInfoValue("run_time")
    _, node_count = h.getInfoValue("mip_node_count")

    if status == SolveStatus.OPTIMAL:
        sol = h.getSolution()
        x = np.array(sol.col_value, dtype=np.float64)
        _, obj = h.getInfoValue("objective_function_value")
        _, gap = h.getInfoValue("mip_gap")
        return MILPResult(
            status=status,
            x=x,
            objective=obj,
            gap=gap,
            node_count=int(node_count),
            wall_time=float(wall_time),
        )

    return MILPResult(
        status=status,
        node_count=int(node_count),
        wall_time=float(wall_time),
    )
