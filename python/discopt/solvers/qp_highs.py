"""HiGHS QP/MIQP solver wrapper.

Solves quadratic programs of the form::

    min  0.5 * x^T Q x + c^T x
    s.t. A_ub @ x <= b_ub
         A_eq @ x == b_eq
         lb <= x <= ub
         x[i] integer where integrality[i] == 1

Uses the HiGHS QP solver via ``passHessian()`` for the quadratic objective.
"""

from __future__ import annotations

from typing import Optional, Union

import highspy
import numpy as np
import scipy.sparse as sp

from discopt.solvers import QPResult, SolveStatus
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


def _extract_lower_triangle(Q: np.ndarray):
    """Extract lower-triangular entries from a symmetric Q matrix.

    HiGHS expects the Hessian in lower-triangular CSC sparse format
    (row >= col). Returns (row_indices, col_indices, values) for
    non-zero entries where row >= col.
    """
    n = Q.shape[0]
    rows = []
    cols = []
    vals = []
    for j in range(n):
        for i in range(j, n):
            val = Q[i, j]
            if abs(val) > 1e-15:
                rows.append(i)
                cols.append(j)
                vals.append(val)
    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float64),
    )


def solve_qp(
    Q: np.ndarray,
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
) -> QPResult:
    """Solve a (mixed-integer) quadratic program using HiGHS.

    Minimizes  0.5 * x^T Q x + c^T x  subject to:
        A_ub @ x <= b_ub   (if provided)
        A_eq @ x == b_eq   (if provided)
        bounds[i][0] <= x[i] <= bounds[i][1]  (if provided)
        x[i] integer where integrality[i] == 1

    Parameters
    ----------
    Q : np.ndarray
        Symmetric Hessian matrix of shape ``(n, n)``. The objective is
        ``0.5 * x^T Q x + c^T x``.
    c : np.ndarray
        Linear objective coefficients, shape ``(n,)``.
    A_ub, b_ub : optional
        Inequality constraints ``A_ub @ x <= b_ub``.
    A_eq, b_eq : optional
        Equality constraints ``A_eq @ x == b_eq``.
    bounds : optional
        Per-variable ``(lower, upper)`` bounds. Defaults to
        ``(-inf, +inf)`` for each variable.
    integrality : np.ndarray, optional
        Array of length ``n``: 0 for continuous, 1 for integer.
        If None, all variables are continuous (pure QP).
    time_limit : float, optional
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative MIP gap tolerance (default 1e-4).

    Returns
    -------
    QPResult
    """
    Q_arr = np.asarray(Q, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)

    if Q_arr.shape != (n, n):
        raise ValueError(f"Q has shape {Q_arr.shape} but c has {n} elements")

    # ---- variable bounds -----------------------------------------------------
    inf = highspy.kHighsInf
    if bounds is not None:
        col_lower = np.array([lb for lb, _ in bounds], dtype=np.float64)
        col_upper = np.array([ub for _, ub in bounds], dtype=np.float64)
    else:
        col_lower = np.full(n, -inf, dtype=np.float64)
        col_upper = np.full(n, inf, dtype=np.float64)

    # Bounds with magnitude ~1e19 fall just below HiGHS's internal infinity
    # threshold (~1e20) but are large enough to cause barrier ill-conditioning
    # and false-optimal results on convex QPs. Translate anything beyond the
    # discopt "very large" threshold to HiGHS infinity so unbounded variables
    # are treated as truly unbounded.
    _FINITE_BOUND_THRESHOLD = 1e15
    col_lower = np.where(col_lower <= -_FINITE_BOUND_THRESHOLD, -inf, col_lower)
    col_upper = np.where(col_upper >= _FINITE_BOUND_THRESHOLD, inf, col_upper)

    # ---- build constraint matrix ---------------------------------------------
    row_lower, row_upper, csc, m = _build_constraint_matrix(A_ub, b_ub, A_eq, b_eq, n)

    # ---- build HiGHS LP model (linear part) ----------------------------------
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

    if integrality is not None:
        h.setOptionValue("mip_rel_gap", float(gap_tolerance))

    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))

    h.passModel(lp)

    # ---- pass Hessian (quadratic objective) ----------------------------------
    q_rows, q_cols, q_vals = _extract_lower_triangle(Q_arr)
    if len(q_vals) > 0:
        # Build column-compressed upper-triangular Hessian.
        # HiGHS expects start array of length dim+1 with column pointers.
        start = np.zeros(n + 1, dtype=np.int32)
        for col in q_cols:
            start[col + 1] += 1
        for i in range(1, n + 1):
            start[i] += start[i - 1]

        # Sort entries by column, then row within each column
        order = np.lexsort((q_rows, q_cols))
        index = q_rows[order].astype(np.int32)
        value = q_vals[order].astype(np.float64)

        hessian = highspy.HighsHessian()
        hessian.dim_ = n
        hessian.format_ = highspy.HessianFormat.kTriangular
        hessian.start_ = start
        hessian.index_ = index
        hessian.value_ = value
        h.passHessian(hessian)

    # ---- solve ---------------------------------------------------------------
    h.run()

    model_status = h.getModelStatus()
    status = _STATUS_MAP.get(model_status, SolveStatus.ERROR)

    _, wall_time = h.getInfoValue("run_time")

    if status == SolveStatus.OPTIMAL:
        sol = h.getSolution()
        x = np.array(sol.col_value, dtype=np.float64)
        _, obj = h.getInfoValue("objective_function_value")

        node_count = 0
        if integrality is not None:
            _, node_count_f = h.getInfoValue("mip_node_count")
            node_count = int(node_count_f)

        return QPResult(
            status=status,
            x=x,
            objective=obj,
            node_count=node_count,
            wall_time=float(wall_time),
        )

    return QPResult(
        status=status,
        wall_time=float(wall_time),
    )
