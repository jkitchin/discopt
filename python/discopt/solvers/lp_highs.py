"""HiGHS LP solver wrapper with warm-start support."""

from typing import List, Optional, Tuple, Union

import highspy
import numpy as np
import scipy.sparse as sp

from discopt.solvers import LPResult, SolveStatus

# Mapping from HiGHS model status to our SolveStatus enum.
_STATUS_MAP = {
    highspy.HighsModelStatus.kOptimal: SolveStatus.OPTIMAL,
    highspy.HighsModelStatus.kInfeasible: SolveStatus.INFEASIBLE,
    highspy.HighsModelStatus.kUnbounded: SolveStatus.UNBOUNDED,
    highspy.HighsModelStatus.kUnboundedOrInfeasible: SolveStatus.UNBOUNDED,
    highspy.HighsModelStatus.kIterationLimit: SolveStatus.ITERATION_LIMIT,
    highspy.HighsModelStatus.kTimeLimit: SolveStatus.TIME_LIMIT,
}


def _to_csc(
    A: Union[np.ndarray, sp.spmatrix],
) -> sp.csc_matrix:
    """Convert a matrix to scipy CSC format."""
    if sp.issparse(A):
        return sp.csc_matrix(A)
    return sp.csc_matrix(np.asarray(A, dtype=np.float64))


def _build_constraint_matrix(
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, sp.csc_matrix, int]:
    """Stack inequality and equality constraints into HiGHS row format.

    Returns (row_lower, row_upper, csc_matrix, num_rows).
    """
    inf = highspy.kHighsInf
    parts_A = []
    parts_rl: list[np.ndarray] = []
    parts_ru: list[np.ndarray] = []

    if A_ub is not None and b_ub is not None:
        b_ub_arr = np.asarray(b_ub, dtype=np.float64).ravel()
        parts_rl.append(np.full(len(b_ub_arr), -inf))
        parts_ru.append(b_ub_arr)
        parts_A.append(_to_csc(A_ub))

    if A_eq is not None and b_eq is not None:
        b_eq_arr = np.asarray(b_eq, dtype=np.float64).ravel()
        parts_rl.append(b_eq_arr)
        parts_ru.append(b_eq_arr)
        parts_A.append(_to_csc(A_eq))

    if not parts_A:
        empty = sp.csc_matrix((0, n), dtype=np.float64)
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), empty, 0

    if len(parts_A) == 1:
        combined = parts_A[0]
    else:
        combined = sp.vstack(parts_A, format="csc")

    row_lower = np.concatenate(parts_rl)
    row_upper = np.concatenate(parts_ru)
    m = combined.shape[0]
    return row_lower, row_upper, combined, m


def _require_finite(name: str, value: Union[np.ndarray, sp.spmatrix]) -> None:
    if sp.issparse(value):
        arr = np.asarray(value.data, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    warm_basis: Optional[object] = None,
    time_limit: Optional[float] = None,
) -> LPResult:
    """Solve a linear program using HiGHS.

    Minimizes ``c^T x`` subject to ``A_ub @ x <= b_ub`` (if provided),
    ``A_eq @ x == b_eq`` (if provided), and
    ``bounds[i][0] <= x[i] <= bounds[i][1]`` (if provided).

    Args:
        c: Objective coefficients, shape (n,).
        A_ub: Inequality constraint matrix (m_ub, n), dense or sparse. None if no inequalities.
        b_ub: Inequality right-hand side (m_ub,). Required if A_ub is given.
        A_eq: Equality constraint matrix (m_eq, n), dense or sparse. None if no equalities.
        b_eq: Equality right-hand side (m_eq,). Required if A_eq is given.
        bounds: Per-variable (lower, upper) bounds. Defaults to (0, +inf) when None.
        warm_basis: ``HighsBasis`` from a previous ``LPResult.basis`` for warm-starting.
        time_limit: Wall-clock time limit in seconds, or None for no limit.

    Returns:
        LPResult containing status, primal solution, objective value, dual
        values, basis (for warm-starting the next solve), iteration count, and
        wall-clock time.

    Raises:
        ValueError: If matrix dimensions are inconsistent.
    """
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    _require_finite("c", c_arr)

    # ---- validate dimensions ------------------------------------------------
    if A_ub is not None:
        shape = A_ub.shape if sp.issparse(A_ub) else np.asarray(A_ub).shape
        if len(shape) != 2 or shape[1] != n:
            raise ValueError(f"A_ub has {shape[1]} columns but c has {n} elements")
        if b_ub is None:
            raise ValueError("b_ub is required when A_ub is provided")
        b_ub_arr = np.asarray(b_ub).ravel()
        if b_ub_arr.shape[0] != shape[0]:
            raise ValueError(f"A_ub has {shape[0]} rows but b_ub has {b_ub_arr.shape[0]} elements")
        _require_finite("A_ub", A_ub)
        _require_finite("b_ub", b_ub_arr)
    if A_eq is not None:
        shape = A_eq.shape if sp.issparse(A_eq) else np.asarray(A_eq).shape
        if len(shape) != 2 or shape[1] != n:
            raise ValueError(f"A_eq has {shape[1]} columns but c has {n} elements")
        if b_eq is None:
            raise ValueError("b_eq is required when A_eq is provided")
        b_eq_arr = np.asarray(b_eq).ravel()
        if b_eq_arr.shape[0] != shape[0]:
            raise ValueError(f"A_eq has {shape[0]} rows but b_eq has {b_eq_arr.shape[0]} elements")
        _require_finite("A_eq", A_eq)
        _require_finite("b_eq", b_eq_arr)
    if bounds is not None and len(bounds) != n:
        raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")

    # ---- variable bounds -----------------------------------------------------
    inf = highspy.kHighsInf
    if bounds is not None:
        col_lower = np.array([lb for lb, _ in bounds], dtype=np.float64)
        col_upper = np.array([ub for _, ub in bounds], dtype=np.float64)
    else:
        col_lower = np.zeros(n, dtype=np.float64)
        col_upper = np.full(n, inf, dtype=np.float64)
    if np.any(np.isnan(col_lower)) or np.any(np.isnan(col_upper)):
        raise ValueError("bounds contain NaN values")

    # Very large finite bounds (~1e19) fall just below HiGHS's internal
    # infinity threshold (~1e20) and can cause numerical issues. Translate
    # anything beyond the discopt "very large" threshold to HiGHS infinity
    # so unbounded variables are handled via the same code path as truly
    # unbounded ones.
    _FINITE_BOUND_THRESHOLD = 1e15
    col_lower = np.where(col_lower <= -_FINITE_BOUND_THRESHOLD, -inf, col_lower)
    col_upper = np.where(col_upper >= _FINITE_BOUND_THRESHOLD, inf, col_upper)

    # ---- build constraint matrix ---------------------------------------------
    row_lower, row_upper, csc, m = _build_constraint_matrix(A_ub, b_ub, A_eq, b_eq, n)

    # ---- non-finite guard ----------------------------------------------------
    # HiGHS has no defined behaviour for NaN in the model: depending on platform
    # it either crashes (SIGBUS) or spins forever in the simplex (NaN comparisons
    # never satisfy the termination test, and the time limit check is bypassed),
    # which manifests as a multi-hour CI hang rather than a failure. A NaN here is
    # always an upstream modelling/linearisation bug, so fail loudly and
    # immediately instead of handing undefined data to the native solver. Inf is
    # legitimate in column bounds (free / one-sided variables) and row bounds
    # (one-sided constraints), so only NaN is rejected for those; objective and
    # matrix coefficients must be fully finite.
    def _reject_nonfinite(name: str, arr: np.ndarray, *, allow_inf: bool) -> None:
        if arr.size == 0:
            return
        bad = ~np.isfinite(arr)
        if allow_inf:
            bad &= ~np.isinf(arr)  # keep only NaN
        if bad.any():
            k = int(bad.sum())
            idx = np.argmax(bad)
            kind = "non-finite" if not allow_inf else "NaN"
            raise ValueError(
                f"LP {name} contains {k} {kind} value(s) (first at index {idx}); "
                "refusing to pass to HiGHS. This indicates an upstream modelling "
                "or linearisation bug producing NaN/Inf coefficients."
            )

    _reject_nonfinite("objective (c)", c_arr, allow_inf=False)
    _reject_nonfinite("column lower bounds", col_lower, allow_inf=True)
    _reject_nonfinite("column upper bounds", col_upper, allow_inf=True)
    _reject_nonfinite("row lower bounds", np.asarray(row_lower), allow_inf=True)
    _reject_nonfinite("row upper bounds", np.asarray(row_upper), allow_inf=True)
    _reject_nonfinite("constraint matrix", np.asarray(csc.data), allow_inf=False)

    # ---- build HighsLp object ------------------------------------------------
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

    # ---- create HiGHS solver -------------------------------------------------
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))

    # Turn off presolve when warm-starting so the basis is used directly.
    if warm_basis is not None:
        h.setOptionValue("presolve", "off")

    h.passModel(lp)

    # ---- warm-start ----------------------------------------------------------
    if warm_basis is not None:
        h.setBasis(warm_basis)

    # ---- solve ---------------------------------------------------------------
    h.run()

    model_status = h.getModelStatus()
    status = _STATUS_MAP.get(model_status, SolveStatus.ERROR)

    _, iters = h.getInfoValue("simplex_iteration_count")
    _, wall_time = h.getInfoValue("run_time")

    if status == SolveStatus.OPTIMAL:
        sol = h.getSolution()
        x = np.array(sol.col_value, dtype=np.float64)
        _, obj = h.getInfoValue("objective_function_value")
        dual = np.array(sol.row_dual, dtype=np.float64)
        col_dual_raw = np.array(getattr(sol, "col_dual", []), dtype=np.float64)
        col_dual = col_dual_raw if col_dual_raw.size else None
        basis = h.getBasis()
        return LPResult(
            status=status,
            x=x,
            objective=obj,
            dual_values=dual,
            reduced_costs=col_dual,
            basis=basis,
            iterations=int(iters),
            wall_time=float(wall_time),
        )

    return LPResult(
        status=status,
        iterations=int(iters),
        wall_time=float(wall_time),
    )
