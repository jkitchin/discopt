"""Gurobi LP/MILP/QP/MIQP solver wrappers.

This module mirrors the matrix-form HiGHS wrappers: callers pass extracted
standard-form arrays and receive discopt result dataclasses. ``gurobipy`` stays
optional and is imported only inside the solver path.

Quadratic objectives use the discopt/HiGHS convention
``0.5 * x.T @ Q @ x + c.T @ x``. Nonconvex ``Q`` matrices are rejected by
default; pass an explicit Gurobi ``NonConvex`` parameter through ``options`` to
delegate nonconvex QP/MIQP handling to Gurobi.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import LPResult, MILPResult, QPResult, SolveStatus

_FINITE_BOUND_THRESHOLD = 1e15


def _load_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError(
            "gurobipy is required for solver='gurobi'. Install gurobipy and "
            "configure a working Gurobi license."
        ) from exc
    return gp, GRB


def _to_csr(A: Union[np.ndarray, sp.spmatrix]) -> sp.csr_matrix:
    if sp.issparse(A):
        return sp.csr_matrix(A, dtype=np.float64)
    return sp.csr_matrix(np.asarray(A, dtype=np.float64))


def _validate_linear_data(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    bounds: Optional[list[tuple[float, float]]],
) -> tuple[np.ndarray, int]:
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)

    if A_ub is not None:
        shape = A_ub.shape if sp.issparse(A_ub) else np.asarray(A_ub).shape
        if len(shape) != 2 or shape[1] != n:
            raise ValueError(f"A_ub has {shape[1]} columns but c has {n} elements")
        if b_ub is None:
            raise ValueError("b_ub is required when A_ub is provided")
        if np.asarray(b_ub).ravel().shape[0] != shape[0]:
            raise ValueError(f"A_ub has {shape[0]} rows but b_ub has wrong length")

    if A_eq is not None:
        shape = A_eq.shape if sp.issparse(A_eq) else np.asarray(A_eq).shape
        if len(shape) != 2 or shape[1] != n:
            raise ValueError(f"A_eq has {shape[1]} columns but c has {n} elements")
        if b_eq is None:
            raise ValueError("b_eq is required when A_eq is provided")
        if np.asarray(b_eq).ravel().shape[0] != shape[0]:
            raise ValueError(f"A_eq has {shape[0]} rows but b_eq has wrong length")

    if bounds is not None and len(bounds) != n:
        raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")

    return c_arr, n


def _validate_qp_data(
    Q: Union[np.ndarray, sp.spmatrix],
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    bounds: Optional[list[tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    c_arr, n = _validate_linear_data(c, A_ub, b_ub, A_eq, b_eq, bounds)
    if sp.issparse(Q):
        Q_arr = sp.csr_matrix(Q, dtype=np.float64).toarray()
    else:
        Q_arr = np.asarray(Q, dtype=np.float64)
    if Q_arr.shape != (n, n):
        raise ValueError(f"Q has shape {Q_arr.shape} but c has {n} elements")
    if not np.all(np.isfinite(Q_arr)) or not np.all(np.isfinite(c_arr)):
        raise ValueError("Q and c must contain only finite values")
    if not np.allclose(Q_arr, Q_arr.T, rtol=1e-10, atol=1e-12):
        Q_arr = 0.5 * (Q_arr + Q_arr.T)
    return Q_arr, c_arr, n


def _has_nonconvex_objective(Q: np.ndarray) -> bool:
    if Q.size == 0:
        return False
    try:
        min_eig = float(np.min(np.linalg.eigvalsh(Q)))
    except np.linalg.LinAlgError:
        return True
    scale = max(1.0, float(np.max(np.abs(Q))))
    return min_eig < -1e-9 * scale


def _has_explicit_nonconvex_option(options: Optional[dict]) -> bool:
    return bool(options) and any(str(key).lower() == "nonconvex" for key in options)


def _normalise_bounds(
    bounds: Optional[list[tuple[float, float]]],
    n: int,
    inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, inf, dtype=np.float64)
    else:
        lb = np.array([lo for lo, _ in bounds], dtype=np.float64)
        ub = np.array([hi for _, hi in bounds], dtype=np.float64)

    lb = np.where(lb <= -_FINITE_BOUND_THRESHOLD, -inf, lb)
    ub = np.where(ub >= _FINITE_BOUND_THRESHOLD, inf, ub)
    return lb, ub


def _status_map(GRB) -> dict[int, SolveStatus]:
    mapping = {
        GRB.OPTIMAL: SolveStatus.OPTIMAL,
        GRB.INFEASIBLE: SolveStatus.INFEASIBLE,
        GRB.UNBOUNDED: SolveStatus.UNBOUNDED,
        GRB.ITERATION_LIMIT: SolveStatus.ITERATION_LIMIT,
        GRB.TIME_LIMIT: SolveStatus.TIME_LIMIT,
    }
    for name in ("NODE_LIMIT", "SOLUTION_LIMIT"):
        code = getattr(GRB, name, None)
        if code is not None:
            mapping[code] = SolveStatus.ITERATION_LIMIT
    return mapping


def _optimize(model, GRB) -> int:
    model.optimize()
    status = int(model.Status)
    if status == GRB.INF_OR_UNBD:
        model.setParam("DualReductions", 0)
        model.optimize()
        status = int(model.Status)
    return status


def _set_common_params(model, time_limit, threads, options) -> None:
    model.setParam("OutputFlag", 0)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    if threads is not None:
        model.setParam("Threads", int(threads))
    if options:
        for key, value in options.items():
            model.setParam(str(key), value)


def _safe_attr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _build_model(
    gp,
    GRB,
    *,
    name: str,
    c: np.ndarray,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    bounds,
    integrality,
    time_limit,
    gap_tolerance,
    threads,
    options,
):
    env = gp.Env(empty=True)
    try:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(name, env=env)
    except Exception:
        env.dispose()
        raise

    _set_common_params(model, time_limit, threads, options)
    if gap_tolerance is not None:
        model.setParam("MIPGap", float(gap_tolerance))

    lb, ub = _normalise_bounds(bounds, len(c), GRB.INFINITY)

    vtype = GRB.CONTINUOUS
    if integrality is not None:
        int_arr = np.asarray(integrality, dtype=np.int32).ravel()
        if int_arr.shape[0] != len(c):
            model.dispose()
            env.dispose()
            raise ValueError(f"integrality has {int_arr.shape[0]} entries but c has {len(c)}")
        vtype = []
        for is_int, lo, hi in zip(int_arr == 1, lb, ub):
            if not is_int:
                vtype.append(GRB.CONTINUOUS)
            elif lo == 0 and hi == 1:
                vtype.append(GRB.BINARY)
            else:
                vtype.append(GRB.INTEGER)

    x = model.addMVar(shape=len(c), lb=lb, ub=ub, vtype=vtype, name="x")
    model.setObjective(c @ x, GRB.MINIMIZE)

    ub_con = None
    eq_con = None
    if A_ub is not None and b_ub is not None:
        A_ub_csr = _to_csr(A_ub)
        b_ub_arr = np.asarray(b_ub, dtype=np.float64).ravel()
        if b_ub_arr.size:
            ub_con = model.addConstr(A_ub_csr @ x <= b_ub_arr, name="ub")
    if A_eq is not None and b_eq is not None:
        A_eq_csr = _to_csr(A_eq)
        b_eq_arr = np.asarray(b_eq, dtype=np.float64).ravel()
        if b_eq_arr.size:
            eq_con = model.addConstr(A_eq_csr @ x == b_eq_arr, name="eq")

    return env, model, x, ub_con, eq_con


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    warm_basis: Optional[object] = None,
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
    options: Optional[dict] = None,
) -> LPResult:
    """Solve a linear program using Gurobi."""
    if warm_basis is not None:
        raise NotImplementedError("Gurobi LP warm-start basis support is not implemented yet.")

    c_arr, _n = _validate_linear_data(c, A_ub, b_ub, A_eq, b_eq, bounds)
    gp, GRB = _load_gurobi()

    env, model, x, ub_con, eq_con = _build_model(
        gp,
        GRB,
        name="discopt_lp",
        c=c_arr,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=None,
        time_limit=time_limit,
        gap_tolerance=None,
        threads=threads,
        options=options,
    )
    try:
        status_code = _optimize(model, GRB)
        status = _status_map(GRB).get(status_code, SolveStatus.ERROR)
        iters = int(_safe_attr(model, "IterCount") or 0)
        wall_time = float(_safe_attr(model, "Runtime") or 0.0)

        if status == SolveStatus.OPTIMAL:
            row_duals = []
            if ub_con is not None:
                pi = _safe_attr(ub_con, "Pi")
                if pi is not None:
                    row_duals.extend(np.asarray(pi, dtype=np.float64).ravel().tolist())
            if eq_con is not None:
                pi = _safe_attr(eq_con, "Pi")
                if pi is not None:
                    row_duals.extend(np.asarray(pi, dtype=np.float64).ravel().tolist())
            rc = _safe_attr(x, "RC")
            reduced_costs = np.asarray(rc, dtype=np.float64).ravel() if rc is not None else None
            return LPResult(
                status=status,
                x=np.asarray(x.X, dtype=np.float64).ravel(),
                objective=float(model.ObjVal),
                dual_values=np.asarray(row_duals, dtype=np.float64) if row_duals else None,
                reduced_costs=reduced_costs,
                iterations=iters,
                wall_time=wall_time,
            )

        return LPResult(status=status, iterations=iters, wall_time=wall_time)
    finally:
        model.dispose()
        env.dispose()


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
    threads: Optional[int] = None,
    options: Optional[dict] = None,
) -> MILPResult:
    """Solve a mixed-integer linear program using Gurobi."""
    c_arr, _n = _validate_linear_data(c, A_ub, b_ub, A_eq, b_eq, bounds)
    gp, GRB = _load_gurobi()

    env, model, x, _ub_con, _eq_con = _build_model(
        gp,
        GRB,
        name="discopt_milp",
        c=c_arr,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=integrality,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        threads=threads,
        options=options,
    )
    try:
        status_code = _optimize(model, GRB)
        status = _status_map(GRB).get(status_code, SolveStatus.ERROR)
        node_count = int(_safe_attr(model, "NodeCount") or 0)
        wall_time = float(_safe_attr(model, "Runtime") or 0.0)

        objective = None
        x_val = None
        if int(_safe_attr(model, "SolCount") or 0) > 0:
            objective = float(model.ObjVal)
            x_val = np.asarray(x.X, dtype=np.float64).ravel()

        bound = _safe_attr(model, "ObjBound")
        bound = float(bound) if bound is not None and np.isfinite(bound) else None
        gap = _safe_attr(model, "MIPGap")
        gap = float(gap) if gap is not None and np.isfinite(gap) else None

        if status == SolveStatus.OPTIMAL:
            assert objective is not None and x_val is not None
            return MILPResult(
                status=status,
                x=x_val,
                objective=objective,
                bound=objective if bound is None else bound,
                gap=0.0 if gap is None else gap,
                node_count=node_count,
                wall_time=wall_time,
            )

        return MILPResult(
            status=status,
            x=x_val,
            objective=objective,
            bound=bound,
            gap=gap,
            node_count=node_count,
            wall_time=wall_time,
        )
    finally:
        model.dispose()
        env.dispose()


def solve_qp(
    Q: Union[np.ndarray, sp.spmatrix],
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    threads: Optional[int] = None,
    options: Optional[dict] = None,
) -> QPResult:
    """Solve a continuous or mixed-integer quadratic program using Gurobi.

    The objective is ``0.5 * x.T @ Q @ x + c.T @ x``. Convex QP/MIQP models
    solve without extra options. Nonconvex objectives require an explicit
    ``NonConvex`` Gurobi parameter in ``options`` so users opt into Gurobi's
    nonconvex quadratic machinery intentionally.
    """
    Q_arr, c_arr, n = _validate_qp_data(Q, c, A_ub, b_ub, A_eq, b_eq, bounds)
    if _has_nonconvex_objective(Q_arr) and not _has_explicit_nonconvex_option(options):
        raise ValueError(
            "Gurobi QP/MIQP nonconvex objective detected. Pass "
            "options={'NonConvex': 2} to solver='gurobi' if Gurobi should solve it."
        )

    gp, GRB = _load_gurobi()

    has_integer = integrality is not None and np.any(np.asarray(integrality, dtype=np.int32) == 1)
    qp_bounds = bounds
    if qp_bounds is None:
        qp_bounds = [(-GRB.INFINITY, GRB.INFINITY)] * n

    env, model, x, ub_con, eq_con = _build_model(
        gp,
        GRB,
        name="discopt_miqp" if has_integer else "discopt_qp",
        c=c_arr,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=qp_bounds,
        integrality=integrality,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance if has_integer else None,
        threads=threads,
        options=options,
    )
    try:
        model.setObjective(0.5 * (x @ Q_arr @ x) + c_arr @ x, GRB.MINIMIZE)
        status_code = _optimize(model, GRB)
        status = _status_map(GRB).get(status_code, SolveStatus.ERROR)
        iters = int(_safe_attr(model, "IterCount") or 0)
        node_count = int(_safe_attr(model, "NodeCount") or 0)
        wall_time = float(_safe_attr(model, "Runtime") or 0.0)

        objective = None
        x_val = None
        if int(_safe_attr(model, "SolCount") or 0) > 0:
            objective = float(model.ObjVal)
            x_val = np.asarray(x.X, dtype=np.float64).ravel()

        bound = _safe_attr(model, "ObjBound")
        bound = float(bound) if bound is not None and np.isfinite(bound) else None
        gap = _safe_attr(model, "MIPGap")
        gap = float(gap) if gap is not None and np.isfinite(gap) else None

        if status == SolveStatus.OPTIMAL:
            assert objective is not None and x_val is not None
            row_duals = None
            reduced_costs = None
            if not has_integer:
                dual_values = []
                if ub_con is not None:
                    pi = _safe_attr(ub_con, "Pi")
                    if pi is not None:
                        dual_values.extend(np.asarray(pi, dtype=np.float64).ravel().tolist())
                if eq_con is not None:
                    pi = _safe_attr(eq_con, "Pi")
                    if pi is not None:
                        dual_values.extend(np.asarray(pi, dtype=np.float64).ravel().tolist())
                row_duals = np.asarray(dual_values, dtype=np.float64) if dual_values else None
                rc = _safe_attr(x, "RC")
                reduced_costs = np.asarray(rc, dtype=np.float64).ravel() if rc is not None else None

            return QPResult(
                status=status,
                x=x_val,
                objective=objective,
                bound=objective if bound is None else bound,
                gap=0.0 if gap is None else gap,
                dual_values=row_duals,
                reduced_costs=reduced_costs,
                node_count=node_count,
                iterations=iters,
                wall_time=wall_time,
            )

        return QPResult(
            status=status,
            x=x_val,
            objective=objective,
            bound=bound,
            gap=gap,
            node_count=node_count,
            iterations=iters,
            wall_time=wall_time,
        )
    finally:
        model.dispose()
        env.dispose()
