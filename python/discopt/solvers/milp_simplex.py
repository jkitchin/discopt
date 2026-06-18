"""Warm-started-simplex MILP backend (Rust ``solve_milp_py``).

A ``solve_milp(c, A_ub, b_ub, ..., integrality, ...)`` adapter, signature- and
``MILPResult``-compatible with :mod:`discopt.solvers.milp_highs` and
:mod:`discopt.solvers.milp_pounce`, so it can be selected through
:func:`discopt.solvers.lp_backend.get_milp_solver`. It marshals the ``A_ub x <= b_ub``
form into the engine's standard form ``A_eq z = b`` (one explicit slack per row) and runs
the pure-Rust warm-started-simplex branch-and-bound.

Soundness: ``MILPResult.objective`` is the incumbent (an upper bound on a non-optimal
exit) and ``MILPResult.bound`` is the engine's **dual lower bound** — equal to the
incumbent once the solve is proven optimal, and a valid lower bound otherwise. Callers
that need a lower bound (AMP/OA/GDP-LOA) must read ``bound``, never ``objective``. If the
Rust binding is unavailable it raises :class:`SimplexBackendUnavailable` so the selector
can fall back.
"""

from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus


class SimplexBackendUnavailable(RuntimeError):
    """Raised when the Rust ``solve_milp_py`` binding cannot be imported."""


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
    max_nodes: int = 1_000_000,
) -> MILPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq, bounds, integrality``
    with the Rust warm-started-simplex B&B.

    Mirrors :func:`discopt.solvers.milp_highs.solve_milp`. The returned
    ``objective`` is the engine's dual lower bound (see module docstring).
    """
    try:
        from discopt._rust import solve_milp_py
    except ImportError as err:  # pragma: no cover - exercised via the selector
        raise SimplexBackendUnavailable(
            "discopt._rust.solve_milp_py is unavailable; build the Rust extension"
        ) from err

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    # Assemble all rows as `<=` (A_eq becomes a pair of `<=` rows) then slack.
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        a = (
            cast("sp.spmatrix", A_ub).toarray()
            if sp.issparse(A_ub)
            else np.asarray(A_ub, dtype=np.float64)
        )
        rows.append(a.reshape(-1, n))
        rhs.extend(np.asarray(b_ub, dtype=np.float64).ravel().tolist())
    if A_eq is not None and b_eq is not None and np.size(b_eq) > 0:
        a = (
            cast("sp.spmatrix", A_eq).toarray()
            if sp.issparse(A_eq)
            else np.asarray(A_eq, dtype=np.float64)
        )
        a = a.reshape(-1, n)
        be = np.asarray(b_eq, dtype=np.float64).ravel()
        rows.append(a)
        rhs.extend(be.tolist())
        rows.append(-a)
        rhs.extend((-be).tolist())

    if rows:
        a_ub = np.vstack(rows)
    else:
        a_ub = np.zeros((0, n), dtype=np.float64)
    b_vec = np.asarray(rhs, dtype=np.float64)
    m = a_ub.shape[0]

    # Standard form A_eq z = b with one slack per row: [A_ub | I] z = b_ub.
    a_std = np.zeros((m, n + m), dtype=np.float64)
    if m > 0:
        a_std[:, :n] = a_ub
        a_std[:, n:] = np.eye(m)

    if bounds is not None:
        lb = np.array([lo for lo, _ in bounds], dtype=np.float64)
        ub = np.array([hi for _, hi in bounds], dtype=np.float64)
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, 1e20, dtype=np.float64)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, 1e20)])
    c_std = np.concatenate([c_arr, np.zeros(m)])

    if integrality is not None:
        int_mask = np.asarray(integrality, dtype=np.int64).ravel()
        int_cols = np.flatnonzero(int_mask != 0).astype(np.int64)
    else:
        int_cols = np.zeros(0, dtype=np.int64)

    status, x_full, obj, bound, nodes, _iters = solve_milp_py(
        np.ascontiguousarray(c_std),
        np.ascontiguousarray(a_std),
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        np.ascontiguousarray(int_cols),
        n,  # n_struct: structural columns precede the slacks
        0.0,  # obj_const: caller (MilpRelaxationModel) applies its own offset
        int(max_nodes),
        float(gap_tolerance),
        time_limit_s=0.0 if time_limit is None else max(0.0, float(time_limit)),
    )

    if status == "infeasible":
        return MILPResult(status=SolveStatus.INFEASIBLE, node_count=int(nodes))
    if status == "unbounded":
        return MILPResult(status=SolveStatus.UNBOUNDED, node_count=int(nodes))

    x_struct = np.asarray(x_full, dtype=np.float64)[:n]
    if status == "optimal":
        # Proven optimum: incumbent == dual bound, a tight valid lower bound.
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=x_struct,
            objective=float(obj),
            bound=float(obj),
            node_count=int(nodes),
        )

    # node_limit / feasible: ``objective`` is the incumbent (upper bound) and
    # ``bound`` is the engine's dual lower bound (sound) if finite. Callers that
    # need a lower bound must read ``bound``, never ``objective``.
    return MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=x_struct,
        objective=float(obj) if np.isfinite(obj) else None,
        bound=float(bound) if np.isfinite(bound) else None,
        node_count=int(nodes),
    )


def solve_lp_warm_std(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    bounds: Optional[list[tuple[float, float]]],
    in_basis: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[Optional[MILPResult], Optional[tuple[np.ndarray, np.ndarray]]]:
    """Warm-startable **pure-LP** solve of ``min c^T x s.t. A_ub x <= b_ub, bounds``.

    Marshals the ``A_ub x <= b_ub`` form into standard form ``[A_ub | I] z = b_ub``
    (one explicit slack per row, structural columns first) and calls the Rust
    ``solve_lp_warm_py``. ``in_basis`` is a ``(col_status, basic_vars)`` pair from a
    previous solve of a *prefix* of the same column set (rows since appended); Rust
    extends it by making the appended slacks basic and dual-simplex re-optimizes.

    Returns ``(result, out_basis)``. ``out_basis`` is the final ``(col_status,
    basic_vars)`` to thread into the next re-solve (``None`` when the LP is not
    optimal). ``result`` is ``None`` for ``iter_limit``/``numerical`` exits so the
    caller can fall back to a cold/HiGHS path. Soundness: the dual simplex
    converges to the LP optimum exactly as a cold solve (a bad basis is ignored
    inside Rust), so the returned objective/bound is unchanged — only the speed is.
    """
    from discopt._rust import solve_lp_warm_py

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        a = (
            cast("sp.spmatrix", A_ub).toarray()
            if sp.issparse(A_ub)
            else np.asarray(A_ub, dtype=np.float64)
        )
        a_ub = a.reshape(-1, n)
        b_vec = np.asarray(b_ub, dtype=np.float64).ravel()
    else:
        a_ub = np.zeros((0, n), dtype=np.float64)
        b_vec = np.zeros(0, dtype=np.float64)
    m = a_ub.shape[0]

    a_std = np.zeros((m, n + m), dtype=np.float64)
    if m > 0:
        a_std[:, :n] = a_ub
        a_std[:, n:] = np.eye(m)

    if bounds is not None:
        lb = np.array([lo for lo, _ in bounds], dtype=np.float64)
        ub = np.array([hi for _, hi in bounds], dtype=np.float64)
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, 1e20, dtype=np.float64)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, 1e20)])
    c_std = np.concatenate([c_arr, np.zeros(m)])

    cs0 = None if in_basis is None else np.ascontiguousarray(in_basis[0], dtype=np.int8)
    bv0 = None if in_basis is None else np.ascontiguousarray(in_basis[1], dtype=np.int64)

    status, x_full, obj, _iters, cs, bv = solve_lp_warm_py(
        np.ascontiguousarray(c_std),
        np.ascontiguousarray(a_std),
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        cs0,
        bv0,
    )

    x_struct = np.asarray(x_full, dtype=np.float64)[:n]
    if status == "optimal":
        return (
            MILPResult(
                status=SolveStatus.OPTIMAL,
                x=x_struct,
                objective=float(obj),
                bound=float(obj),
                node_count=0,
            ),
            (np.asarray(cs), np.asarray(bv)),
        )
    if status == "infeasible":
        return MILPResult(status=SolveStatus.INFEASIBLE, node_count=0), None
    if status == "unbounded":
        return MILPResult(status=SolveStatus.UNBOUNDED, node_count=0), None
    # iter_limit / numerical: signal the caller to fall back to the generic path.
    return None, None
