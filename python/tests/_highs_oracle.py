"""An independent HiGHS feasibility oracle for tests (#2332).

One HiGHS verdict is not an oracle. HiGHS 1.11.0-1.15.1 return ``kInfeasible`` with
presolve off on a feasible 5-column MILP that presolve on solves (``ISSUE_2332``
below; fixed in 1.16.0), and #1634 measured false labels from each configuration
on its own. A test helper that asks "is this subsystem infeasible?" with a single
presolve-off solve would pass a wrong IIS on such a model.

:func:`milp_verdict` therefore answers:

* ``"feasible"`` when either configuration returns a point that, with the integer
  columns rounded, satisfies every row and bound on independent re-evaluation;
* ``"infeasible"`` only when presolve on AND presolve off both say ``kInfeasible``;
* ``None`` otherwise (a disagreement with no verified point, a time limit, ...).

Callers assert on the string, so ``None`` fails an ``== "infeasible"`` check
loudly instead of passing as one.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp

#: discopt's unbounded-bound sentinel; anything at or past it is infinite here.
_INF = 1e20

#: The #2332 reproducer: ``row_lo <= A x <= row_hi``, ``lb <= x <= ub``, x0 and x3
#: integer. ``witness`` satisfies it exactly; HiGHS 1.11.0-1.15.1 answer
#: ``kInfeasible`` with presolve off.
ISSUE_2332 = {
    "A": np.array(
        [
            [0.0, 0.0, 2.0, -3.0, 3.0],
            [2.0, -2.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0, -1.5],
            [0.0, 0.0, 0.0, 0.0, -1.0],
        ]
    ),
    "row_lo": np.array([-3.0, -_INF, -0.5, -2.0]),
    "row_hi": np.array([-3.0, 1.0, -0.5, _INF]),
    "lb": np.array([1.0, -2.0, -2.0, 0.0, -2.0]),
    "ub": np.array([2.5, _INF, 1.0, 3.0, _INF]),
    "ints": (0, 3),
    "witness": np.array([2.0, 1.5, -1.5, 0.0, 0.0]),
}


def _int_mask(ints, n: int) -> np.ndarray:
    a = np.asarray(ints if ints is not None else [])
    if a.dtype == bool:
        if a.size != n:
            raise ValueError(f"integrality mask has {a.size} entries for {n} columns")
        return a.copy()
    mask = np.zeros(n, dtype=bool)
    mask[a.astype(np.int64)] = True
    return mask


def satisfies(A, row_lo, row_hi, lb, ub, ints, x, tol: float = 1e-7) -> bool:
    """Does ``x`` meet every row and bound within ``tol``, with exact integers where
    ``ints`` holds? (Infinite sides, or sides at the 1e20 sentinel, are absent.)"""
    A = sp.csr_matrix(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    held = _int_mask(ints, x.size)
    if not np.all(x[held] == np.round(x[held])):
        return False
    act = A @ x
    lo, hi = np.asarray(row_lo, float), np.asarray(row_hi, float)
    lbv, ubv = np.asarray(lb, float), np.asarray(ub, float)
    return bool(
        np.all((lo <= -_INF) | (act >= lo - tol))
        and np.all((hi >= _INF) | (act <= hi + tol))
        and np.all((lbv <= -_INF) | (x >= lbv - tol))
        and np.all((ubv >= _INF) | (x <= ubv + tol))
    )


def milp_verdict(
    A,
    row_lo,
    row_hi,
    lb,
    ub,
    ints: Optional[Iterable[int]] = None,
    *,
    time_limit: float = 10.0,
    tol: float = 1e-7,
) -> Optional[str]:
    """``"feasible"``, ``"infeasible"`` or ``None`` for ``row_lo <= A x <= row_hi``,
    ``lb <= x <= ub``, integer where ``ints`` (indices or a boolean mask) holds.
    Infinite values and the 1e20 sentinel both mean "no bound"."""
    import highspy

    A = sp.csr_matrix(A, dtype=np.float64)
    m, n = A.shape
    held = _int_mask(ints, n)
    inf = highspy.kHighsInf

    def side(v, lower: bool) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return np.where(v <= -_INF, -inf, v) if lower else np.where(v >= _INF, inf, v)

    rlo, rhi = side(row_lo, True), side(row_hi, False)
    clb, cub = side(lb, True), side(ub, False)
    idx = np.flatnonzero(held).astype(np.int32)
    statuses = []
    for presolve in ("on", "off"):
        h = highspy.Highs()
        h.silent()
        for key, val in (("presolve", presolve), ("time_limit", float(time_limit))):
            if h.setOptionValue(key, val) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS rejected option {key}={val!r}")
        h.addVars(n, clb, cub)
        for r in range(m):
            s, e = A.indptr[r], A.indptr[r + 1]
            h.addRow(rlo[r], rhi[r], e - s, A.indices[s:e].astype(np.int32), A.data[s:e])
        if idx.size:
            h.changeColsIntegrality(
                idx.size, idx, np.array([highspy.HighsVarType.kInteger] * idx.size)
            )
        h.run()
        st = h.getModelStatus()
        statuses.append(st)
        if st == highspy.HighsModelStatus.kOptimal:
            x = np.array(h.getSolution().col_value, dtype=np.float64)
            x[held] = np.round(x[held])
            if satisfies(A, row_lo, row_hi, lb, ub, held, x, tol):
                return "feasible"
    if all(st == highspy.HighsModelStatus.kInfeasible for st in statuses):
        return "infeasible"
    return None
