"""MPS/LP -> discopt Model loader for the LP/MILP routing panel.

Stage 0 of ``docs/dev/lp-milp-highs-routing-plan.md``. HiGHS reads the instance
(``readModel`` -> ``getLp``); the arrays are written as a *text* AMPL ``.nl``
(integer columns last, as the format requires) and loaded with
``discopt.modeling.core.from_nl`` -- the same parser user ``.nl`` files take.

Everything here refuses loudly (CLAUDE.md §3, §7): an unsupported model feature
raises instead of being dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LinearInstance:
    name: str
    c: np.ndarray  # original column order
    offset: float
    maximize: bool
    A: sp.csr_matrix  # rows x cols, original order
    row_lo: np.ndarray
    row_hi: np.ndarray
    col_lo: np.ndarray
    col_hi: np.ndarray
    is_int: np.ndarray  # bool, original order

    @property
    def n(self) -> int:
        return int(self.c.shape[0])

    @property
    def m(self) -> int:
        return int(self.A.shape[0])


def read_instance(path: str) -> LinearInstance:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    st = h.readModel(path)
    if st == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS could not read {path}")
    if h.getModel().hessian_.dim_ > 0:
        raise ValueError(f"{path}: quadratic objective is out of scope for the LP/MILP panel")
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    if a.format_ != highspy.MatrixFormat.kColwise:
        raise ValueError(f"{path}: expected a column-wise matrix, got {a.format_}")
    A = sp.csc_matrix(  # noqa: N806
        (np.asarray(a.value_, float), np.asarray(a.index_), np.asarray(a.start_)), shape=(m, n)
    ).tocsr()
    integ = list(lp.integrality_) if len(lp.integrality_) else []
    is_int = np.zeros(n, dtype=bool)
    for j, t in enumerate(integ):
        if t == highspy.HighsVarType.kInteger:
            is_int[j] = True
        elif t != highspy.HighsVarType.kContinuous:
            raise ValueError(f"{path}: column {j} has unsupported type {t}")
    return LinearInstance(
        name=os.path.splitext(os.path.basename(path))[0],
        c=np.asarray(lp.col_cost_, float),
        offset=float(lp.offset_),
        maximize=lp.sense_ == highspy.ObjSense.kMaximize,
        A=A,
        row_lo=np.asarray(lp.row_lower_, float),
        row_hi=np.asarray(lp.row_upper_, float),
        col_lo=np.asarray(lp.col_lower_, float),
        col_hi=np.asarray(lp.col_upper_, float),
        is_int=is_int,
    )


def _num(v: float) -> str:
    if not np.isfinite(v):
        raise ValueError(f"non-finite value {v} reached the .nl writer")
    return repr(float(v))


def _bound_line(lo: float, hi: float) -> str:
    flo, fhi = np.isfinite(lo), np.isfinite(hi)
    if flo and fhi:
        if lo > hi:
            raise ValueError(f"crossed bounds [{lo}, {hi}]")
        return f"4 {_num(lo)}" if lo == hi else f"0 {_num(lo)} {_num(hi)}"
    if fhi:
        return f"1 {_num(hi)}"
    if flo:
        return f"2 {_num(lo)}"
    return "3"


def nl_column_order(inst: LinearInstance) -> np.ndarray:
    """``perm[k]`` = original column placed at ``.nl`` position ``k`` (integers last)."""
    return np.concatenate([np.flatnonzero(~inst.is_int), np.flatnonzero(inst.is_int)])


def write_nl(inst: LinearInstance, path: str) -> np.ndarray:
    """Write ``inst`` as a text ``.nl``; return the column permutation used."""
    perm = nl_column_order(inst)
    n, niv = inst.n, int(inst.is_int.sum())
    A = inst.A.tocsc()[:, perm].tocsr()  # noqa: N806
    A.eliminate_zeros()
    row_nnz = np.diff(A.indptr)
    keep = np.flatnonzero(row_nnz > 0)
    for i in np.flatnonzero(row_nnz == 0):
        # An empty row is a pure feasibility statement about 0; refuse rather than drop
        # one that is violated.
        if inst.row_lo[i] > 0 or inst.row_hi[i] < 0:
            raise ValueError(f"{inst.name}: empty row {i} is infeasible")
    A = A[keep]  # noqa: N806
    lo, hi = inst.row_lo[keep], inst.row_hi[keep]
    m = A.shape[0]
    fin = np.isfinite(lo) & np.isfinite(hi)
    n_eq = int(np.sum(fin & (lo == hi)))
    n_rng = int(np.sum(fin & (lo < hi)))
    c = inst.c[perm]
    nzo = int(np.count_nonzero(c))
    col_counts = np.diff(A.tocsc().indptr)

    out = []
    out.append("g3 1 1 0\n")
    out.append(f" {n} {m} 1 {n_rng} {n_eq}\n")
    out.append(" 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n")
    out.append(f" 0 {niv} 0 0 0\n")
    out.append(f" {A.nnz} {nzo}\n 0 0\n 0 0 0 0 0\n")
    for i in range(m):
        out.append(f"C{i}\nn0\n")
    out.append(f"O0 {1 if inst.maximize else 0}\nn{_num(inst.offset)}\n")
    if m:
        out.append("r\n")
        out.extend(_bound_line(lo[i], hi[i]) + "\n" for i in range(m))
    out.append("b\n")
    out.extend(_bound_line(inst.col_lo[j], inst.col_hi[j]) + "\n" for j in perm)
    out.append(f"k{n - 1}\n")
    out.extend(f"{int(v)}\n" for v in np.cumsum(col_counts)[:-1])
    for i in range(m):
        s, e = A.indptr[i], A.indptr[i + 1]
        out.append(f"J{i} {e - s}\n")
        out.extend(f"{A.indices[k]} {_num(A.data[k])}\n" for k in range(s, e))
    if nzo:
        out.append(f"G0 {nzo}\n")
        out.extend(f"{j} {_num(c[j])}\n" for j in np.flatnonzero(c))
    with open(path, "w") as f:
        f.write("".join(out))
    return perm


def dense_max_violation(inst: LinearInstance, x: np.ndarray) -> float:
    """Max row/bound/integrality violation of ``x`` (original order), checked densely.

    Row activities use a dense per-row dot product, not a scipy sparse matvec, whose
    FMA contraction fabricates residuals on sentinel-valued entries (#1229).
    """
    x = np.asarray(x, float).ravel()
    if x.shape[0] != inst.n:
        raise ValueError(f"x has {x.shape[0]} entries, instance has {inst.n}")
    if not np.all(np.isfinite(x)):
        return float("inf")
    viol = 0.0
    viol = max(viol, float(np.max(np.maximum(inst.col_lo - x, 0.0), initial=0.0)))
    viol = max(viol, float(np.max(np.maximum(x - inst.col_hi, 0.0), initial=0.0)))
    for i in range(inst.m):
        s, e = inst.A.indptr[i], inst.A.indptr[i + 1]
        act = float(np.dot(inst.A.data[s:e], x[inst.A.indices[s:e]]))
        viol = max(viol, inst.row_lo[i] - act, act - inst.row_hi[i])
    if inst.is_int.any():
        xi = x[inst.is_int]
        viol = max(viol, float(np.max(np.abs(xi - np.round(xi)))))
    return viol


def objective_value(inst: LinearInstance, x: np.ndarray) -> float:
    return float(np.dot(inst.c, x)) + inst.offset


def highs_reference(path: str, time_limit: float = 60.0) -> dict:
    """Raw HiGHS on the original file: the oracle for loader validation."""
    import highspy

    h = highspy.Highs()
    # threads stays at the default, as in H0 and the route. HiGHS's scheduler is
    # process-global: the first run sizes it, and a later run with a different nonzero
    # threads value fails with kError.
    opts = (("output_flag", False), ("time_limit", float(time_limit)))
    for k, v in opts:
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {k}")
    if h.readModel(path) == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS could not read {path}")
    st = h.run()
    if st == highspy.HighsStatus.kError:
        raise RuntimeError(f"{path}: HiGHS run returned {st}")
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    obj = float(info.objective_function_value)
    x = np.asarray(h.getSolution().col_value, float)
    return {
        "status": status,
        "objective": obj,
        "x": x,
        "nodes": int(info.mip_node_count),
    }
