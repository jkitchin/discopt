"""Matrix-form IIS engine for linear models, with integrality as an element.

:func:`discopt.infeasibility.compute_iis` filters ``Constraint`` objects and
whole-variable bounds by re-solving the model, which cannot express the element
that dominates the size of a MIP's infeasible subsystem: the **integrality of a
single column**. (A discopt ``Variable`` carries one ``var_type`` for all its
elements.) For a linear model this module works on the extracted matrix instead,
with one persistent HiGHS model whose row bounds, column bounds and per-column
integrality are toggled in place for each probe.

Elements (Guieu & Chinneck 1999; the MIPcc27 competition uses the same set):

* a **constraint** (all rows of one ``model._constraints`` entry);
* the **integrality** of one integer or binary column;
* a finite **bound** of a column that is *not* held integer. The bounds of a
  column whose integrality is kept always apply and are not elements -- for a
  binary they *are* its domain. Once a column's integrality is relaxed its
  bounds become ordinary elements (a relaxed binary keeps ``[0, 1]`` only if
  those bounds are themselves kept).

Algorithm: deletion filtering (Chinneck & Dravnieks 1991) over the order
integrality -> constraints -> bounds. Before filtering integralities one at a
time, the LP relaxation is tested (Guieu & Chinneck's first step): if it is
already infeasible, every integrality is dropped in a single probe.

Soundness contract (same as ``compute_iis``): an element is dropped only on a
**proven** infeasible status; a probe that hits its time limit keeps the element
and clears ``proven_irreducible``. "Proven" means HiGHS returned ``Infeasible``
AND a presolve-free re-solve agreed -- the #1634 cross-check discopt's own HiGHS
MILP route requires before an infeasibility claim may stand.

Certificates come for free. A kept element ``e`` was kept because the active set
minus ``e`` was feasible, and the probe returned a point ``x``. Elements are only
ever removed afterwards, so ``x`` satisfies the *final* subsystem minus ``e`` as
well. Each such point is re-checked arithmetically against the final subsystem
(every other kept element within ``cert_tol``, ``e`` violated by more than
``cert_tol``) before it is reported; a point that fails the check is dropped,
never "repaired".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model

_INF = 1e20  # discopt's unbounded-bound sentinel


@dataclass
class MilpIIS:
    """Result of :func:`milp_iis`, in flat-column / constraint-index terms."""

    constraints: list[int]  # indices into model._constraints
    bounds: list[tuple[int, str]]  # (flat column, "lower"|"upper")
    integrality: list[int]  # flat columns
    certificates: dict[tuple, np.ndarray] = field(default_factory=dict)
    n_solves: int = 0
    proven_irreducible: bool = True
    n_cross_checks: int = 0  # infeasibility claims re-solved with presolve off (#1634)
    n_refuted: int = 0  # ... that the presolve-free solve found feasible


class _Engine:
    """One HiGHS model; elements are toggled by changing bounds/integrality."""

    def __init__(self, A, row_lo, row_hi, col_lb, col_ub, is_int, row_groups, probe_tl):
        import highspy

        self.hs = highspy
        self.A = A.tocsr()
        self.row_lo, self.row_hi = row_lo, row_hi
        self.col_lb, self.col_ub = col_lb, col_ub
        self.is_int = is_int
        self.row_groups = row_groups  # constraint index -> np.ndarray of row ids
        self.probe_tl = probe_tl
        self.n_solves = 0
        self.n_cross_checks = 0  # infeasibility claims re-solved with presolve off
        self.n_refuted = 0  # ... of which the presolve-free solve found feasible
        n, m = col_lb.size, row_lo.size
        inf = highspy.kHighsInf
        self._inf = inf
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("output_flag", False)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n, m
        lp.col_cost_ = np.zeros(n)
        lp.col_lower_ = np.where(col_lb <= -_INF, -inf, col_lb)
        lp.col_upper_ = np.where(col_ub >= _INF, inf, col_ub)
        lp.row_lower_ = np.where(row_lo <= -_INF, -inf, row_lo)
        lp.row_upper_ = np.where(row_hi >= _INF, inf, row_hi)
        csc = A.tocsc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = csc.data.astype(np.float64)
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous for b in is_int
        ]
        h.passModel(lp)
        self.h = h

    def probe(
        self,
        active_cons: set[int],
        active_bounds: set[tuple[int, str]],
        active_int: set[int],
    ) -> tuple[Optional[bool], Optional[np.ndarray]]:
        """``(True, None)`` proven infeasible; ``(False, x)`` feasible at ``x``;
        ``(None, None)`` inconclusive (time limit or another non-proof status)."""
        h, inf = self.h, self._inf
        n, m = self.col_lb.size, self.row_lo.size
        rlo = np.full(m, -inf)
        rhi = np.full(m, inf)
        for ci in active_cons:
            rows = self.row_groups[ci]
            rlo[rows] = np.where(self.row_lo[rows] <= -_INF, -inf, self.row_lo[rows])
            rhi[rows] = np.where(self.row_hi[rows] >= _INF, inf, self.row_hi[rows])
        lb = np.full(n, -inf)
        ub = np.full(n, inf)
        integ = np.zeros(n, dtype=bool)
        for j in active_int:
            integ[j] = True
        # A column held integer keeps its declared bounds unconditionally.
        held = np.flatnonzero(integ)
        lb[held] = self.col_lb[held]
        ub[held] = self.col_ub[held]
        for j, side in active_bounds:
            if integ[j]:
                continue
            if side == "lower":
                lb[j] = self.col_lb[j]
            else:
                ub[j] = self.col_ub[j]
        lb = np.where(lb <= -_INF, -inf, lb)
        ub = np.where(ub >= _INF, inf, ub)
        idx_m = np.arange(m, dtype=np.int32)
        idx_n = np.arange(n, dtype=np.int32)
        h.clearSolver()
        if m:
            h.changeRowsBounds(m, idx_m, rlo, rhi)
        if n:
            h.changeColsBounds(n, idx_n, lb, ub)
            vt = self.hs.HighsVarType
            h.changeColsIntegrality(
                n, idx_n, np.array([vt.kInteger if b else vt.kContinuous for b in integ])
            )
        h.setOptionValue("time_limit", float(self.probe_tl))
        h.setOptionValue("presolve", "choose")
        self.n_solves += 1
        h.run()
        ms = h.getModelStatus()
        MS = self.hs.HighsModelStatus
        if ms == MS.kOptimal:
            return False, np.array(h.getSolution().col_value, dtype=np.float64)
        if ms not in (MS.kInfeasible, MS.kUnboundedOrInfeasible):
            return None, None
        # An infeasibility claim is what lets the filter DROP an element, so it
        # stands only when a presolve-free solve agrees -- the #1634 policy of
        # discopt's own HiGHS MILP route, where presolve-on and presolve-off each
        # produced false claims on a 1200-instance panel but never on the same
        # instance. (``kUnboundedOrInfeasible`` cannot be "unbounded" under a zero
        # objective; the presolve-free solve decides it the same way.)
        self.n_cross_checks += 1
        h.setOptionValue("presolve", "off")
        h.clearSolver()
        self.n_solves += 1
        h.run()
        ms2 = h.getModelStatus()
        h.setOptionValue("presolve", "choose")
        if ms2 == MS.kInfeasible:
            return True, None
        if ms2 == MS.kOptimal:
            if ms == MS.kInfeasible:
                self.n_refuted += 1
            return False, np.array(h.getSolution().col_value, dtype=np.float64)
        return None, None


def _violation(x, eng: _Engine, rows=None, cols_lb=None, cols_ub=None, cols_int=None):
    """Max violation of the given rows / bounds / integralities at ``x``."""
    v = 0.0
    if rows is not None and len(rows):
        act = eng.A[rows] @ x
        lo, hi = eng.row_lo[rows], eng.row_hi[rows]
        v = max(v, float(np.max(np.where(lo > -_INF, lo - act, 0.0), initial=0.0)))
        v = max(v, float(np.max(np.where(hi < _INF, act - hi, 0.0), initial=0.0)))
    if cols_lb is not None and len(cols_lb):
        v = max(v, float(np.max(eng.col_lb[cols_lb] - x[cols_lb], initial=0.0)))
    if cols_ub is not None and len(cols_ub):
        v = max(v, float(np.max(x[cols_ub] - eng.col_ub[cols_ub], initial=0.0)))
    if cols_int is not None and len(cols_int):
        xi = x[cols_int]
        v = max(v, float(np.max(np.abs(xi - np.round(xi)), initial=0.0)))
    return v


def _check_certificate(x, key, eng, cons, bounds, ints, tol) -> bool:
    """``x`` satisfies the final subsystem minus ``key`` and violates ``key``."""
    kind = key[0]
    cons_o = [c for c in cons if not (kind == "constraint" and c == key[1])]
    ints_o = [j for j in ints if not (kind == "integrality" and j == key[1])]
    held = set(ints_o)
    # Bounds of a column held integer apply; for the integrality element itself
    # its own bounds still apply too (relaxing integrality keeps the box).
    lb_cols = sorted(held | {j for j, s in bounds if s == "lower"})
    ub_cols = sorted(held | {j for j, s in bounds if s == "upper"})
    if kind == "integrality":
        lb_cols = sorted(set(lb_cols) | {key[1]})
        ub_cols = sorted(set(ub_cols) | {key[1]})
    if kind == "bound":
        j, s = key[1], key[2]
        if s == "lower":
            lb_cols = [c for c in lb_cols if c != j]
        else:
            ub_cols = [c for c in ub_cols if c != j]
    rows = np.concatenate([eng.row_groups[c] for c in cons_o]) if cons_o else np.zeros(0, int)
    lb_cols = [c for c in lb_cols if eng.col_lb[c] > -_INF]
    ub_cols = [c for c in ub_cols if eng.col_ub[c] < _INF]
    if _violation(x, eng, rows, lb_cols, ub_cols, ints_o) > tol:
        return False
    # ``key`` itself must be violated by more than tol.
    if kind == "constraint":
        e = _violation(x, eng, rows=eng.row_groups[key[1]])
    elif kind == "integrality":
        e = _violation(x, eng, cols_int=[key[1]])
    else:
        j, s = key[1], key[2]
        e = _violation(
            x, eng, cols_lb=[j] if s == "lower" else None, cols_ub=[j] if s == "upper" else None
        )
    return bool(e > tol)


def milp_iis(
    model: "Model",
    *,
    include_bounds: bool = True,
    time_limit: float = 30.0,
    cert_tol: float = 1e-6,
) -> MilpIIS:
    """IIS of a linear (LP/MILP) ``model`` with integrality as an element.

    Raises ``NotImplementedError`` (from the linear extractor) for a model with a
    nonlinear or non-algebraic constraint, and ``ValueError`` when the full model
    is not proven infeasible.
    """
    import scipy.sparse as sp

    from discopt.decomposition._linear import extract_linear
    from discopt.modeling.core import VarType

    lin = extract_linear(model)
    A = sp.csr_matrix(lin.A)
    b = np.asarray(lin.b, dtype=np.float64)
    m = lin.m
    row_lo = np.full(m, -_INF)
    row_hi = np.full(m, _INF)
    for r, s in enumerate(lin.sense):
        if s in ("<=", "=="):
            row_hi[r] = b[r]
        if s in (">=", "=="):
            row_lo[r] = b[r]
    rows_of: dict[int, list[int]] = {}
    for r, src in enumerate(lin.row_source):
        rows_of.setdefault(src, []).append(r)
    row_groups = {k: np.asarray(v, dtype=np.int64) for k, v in rows_of.items()}

    lbs, ubs, ints = [], [], []
    for v in model._variables:
        lbs.append(np.broadcast_to(np.asarray(v.lb, dtype=np.float64), v.shape).reshape(-1))
        ubs.append(np.broadcast_to(np.asarray(v.ub, dtype=np.float64), v.shape).reshape(-1))
        ints.append(np.full(v.size, v.var_type in (VarType.INTEGER, VarType.BINARY)))
    col_lb = np.concatenate(lbs) if lbs else np.zeros(0)
    col_ub = np.concatenate(ubs) if ubs else np.zeros(0)
    is_int = np.concatenate(ints) if ints else np.zeros(0, dtype=bool)
    if col_lb.size != lin.n:
        raise AssertionError(f"column count mismatch: {col_lb.size} vs extractor {lin.n}")

    eng = _Engine(A, row_lo, row_hi, col_lb, col_ub, is_int, row_groups, time_limit)
    cons = set(row_groups)
    ints_active = set(int(j) for j in np.flatnonzero(is_int))
    # Finite bounds, all applied at the start. They are elements (filtered in
    # step 3) only for columns not held integer -- the engine applies a held
    # column's box regardless -- and only when include_bounds is set.
    finite_bounds: list[tuple[int, str]] = []
    for j in range(col_lb.size):
        if col_lb[j] > -_INF:
            finite_bounds.append((j, "lower"))
        if col_ub[j] < _INF:
            finite_bounds.append((j, "upper"))
    bounds = set(finite_bounds)
    proven = True
    pending: dict[tuple, np.ndarray] = {}

    base, _ = eng.probe(cons, bounds, ints_active)
    if base is not True:
        if base is False:
            raise ValueError(
                "compute_iis: the model is feasible — there is no infeasibility to explain."
            )
        raise ValueError(
            "compute_iis: could not prove the model infeasible within the time limit; "
            "an IIS is only defined for an infeasible model."
        )

    def filter_one(key, drop, restore):
        nonlocal proven
        drop()
        res, x = eng.probe(cons, bounds, ints_active)
        if res is True:
            return
        restore()
        if res is None:
            proven = False
        else:
            pending[key] = x

    # 1. Integrality. Guieu & Chinneck: if the LP relaxation is infeasible, all
    #    integrality goes in one probe.
    if ints_active:
        res, _ = eng.probe(cons, bounds, set())
        if res is True:
            ints_active.clear()
        else:
            for j in sorted(ints_active):
                filter_one(
                    ("integrality", j),
                    lambda j=j: ints_active.discard(j),
                    lambda j=j: ints_active.add(j),
                )
    # 2. Constraints.
    for ci in sorted(cons):
        filter_one(("constraint", ci), lambda ci=ci: cons.discard(ci), lambda ci=ci: cons.add(ci))
    # 3. Bounds of columns not held integer.
    if include_bounds:
        for j, side in finite_bounds:
            if j in ints_active:
                continue  # held integer: its box is not an element
            item = (j, side)
            filter_one(
                ("bound", j, side),
                lambda item=item: bounds.discard(item),
                lambda item=item: bounds.add(item),
            )

    # Bounds the final subsystem APPLIES (beyond held-integer boxes). With
    # include_bounds=False that is every finite bound, but none is an element.
    applied_bounds = sorted((j, s) for j, s in bounds if j not in ints_active)
    kept_bounds = applied_bounds if include_bounds else []
    certs = {}
    for key, x in pending.items():
        if _check_certificate(
            x, key, eng, sorted(cons), applied_bounds, sorted(ints_active), cert_tol
        ):
            certs[key] = x
    return MilpIIS(
        constraints=sorted(cons),
        bounds=kept_bounds,
        integrality=sorted(ints_active),
        certificates=certs,
        n_solves=eng.n_solves,
        proven_irreducible=proven,
        n_cross_checks=eng.n_cross_checks,
        n_refuted=eng.n_refuted,
    )
