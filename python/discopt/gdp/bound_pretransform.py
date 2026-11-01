"""``gdp.bound_pretransformation``: disjunct-local variable bounds as global rows (#1775 G2.4).

Pyomo's ``gdp.bound_pretransformation``. A disjunct row that bounds a single variable
-- ``a*x + b <= 0``, ``>= 0`` or ``== 0`` -- is an implication "if this disjunct is
selected then ``x`` is in ``[L_k, U_k]``". Under SELECT_ONE exactly one indicator
``y_k`` is 1, so all of a disjunction's bounds on ``x`` are exactly

    sum_k L_k y_k  <=  x  <=  sum_k U_k y_k

where a disjunct with no bound of its own on ``x`` contributes the variable's
global bound. The transformation emits those two rows and removes the univariate
rows they replace; a GDP lowering must still follow for the rest of the model.

Rules, each of which keeps the transformation exact:

* **SELECT_ONE only.** Under OR several ``y_k`` may be 1 and the sums are not a
  bound; an OR disjunction is refused, as Pyomo refuses it.
* **Sign-correct.** ``a*x + b <= 0`` bounds ``x`` from above when ``a > 0`` and
  from below when ``a < 0`` (Pyomo's mbigm shortcut ignores the sign -- spec §3.6).
* **No defaulted infinity** (spec §0.2 item 5). A side is emitted only when every
  disjunct's ``L_k`` (or ``U_k``) is finite; otherwise the rows on that side stay
  in their disjuncts, untouched.
* **Array rows.** An array-valued row (``x <= 3`` over a vector ``x``) is read
  element by element from its exact affine map and emitted as one array-valued
  bound row, so it lowers exactly as the per-element rows it stands for while
  staying one object (#1215). It is dropped only when every element is univariate
  and every side it bounds was emitted.
* **User big-Ms.** Inside the big-M lowerings a disjunction carrying user
  ``big_m`` values (#1775 G2.6) is not pretransformed: the user's M stands.
* **Scope.** ``either_or`` disjunctions and complete top-level
  ``add_disjunction`` block groups without nesting. A disjunction that contains a
  nested one, a block nested in a disjunct, and anything else is left as it is --
  a partial transformation may skip what it does not cover. (The spec's ancestral
  propagation for nesting is not done here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constraint,
    DisjunctionSemantics,
    Expression,
    IndexExpression,
    Model,
    UnaryOp,
    Variable,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
)

#: Magnitudes at or above this are "no bound" (the GDP passes' sentinel).
_INF = 1e19


@dataclass
class PretransformReport:
    """What :func:`bound_pretransform` did: disjunctions transformed, univariate
    rows removed, global bound rows added, and why any disjunction was skipped."""

    disjunctions: int = 0
    rows_removed: int = 0
    rows_added: int = 0
    skipped: list[str] = field(default_factory=list)


def _slot_box(model: Model) -> dict[int, tuple[float, float]]:
    box: dict[int, tuple[float, float]] = {}
    off = 0
    for v in model._variables:
        lb = np.broadcast_to(np.asarray(v.lb, dtype=np.float64), v.shape).ravel()
        ub = np.broadcast_to(np.asarray(v.ub, dtype=np.float64), v.shape).ravel()
        for i in range(v.size):
            box[off + i] = (float(lb[i]), float(ub[i]))
        off += v.size
    return box


def _leaf_for_slot(expr: Expression, slot: int, model: Model) -> Expression | None:
    """The ``Variable`` / ``IndexExpression`` node of ``expr`` that names ``slot``."""
    from discopt._flat_index import resolve_scalar_slot

    stack: list[Any] = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, (Variable, IndexExpression)):
            if resolve_scalar_slot(e, model) == slot:
                return e
            if isinstance(e, IndexExpression):
                continue
        if isinstance(e, BinaryOp):
            stack.extend((e.left, e.right))
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
    return None


def _univariate(con: Constraint, model: Model, n_vars: int):
    """``(slot, leaf, lower, upper)`` when ``con`` bounds exactly one scalar slot, else None.

    ``lower``/``upper`` is the bound it implies on that slot (``None`` for no bound on
    that side). Only operator-built rows (``rhs == 0``, scalar) are read."""
    from discopt._relax.gdp_reformulate import _extract_body_coeffs

    if np.size(con.rhs) != 1 or float(np.asarray(con.rhs).ravel()[0]) != 0.0:
        return None
    coeffs = _extract_body_coeffs(con.body, model, n_vars)
    if coeffs is None:
        return None
    c_vec, off = coeffs
    nz = np.flatnonzero(c_vec)
    if nz.size != 1:
        return None
    slot = int(nz[0])
    leaf = _leaf_for_slot(con.body, slot, model)
    if leaf is None:
        return None
    return _bound_of(con.sense, slot, leaf, float(c_vec[slot]), off)


def _bound_of(sense: str, slot: int, leaf: Expression, a: float, off: float):
    """``(slot, leaf, lower, upper)`` of the row ``a*x + off  (sense)  0``."""
    value = -off / a  # a*x + off  (sense)  0   =>   x  vs  -off/a
    if sense == "==":
        return slot, leaf, value, value
    upper_side = (sense == "<=") == (a > 0)
    return (slot, leaf, None, value) if upper_side else (slot, leaf, value, None)


def _emit(leaf, vals, selectors, sense, out_rows, report) -> None:
    """Append ``leaf - sum_k vals[k] * y_k  (sense)  0``; a zero term is left out."""
    total: Expression = leaf
    for v, y in zip(vals, selectors):
        if np.ndim(v) == 0:
            if v != 0.0:
                total = total - float(v) * y
        elif np.any(v != 0.0):
            total = total - np.asarray(v, dtype=np.float64) * y
    out_rows.append(Constraint(body=total, sense=sense, rhs=0.0))
    report.rows_added += 1


def _slot_var(slot: int, model: Model) -> tuple[int, int]:
    """``(position in model._variables, flat element)`` of the flat ``slot``."""
    off = 0
    for i, v in enumerate(model._variables):
        if slot < off + v.size:
            return i, slot - off
        off += v.size
    raise IndexError(f"slot {slot} is past the model's {off} columns")


def _slot_leaf(slot: int, model: Model) -> Expression | None:
    """``x[i]`` for the flat ``slot`` (the variable itself when it is scalar)."""
    off = 0
    for v in model._variables:
        if slot < off + v.size:
            if v.shape == ():
                return v
            leaf: Expression = v[np.unravel_index(slot - off, v.shape)]
            return leaf
        off += v.size
    return None


def _row_bounds(con: Constraint, model: Model, n_vars: int):
    """Every element of ``con`` as ``(slot, leaf, lower, upper)``, and whether that
    covers the row.

    A scalar row is :func:`_univariate`. An array-valued row (``x <= 3`` over a
    vector ``x``; #1215 keeps it one row) is read element by element from its exact
    affine map (:func:`discopt._relax.affine_array.affine_of`), so it is
    pretransformed exactly as the per-element rows ``x[i] <= 3`` it stands for. The
    row is covered only when every element bounds exactly one slot; a row that is
    not covered is never dropped, and the bounds read from it are still valid.
    """
    if _is_array(con):
        return _array_row_bounds(con, model, n_vars)
    u = _univariate(con, model, n_vars)
    return ([u], True) if u is not None else ([], False)


def _is_array(con: Constraint) -> bool:
    shape = getattr(con.body, "shape", ())
    return bool(shape) and int(np.prod(shape)) > 1


def _array_row_bounds(con: Constraint, model: Model, n_vars: int):
    from discopt._relax.affine_array import NotAffineError, affine_of

    try:
        aff = affine_of(con.body, model._flat_var_offset, n_vars)
    except NotAffineError:
        return [], False
    size = int(np.prod(aff.shape)) if aff.shape else 1
    if aff.M is None or aff.M.shape[0] != size:
        return [], False
    rhs = np.broadcast_to(np.asarray(con.rhs, dtype=np.float64), aff.shape).ravel()
    b = np.broadcast_to(np.asarray(aff.b, dtype=np.float64), aff.shape).ravel()
    M = aff.M.tocsr()
    out: list[tuple[int, Expression, float | None, float | None]] = []
    for i in range(size):
        lo, hi = M.indptr[i], M.indptr[i + 1]
        cols, vals = M.indices[lo:hi], M.data[lo:hi]
        keep = vals != 0.0
        cols, vals = cols[keep], vals[keep]
        if cols.size != 1:
            return out, False
        slot = int(cols[0])
        leaf = _slot_leaf(slot, model)
        if leaf is None:
            return out, False
        out.append(_bound_of(con.sense, slot, leaf, float(vals[0]), float(b[i] - rhs[i])))
    return out, True


def _transform_one(
    disjuncts: list[list[Constraint]],
    selectors: list[Variable],
    model: Model,
    n_vars: int,
    box: dict[int, tuple[float, float]],
    report: PretransformReport,
    out_rows: list[Constraint],
) -> list[set[int]]:
    """Append the bound rows of one disjunction to ``out_rows``; return, per
    disjunct, the ``id`` of each row the appended rows replace."""
    # per slot: leaf, and per disjunct the tightest bound each side
    found: dict[int, dict[str, Any]] = {}
    parsed: dict[int, tuple[list, bool]] = {}
    for k, rows in enumerate(disjuncts):
        for con in rows:
            entries, covered = _row_bounds(con, model, n_vars)
            parsed[id(con)] = (entries, covered)
            from_array = _is_array(con)
            for slot, leaf, lo, hi in entries:
                rec = found.setdefault(slot, {"leaf": leaf, "lo": {}, "hi": {}, "array": True})
                rec["array"] = rec["array"] and from_array
                if lo is not None:
                    rec["lo"][k] = max(rec["lo"].get(k, -np.inf), lo)
                if hi is not None:
                    rec["hi"][k] = min(rec["hi"].get(k, np.inf), hi)

    emitted: set[tuple[int, str]] = set()
    # (variable position, side) -> {element: per-disjunct values}, for slots read
    # only from array rows; emitted below as one array row per variable (#1215).
    grouped: dict[tuple[int, str], dict[int, list[float]]] = {}
    for slot, rec in found.items():
        glo, ghi = box[slot]
        for side, per, fallback, sense in (
            ("lo", rec["lo"], glo, ">="),
            ("hi", rec["hi"], ghi, "<="),
        ):
            if not per:
                continue
            vals = [per.get(k, fallback) for k in range(len(disjuncts))]
            if not all(np.isfinite(v) and abs(v) < _INF for v in vals):
                continue  # a disjunct would contribute an infinite bound: keep the rows
            emitted.add((slot, side))
            if rec["array"]:
                vi, pos = _slot_var(slot, model)
                grouped.setdefault((vi, side), {})[pos] = vals
                continue
            _emit(rec["leaf"], vals, selectors, sense, out_rows, report)
    for (vi, side), per_pos in grouped.items():
        var = model._variables[vi]
        sense = ">=" if side == "lo" else "<="
        if len(per_pos) == var.size:
            # Every element bounded on this side: one array row over the whole
            # variable, C order -- the per-element rows, unexpanded.
            table = np.array([per_pos[i] for i in range(var.size)], dtype=np.float64)
            cols = [table[:, k].reshape(var.shape) for k in range(len(disjuncts))]
            _emit(var, cols, selectors, sense, out_rows, report)
        else:
            for pos, vals in sorted(per_pos.items()):
                leaf = var[np.unravel_index(pos, var.shape)]
                _emit(leaf, vals, selectors, sense, out_rows, report)

    # A row goes only when every side it bounds was emitted (an equality bounds both).
    drop: list[set[int]] = [set() for _ in disjuncts]
    for k, rows in enumerate(disjuncts):
        for con in rows:
            entries, covered = parsed[id(con)]
            if not covered:
                continue
            if all(
                (slot, side) in emitted
                for slot, _leaf, lo, hi in entries
                for side, b in (("lo", lo), ("hi", hi))
                if b is not None
            ):
                drop[k].add(id(con))
    return drop


def bound_pretransform(model: Model) -> PretransformReport:
    """Apply the bound pretransformation to ``model`` in place (#1775 G2.4).

    Raises ``NotImplementedError`` on an OR disjunction. Returns a
    :class:`PretransformReport`. See the module docstring for the rules.
    """
    from discopt._relax.gdp_reformulate import (
        _collect_indicator_hull_groups,
        _precompute_lp_relaxation,
    )

    report = PretransformReport()
    for c in model._constraints:
        sem = getattr(c, "semantics", None)
        if isinstance(c, (_DisjunctiveConstraint, _IndicatorConstraint)) and sem is (
            DisjunctionSemantics.OR
        ):
            name = getattr(c, "name", None) or "<anon>"
            raise NotImplementedError(
                f"gdp.bound_pretransformation: disjunction {name!r} has "
                "DisjunctionSemantics.OR. The bound rows sum the disjuncts' bounds "
                "weighted by their indicators, which is a bound only when exactly one "
                "is selected; Pyomo's transformation refuses OR the same way."
            )

    lp = _precompute_lp_relaxation(model)
    n_vars = lp[4] if lp is not None else sum(v.size for v in model._variables)
    box = _slot_box(model)

    # either_or disjunctions (top level, no nesting).
    for c in list(model._constraints):
        if not isinstance(c, _DisjunctiveConstraint) or c.parent_indicator is not None:
            continue
        if c.semantics is not DisjunctionSemantics.SELECT_ONE:
            report.skipped.append(f"{c.name or '<anon>'}: semantics {c.semantics.name}")
            continue
        if any(isinstance(it, _DisjunctiveConstraint) for d in c.disjuncts for it in d):
            report.skipped.append(f"{c.name or '<anon>'}: contains a nested disjunction")
            continue
        if not any(_univariate(r, model, n_vars) for d in c.disjuncts for r in d):
            continue
        selectors = [b.variable for b in c.indicators]  # allocates them + exactly(1)
        drop = _transform_one(
            c.disjuncts, selectors, model, n_vars, box, report, model._constraints
        )
        if any(drop):
            report.disjunctions += 1
            if c.big_m is not None:  # #1775 G2.6: keep the user Ms aligned with the rows
                c.big_m = tuple(
                    tuple(m for r, m in zip(d, ms) if id(r) not in drop[k])
                    for k, (d, ms) in enumerate(zip(c.disjuncts, c.big_m))
                )
            for k, d in enumerate(c.disjuncts):
                kept = [r for r in d if id(r) not in drop[k]]
                report.rows_removed += len(d) - len(kept)
                d[:] = kept

    # add_disjunction block groups (complete, top level, no nesting).
    for key, (dc, sels) in _collect_indicator_hull_groups(model).groups.items():
        if any(isinstance(it, _DisjunctiveConstraint) for d in dc.disjuncts for it in d):
            report.skipped.append(f"{key}: contains a nested disjunction")
            continue
        drop = _transform_one(dc.disjuncts, sels, model, n_vars, box, report, model._constraints)
        if not any(drop):
            continue
        report.disjunctions += 1
        before = len(model._constraints)
        # Drop a row only as a member of *this* group's disjunct k. The same
        # Constraint object may also sit in an if_then or another block
        # (#1794 B1); matching on ``id(r.constraint)`` alone deleted that row
        # too, and the solve certified an incumbent violating it.
        model._constraints[:] = [
            r
            for r in model._constraints
            if not (
                isinstance(r, _IndicatorConstraint)
                and r.disjunction is not None
                and r.disjunction[0] == key
                and id(r.constraint) in drop[r.disjunction[1]]
            )
        ]
        report.rows_removed += before - len(model._constraints)
    return report


def pretransform_disjunction(
    disjuncts: list[list[Constraint]],
    selectors: list[Variable],
    model: Model,
    out_rows: list[Constraint],
) -> list[set[int]]:
    """The pretransformation of one SELECT_ONE disjunction, for a GDP lowering.

    Appends the bound rows to ``out_rows`` and returns, per disjunct, the ``id`` of
    each row they replace -- rows the caller then does not lower. ``selectors`` are
    the lowering's own (one per disjunct, exactly one of them 1); ``model`` supplies
    the columns and the global box. Used by ``reformulate_gdp`` under
    ``DISCOPT_GDP_BOUND_PRETRANSFORM`` (BC, default ON).
    """
    n_vars = sum(v.size for v in model._variables)
    return _transform_one(
        disjuncts, selectors, model, n_vars, _slot_box(model), PretransformReport(), out_rows
    )


def _bound_pretransform_in_place(model: Model) -> None:
    """The registry's in-place entry point (``transformations.apply_to``)."""
    bound_pretransform(model)


__all__ = ["PretransformReport", "bound_pretransform", "pretransform_disjunction"]
