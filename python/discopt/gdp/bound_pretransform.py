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
    a = float(c_vec[slot])
    leaf = _leaf_for_slot(con.body, slot, model)
    if leaf is None:
        return None
    value = -off / a  # a*x + off  (sense)  0   =>   x  vs  -off/a
    if con.sense == "==":
        return slot, leaf, value, value
    upper_side = (con.sense == "<=") == (a > 0)
    return (slot, leaf, None, value) if upper_side else (slot, leaf, value, None)


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
    rows_of: dict[tuple[int, int, str], list[int]] = {}
    for k, rows in enumerate(disjuncts):
        for con in rows:
            u = _univariate(con, model, n_vars)
            if u is None:
                continue
            slot, leaf, lo, hi = u
            rec = found.setdefault(slot, {"leaf": leaf, "lo": {}, "hi": {}})
            if lo is not None:
                rec["lo"][k] = max(rec["lo"].get(k, -np.inf), lo)
                rows_of.setdefault((slot, k, "lo"), []).append(id(con))
            if hi is not None:
                rec["hi"][k] = min(rec["hi"].get(k, np.inf), hi)
                rows_of.setdefault((slot, k, "hi"), []).append(id(con))

    emitted: set[tuple[int, str]] = set()
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
            total: Expression = rec["leaf"]
            for v, y in zip(vals, selectors):
                if v != 0.0:
                    total = total - float(v) * y
            out_rows.append(Constraint(body=total, sense=sense, rhs=0.0))
            report.rows_added += 1
            emitted.add((slot, side))

    # A row goes only when every side it bounds was emitted (an equality bounds both).
    drop: list[set[int]] = [set() for _ in disjuncts]
    for k, rows in enumerate(disjuncts):
        for con in rows:
            u = _univariate(con, model, n_vars)
            if u is None:
                continue
            slot, _leaf, lo, hi = u
            sides = [s for s, b in (("lo", lo), ("hi", hi)) if b is not None]
            if all((slot, s) in emitted for s in sides):
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
        gone = set().union(*drop)
        if not gone:
            continue
        report.disjunctions += 1
        before = len(model._constraints)
        model._constraints[:] = [
            r
            for r in model._constraints
            if not (isinstance(r, _IndicatorConstraint) and id(r.constraint) in gone)
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
    ``DISCOPT_GDP_BOUND_PRETRANSFORM`` (BC, default OFF).
    """
    n_vars = sum(v.size for v in model._variables)
    return _transform_one(
        disjuncts, selectors, model, n_vars, _slot_box(model), PretransformReport(), out_rows
    )


def _bound_pretransform_in_place(model: Model) -> None:
    """The registry's in-place entry point (``transformations.apply_to``)."""
    bound_pretransform(model)


__all__ = ["PretransformReport", "bound_pretransform", "pretransform_disjunction"]
