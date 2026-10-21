"""``gdp.partition_disjuncts`` and ``gdp.between_steps``: P-split (#1775 G2.8).

Pyomo's P-split (Kronqvist, Misener & Tsay 2021). A disjunct row whose body is
additively separable,

    sum_p h_p(x_p) + c <= 0      (a ``>=`` row is negated first),

over P groups of variables ``x_p`` is replaced by

    sum_p a_p + c <= 0           in the disjunct, and
    h_p(x_p) <= a_p              for each p, as *global* rows,

with each auxiliary ``a_p`` in ``[lo_p, hi_p]``, the interval enclosure of ``h_p``
over the variable box. This is **exact for any h_p** -- convex or not:

* disjunct selected: the global rows give ``sum h_p <= sum a_p <= -c``; and any
  point of the original row is reached with ``a_p = h_p(x_p)``, which lies in
  ``[lo_p, hi_p]``;
* disjunct not selected: ``a_p = hi_p >= h_p(x_p)`` satisfies every global row
  wherever ``x`` is, so the split removes no point there.

Its *strength* (a hull of the split disjuncts is between big-M and the full hull,
the "between steps") is the paper's claim for convex rows only. ``between_steps``
is the partition followed by the hull lowering.

What is split: every scalar ``<=``/``>=`` row of an ``either_or`` / nested
disjunction and every indicator row (``if_then``, ``add_disjunction`` blocks) whose
summands fall into at least two groups. An ``==`` row, an array-valued row and a
row that groups into one part are left as they are, and so is a row that
depends on a Parameter (its auxiliaries' bounds would be frozen at today's
value). Refused, loudly: a summand
whose variables straddle two user-given partitions, a variable of a row in no
partition, an ``h_p`` with no finite upper bound (``a_p`` would have none; spec
§0.2 item 5), and an ``h_p`` undefined on part of the box (the global row keeps it
everywhere, #1779). A user ``big_m`` (#1775 G2.6) on a split row is dropped: it
bounded the old body, not ``sum a_p + c``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from discopt.constants import DEFAULT_VARIABLE_BOUND
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    Parameter,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
)


@dataclass
class PartitionReport:
    """What :func:`partition_disjuncts` did: rows split, auxiliaries added, and
    the rows left alone with the reason."""

    rows_split: int = 0
    aux_vars: int = 0
    skipped: list[str] = field(default_factory=list)


def _summands(expr: Expression, sign: float, out: list) -> None:
    """Flatten ``expr`` into ``(sign, term)`` summands over ``+``/``-``/negation."""
    if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
        _summands(expr.left, sign, out)
        _summands(expr.right, sign if expr.op == "+" else -sign, out)
    elif isinstance(expr, UnaryOp) and expr.op == "neg":
        _summands(expr.operand, -sign, out)
    elif isinstance(expr, SumOverExpression):
        for t in expr.terms:
            _summands(t, sign, out)
    else:
        out.append((sign, expr))


def _var_names(expr: Any, out: set) -> set:
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, Variable):
            out.add(e.name)
        elif isinstance(e, IndexExpression):
            stack.append(e.base)
        elif isinstance(e, (Constant, Parameter)):
            continue
        else:
            for attr in ("left", "right", "operand", "base"):
                child = getattr(e, attr, None)
                if child is not None:
                    stack.append(child)
            for coll in ("terms", "args"):
                stack.extend(getattr(e, coll, None) or [])
    return out


def _groups(
    terms: list, partitions: Optional[list[set[str]]], num_partitions: int, where: str
) -> list[list]:
    """Assign summands to parts; return the non-empty parts (each a list of terms)."""
    if partitions is not None:
        parts: list[list] = [[] for _ in partitions]
        for sign, t, names in terms:
            hit = {i for i, p in enumerate(partitions) for n in names if n in p}
            missing = [n for n in names if not any(n in p for p in partitions)]
            if missing:
                raise ValueError(
                    f"gdp.partition_disjuncts: {where} uses {sorted(missing)}, which is in "
                    "no partition; give every variable of a split row a partition."
                )
            if len(hit) > 1:
                raise ValueError(
                    f"gdp.partition_disjuncts: a summand of {where} uses variables of "
                    f"partitions {sorted(hit)}; a summand must lie in one partition "
                    "(P-split splits a sum, not a term)."
                )
            parts[hit.pop()].append((sign, t))
        return [p for p in parts if p]
    # Automatic: summands sharing a variable stay together (union-find), then the
    # components are packed into num_partitions bins, fewest summands first.
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, _, names in terms:
        ns = sorted(names)
        for n in ns[1:]:
            parent[find(n)] = find(ns[0])
    comps: dict[str, list] = {}
    order: list[str] = []
    for sign, t, names in terms:
        root = find(sorted(names)[0])
        if root not in comps:
            comps[root] = []
            order.append(root)
        comps[root].append((sign, t))
    bins: list[list] = [[] for _ in range(min(num_partitions, len(order)))]
    for root in order:
        min(bins, key=len).extend(comps[root])
    return [b for b in bins if b]


def _has_parameter(expr: Any) -> bool:
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, Parameter):
            return True
        for attr in ("left", "right", "operand", "base"):
            child = getattr(e, attr, None)
            if child is not None:
                stack.append(child)
        for coll in ("terms", "args"):
            stack.extend(getattr(e, coll, None) or [])
    return False


def _sum(signed: list) -> Expression:
    total: Optional[Expression] = None
    for sign, t in signed:
        term = t if sign > 0 else -t
        total = term if total is None else total + term
    assert total is not None
    return total


def _split_row(
    con: Constraint,
    model: Model,
    partitions: Optional[list[set[str]]],
    num_partitions: int,
    report: PartitionReport,
    globals_out: list[Constraint],
    where: str,
    local_names: set[str],
) -> Constraint:
    """The split replacement of ``con`` (or ``con`` itself when not split)."""
    from discopt._relax.gdp_reformulate import (
        _BIGM_SENTINEL,
        _bound_expression,
        _domain_gap,
        _domain_gap_error,
    )

    if con.sense not in ("<=", ">=") or np.size(con.rhs) != 1:
        report.skipped.append(f"{where}: sense {con.sense!r}")
        return con
    if tuple(getattr(con.body, "shape", ()) or ()) != ():
        report.skipped.append(f"{where}: array-valued")
        return con
    if local_names & _var_names(con.body, set()):
        # The global rows h_p <= a_p would carry a Disjunct.variable out of its
        # block, which Model.validate refuses (#1775 G1.2).
        report.skipped.append(f"{where}: uses a disjunct-local variable")
        return con
    if _has_parameter(con.body):
        # The auxiliaries' bounds and the constant would be frozen at today's
        # values; a later parameter update could then cut feasible points.
        report.skipped.append(f"{where}: depends on a Parameter")
        return con
    sign0 = 1.0 if con.sense == "<=" else -1.0
    raw: list = []
    _summands(con.body, sign0, raw)
    const = 0.0
    terms = []
    for sign, t in raw:
        names = _var_names(t, set())
        if names:
            terms.append((sign, t, names))
        else:
            from discopt._relax.gdp_reformulate import _body_at_zero

            const += sign * float(_body_at_zero(t, {}))
    rhs = float(np.asarray(con.rhs, dtype=np.float64).ravel()[0])
    const -= sign0 * rhs
    parts = _groups(terms, partitions, num_partitions, where)
    if len(parts) < 2:
        report.skipped.append(f"{where}: one part")
        return con
    alphas: list[Variable] = []
    for p, signed in enumerate(parts):
        h = _sum(signed)
        gap = _domain_gap(h, model)
        if gap is not None:
            raise _domain_gap_error(con, gap, "gdp.partition_disjuncts")
        lo, hi = (float(v) for v in _bound_expression(h, model))
        if not (np.isfinite(hi) and abs(hi) < _BIGM_SENTINEL):
            raise ValueError(
                f"gdp.partition_disjuncts: part {p} of {where} has no finite upper bound "
                f"(interval enclosure [{lo:g}, {hi:g}]), so its auxiliary would have none "
                "and an unselected disjunct could not absorb it. Bound its variables."
            )
        lb = lo if np.isfinite(lo) and abs(lo) < _BIGM_SENTINEL else None
        a = Variable(
            model._fresh_name("_gdp_psplit_a"),
            VarType.CONTINUOUS,
            (),
            -DEFAULT_VARIABLE_BOUND if lb is None else lb,
            hi,
            model,
        )
        model._register_generated_variable(a)
        alphas.append(a)
        globals_out.append(Constraint(body=h - a, sense="<=", rhs=0.0))
        report.aux_vars += 1
    body: Expression = alphas[0]
    for a in alphas[1:]:
        body = body + a
    if const != 0.0:
        body = body + const
    report.rows_split += 1
    return Constraint(body=body, sense="<=", rhs=0.0, name=con.name)


def partition_disjuncts(
    model: Model,
    *,
    num_partitions: int = 2,
    partitions: Optional[Sequence[Sequence[Variable]]] = None,
) -> PartitionReport:
    """Apply P-split to ``model`` in place (#1775 G2.8); see the module docstring.

    ``partitions`` (a list of variable groups) is used for every split row when
    given; otherwise the summands of each row are grouped automatically (summands
    sharing a variable stay together) into at most ``num_partitions`` parts.
    """
    if num_partitions < 2 and partitions is None:
        raise ValueError(
            f"gdp.partition_disjuncts: num_partitions must be >= 2, got {num_partitions!r}"
        )
    parts = None if partitions is None else [{v.name for v in p} for p in partitions]
    report = PartitionReport()
    globals_out: list[Constraint] = []
    local_names = {v.name for d in model._gdp_factory_disjuncts for v in d._local_vars}
    replaced: dict[int, Constraint] = {}

    def walk(dc: _DisjunctiveConstraint) -> None:
        label = dc.name or "<anon>"
        big_m = [list(r) for r in dc.big_m] if dc.big_m is not None else None
        for k, d in enumerate(dc.disjuncts):
            for j, item in enumerate(d):
                if isinstance(item, _DisjunctiveConstraint):
                    walk(item)
                    continue
                new = _split_row(
                    item,
                    model,
                    parts,
                    num_partitions,
                    report,
                    globals_out,
                    f"row {j} of disjunct {k} of {label!r}",
                    local_names,
                )
                if new is not item:
                    d[j] = new
                    replaced[id(item)] = new
                    if big_m is not None:
                        big_m[k][j] = None  # it bounded the old body
        if big_m is not None:
            kept = any(v is not None for r in big_m for v in r)
            dc.big_m = tuple(tuple(r) for r in big_m) if kept else None

    for i, c in enumerate(list(model._constraints)):
        if isinstance(c, _DisjunctiveConstraint):
            walk(c)
        elif isinstance(c, _IndicatorConstraint):
            new = _split_row(
                c.constraint,
                model,
                parts,
                num_partitions,
                report,
                globals_out,
                f"indicator row {c.constraint.name or i}",
                local_names,
            )
            if new is not c.constraint:
                replaced[id(c.constraint)] = new
                c.constraint = new
    # A Disjunct block keeps its own rows (serialization rebuilds blocks from them,
    # and validate scopes local variables by them): point them at the new rows.
    for blk in model._gdp_factory_disjuncts:
        blk._constraints = [replaced.get(id(r), r) for r in blk._constraints]
    model._constraints.extend(globals_out)
    return report


def _partition_in_place(model: Model, **options: Any) -> None:
    """The registry's in-place entry point (``transformations``)."""
    partition_disjuncts(model, **options)


def between_steps(model: Model, **options: Any) -> Model:
    """P-split a copy of ``model``, then lower it by hull (Pyomo ``gdp.between_steps``).

    ``model`` is not changed. The result owns its variables (they are the copy's,
    re-pointed), like any other lowering's output.
    """
    import copy

    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.modeling.core import carry_validation_guards

    work = copy.deepcopy(model)
    partition_disjuncts(work, **options)
    out = reformulate_gdp(work, "hull")
    holders: list[Any] = [*out._variables, *out._parameters]
    for holder in holders:
        if holder.model is work:
            holder.model = out
    carry_validation_guards(model, out)
    return out


__all__ = ["PartitionReport", "between_steps", "partition_disjuncts"]
