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
value). Refused, loudly: a summand whose variables lie in two user-given
partitions (this grouping cannot split the row: P-split assigns each summand to
one part), a variable listed in two partitions, a variable of a row in no
partition, an ``h_p`` with no finite upper bound (``a_p`` would have none; spec
§0.2 item 5), and an ``h_p`` undefined on part of the box (the global row keeps it
everywhere, #1779). A user ``big_m`` (#1775 G2.6) on a split row is dropped: it
bounded the old body, not ``sum a_p + c``.

**Atomic.** Every replacement row, auxiliary and global row is built before the
model is touched, so a refusal leaves the model exactly as it was. The user's
disjunct containers are never assigned into: a disjunction whose rows change gets
new lists (a tuple disjunct a new tuple), and its Disjunct blocks are re-pointed.

**Frozen to the variable bounds of the call.** ``[lo_p, hi_p]`` is the interval
enclosure of ``h_p`` over the box *at the time of the call* (``evaluate_interval``,
rounded outward). Widening a bound afterwards can push ``h_p`` past ``hi_p``, and
then an unselected disjunct's global row ``h_p <= a_p <= hi_p`` cuts points that
have become feasible. Each enclosure is therefore recorded as a validation guard
(``Model._piecewise_domains``, the same re-check a ``piecewise`` input's breakpoint
span gets), so :meth:`Model.validate` -- and with it every solve -- refuses the
model once a bound has moved an enclosure outside its auxiliary's bounds.
Tightening a bound is fine. The guard travels with ``deepcopy``, ``pickle``,
:mod:`discopt.serialize` and every lowering (``carry_validation_guards``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from discopt.constants import DEFAULT_VARIABLE_BOUND
from discopt.modeling._piecewise import DOMAINS_ATTR, PSPLIT_GUARD_PREFIX
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
                    f"partitions {sorted(hit)} ({sorted(names)}). P-split assigns each "
                    "summand to one part, so these partitions cannot split this row. Put "
                    "those variables in one partition, or leave partitions=None: the "
                    "automatic grouping keeps summands that share a variable together."
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


class _Plan:
    """Everything :func:`partition_disjuncts` will add, built without touching the
    model: the auxiliaries (indexed as they will be once registered), the global
    rows, and the enclosure guards."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.aux: list[Variable] = []
        self.globals_out: list[Constraint] = []
        self.guards: list[tuple[Expression, float, float, str]] = []
        self._reserved: set[str] = set()
        self.var_by_name = {v.name: v for v in model._variables}

    def _fresh_name(self, base: str) -> str:
        # Model._fresh_name, also avoiding the names this plan has taken already.
        taken = self.model._names
        name, k = base, 2
        while name in taken or name in self._reserved:
            name, k = f"{base}_{k}", k + 1
        self._reserved.add(name)
        return name

    def new_aux(self, lb: float, ub: float) -> Variable:
        a = Variable(self._fresh_name("_gdp_psplit_a"), VarType.CONTINUOUS, (), lb, ub, self.model)
        # Its position once the commit registers the auxiliaries in order.
        a._index = len(self.model._variables) + len(self.aux)
        self.aux.append(a)
        return a


def _one_part_reason(terms: list, partitions: Optional[list[set[str]]], plan: _Plan) -> str:
    if len(terms) < 2:
        return "one part (a single variable term)"
    if partitions is not None:
        why = "one part (every summand is in the same partition)"
    else:
        why = "one part (its summands are linked through shared variables)"
    vector = sorted(
        {
            n
            for _, _, names in terms
            for n in names
            if n in plan.var_by_name and plan.var_by_name[n].shape != ()
        }
    )
    if vector:
        why += (
            f"; the elements of the vector variable(s) {vector} count as one variable, "
            "so a row over several of them is not split across them. Declare them as "
            "separate scalar variables to split it"
        )
    return why


def _split_row(
    con: Constraint,
    plan: _Plan,
    partitions: Optional[list[set[str]]],
    num_partitions: int,
    report: PartitionReport,
    where: str,
    local_names: set[str],
) -> Constraint:
    """The split replacement of ``con`` (or ``con`` itself when not split).

    Adds to ``plan`` only; the model is not changed.
    """
    from discopt._relax.convexity.interval_eval import evaluate_interval
    from discopt._relax.gdp_reformulate import (
        _BIGM_SENTINEL,
        _body_at_zero,
        _domain_gap,
        _domain_gap_error,
    )

    model = plan.model
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
    # body >= rhs  <=>  -body + rhs <= 0;  body <= rhs  <=>  body - rhs <= 0.
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
            const += sign * float(_body_at_zero(t, {}))
    rhs = float(np.asarray(con.rhs, dtype=np.float64).ravel()[0])
    const -= sign0 * rhs
    parts = _groups(terms, partitions, num_partitions, where)
    if len(parts) < 2:
        report.skipped.append(f"{where}: {_one_part_reason(terms, partitions, plan)}")
        return con
    alphas: list[Variable] = []
    pending: list[tuple[Expression, float, float]] = []
    for p, signed in enumerate(parts):
        h = _sum(signed)
        gap = _domain_gap(h, model)
        if gap is not None:
            raise _domain_gap_error(con, gap, "gdp.partition_disjuncts")
        # evaluate_interval rounds outward, so [lo, hi] contains h over the box
        # exactly, not up to float error; it is also what the validate-time guard
        # recomputes, so the guard holds at the call by construction.
        iv = evaluate_interval(h, model)
        lo, hi = float(np.min(iv.lo)), float(np.max(iv.hi))
        if not (np.isfinite(hi) and abs(hi) < _BIGM_SENTINEL):
            raise ValueError(
                f"gdp.partition_disjuncts: part {p} of {where} has no finite upper bound "
                f"(interval enclosure [{lo:g}, {hi:g}]), so its auxiliary would have none "
                "and an unselected disjunct could not absorb it. Bound its variables."
            )
        pending.append((h, lo, hi))
    # Only now, with every part of the row validated, is anything planned.
    for h, lo, hi in pending:
        finite_lo = bool(np.isfinite(lo) and abs(lo) < _BIGM_SENTINEL)
        a = plan.new_aux(lo if finite_lo else -DEFAULT_VARIABLE_BOUND, hi)
        alphas.append(a)
        plan.globals_out.append(Constraint(body=h - a, sense="<=", rhs=0.0))
        plan.guards.append(
            (
                h,
                lo if finite_lo else -np.inf,
                hi,
                f"{PSPLIT_GUARD_PREFIX} auxiliary {a.name} of {where}",
            )
        )
        report.aux_vars += 1
    body: Expression = alphas[0]
    for a in alphas[1:]:
        body = body + a
    if const != 0.0:
        body = body + const
    report.rows_split += 1
    return Constraint(body=body, sense="<=", rhs=0.0, name=con.name)


def _partition_names(model: Model, partitions: Sequence[Sequence[Variable]]) -> list[set[str]]:
    """The user's partitions as name sets, refusing anything P-split cannot use."""
    owner: dict[str, int] = {}
    out: list[set[str]] = []
    for i, group in enumerate(partitions):
        if isinstance(group, Expression):
            raise ValueError(
                f"gdp.partition_disjuncts: partition {i} is {group!r}, not a list of "
                "variables; pass partitions=[[x, y], [z], ...]."
            )
        names: set[str] = set()
        for v in group:
            if not isinstance(v, Variable):
                hint = ""
                if isinstance(v, IndexExpression) and isinstance(v.base, Variable):
                    hint = (
                        f" It is an element of the vector variable {v.base.name!r}: a "
                        "partition holds whole variables, so the elements of one vector "
                        "variable cannot go into different parts. List the whole variable, "
                        "or declare the elements as separate scalar variables."
                    )
                raise ValueError(
                    f"gdp.partition_disjuncts: partition {i} holds {v!r} "
                    f"({type(v).__name__}), not a Variable.{hint}"
                )
            if v.model is not model:
                raise ValueError(
                    f"gdp.partition_disjuncts: partition {i} holds {v.name!r}, which is "
                    "not a variable of this model."
                )
            if owner.get(v.name, i) != i:
                raise ValueError(
                    f"gdp.partition_disjuncts: duplicate variable {v.name!r}: it is in "
                    f"partitions {owner[v.name]} and {i}. Each variable goes in one partition."
                )
            owner[v.name] = i
            names.add(v.name)
        out.append(names)
    return out


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

    Atomic: a refusal leaves ``model`` unchanged. The auxiliaries' bounds are the
    enclosures over today's variable box; :meth:`Model.validate` refuses the model
    if a bound is later widened past one of them (module docstring).
    """
    if num_partitions < 2 and partitions is None:
        raise ValueError(
            f"gdp.partition_disjuncts: num_partitions must be >= 2, got {num_partitions!r}"
        )
    parts = None if partitions is None else _partition_names(model, partitions)
    report = PartitionReport()
    plan = _Plan(model)
    local_names = {v.name for d in model._gdp_factory_disjuncts for v in d._local_vars}
    replaced: dict[int, Constraint] = {}
    new_disjuncts: list[tuple[_DisjunctiveConstraint, list, Any]] = []
    new_indicators: list[tuple[_IndicatorConstraint, Constraint]] = []
    seen: set[int] = set()

    # Phase 1: plan. Nothing below assigns into the model or the user's containers.
    def walk(dc: _DisjunctiveConstraint) -> None:
        if id(dc) in seen:
            return
        seen.add(id(dc))
        label = dc.name or "<anon>"
        big_m = [list(r) for r in dc.big_m] if dc.big_m is not None else None
        out: list = []
        changed = False
        for k, d in enumerate(dc.disjuncts):
            rows = list(d)
            d_changed = False
            for j, item in enumerate(rows):
                if isinstance(item, _DisjunctiveConstraint):
                    walk(item)
                    continue
                new = _split_row(
                    item,
                    plan,
                    parts,
                    num_partitions,
                    report,
                    f"row {j} of disjunct {k} of {label!r}",
                    local_names,
                )
                if new is not item:
                    rows[j] = new
                    replaced[id(item)] = new
                    d_changed = True
                    if big_m is not None:
                        big_m[k][j] = None  # it bounded the old body
            out.append((tuple(rows) if isinstance(d, tuple) else rows) if d_changed else d)
            changed = changed or d_changed
        if changed:
            new_big_m = dc.big_m
            if big_m is not None:
                kept = any(v is not None for r in big_m for v in r)
                new_big_m = tuple(tuple(r) for r in big_m) if kept else None
            new_disjuncts.append((dc, out, new_big_m))

    for i, c in enumerate(model._constraints):
        if isinstance(c, _DisjunctiveConstraint):
            walk(c)
        elif isinstance(c, _IndicatorConstraint):
            new = _split_row(
                c.constraint,
                plan,
                parts,
                num_partitions,
                report,
                f"indicator row {c.constraint.name or i}",
                local_names,
            )
            if new is not c.constraint:
                replaced[id(c.constraint)] = new
                new_indicators.append((c, new))
    # A Disjunct block keeps its own rows (serialization rebuilds blocks from them,
    # and validate scopes local variables by them): it is pointed at the new rows.
    new_blocks = [
        (blk, [replaced.get(id(r), r) for r in blk._constraints])
        for blk in model._gdp_factory_disjuncts
        if any(id(r) in replaced for r in blk._constraints)
    ]

    # Phase 2: commit. Plain assignments and appends; nothing here can refuse.
    for a in plan.aux:
        assert a._index == len(model._variables)
        model._register_generated_variable(a)
    for dc, out, new_big_m in new_disjuncts:
        dc.disjuncts = out
        dc.big_m = new_big_m
    for ic, new in new_indicators:
        ic.constraint = new
    for blk, rows in new_blocks:
        blk._constraints = rows
    model._constraints.extend(plan.globals_out)
    guards = getattr(model, DOMAINS_ATTR, None)
    if guards is None:
        guards = []
        setattr(model, DOMAINS_ATTR, guards)
    guards.extend(plan.guards)
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
