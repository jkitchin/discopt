"""A read-only, normalized tree of a model's disjunctive structure (#1775 G1.1).

Three modeling APIs produce GDP, and before this module each consumer walked
``model._constraints`` and re-derived the structure itself:

* :meth:`Model.either_or` adds a ``_DisjunctiveConstraint``; nesting is a
  ``_DisjunctiveConstraint`` inside a disjunct;
* :meth:`Model.add_disjunction` adds ``_IndicatorConstraint`` rows tagged
  ``.disjunction = (key, k, n)`` plus an ``exactly(1)`` row over its indicators;
* :meth:`Model.if_then` adds untagged ``_IndicatorConstraint`` rows.

:func:`gdp_tree` maps all three onto one shape::

    GDPTree
      disjunctions: list[DisjunctionNode]        # roots, in model order
      logic:        list[_LogicalConstraint]
      indicators:   dict[DisjunctNode, BooleanVar]
    DisjunctionNode(id, name, kind, semantics, disjuncts, parent, origin)
    DisjunctNode(id, name, indicator, active_value, rows, children, local_vars, parent)

An ``if_then`` becomes a one-disjunct ``"implication"`` node, read as
``[y => rows] v [not y]``. The tree never changes the model and allocates nothing:
an ``either_or`` disjunct whose indicators were never read has ``indicator=None``
(reading :attr:`_DisjunctiveConstraint.indicators` is what allocates them).

Nesting (#1775 G1.2): a disjunction nested in an ``either_or`` disjunct is a child
of that disjunct (its disjuncts have ``indicator=None``). A block disjunction or a
``Model.disjunction`` nested in a ``Disjunct`` block (``Disjunct.add_disjunction``,
``Disjunct.subject_to``) is lowered at model level carrying its parent's indicator
column; the tree attaches it as a child of the disjunct that owns that column.
``local_vars`` lists the variables declared with ``Disjunct.variable``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from discopt.modeling.core import (
    BooleanVar,
    Constraint,
    DisjunctionSemantics,
    Model,
    Variable,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _LogicalConstraint,
)

Kind = Literal["either_or", "add_disjunction", "implication"]


@dataclass(frozen=True)
class Origin:
    """Where a disjunction came from: the index of its first row in
    ``model._constraints`` and the source object (the ``_DisjunctiveConstraint``,
    or the ``add_disjunction`` key / first ``_IndicatorConstraint``). A
    disjunction nested inside an ``either_or`` disjunct has no row of its own:
    its index is ``-1`` and its source is the nested ``_DisjunctiveConstraint``."""

    index: int
    source: Any


@dataclass(eq=False)
class DisjunctNode:
    id: int
    name: Optional[str]
    indicator: Optional[BooleanVar]
    #: The indicator value that activates the rows: 0 for ``if_then(1 - z, ...)``.
    active_value: int
    rows: list[Constraint]
    children: list[DisjunctionNode] = field(default_factory=list)
    local_vars: list[Variable] = field(default_factory=list)
    parent: Optional[DisjunctionNode] = None

    def __repr__(self) -> str:
        ind = self.indicator.variable.name if self.indicator is not None else None
        return f"DisjunctNode(id={self.id}, name={self.name!r}, indicator={ind!r})"


@dataclass(eq=False)
class DisjunctionNode:
    id: int
    name: Optional[str]
    kind: Kind
    semantics: DisjunctionSemantics
    disjuncts: list[DisjunctNode]
    parent: Optional[DisjunctNode]
    origin: Optional[Origin]

    def __repr__(self) -> str:
        return (
            f"DisjunctionNode(id={self.id}, name={self.name!r}, kind={self.kind!r}, "
            f"n={len(self.disjuncts)})"
        )


@dataclass
class GDPTree:
    disjunctions: list[DisjunctionNode]
    logic: list[_LogicalConstraint]
    indicators: dict[DisjunctNode, BooleanVar]

    def walk(self) -> Iterator[DisjunctionNode]:
        """Every disjunction, roots first in model order, each followed by its subtree."""

        def rec(node: DisjunctionNode) -> Iterator[DisjunctionNode]:
            yield node
            for d in node.disjuncts:
                for child in d.children:
                    yield from rec(child)

        for root in self.disjunctions:
            yield from rec(root)

    def node_for(self, source: object) -> DisjunctionNode:
        """The node built from ``source`` (a ``_DisjunctiveConstraint`` or a key).

        Searches every node that came from a model row, roots and the children
        attached under a ``Disjunct`` block (#1775 G1.2) alike.
        """
        for node in self.walk():
            if node.origin is not None and node.origin.source is source:
                return node
        raise KeyError(f"no disjunction was built from {source!r}")


def _fill_empty_block_indicators(model: Model, groups: dict) -> None:
    """Name the indicator of a block disjunct that emitted no row (#1775 G4.1).

    ``add_disjunction`` tags each block's rows with its indicator, so an empty
    block -- Pyomo's "do nothing" disjunct, ``[x <= u] v []`` -- is tagged on no
    row and its node had ``indicator=None``, though the model owns the column.
    The disjunction's selector row (``exactly(1, indicators)`` at top level) holds
    every block's indicator with coefficient 1: when exactly one disjunct of a
    node is unnamed and exactly one ``== 1`` row is the node's known indicators
    plus one other binary column, that column is the empty block's indicator.
    Anything less certain leaves the node as it was.
    """
    from discopt._relax.milp_relaxation import _linearize_affine_expr_sparse
    from discopt.modeling.core import VarType

    todo = [
        node
        for node in groups.values()
        if node.kind == "add_disjunction"
        and sum(d.indicator is None for d in node.disjuncts) == 1
        and any(d.indicator is not None for d in node.disjuncts)
    ]
    if not todo:
        return
    cols: list[tuple[Variable, int]] = []
    for v in model._variables:
        cols.extend((v, k) for k in range(int(v.size)))
    n = len(cols)

    def col_of(var: Variable) -> Optional[int]:
        off = 0
        for v in model._variables:
            if v is var:
                return off if v.size == 1 else None
            off += int(v.size)
        return None

    rows: list[frozenset[int]] = []
    for c in model._constraints:
        if getattr(c, "sense", None) != "==" or type(c) is not Constraint:
            continue
        try:
            terms, const = _linearize_affine_expr_sparse(c.body, model, n)
        except (ValueError, TypeError):
            continue
        nz = {j: t for j, t in terms.items() if abs(t) > 1e-12}
        if abs(float(const) + 1.0) > 1e-12 or any(abs(t - 1.0) > 1e-12 for t in nz.values()):
            continue
        rows.append(frozenset(nz))
    for node in todo:
        known = [col_of(d.indicator.variable) for d in node.disjuncts if d.indicator is not None]
        if any(j is None for j in known):
            continue
        want = frozenset(j for j in known if j is not None)
        hits = [r for r in rows if want < r and len(r) == len(want) + 1]
        if len(hits) != 1:
            continue
        (extra,) = hits[0] - want
        var, k = cols[extra]
        if var.var_type is not VarType.BINARY or var.size != 1:
            continue
        d = next(d for d in node.disjuncts if d.indicator is None)
        d.indicator, d.name = BooleanVar(var), var.name


def gdp_tree(model: Model) -> GDPTree:
    """Build the :class:`GDPTree` of ``model`` (read-only; allocates nothing)."""
    counter = [0]

    def next_id() -> int:
        counter[0] += 1
        return counter[0] - 1

    def from_either_or(
        dc: _DisjunctiveConstraint, parent: Optional[DisjunctNode], origin: Optional[Origin]
    ) -> DisjunctionNode:
        node = DisjunctionNode(next_id(), dc.name, "either_or", dc.semantics, [], parent, origin)
        inds = dc._indicator_vars
        for k, items in enumerate(dc.disjuncts):
            d = DisjunctNode(
                next_id(),
                f"{dc.name}[{k}]" if dc.name else None,
                BooleanVar(inds[k]) if inds is not None else None,
                1,
                [],
                parent=node,
            )
            for item in items:
                if isinstance(item, _DisjunctiveConstraint):
                    d.children.append(from_either_or(item, d, Origin(-1, item)))
                elif isinstance(item, Constraint):
                    d.rows.append(item)
                else:
                    raise TypeError(
                        f"disjunct item of type {type(item).__name__} in "
                        f"{dc.name or '<anon>'!r} is not a constraint or a disjunction"
                    )
            node.disjuncts.append(d)
        return node

    roots: list[DisjunctionNode] = []
    logic: list[_LogicalConstraint] = []
    groups: dict[str, DisjunctionNode] = {}
    last_implication: Optional[tuple[int, int, DisjunctionNode]] = None
    parents: dict[DisjunctionNode, Variable] = {}
    for ci, c in enumerate(model._constraints):
        if isinstance(c, _DisjunctiveConstraint):
            node = from_either_or(c, None, Origin(ci, c))
            roots.append(node)
            if c.parent_indicator is not None:
                parents[node] = c.parent_indicator
            last_implication = None
        elif isinstance(c, _IndicatorConstraint) and c.disjunction is not None:
            key, k, n = c.disjunction
            group = groups.get(key)
            if group is None:
                group = DisjunctionNode(
                    next_id(),
                    key,
                    "add_disjunction",
                    c.semantics,
                    [],
                    None,
                    Origin(ci, key),
                )
                # An empty disjunct carries no row, so its indicator is unknown here.
                group.disjuncts = [
                    DisjunctNode(next_id(), None, None, 1, [], parent=group) for _ in range(n)
                ]
                groups[key] = group
                roots.append(group)
                if c.parent is not None:
                    parents[group] = c.parent
            d = group.disjuncts[k]
            ind = BooleanVar(c.indicator)
            if d.indicator is None:
                d.indicator, d.active_value, d.name = ind, int(c.active_value), c.indicator.name
            elif d.indicator.variable is not c.indicator:
                raise ValueError(f"disjunct {k} of {key!r} has rows under two different indicators")
            d.rows.append(c.constraint)
            last_implication = None
        elif isinstance(c, _IndicatorConstraint):
            # One ``if_then`` call emits one row per constraint, consecutively and
            # under the same indicator; a run of such rows is one implication.
            if (
                last_implication is not None
                and last_implication[0] == id(c.indicator)
                and last_implication[1] == int(c.active_value)
            ):
                last_implication[2].disjuncts[0].rows.append(c.constraint)
                continue
            node = DisjunctionNode(
                next_id(),
                c.name,
                "implication",
                DisjunctionSemantics.SELECT_ONE,
                [],
                None,
                Origin(ci, c),
            )
            node.disjuncts.append(
                DisjunctNode(
                    next_id(),
                    c.name,
                    BooleanVar(c.indicator),
                    int(c.active_value),
                    [c.constraint],
                    parent=node,
                )
            )
            roots.append(node)
            last_implication = (id(c.indicator), int(c.active_value), node)
        elif isinstance(c, _LogicalConstraint):
            logic.append(c)
            last_implication = None
        else:
            last_implication = None

    # #1775 G1.2: fill the indicators of block disjuncts that emitted no row, then
    # move every root that names a parent indicator under the disjunct owning it.
    members = getattr(model, "_gdp_block_members", {})
    # A block disjunction none of whose disjuncts has a row of its own (each holds
    # only nested disjunctions, or nothing) is tagged on no row, so the loop above
    # never saw it (#1790); build its node from the emission record, at the place
    # in model order where its emission began.
    rowless: list[DisjunctionNode] = []
    for key, rec in members.items():
        if key in groups:
            continue
        group = DisjunctionNode(
            next_id(),
            key,
            "add_disjunction",
            rec.semantics,
            [],
            None,
            Origin(rec.start, key),
        )
        group.disjuncts = [
            DisjunctNode(next_id(), None, None, 1, [], parent=group) for _ in rec.indicators
        ]
        groups[key] = group
        rowless.append(group)
        if rec.parent is not None:
            parents[group] = rec.parent
    if rowless:
        # Stable, and a row-less node sorts before a root starting at the same index.
        # Every root built here has an origin; -1 would only ever sort a stray first.
        roots = sorted(rowless + roots, key=lambda r: r.origin.index if r.origin else -1)
    for key, node in groups.items():
        rec = members.get(key)
        for d, v in zip(node.disjuncts, rec.indicators if rec is not None else ()):
            if d.indicator is None:
                d.indicator, d.name = BooleanVar(v), v.name
    _fill_empty_block_indicators(model, groups)
    owner: dict[int, DisjunctNode] = {}
    for node in roots:
        for d in node.disjuncts:
            if d.indicator is not None:
                owner.setdefault(id(d.indicator.variable), d)
    for d0 in getattr(model, "_gdp_factory_disjuncts", ()):
        if d0._local_vars and id(d0.indicator.variable) in owner:
            owner[id(d0.indicator.variable)].local_vars = list(d0._local_vars)
    attached: list[DisjunctionNode] = []
    for node, p in parents.items():
        found = owner.get(id(p))
        if found is None:
            raise ValueError(
                f"disjunction {node.name!r} is nested in a disjunct whose indicator "
                f"{p.name!r} belongs to no disjunction of this model"
            )
        node.parent = found
        found.children.append(node)
        attached.append(node)
    roots = [r for r in roots if not any(r is a for a in attached)]

    tree = GDPTree(roots, logic, {})
    for node in tree.walk():
        for d in node.disjuncts:
            if d.indicator is not None:
                tree.indicators[d] = d.indicator
    return tree
