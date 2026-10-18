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
(reading :attr:`_DisjunctiveConstraint.indicators` is what allocates them). Nested
disjunctions always have ``indicator=None`` until #1775 G1.2.
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
    """Where a root disjunction came from: the index of its first row in
    ``model._constraints`` and the source object (the ``_DisjunctiveConstraint``,
    or the ``add_disjunction`` key / first ``_IndicatorConstraint``)."""

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
        """The root built from ``source`` (a ``_DisjunctiveConstraint`` or a key)."""
        for root in self.disjunctions:
            if root.origin is not None and root.origin.source is source:
                return root
        raise KeyError(f"no root disjunction was built from {source!r}")


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
                    d.children.append(from_either_or(item, d, None))
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
    for ci, c in enumerate(model._constraints):
        if isinstance(c, _DisjunctiveConstraint):
            roots.append(from_either_or(c, None, Origin(ci, c)))
            last_implication = None
        elif isinstance(c, _IndicatorConstraint) and c.disjunction is not None:
            key, k, n = c.disjunction
            node = groups.get(key)
            if node is None:
                node = DisjunctionNode(
                    next_id(),
                    key,
                    "add_disjunction",
                    DisjunctionSemantics.SELECT_ONE,
                    [],
                    None,
                    Origin(ci, key),
                )
                # An empty disjunct carries no row, so its indicator is unknown here.
                node.disjuncts = [
                    DisjunctNode(next_id(), None, None, 1, [], parent=node) for _ in range(n)
                ]
                groups[key] = node
                roots.append(node)
            d = node.disjuncts[k]
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

    tree = GDPTree(roots, logic, {})
    for node in tree.walk():
        for d in node.disjuncts:
            if d.indicator is not None:
                tree.indicators[d] = d.indicator
    return tree
