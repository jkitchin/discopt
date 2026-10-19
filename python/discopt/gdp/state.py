"""Fixing disjuncts: the reduced subproblem and the current disjunctive state (#1775 G1.3).

A *selection* names, for some disjunctions of a model, the disjunct that is
selected -- as :class:`~discopt.gdp.tree.DisjunctNode` objects from
:func:`~discopt.gdp.tree.gdp_tree`. Relative to it every disjunction of the GDP
tree is in one of four states:

* **decided** -- active and a disjunct of it is selected. That disjunct's rows
  become ordinary rows, the other disjuncts' rows are dropped, and every known
  indicator is pinned (1 for the selected disjunct, 0 for the others).
* **inactive** -- nested under a disjunct that is not selected. All of its rows
  are dropped and its known indicators are pinned to 0.
* **undecided** -- active, nothing selected. It stays a disjunction. A
  disjunction nested in a selected ``either_or`` disjunct is promoted to the top
  level, since it now applies unconditionally.
* an ``if_then`` (implication) node is decided by selecting its one disjunct
  (its rows become ordinary rows and its indicator is pinned to its active
  value); left unselected, its rows stay indicator rows.

Indicators are pinned with equality rows, never by changing a variable's bounds,
so no shared :class:`Variable` is mutated. Logical constraints are kept: over
pinned indicators they are either satisfied or make the model infeasible.

Two entry points, both registered in :mod:`discopt.transformations`:

* :func:`reduced_subproblem` (``"gdp.fix_disjuncts"``) -- spec §5.2. It needs a
  **full** configuration (every active disjunction other than an implication
  decided) and returns a new model sharing the input's variables. Dropping the
  inactive rows, rather than relaxing them with a big-M, is what makes a
  logic-based algorithm logic-based (Türkay & Grossmann 1996): an inactive row
  with a domain-restricted function is never evaluated. Variables local to an
  inactive disjunct (``Disjunct.variable``) are then unconstrained except by
  their bounds; their value in a solution means nothing.
* :func:`transform_current_disjunctive_state`
  (``"gdp.transform_current_disjunctive_state"``) -- Pyomo's transformation of
  the same name. It works **in place**, accepts a partial selection (by default
  the indicators currently fixed at 1, i.e. ``lb == ub == 1``), and returns a
  :class:`DisjunctiveStateToken` whose :meth:`~DisjunctiveStateToken.reverse`
  restores the model exactly.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from discopt.gdp.tree import DisjunctionNode, DisjunctNode, GDPTree, gdp_tree
from discopt.modeling.core import (
    Constraint,
    Model,
    Objective,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    carry_validation_guards,
)

# A disjunct's address: the root disjunction's row index, then alternating
# (disjunct index, child index) steps, ending with the disjunct index. It is
# stable under ``Model.clone`` and ``create_using``'s deep copy, so a selection
# taken from one model's tree can be resolved on a copy's tree. An address alone
# is not trusted (#1794 S2): every disjunction on the path must also match the
# one it resolves to (:func:`_signature`), so a node from a stale tree or another
# model is refused rather than silently resolved to a different disjunction.
Address = tuple


def _signature(node: DisjunctionNode) -> tuple:
    """What a disjunction is, independent of object identity: kind, name, and per
    disjunct its name, rows (as text) and number of nested disjunctions."""
    return (
        node.kind,
        node.name,
        tuple((d.name, tuple(str(r) for r in d.rows), len(d.children)) for d in node.disjuncts),
    )


def _path(d: DisjunctNode) -> list[DisjunctionNode]:
    """The disjunctions from ``d``'s root down to ``d``'s own."""
    out: list[DisjunctionNode] = []
    cur: Optional[DisjunctNode] = d
    while cur is not None:
        node = cur.parent
        if node is None:
            raise ValueError(f"{cur!r} is not part of a GDP tree")
        out.append(node)
        cur = node.parent
    return out[::-1]


def _address(d: DisjunctNode) -> Address:
    path: list[Any] = []
    cur = d
    while True:
        node = cur.parent
        if node is None:
            raise ValueError(f"{cur!r} is not part of a GDP tree")
        path.append(next(i for i, x in enumerate(node.disjuncts) if x is cur))
        if node.parent is None:
            if node.origin is None or node.origin.index < 0:
                raise ValueError(f"{node!r} is a root without a model row")
            path.append(node.origin.index)
            return tuple(reversed(path))
        parent = node.parent
        path.append(next(i for i, x in enumerate(parent.children) if x is node))
        cur = parent


def _resolve(tree: GDPTree, selected: DisjunctNode) -> DisjunctNode:
    """``selected``'s counterpart on ``tree``, refused unless the structure matches."""
    address = _address(selected)
    expected = _path(selected)
    stale = ValueError(
        f"{selected!r} is not a disjunct of this model's current GDP tree (it comes from "
        "another model, or from a tree taken before the model was changed); take the "
        "selection from gdp_tree(model)"
    )
    # A row-less block disjunction (#1790) starts where its first nested row does,
    # so two roots can share an index: pick the one whose structure matches, and
    # refuse when that is still ambiguous (#1794 S3).
    want = _signature(expected[0])
    candidates = [
        r
        for r in tree.disjunctions
        if r.origin is not None and r.origin.index == address[0] and _signature(r) == want
    ]
    if not candidates:
        raise stale
    if len(candidates) > 1:
        raise ValueError(
            f"{len(candidates)} identical disjunctions start at model row {address[0]}; "
            f"{selected!r} cannot be resolved to one of them"
        )
    node = candidates[0]
    steps: list[int] = [int(i) for i in address[1:]]
    depth = 0
    while True:
        if _signature(node) != _signature(expected[depth]):
            raise stale
        k = steps.pop(0)
        if k >= len(node.disjuncts):
            raise stale
        d: DisjunctNode = node.disjuncts[k]
        if not steps:
            return d
        j = steps.pop(0)
        if j >= len(d.children):
            raise stale
        node = d.children[j]
        depth += 1


def _chosen(tree: GDPTree, selected: Iterable[DisjunctNode]) -> dict[int, int]:
    """``id(DisjunctionNode) -> selected disjunct index`` on ``tree``."""
    chosen: dict[int, int] = {}
    for d in selected:
        if not isinstance(d, DisjunctNode):
            raise TypeError(
                f"a selection is DisjunctNode objects from gdp_tree(model), not {type(d).__name__}"
            )
        mine = _resolve(tree, d)
        node = mine.parent
        assert node is not None
        k = next(i for i, x in enumerate(node.disjuncts) if x is mine)
        if chosen.get(id(node), k) != k:
            raise ValueError(f"two disjuncts of {node.name!r} are selected")
        chosen[id(node)] = k
    # A selected disjunct must be reachable: every enclosing disjunct selected.
    for node in tree.walk():
        if id(node) in chosen and node.parent is not None:
            outer = node.parent.parent
            assert outer is not None
            k_outer = chosen.get(id(outer))
            if k_outer is None or outer.disjuncts[k_outer] is not node.parent:
                raise ValueError(
                    f"a disjunct of {node.name!r} is selected, but the disjunct it is "
                    "nested in is not; select the enclosing disjunct too"
                )
    return chosen


@dataclass
class _Plan:
    drop: set[int]  # indices into model._constraints of rows to remove
    add: list[Any]  # ordinary rows and promoted disjunctions, in order
    pins: list[tuple[Any, float]]  # (indicator expression, value)
    # index -> replacement row: a disjunction nested in a selected Disjunct block,
    # left undecided, is re-emitted without its (now pinned) parent link.
    replace: dict[int, Any] = field(default_factory=dict)
    # ``_gdp_block_members`` keys of block disjunctions whose rows are gone.
    retired_keys: set[str] = field(default_factory=set)
    promoted_keys: set[str] = field(default_factory=set)

    def rows(self, model: Model) -> list[Any]:
        out = [
            self.replace.get(i, c) for i, c in enumerate(model._constraints) if i not in self.drop
        ]
        out.extend(self.add)
        out.extend(_pin_row(e, v) for e, v in self.pins)
        return out


def _plan(model: Model, selected: Iterable[DisjunctNode], *, full: bool) -> _Plan:
    # A model that would not validate is refused before it is transformed: the
    # transformation must not turn a forgotten block (#1430) into a silent drop.
    model._reject_unattached_gdp_blocks()
    tree = gdp_tree(model)
    chosen = _chosen(tree, selected)
    plan = _Plan(set(), [], [])
    rows = model._constraints
    by_key: dict[str, list[int]] = {}
    for i, c in enumerate(rows):
        if isinstance(c, _IndicatorConstraint) and c.disjunction is not None:
            by_key.setdefault(c.disjunction[0], []).append(i)

    def own_rows(node: DisjunctionNode) -> list[int]:
        """Indices of the model rows that make up ``node`` -- by row identity, never
        by the wrapped constraint, which another disjunction or an ``if_then`` may
        share (#1794 B1)."""
        assert node.origin is not None
        if isinstance(node.origin.source, _DisjunctiveConstraint):
            return [node.origin.index]
        if node.kind == "add_disjunction":
            return by_key.get(node.origin.source, [])
        # An implication is the run of consecutive rows starting at its origin.
        n = len(node.disjuncts[0].rows)
        run = list(range(node.origin.index, node.origin.index + n))
        for i, r in zip(run, node.disjuncts[0].rows):
            row: Any = rows[i]
            if not (isinstance(row, _IndicatorConstraint) and row.constraint is r):
                raise RuntimeError(f"implication {node.name!r} is not the row run gdp_tree read")
        return run

    def drop_rows(node: DisjunctionNode) -> None:
        plan.drop.update(own_rows(node))
        if node.kind == "add_disjunction":
            assert node.origin is not None
            plan.retired_keys.add(node.origin.source)

    def promote(node: DisjunctionNode) -> None:
        # Nested in a selected Disjunct block, whose indicator is now pinned to 1:
        # the parent link is satisfied, so re-emit the rows without it. Copies,
        # never mutation: the token's ``before`` keeps the original rows.
        for i in own_rows(node):
            c: Any = copy.copy(rows[i])
            if isinstance(c, _DisjunctiveConstraint):
                c.parent_indicator = None
            else:
                c.parent = None
            plan.replace[i] = c
        if node.kind == "add_disjunction":
            assert node.origin is not None
            plan.promoted_keys.add(node.origin.source)

    def pin(d: DisjunctNode, on: bool) -> None:
        if d.indicator is not None:
            value = d.active_value if on else 1 - d.active_value
            plan.pins.append((d.indicator.variable, float(value)))

    def visit(node: DisjunctionNode, active: bool) -> None:
        k = chosen.get(id(node))
        if node.kind == "implication":
            if k is not None:
                drop_rows(node)
                plan.add.extend(node.disjuncts[0].rows)
                pin(node.disjuncts[0], True)
            return
        top_level = node.origin is not None and node.origin.index >= 0
        if not active:
            if top_level:
                drop_rows(node)
            for d in node.disjuncts:
                pin(d, False)
                for child in d.children:
                    visit(child, False)
            return
        if k is None:
            if full:
                raise ValueError(
                    f"no disjunct of {node.name!r} is selected; a reduced subproblem "
                    "needs one per active disjunction"
                )
            if not top_level:
                # Nested in a selected either_or disjunct: now unconditional.
                assert node.origin is not None
                plan.add.append(node.origin.source)
            elif node.parent is not None:
                promote(node)
            return
        if top_level:
            drop_rows(node)
        for j, d in enumerate(node.disjuncts):
            pin(d, j == k)
            if j == k:
                plan.add.extend(d.rows)
            for child in d.children:
                visit(child, j == k)

    for root in tree.disjunctions:
        visit(root, True)
    return plan


def _pin_row(expr: Any, value: float) -> Constraint:
    row = expr == value
    assert isinstance(row, Constraint)
    return row


def reduced_subproblem(model: Model, selected: Iterable[DisjunctNode]) -> Model:
    """The model with every disjunction fixed by ``selected`` (spec §5.2).

    ``selected`` must decide every active disjunction (implications may be left
    alone). Returns a new model sharing ``model``'s variables: global rows, the
    selected disjuncts' rows as ordinary rows, logical constraints, and one
    equality row per known indicator. The inactive disjuncts' rows are gone.
    Fast-API rows (``add_linear_constraints``) become ordinary rows, a fast-API
    objective an ordinary objective, and the complementarity records and
    ``Disjunct`` blocks are carried. ``model`` is not changed.
    """
    from discopt.mpec import carry_complementarities

    plan = _plan(model, selected, full=True)
    out = Model(model.name)
    out._variables = list(model._variables)
    out._parameters = list(model._parameters)
    out._rebuild_name_index()
    out._objective = model._objective
    if model._objective is not None and getattr(model._objective, "_is_placeholder", False):
        # ``add_linear_objective`` / ``add_quadratic_objective`` keep the real
        # objective in the source's Rust builder (#1794 B2).
        out._objective = Objective(model._objective_expression(), model._objective.sense)
    out._constraints = plan.rows(model)
    # Fast-API rows live only in the source's builder; carry them as ordinary
    # rows, or the subproblem silently loses them (#1794 B2).
    out._constraints.extend(model._builder_linear_constraints())
    out._gdp_factory_disjuncts = list(model._gdp_factory_disjuncts)
    carry_complementarities(model, out, pass_name="gdp.fix_disjuncts")
    carry_validation_guards(model, out)
    return out


#: Pyomo's name for the same transformation (``TransformationFactory("gdp.fix_disjuncts")``).
fix_disjuncts = reduced_subproblem


@dataclass
class DisjunctiveStateToken:
    """Undo record of :func:`transform_current_disjunctive_state`."""

    model: Model
    before: list
    after: list
    #: ``_gdp_factory_disjunctions`` and ``_gdp_block_members`` before the
    #: transformation (#1794 S1/S3), restored by :meth:`reverse`.
    factory_before: list = field(default_factory=list)
    members_before: Optional[dict] = None

    def reverse(self) -> Model:
        """Restore the model's rows to what they were before the transformation.

        Refused when the rows were changed since: restoring would silently drop
        that change.
        """
        rows = self.model._constraints
        if len(rows) != len(self.after) or any(a is not b for a, b in zip(rows, self.after)):
            raise ValueError(
                "the model's rows changed after transform_current_disjunctive_state; "
                "reversing would drop those changes"
            )
        self.model._constraints = list(self.before)
        reg = self.model._gdp_factory_disjunctions
        later = [dc for dc in reg if not any(dc is f for f in self.factory_before)]
        self.model._gdp_factory_disjunctions = list(self.factory_before) + later
        if self.members_before is None:
            self.model.__dict__.pop("_gdp_block_members", None)
        else:
            self.model.__dict__["_gdp_block_members"] = dict(self.members_before)
        return self.model


def _fixed_selection(model: Model) -> list[DisjunctNode]:
    """Disjuncts whose indicator is fixed at its active value (``lb == ub``)."""
    from discopt.validation.feasibility import _column_indices

    lb = np.concatenate([np.ravel(np.asarray(v.lb, dtype=np.float64)) for v in model._variables])
    ub = np.concatenate([np.ravel(np.asarray(v.ub, dtype=np.float64)) for v in model._variables])
    out: list[DisjunctNode] = []
    for node in gdp_tree(model).walk():
        for d in node.disjuncts:
            if d.indicator is None:
                continue
            cols = _column_indices(d.indicator.variable, model)
            if cols.size == 1 and lb[cols[0]] == ub[cols[0]] == d.active_value:
                out.append(d)
    return out


def transform_current_disjunctive_state(
    model: Model, selected: Optional[Iterable[DisjunctNode]] = None
) -> DisjunctiveStateToken:
    """Apply a (partial) selection to ``model`` in place; return the undo token.

    ``selected`` defaults to the disjuncts whose indicator is currently fixed at
    1 (Pyomo's reading of "current state"). Undecided disjunctions stay
    disjunctions; see the module docstring for the four states.
    """
    chosen = _fixed_selection(model) if selected is None else list(selected)
    plan = _plan(model, chosen, full=False)
    before = list(model._constraints)
    after = plan.rows(model)
    reachable_before = _reachable_disjunctions(before)
    reachable_after = _reachable_disjunctions(after)
    factory_before = list(model._gdp_factory_disjunctions)
    # A Model.disjunction the transformation decided or dropped is consumed, not
    # forgotten: without this, validate (#1430) refuses the transformed model.
    consumed = [
        dc for dc in factory_before if id(dc) in reachable_before and id(dc) not in reachable_after
    ]
    members = model.__dict__.get("_gdp_block_members")
    model._constraints = after
    model._gdp_factory_disjunctions = [
        dc for dc in factory_before if not any(dc is c for c in consumed)
    ]
    if members is not None:
        # Keep the nesting record in step with the rows (#1794 S3): a decided or
        # dropped block disjunction is gone, a promoted one has no parent now.
        new_members = {k: v for k, v in members.items() if k not in plan.retired_keys}
        for k in plan.promoted_keys & set(new_members):
            rec = new_members[k]
            if hasattr(rec, "_replace") and getattr(rec, "parent", None) is not None:
                new_members[k] = rec._replace(parent=None)
        model.__dict__["_gdp_block_members"] = new_members
    return DisjunctiveStateToken(
        model,
        before,
        list(after),
        factory_before=factory_before,
        members_before=None if members is None else dict(members),
    )


def _reachable_disjunctions(rows: list) -> set[int]:
    """``id()`` of every ``_DisjunctiveConstraint`` reachable from ``rows``."""
    seen: set[int] = set()
    stack = [c for c in rows if isinstance(c, _DisjunctiveConstraint)]
    while stack:
        dc = stack.pop()
        if id(dc) in seen:
            continue
        seen.add(id(dc))
        for group in dc.disjuncts:
            stack.extend(it for it in group if isinstance(it, _DisjunctiveConstraint))
    return seen


def _transform_current_disjunctive_state_in_place(
    model: Model, selected: Optional[Iterable[DisjunctNode]] = None
) -> None:
    """Registry adapter: the in-place contract returns nothing; the undo token is
    kept on the model for :func:`reverse_current_disjunctive_state`.

    The token lives in ``model._gdp_state_tokens``, which :mod:`discopt.serialize`
    does not record, so :meth:`Model.save` refuses the model (``SerializationError``)
    until every transformation is reversed: saving would reload a model that can no
    longer be reversed. ``pickle`` and ``Model.clone`` carry the tokens.
    """
    token = transform_current_disjunctive_state(model, selected)
    model.__dict__.setdefault("_gdp_state_tokens", []).append(token)


def reverse_current_disjunctive_state(model: Model) -> Model:
    """Undo the latest ``gdp.transform_current_disjunctive_state`` applied through
    :mod:`discopt.transformations`."""
    tokens = model.__dict__.get("_gdp_state_tokens") or []
    if not tokens:
        raise ValueError("no transform_current_disjunctive_state to reverse on this model")
    tokens[-1].reverse()
    tokens.pop()
    return model


__all__ = [
    "DisjunctiveStateToken",
    "fix_disjuncts",
    "reduced_subproblem",
    "reverse_current_disjunctive_state",
    "transform_current_disjunctive_state",
]
