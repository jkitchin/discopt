"""Fixing disjuncts: the reduced subproblem and the current disjunctive state (#1775 G1.3).

A *selection* names, for some disjunctions of a model, the disjunct that is
selected -- as :class:`~discopt.gdp.tree.DisjunctNode` objects from
:func:`~discopt.gdp.tree.gdp_tree`. Relative to it every disjunction of the GDP
tree is in one of four states:

* **decided** -- active and a disjunct of it is selected (several, under OR
  semantics). The selected disjuncts' rows become ordinary rows, the others'
  rows are dropped, and every known indicator is pinned (1 selected, 0 not).
  Under OR, :func:`reduced_subproblem` reads ``selected`` as exactly the set
  enforced; :func:`transform_current_disjunctive_state` refuses a selection of
  only some of an OR disjunction's disjuncts, since the rest would be undecided
  rather than off.
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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from discopt.gdp.tree import DisjunctionNode, DisjunctNode, GDPTree, gdp_tree
from discopt.modeling.core import (
    Constraint,
    Model,
    SelectorCardinality,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    carry_validation_guards,
)

# A disjunct's address: the root disjunction's row index, then alternating
# (disjunct index, child index) steps, ending with the disjunct index. It is
# stable under ``Model.clone`` and ``create_using``'s deep copy, so a selection
# taken from one model's tree can be resolved on a copy's tree.
Address = tuple


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


def _resolve(tree: GDPTree, address: Address) -> DisjunctNode:
    roots = {r.origin.index: r for r in tree.disjunctions if r.origin is not None}
    node = roots.get(address[0])
    if node is None:
        raise ValueError(f"no disjunction at model row {address[0]}")
    steps: list[int] = [int(i) for i in address[1:]]
    while True:
        d: DisjunctNode = node.disjuncts[steps.pop(0)]
        if not steps:
            return d
        node = d.children[steps.pop(0)]


def _chosen(tree: GDPTree, selected: Iterable[DisjunctNode]) -> dict[int, set[int]]:
    """``id(DisjunctionNode) -> selected disjunct indices`` on ``tree``.

    One index per disjunction, except under OR semantics (#1775 G1.4), where
    several disjuncts may be selected together.
    """
    chosen: dict[int, set[int]] = {}
    for d in selected:
        if not isinstance(d, DisjunctNode):
            raise TypeError(
                f"a selection is DisjunctNode objects from gdp_tree(model), not {type(d).__name__}"
            )
        mine = _resolve(tree, _address(d))
        node = mine.parent
        assert node is not None
        k = next(i for i, x in enumerate(node.disjuncts) if x is mine)
        picked = chosen.setdefault(id(node), set())
        picked.add(k)
        if len(picked) > 1 and node.semantics.cardinality is not SelectorCardinality.AT_LEAST_ONE:
            raise ValueError(f"two disjuncts of {node.name!r} are selected")
    # A selected disjunct must be reachable: every enclosing disjunct selected.
    for node in tree.walk():
        if id(node) in chosen and node.parent is not None:
            outer = node.parent.parent
            assert outer is not None
            ks = chosen.get(id(outer), set())
            if not any(outer.disjuncts[k] is node.parent for k in ks):
                raise ValueError(
                    f"a disjunct of {node.name!r} is selected, but the disjunct it is "
                    "nested in is not; select the enclosing disjunct too"
                )
    return chosen


@dataclass
class _Plan:
    drop: set[int]  # id() of model rows (or row-holding objects) to remove
    add: list[Any]  # ordinary rows and promoted disjunctions, in order
    pins: list[tuple[Any, float]]  # (indicator expression, value)


def _plan(model: Model, selected: Iterable[DisjunctNode], *, full: bool) -> _Plan:
    tree = gdp_tree(model)
    chosen = _chosen(tree, selected)
    plan = _Plan(set(), [], [])
    indicator_rows = [c for c in model._constraints if isinstance(c, _IndicatorConstraint)]

    def drop_rows(node: DisjunctionNode) -> None:
        # An either_or / nested Model.disjunction is one object in the model; a
        # block or an implication is a set of indicator rows.
        if node.origin is not None and isinstance(node.origin.source, _DisjunctiveConstraint):
            plan.drop.add(id(node.origin.source))
        else:
            mine = {id(r) for d in node.disjuncts for r in d.rows}
            plan.drop.update(id(c) for c in indicator_rows if id(c.constraint) in mine)

    def pin(d: DisjunctNode, on: bool) -> None:
        if d.indicator is not None:
            value = d.active_value if on else 1 - d.active_value
            plan.pins.append((d.indicator.variable, float(value)))

    def visit(node: DisjunctionNode, active: bool) -> None:
        ks = chosen.get(id(node))
        if node.kind == "implication":
            if ks:
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
        if not ks:
            if full:
                raise ValueError(
                    f"no disjunct of {node.name!r} is selected; a reduced subproblem "
                    "needs one per active disjunction"
                )
            if not top_level:
                # Nested in a selected either_or disjunct: now unconditional.
                assert node.origin is not None
                plan.add.append(node.origin.source)
            return
        if (
            not full
            and node.semantics.cardinality is SelectorCardinality.AT_LEAST_ONE
            and len(ks) < len(node.disjuncts)
        ):
            # Under OR, a disjunct left out of the selection is undecided, not
            # off: pinning it to 0 would shrink the feasible set and certify a
            # different optimum. A partial OR state has no representation here.
            raise ValueError(
                f"only some disjuncts of the OR disjunction {node.name!r} are "
                "selected; the others stay undecided under OR semantics, which an "
                "in-place transform cannot represent. Select all of them, or use "
                "reduced_subproblem for an exact selection"
            )
        if top_level:
            drop_rows(node)
        for j, d in enumerate(node.disjuncts):
            pin(d, j in ks)
            if j in ks:
                plan.add.extend(d.rows)
            for child in d.children:
                visit(child, j in ks)

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
    ``model`` is not changed.
    """
    plan = _plan(model, selected, full=True)
    out = Model(model.name)
    out._variables = list(model._variables)
    out._parameters = list(model._parameters)
    out._rebuild_name_index()
    out._objective = model._objective
    out._constraints = [c for c in model._constraints if id(c) not in plan.drop]
    out._constraints.extend(plan.add)
    out._constraints.extend(_pin_row(e, v) for e, v in plan.pins)
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
    after = [c for c in before if id(c) not in plan.drop]
    after.extend(plan.add)
    after.extend(_pin_row(e, v) for e, v in plan.pins)
    model._constraints = after
    return DisjunctiveStateToken(model, before, list(after))


def _transform_current_disjunctive_state_in_place(
    model: Model, selected: Optional[Iterable[DisjunctNode]] = None
) -> None:
    """Registry adapter: the in-place contract returns nothing; the undo token is
    kept on the model for :func:`reverse_current_disjunctive_state`."""
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
