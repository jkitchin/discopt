"""``gdp.basic_step``: the Cartesian product of disjunctions (#1775 G2.9).

Pyomo's ``apply_basic_step`` (Ruiz & Grossmann). Two or more SELECT_ONE
disjunctions are replaced by one disjunction with a disjunct per combination
``(j_1, ..., j_k)`` -- the rows of disjunct ``j_1`` of the first, ``j_2`` of the
second, and so on. The feasible set is unchanged: exactly one disjunct of each
original is selected iff exactly one combination is. The hull of the product is
at least as tight as the intersection of the hulls (the point of a basic step);
the price is ``prod_i n_i`` disjuncts.

When the indicators of an original disjunction were read (to be used in logic,
fixed or priced), they stay in the model and are tied to the new selectors:
``y_{i,j} = sum of Y over the combinations whose i-th entry is j``.

Refused, loudly: fewer than two disjunctions, one that is not a top-level
``either_or`` of this model (or appears twice), OR semantics (the product of OR
disjunctions is not a SELECT_ONE disjunction), a disjunct holding a nested
disjunction, a user ``big_m`` (its rows are copied into many disjuncts), differing
``method`` overrides, and a product with more than ``max_disjuncts`` disjuncts.
Atomic: a refusal leaves the model unchanged.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional, Sequence

from discopt.modeling.core import (
    Constraint,
    DisjunctionSemantics,
    Model,
    _DisjunctiveConstraint,
)


def _copy_row(r: Constraint) -> Constraint:
    # A fresh object per combination: lowerings and serialization treat rows by
    # identity, and one row now sits in many disjuncts.
    return Constraint(body=r.body, sense=r.sense, rhs=r.rhs, name=r.name)


def apply_basic_step(
    model: Model,
    disjunctions: Sequence[_DisjunctiveConstraint],
    *,
    name: Optional[str] = None,
    max_disjuncts: int = 1000,
) -> _DisjunctiveConstraint:
    """Replace ``disjunctions`` by their Cartesian product, in place; return it.

    See the module docstring for what is refused. The new disjunction takes the
    place of the first of ``disjunctions`` in ``model._constraints``.
    """
    dcs = list(disjunctions)
    if len(dcs) < 2:
        raise ValueError("gdp.basic_step: give at least two disjunctions to combine.")
    if len({id(d) for d in dcs}) != len(dcs):
        raise ValueError("gdp.basic_step: a disjunction is listed twice.")
    position = {id(c): i for i, c in enumerate(model._constraints)}
    methods = set()
    size = 1
    for dc in dcs:
        label = dc.name or "<anon>"
        if not isinstance(dc, _DisjunctiveConstraint) or id(dc) not in position:
            raise ValueError(
                f"gdp.basic_step: {label!r} is not an either_or disjunction of this model."
            )
        if dc.parent_indicator is not None:
            raise ValueError(f"gdp.basic_step: {label!r} is nested in a Disjunct block.")
        if dc.semantics is not DisjunctionSemantics.SELECT_ONE:
            raise NotImplementedError(
                f"gdp.basic_step: {label!r} has DisjunctionSemantics.{dc.semantics.name}; "
                "the product of disjunctions is a select-one disjunction only when "
                "each factor is."
            )
        if any(isinstance(it, _DisjunctiveConstraint) for d in dc.disjuncts for it in d):
            raise ValueError(f"gdp.basic_step: {label!r} contains a nested disjunction.")
        if dc.big_m is not None:
            raise ValueError(
                f"gdp.basic_step: {label!r} carries a user big_m; its rows are copied into "
                "many disjuncts, where that M was not checked. Remove big_m first."
            )
        methods.add(dc.method)
        size *= len(dc.disjuncts)
    if len(methods) > 1:
        raise ValueError(
            "gdp.basic_step: the disjunctions declare different methods "
            f"{sorted(map(str, methods))}."
        )
    if size > max_disjuncts:
        raise ValueError(
            f"gdp.basic_step: the product has {size} disjuncts, more than "
            f"max_disjuncts={max_disjuncts}."
        )

    # Phase 1: build. Nothing here touches the model.
    combos = list(itertools.product(*(range(len(dc.disjuncts)) for dc in dcs)))
    disjuncts: list[list[Any]] = [
        [_copy_row(r) for i, j in enumerate(combo) for r in dcs[i].disjuncts[j]] for combo in combos
    ]
    new = _DisjunctiveConstraint(
        disjuncts=disjuncts,
        name=name or "_x_".join(dc.name or "anon" for dc in dcs),
        method=next(iter(methods)),
        _model=model,
    )
    keep = [c for c in model._constraints if id(c) not in {id(d) for d in dcs}]
    first = min(position[id(d)] for d in dcs)
    insert_at = sum(1 for c in model._constraints[:first] if id(c) not in {id(d) for d in dcs})
    read = [(i, dc) for i, dc in enumerate(dcs) if dc._indicator_vars is not None]

    # Phase 2: commit.
    rows: list[Any] = [*keep[:insert_at], new, *keep[insert_at:]]
    model._constraints = rows
    if read:
        sel = [b.variable for b in new.indicators]  # allocates them + exactly(1)
        for i, dc in read:
            assert dc._indicator_vars is not None
            for j, y in enumerate(dc._indicator_vars):
                total: Any = None
                for c, combo in enumerate(combos):
                    if combo[i] == j:
                        total = sel[c] if total is None else total + sel[c]
                model._constraints.append(y - total == 0)
    return new


def _basic_step_in_place(model: Model, **options: Any) -> None:
    """The registry's in-place entry point (``transformations``)."""
    apply_basic_step(model, **options)


__all__ = ["apply_basic_step"]
