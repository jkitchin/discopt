"""Memo of structure-only analyses (linearity, problem class) across one model's solve.

THE COST.  A solve asks "is this model linear?" and "what problem class is it?"
several times on the same, unchanged model: the HiGHS-route pre-check, the
dispatcher and the MILP engine entry each re-run ``_milp_is_exactly_linear``
(a ``model_to_repr`` marshal plus a term-classifier walk), and
``classify_problem`` runs four times. On ``tbfp-network`` read from a file
(72,747 scalar variables, one linear-row block, a 72k-term objective) that was
2.6 s of linearity checks and 1.6 s of classification inside a 20 s budget
(measured 2026-11-14), all answering the same question.

THE KEY is everything the cached answers read, so a hit is the answer the
analysis would give:

* variable identity and order (an added or replaced variable misses), and the
  ``var_type`` of each (``classify_problem`` reads integrality);
* parameter identity and value (a zero parameter can empty a quadratic form);
* constraint identity, order and ``body`` identity (expression nodes are never
  mutated in place; every rewrite builds new nodes);
* the objective's identity, expression identity and placeholder flag;
* the Rust builder's identity, its number of linear row blocks (append-only), and
  the identity of its linear / quadratic objective records (replaced, never
  mutated, by ``add_linear_objective`` / ``add_quadratic_objective``).

Bounds are not in the key: neither analysis reads them. A model with any
constraint that is not a plain :class:`Constraint` (indicator, disjunction,
logical) is never memoised, as in :mod:`discopt._repr_cache`. The entry pins
every object whose ``id`` is in the key, so an id cannot be recycled onto a
different object while the entry lives. The memo lives on the model as
``_structure_memo_cache``, which ``copy`` and ``pickle`` drop.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, cast

from discopt._evaluator_cache import _value_bytes, id_column
from discopt.modeling.core import Constraint, Model

_T = TypeVar("_T")

_ATTR = "_structure_memo_cache"


def _state_key(model: Model) -> tuple[tuple, tuple] | None:
    constraints = tuple(model._constraints)
    if not all(type(c) is Constraint for c in constraints):
        return None
    variables = tuple(model._variables)
    parameters = tuple(model._parameters)
    objective = model._objective
    expr = None if objective is None else objective.expression
    builder = getattr(model, "_builder", None)
    lin_obj = getattr(model, "_builder_linear_objective", None)
    quad_obj = getattr(model, "_builder_quadratic_objective", None)
    bodies = tuple(c.body for c in constraints)
    key = (
        # Packed id columns (#2293); every object they name is pinned below.
        id_column(variables),
        tuple(v.var_type for v in variables),
        tuple((id(p), _value_bytes(p.value)) for p in parameters),
        id_column(constraints),
        id_column(bodies),
        id(objective),
        id(expr),
        None if objective is None else getattr(objective, "_is_placeholder", None),
        id(builder),
        len(getattr(model, "_builder_linear_blocks", None) or ()),
        id(lin_obj),
        id(quad_obj),
    )
    # `bodies` is pinned too. The key holds their ids, and a body can be replaced
    # in place (the #2090 like-term pass does it), which would free the old body
    # and let a new expression reuse its id while this entry lives (#1329).
    pins = (variables, parameters, constraints, bodies, objective, expr, builder, lin_obj, quad_obj)
    return key, pins


def structure_memo(model: Model, name: str, compute: Callable[[Model], _T]) -> _T:
    """``compute(model)``, memoised under *name* on the model's structure (see the
    module docstring). Exceptions propagate and are never cached."""
    state = _state_key(model)
    if state is None:
        return compute(model)
    key, pins = state
    memo = getattr(model, _ATTR, None)
    if memo is None or memo[0] != key:
        memo = (key, pins, {})
        setattr(model, _ATTR, memo)
    values: dict[str, Any] = memo[2]
    if name not in values:
        values[name] = compute(model)
    return cast(_T, values[name])
