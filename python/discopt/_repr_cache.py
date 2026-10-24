"""A read-only memo of ``model_to_repr`` for the passes that only query the repr.

THE COST.  ``model_to_repr`` marshals the whole Python expression DAG into the
Rust IR.  On ``telecomsp_nor_sun`` (21,270 rows, 13.2 M tree nodes) one call is
~12 s, and a single ``solve(time_limit=60)`` made six of them on the *same,
unchanged* model: the convex kernel's linearity pre-check, the arena-tape build,
the scaling diagnostics and three problem classifications -- 74 s of marshalling,
more than the whole time limit, before any search.

THE CONTRACT.  :func:`readonly_model_repr` returns a repr that callers may
*query* and must not *mutate* (``tighten_var_bounds``, ``add_variable``,
``add_linear_constraints``, ``set_*_objective``).  Every call site using it was
audited for that; the root presolve, which does mutate, keeps calling
``model_to_repr`` directly.  A shared repr mutated by one caller would leak into
the next, so a site that ever needs to mutate must not use this.

THE KEY is everything ``model_to_repr`` reads from Python, by value where the
object is mutable in place and by identity where it is not (expression nodes are
never mutated in place; every rewrite builds new nodes):

* variables: identity, order, ``name``, ``var_type`` (implied-integer detection
  rewrites it in place), ``shape``, the ``lb``/``ub`` *values* (B&B rebinds them),
  ``_is_placeholder``, ``_builder_idx``;
* parameters: identity, order, the ``value`` (written through in place);
* constraints: identity, order, and each one's ``body`` identity, ``sense``,
  ``rhs`` value and ``name``;
* the objective: identity, ``expression`` identity, ``sense``.

A model with a Rust builder, fast-API rows, or any constraint that is not a
plain :class:`Constraint` (indicator, disjunction, logical) is never cached:
those carry state outside this key.  A miss costs exactly what the call cost
before, so the only risk is a false hit, and the key is what rules that out.
The entry pins every object whose ``id`` is in the key, so an id cannot be
recycled onto a different object while the entry lives.

The memo lives on the model as ``_readonly_repr_cache``, which ``copy`` and
``pickle`` drop like every other ``*_cache`` attribute.
"""

from __future__ import annotations

from typing import Any

from discopt._evaluator_cache import _value_bytes
from discopt.modeling.core import Constraint, Model

__all__ = ["readonly_model_repr"]

_ATTR = "_readonly_repr_cache"


def _state_key(model: Model) -> tuple[tuple, tuple] | None:
    """``(key, pins)`` for *model*, or ``None`` when it must not be cached."""
    if getattr(model, "_builder", None) is not None:
        return None
    if getattr(model, "_builder_linear_blocks", None):
        return None
    constraints = tuple(model._constraints)
    if not all(type(c) is Constraint for c in constraints):
        return None
    variables = tuple(model._variables)
    parameters = tuple(model._parameters)
    objective = model._objective
    key = (
        tuple(
            (
                id(v),
                v.name,
                v.var_type,
                v.shape,
                _value_bytes(v.lb),
                _value_bytes(v.ub),
                getattr(v, "_is_placeholder", None),
                getattr(v, "_builder_idx", None),
            )
            for v in variables
        ),
        tuple((id(p), _value_bytes(p.value)) for p in parameters),
        tuple((id(c), id(c.body), c.sense, _value_bytes(c.rhs), c.name) for c in constraints),
        None
        if objective is None
        else (id(objective), id(objective.expression), str(objective.sense)),
    )
    pins = (
        variables,
        parameters,
        constraints,
        tuple(c.body for c in constraints),
        objective,
        None if objective is None else objective.expression,
    )
    return key, pins


def readonly_model_repr(model: Model) -> Any:
    """``model_to_repr(model, model._builder)``, memoised on the model's state.

    The returned repr is shared: query it, never mutate it (see the module
    docstring).  Exceptions from ``model_to_repr`` propagate unchanged and are
    never cached.
    """
    from discopt._rust import model_to_repr

    state = _state_key(model)
    if state is None:
        return model_to_repr(model, getattr(model, "_builder", None))
    key, pins = state
    entry = getattr(model, _ATTR, None)
    if entry is not None and entry[0] == key:
        return entry[2]
    repr_ = model_to_repr(model, None)
    setattr(model, _ATTR, (key, pins, repr_))
    return repr_
