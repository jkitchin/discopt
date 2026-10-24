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

import numpy as np

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
    # Column-wise, not one small tuple per variable/row: on
    # ``acopf_caseactivsg25k_qcqp`` (328k variables, 365k rows) the per-object
    # tuples cost 4.4 s -- each is a GC-tracked allocation, and every collection
    # rescans a heap holding the whole expression graph -- against ~1 s for this.
    # Bounds are compared as one float64 vector (``+ 0.0`` folds ``-0.0`` onto
    # ``+0.0``, as ``_value_bytes`` does); the shapes column makes the
    # concatenation unambiguous.
    if variables:
        lb = np.concatenate([np.ravel(np.asarray(v.lb, dtype=np.float64)) for v in variables])
        ub = np.concatenate([np.ravel(np.asarray(v.ub, dtype=np.float64)) for v in variables])
        bounds = ((lb + 0.0).tobytes(), (ub + 0.0).tobytes())
    else:
        bounds = (b"", b"")
    key = (
        tuple(map(id, variables)),
        tuple(v.name for v in variables),
        tuple(v.var_type for v in variables),
        tuple(v.shape for v in variables),
        tuple(getattr(v, "_is_placeholder", None) for v in variables),
        tuple(getattr(v, "_builder_idx", None) for v in variables),
        bounds,
        tuple((id(p), _value_bytes(p.value)) for p in parameters),
        tuple(map(id, constraints)),
        tuple(id(c.body) for c in constraints),
        tuple(c.sense for c in constraints),
        # A scalar rhs compares by value (``-0.0 == 0.0``; a NaN only ever
        # misses); an array rhs by its bytes, so tuple equality never meets an
        # array's ambiguous truth value.
        tuple(
            r if type(r) in (float, int) else _value_bytes(r) for r in (c.rhs for c in constraints)
        ),
        tuple(c.name for c in constraints),
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
