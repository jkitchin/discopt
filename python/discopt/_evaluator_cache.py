"""Evaluator fingerprinting and the per-model LRU, with no JAX import.

Lives outside ``_relax/`` deliberately. ``_relax/nlp_evaluator.py`` imports jax at
module scope, so anything that must run *before* the choice of evaluator backend
— which is exactly this — cannot live there without pulling JAX in and defeating
the point of having a JAX-free backend at all (issue #75).

Both backends share this module so their cache-validity rules cannot drift apart:
a fingerprint that means one thing for the JAX evaluator and another for the tape
would be worse than two separate implementations, because it would look shared.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Model


def evaluator_fingerprint(model: "Model") -> tuple:
    """Structural fingerprint of a model for evaluator-cache validity.

    Captures the object identity of the objective, constraints, variables, and
    parameters, plus the Gauss-Newton flag — but NOT mutable variable bounds or
    ``Parameter.value``. Two models with the same fingerprint can therefore share
    one compiled evaluator across bound changes (every B&B node) and parameter
    re-binds.

    **``Parameter.value`` is excluded on purpose, and that is a live hazard for
    any value-baking backend.** The JAX evaluator reads parameter values on every
    call, so a re-bind needs no rebuild. A tape bakes them in as constants, so a
    tape cached under this fingerprint would serve derivatives for the OLD value
    with no error and no exception. ``TapeNLPEvaluator`` closes that itself by
    snapshotting values and rebuilding when they move; do not "fix" it here by
    adding values to the fingerprint, which would instead defeat the JAX
    evaluator's whole reason for excluding them.
    """
    _blocks = getattr(model, "_builder_linear_blocks", None) or ()
    return (
        id(model._objective),
        tuple(id(c) for c in model._constraints),
        # #840: the fast-path builder rows are part of the evaluator's constraint
        # set, so they must be in the fingerprint — else a model that gains a fast
        # family (or has its builder rows materialized into ``_constraints``)
        # would reuse a stale evaluator built without them.
        tuple((id(A), int(A.shape[0]), sense) for A, _x, sense, _b, _name in _blocks),
        tuple(id(v) for v in model._variables),
        tuple(id(p) for p in model._parameters),
        bool(getattr(model, "_gauss_newton_hessian", False)),
    )


def solution_state_fingerprint(model: "Model") -> tuple:
    """Everything that decides WHICH point a solve of *model* returns.

    :func:`evaluator_fingerprint` plus the two things it deliberately leaves
    out -- variable bounds and ``Parameter.value`` -- because for an evaluator
    they are inputs, while for a *solution* they are part of the problem. A
    recorded result is about this fingerprint's problem and no other.

    Used by ``Model.sensitivity()`` to tell a live reference solution from a
    stale one (#1322): the reference is what decides which basin the derivatives
    describe, and a parameter change alone can move the global optimum to
    another well while the old point remains a perfectly good KKT point with a
    perfectly matching objective value.

    Values, not identities, for bounds and parameters: a bound is rebound with a
    fresh array on every B&B node and a parameter is written through in place,
    so identity would report change where there is none and miss change where
    there is. `tobytes()` gives exact equality -- the right test here, since the
    question is "is this the same problem", not "is it close".
    """
    return (
        evaluator_fingerprint(model),
        tuple(
            (
                np.asarray(v.lb, dtype=np.float64).tobytes(),
                np.asarray(v.ub, dtype=np.float64).tobytes(),
            )
            for v in model._variables
        ),
        tuple(np.asarray(p.value, dtype=np.float64).tobytes() for p in model._parameters),
        None if model._objective is None else str(model._objective.sense),
    )


# Number of distinct-fingerprint evaluators kept per model. A single slot was
# enough for the plain B&B loop (one structural fingerprint for the whole solve),
# but the primal heuristics *temporarily* add a structural row (RENS /
# local-branching sub-solves append a constraint, solve, then remove it) and then
# re-solve the *base* model. That oscillation thrashes a one-slot cache: measured
# on clay0303hfsg, the base evaluator rebuilt 3x per solve (#723).
EVALUATOR_CACHE_MAXSIZE = 8


def cached_by_fingerprint(
    model: "Model",
    cache_attr: str,
    factory: Callable[["Model"], Any],
    maxsize: int = EVALUATOR_CACHE_MAXSIZE,
) -> Any:
    """Return a per-model LRU-cached object built by ``factory``.

    ``cache_attr`` names the attribute the LRU hangs off, so different backends
    keep separate caches on the same model and a backend switch mid-process
    cannot hand one backend's evaluator to the other.
    """
    fp = evaluator_fingerprint(model)
    cache: "OrderedDict[tuple, Any] | None" = getattr(model, cache_attr, None)
    if cache is None:
        cache = OrderedDict()
        setattr(model, cache_attr, cache)
    hit = cache.get(fp)
    if hit is not None:
        cache.move_to_end(fp)
        return hit
    built = factory(model)
    cache[fp] = built
    cache.move_to_end(fp)
    while len(cache) > maxsize:
        cache.popitem(last=False)  # evict least-recently-used
    return built
