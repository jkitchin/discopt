"""Regression for the shared NLPEvaluator cache (perf Stage 1).

The B&B loop, primal heuristics, and POUNCE node solves all evaluate the same
model (only bounds / parameter values change, read live). Constructing a fresh
``NLPEvaluator(model)`` per call re-traces and re-compiles its JAX callables — on
gear4 the ``diving`` heuristic did this ~110×/solve, ~15 s of pure Python.

``cached_evaluator(model)`` returns one evaluator per structural fingerprint so
those callers reuse it. This pins:
  * repeated calls return the *same* object (cache hit),
  * a structural change invalidates it (cache miss),
  * a bound change does *not* invalidate it (the B&B common case),
  * ``solver._make_evaluator`` shares the very same cache.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
from discopt._jax.nlp_evaluator import cached_evaluator, evaluator_fingerprint  # noqa: E402


def _model() -> dm.Model:
    m = dm.Model("c")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2.0) ** 2 + (y - 1.0) ** 2)
    m.subject_to(x + y <= 4.0)
    return m


def test_repeated_calls_reuse_same_evaluator():
    m = _model()
    ev1 = cached_evaluator(m)
    ev2 = cached_evaluator(m)
    assert ev1 is ev2, "cached_evaluator must return the same object on a cache hit"


def test_bound_change_does_not_invalidate():
    m = _model()
    ev1 = cached_evaluator(m)
    # Mutate a variable bound (the per-node B&B case) — fingerprint must be stable.
    m._variables[0].ub = m._variables[0].ub * 0.5
    fp_before = evaluator_fingerprint(m)
    ev2 = cached_evaluator(m)
    assert ev2 is ev1, "a bound change must NOT rebuild the evaluator"
    assert evaluator_fingerprint(m) == fp_before


def test_structural_change_invalidates():
    m = _model()
    ev1 = cached_evaluator(m)
    # Add a constraint — a genuine structural change.
    x = m._variables[0]
    m.subject_to(x >= 0.5)
    ev2 = cached_evaluator(m)
    assert ev2 is not ev1, "a structural change must rebuild the evaluator"


def test_make_evaluator_shares_the_cache():
    import discopt.solver as S

    m = _model()
    ev_main = S._make_evaluator(m)
    ev_heur = cached_evaluator(m)
    assert ev_main is ev_heur, "_make_evaluator and cached_evaluator must share one cache"
