"""FFI-boundary time accounting for the solve-layer profile (issue #74).

Why this module exists
======================

``SolveResult`` reports where a solve's wall clock went. The pre-#921 model had
three fields — ``rust_time`` / ``jax_time`` / ``python_time`` — accumulated by
*phase* timers wrapped around large regions of :mod:`discopt.solver`, with
``python_time`` derived as ``wall - rust - jax``. That model was wrong in two
independent ways, both measured:

1. **The phases were mixed.** One accumulator spanned ~745 lines of
   ``solver.py`` and enclosed the Rust simplex and the numpy relaxation build,
   charging all of it to JAX. Nine corpus instances reported 0.14-1.09 s of
   ``jax_time`` on solves where ``jax`` never entered ``sys.modules`` at all
   (``st_testgr3``: 1.09 s of a 1.13 s solve).

2. **The buckets were not disjoint.** ~96 % of JAX's cost on the solve path is
   *interpreted Python* (tracing, ``core.bind``, primitive dispatch), not native
   XLA — measured on heatexch_gen3 as 13.34 s Python vs 0.56 s XLA. Subtracting
   ``jax_time`` from the wall to obtain ``python_time`` therefore removed Python
   time twice, and POUNCE — 11.69 s of a 15 s tspn08 solve, i.e. 78 % — had no
   bucket at all and was silently absorbed.

The model here instead measures at **FFI boundaries**, never around phases:

* ``rust`` and ``python`` are disjoint and partition the wall clock —
  native Rust (the discopt core *and* POUNCE) versus interpreted Python.
* ``jax`` and ``pounce`` are diagnostic **subsets**: ``jax <= python`` because
  JAX time is Python time, and ``pounce <= rust`` because POUNCE is Rust.

Usage
=====

Take a snapshot before the work and diff after; nesting is handled naturally,
because an inner solve's boundary time genuinely elapsed during the outer one::

    before = timing.snapshot()
    ...
    delta = timing.since(before)      # {"rust": ..., "pounce": ..., "jax": ...}

and charge a boundary with::

    with timing.charge("pounce"):
        x, info = problem.solve(x0)

Charging records **self time**: a nested region's elapsed time is subtracted from
its parent, so a POUNCE solve that calls back into the Python evaluator reports
the two layers separately rather than billing the callbacks to POUNCE.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from typing import Iterator

__all__ = ["charge", "snapshot", "since", "BUCKETS"]


@dataclasses.dataclass(slots=True)
class _Frame:
    """One open ``charge`` region.

    ``child`` accumulates the wall time of nested regions so the parent can
    report *self* time. A plain list was used here first; it typed as
    ``list[object]`` and silently defeated the arithmetic below.
    """

    bucket: str
    started: float
    child: float = 0.0


#: Boundary buckets. ``rust`` and ``python`` partition the wall clock; ``pounce``
#: is a subset of ``rust`` and ``jax`` is a subset of ``python``.
BUCKETS = ("rust", "pounce", "jax")

_state = threading.local()


def _totals() -> dict[str, float]:
    totals = getattr(_state, "totals", None)
    if totals is None:
        totals = _state.totals = dict.fromkeys(BUCKETS, 0.0)
    return totals


def _stack() -> list[_Frame]:
    """Active regions on this thread, innermost last."""
    stack = getattr(_state, "stack", None)
    if stack is None:
        stack = _state.stack = []
    return stack


@contextlib.contextmanager
def charge(bucket: str) -> Iterator[None]:
    """Charge the enclosed block's **self time** to ``bucket``.

    Self time, not wall time: any nested ``charge`` region is subtracted from the
    enclosing one. This is essential rather than cosmetic here, because POUNCE
    calls *back into Python* for every derivative — ``problem.solve()`` wall time
    contains the evaluator callbacks. Charging that wall to ``pounce`` would
    re-create exactly the defect this module exists to fix: a bucket inflated by
    work belonging to another layer.

    So::

        with charge("pounce"):      # 10 s wall
            ...
            with charge("jax"):    # 3 s of callbacks
                ...

    records ``pounce = 7`` and ``jax = 3``, not ``pounce = 10``.

    Re-entrant per bucket: a nested region of the *same* bucket still measures
    self time correctly (its elapsed is subtracted from its parent, then added
    back to the same bucket), so POUNCE's LP path reached from inside an NLP
    solve is not double-counted.

    The bookkeeping is in a ``finally``, so a boundary that raises is still
    charged — an exception costing 2 s of Rust must not vanish from the profile.
    """
    if bucket not in BUCKETS:
        raise ValueError(f"unknown timing bucket {bucket!r}; expected one of {BUCKETS}")
    stack = _stack()
    frame = _Frame(bucket, time.perf_counter())
    stack.append(frame)
    try:
        yield
    finally:
        stack.pop()
        elapsed = time.perf_counter() - frame.started
        self_time = elapsed - frame.child  # minus time already claimed by children
        _totals()[bucket] += self_time
        if stack:
            stack[-1].child += elapsed  # parent must not also count our wall


def snapshot() -> dict[str, float]:
    """Current per-bucket totals for this thread."""
    return dict(_totals())


def since(before: dict[str, float]) -> dict[str, float]:
    """Elapsed per-bucket time since ``before`` (a prior :func:`snapshot`).

    Clamped at zero: a caller that snapshots on one thread and diffs on another
    would otherwise report a negative duration, which reads as a live counter
    rather than the instrumentation bug it is.
    """
    now = _totals()
    return {b: max(0.0, now[b] - before.get(b, 0.0)) for b in BUCKETS}
