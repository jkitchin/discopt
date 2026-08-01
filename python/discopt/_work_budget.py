"""Deterministic work budgets — the machine-speed-independent replacement for
wall-clock "how much work" gates (issue #912).

The problem
-----------

A wall clock answers two very different questions in this solver:

1. *When do we stop?* — the user's ``time_limit``. Reading a clock for this is
   correct by definition: the contract is "return within N seconds".
2. *How much work do we do?* — e.g. "keep perturbing and re-descending until a
   5-second heuristic budget expires". Reading a clock for **this** makes the
   answer a function of machine speed: the same model with the same
   ``time_limit`` explores a different tree on a faster box, under CPU
   contention, or with a different JIT cache state.

Issue #912 measured (2) end to end: ``gear2`` closes in 3 nodes when the root
integer local search gets its full 5 s and in 91 nodes when it gets 3 s, and the
default sits exactly on that cliff because the descent never converges. Every
"node counts unchanged" verdict in this repo is only meaningful if the search is
a pure function of its input, so a clock in role (2) is a correctness-of-process
bug, not a performance detail.

The fix
-------

Count *work*, not seconds. :class:`WorkBudget` counts model-level operations —
constraint/objective evaluations and NLP solves — against a per-kind limit. A
loop gated on ``budget.exhausted()`` performs the same operations on every
machine, so the point it stops at is a function of the model alone.

The clock does not disappear: a :class:`WorkBudget` may also carry the *solve*
deadline (role 1), so a heuristic still cannot run past the user's
``time_limit``. The difference is that the deadline is now a backstop that only
fires when the user's own limit is about to be violated, rather than the primary
gate. :attr:`WorkBudget.stopped_on` records which one fired, so a panel can
assert that a run was decided by work and not by the clock — i.e. that the run
it is about to compare is actually reproducible.

Why per-kind counters and not one currency
------------------------------------------

The obvious design — convert everything to one "work unit" via a fixed cost
ratio — was implemented first and **falsified on this corpus**. Measured over 9
in-repo MINLPLib instances (``item912_work_unit_calibration.py``, 5 interleaved
rounds each): one constraint evaluation costs 0.7-2.7 us, one continuous-repair
sub-NLP solve 1.9-104 ms. The ratio has a geomean of 12 364 but a range of
2 933-77 964 — a 27x spread — because the two operations are not two sizes of the
same thing.

With a single currency at the geomean, one budget cannot serve both regimes, and
the failure was measured rather than predicted: at the converted-unit default
``nvs09`` regressed from 5 nodes / `optimal` to 29 nodes / `feasible` because the
shared budget ran out before its search was done, while the *same* number handed
the more expensive-per-solve ``syn05hfsg`` three times its legacy wall time. The
root cause is that a sub-NLP solve is not a fixed number of evaluations: its cost
varies 5x across instances (13.3 to 136 solves/s measured) while a single
currency assumes it constant. Two counters, each with its own limit, price each
operation on its own terms. See
``docs/dev/work-budget-calibration-2026-08-01.md``.

Usage::

    budget = WorkBudget({EVAL: 20_000, NLP_SOLVE: 128}, deadline=solve_deadline)
    while not budget.exhausted():
        budget.charge(EVAL)
        ...
    logger.debug("stopped on %s after %r", budget.stopped_on, budget.used)
"""

from __future__ import annotations

import time
from typing import Mapping, Optional

__all__ = [
    "EVAL",
    "NLP_SOLVE",
    "WorkBudget",
]

#: One constraint-vector or objective evaluation (0.7-2.7 us measured).
EVAL = "eval"

#: One NLP solve — a continuous-repair sub-NLP or a backend local solve
#: (1.9-104 ms measured).
NLP_SOLVE = "nlp_solve"

class WorkBudget:
    """A deterministic per-kind work counter, backstopped by a wall deadline.

    Args:
        limits: Mapping of work kind to the number of operations of that kind
            this budget allows. A kind absent from the mapping, or mapped to a
            non-positive value, is unlimited — but still counted, so an
            instrumented run can report what an unbounded loop actually did.
            ``None`` means every kind is unlimited.
        deadline: Optional absolute ``time.perf_counter()`` timestamp. This is
            the *user's* limit (role 1 above), not a work gate: it is checked in
            :meth:`exhausted` so a heuristic cannot overrun ``time_limit``, and
            when it is what stopped the loop :attr:`stopped_on` says so.
    """

    __slots__ = ("limits", "deadline", "used", "_stopped_on")

    def __init__(
        self,
        limits: Optional[Mapping[str, int]] = None,
        *,
        deadline: Optional[float] = None,
    ):
        self.limits: dict[str, int] = {
            k: int(v) for k, v in (limits or {}).items() if v is not None and int(v) > 0
        }
        self.deadline: Optional[float] = deadline
        self.used: dict[str, int] = {}
        self._stopped_on: Optional[str] = None

    def charge(self, kind: str, count: int = 1) -> None:
        """Record ``count`` operations of ``kind`` as spent."""
        self.used[kind] = self.used.get(kind, 0) + int(count)

    def exhausted(self) -> bool:
        """True once any counter has reached its limit, or the deadline passed.

        Work is checked before the clock on purpose: when both would fire, the
        deterministic reason is the one worth reporting.
        """
        for kind, limit in self.limits.items():
            if self.used.get(kind, 0) >= limit:
                if self._stopped_on is None:
                    self._stopped_on = f"work:{kind}"
                return True
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            if self._stopped_on is None:
                self._stopped_on = "deadline"
            return True
        return False

    def spent(self, kind: str) -> int:
        """Operations of ``kind`` charged so far."""
        return self.used.get(kind, 0)

    def remaining(self, kind: str) -> Optional[int]:
        """Operations of ``kind`` still affordable, or ``None`` when unlimited.

        This is the deterministic replacement for "how much time do I have
        left?", which callers used to answer by subtracting two clock reads and
        dividing by a *measured* mean cost — making the decision they based on it
        (e.g. local branching's enumeration radius) a function of machine speed.
        """
        limit = self.limits.get(kind)
        return None if limit is None else max(0, limit - self.used.get(kind, 0))

    @property
    def stopped_on(self) -> Optional[str]:
        """``"work:<kind>"``, ``"deadline"``, or ``None`` if :meth:`exhausted`
        never returned True.

        A search that reports a ``work:`` reason (or ``None``) is reproducible
        with respect to this gate; one that reports ``"deadline"`` was cut by the
        machine's speed, and the result it returns is not a function of the model
        alone.
        """
        return self._stopped_on

    @property
    def deterministic(self) -> bool:
        """True iff nothing about this budget's outcome depended on the clock."""
        return self._stopped_on != "deadline"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"WorkBudget(limits={self.limits}, used={self.used}, stopped_on={self._stopped_on!r})"
        )
