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

What role-2 clocks cost when they are left in place (#1116)
----------------------------------------------------------

#912 fixed role (2) at the sites it covered. #1116 measured what the *remaining*
role-2 clocks do to the answer, and it is larger than the node-count cliff above.

``kriging_peaks-full200``, ``max_nodes=1``, one process, one binary, no user time
pressure, three repetitions: root dual bounds of −25371.8 / −28852.0 / −28072.6 —
a 14 % spread — with the incumbent bit-identical to the last digit. At
``max_nodes=300`` the same defect survives as a 13th-digit bound move that flips
the node count 301 ↔ 303, which is enough to make CLAUDE.md §5's bound-neutral
regime (``node_count`` and ``objective`` *exactly* unchanged) inapplicable to the
instance and to turn an A/B on it into a measurement of noise. One #1115 panel row
was published and withdrawn for exactly that reason.

The mechanism is structural, not float noise. The first root LP comes back with a
different number of COLUMNS per repetition (1532 vs 1469 measured), because a
wall-truncated tightening stage handed the relaxation builder a different box: the
root fixpoint returned 894 tightened bounds in one repetition and 898 in the next,
from *different input boxes*, having spent 237.7 s and 247.55 s in OBBT. Three
plausible non-clock explanations were tested and falsified — the Rust hash
containers named in the issue (the whole crate has three iteration sites over hash
containers, all order-insensitive), thread scheduling (pinned to one thread, still
varies), and ``Variable.__hash__ = id(self)`` (replaced with a stable index across
26 M patched calls, still varies). Replacing the clock alone with a deterministic
counter made the solve reproduce bit-exactly, and *tighter* than any wall-bounded
run (−1044.819 vs the −25 000…−34 000 cluster) — an early-truncated tightening
stage is not merely nondeterministic, it is leaving bound on the table.

``solver_tuning.SolverTuning.deterministic`` (``DISCOPT_DETERMINISTIC``, default
OFF) is the switch that renders those budgets inert, via ``solver._role2_budget``
/ ``_role2_deadline`` / ``_role2_horizon`` at the origin of each one. It is the
same doctrine applied by suppression rather than by counting: where #912 replaces
a wall budget with a work counter, the flag removes the wall budget and leaves the
stage to the deterministic caps it already carries (round counts, iteration caps,
the node budget), with the user's ``time_limit`` still stopping the search.

Two role-1 mechanisms are deliberately left in place under that flag: the
phase-entry gates (``_deadline_exhausted()`` / ``_remaining_budget() > x``, which
decide whether an optional preprocessing phase starts at all) and the POUNCE
``max_wall_time = min(30.0, caller_limit)`` stall backstop. Neutralizing either
would let preprocessing overrun the user's ``time_limit`` without bound, which
trades a reproducibility bug for a broken role-1 promise — so the guarantee is
scoped to a solve whose role-1 budget never binds, rather than widened until it
does. Both residuals were measured not to bind on the reproduction instance: the
30 s POUNCE cap was live during the arm that reproduced bit-exactly, and the
solve finished in ~7 min against the default 3600 s limit.

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
from typing import Callable, Mapping, Optional

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
        deadline: Optional absolute ``clock()`` timestamp. This is the *user's*
            limit (role 1 above), not a work gate: it is checked in
            :meth:`exhausted` so a heuristic cannot overrun ``time_limit``, and
            when it is what stopped the loop :attr:`stopped_on` says so.
        clock: The monotonic clock ``deadline`` is measured against. Defaults to
            :func:`time.perf_counter`; production code never passes anything
            else. The seam exists so a *test* of deadline-edge behaviour can
            drive the deadline from a clock it controls instead of racing the
            machine: with the real clock, the truncation a test means to pin is
            a consequence of wall time, so an unrelated scheduling stall (a
            descheduled xdist worker on a loaded runner) silently changes what
            the code under test does and the test flakes (#950). ``deadline``
            and ``clock`` must be expressed in the same time base.
    """

    __slots__ = ("limits", "deadline", "used", "_stopped_on", "_clock")

    def __init__(
        self,
        limits: Optional[Mapping[str, int]] = None,
        *,
        deadline: Optional[float] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.limits: dict[str, int] = {
            k: int(v) for k, v in (limits or {}).items() if v is not None and int(v) > 0
        }
        self.deadline: Optional[float] = deadline
        self.used: dict[str, int] = {}
        self._stopped_on: Optional[str] = None
        self._clock: Callable[[], float] = clock

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
        if self.deadline is not None and self._clock() >= self.deadline:
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
