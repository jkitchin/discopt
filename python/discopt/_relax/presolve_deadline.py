"""A deadline the optional pre-solve structure passes can see (issue #1456).

THE DEFECT.  ``solve_model``'s reformulation block runs a series of optional
structure-detection passes — entropy canonicalization, functional-dependency
detection, binary-multilinear reform, factorable reform, integer-bilinear
reform.  Every one is a walk over the model's expression DAG, every one has an
existing "found nothing, model unchanged" path, and none of them could see the
caller's ``time_limit``.  They ran to completion however long they took, before
branch and bound started, so the time limit could not reach them.  Measured at
``time_limit=5``: ``densitymod`` spent 27.0 s in this block (49 % of its solve),
``pb302095`` 33.0 s, ``truck`` 26.1 s (81 %), ``telecomsp_metro`` 13.2 s (79 %).

THE FIX is the gate the house already uses for the *later* root-setup phases:
``solve_model``'s own ``_deadline_exhausted()``, checked at phase entry.  Those
phases (root presolve, reverse-AD, the eigenvalue diagnostic, root OBBT, the
root cut pool) have carried it since #654; this block sits *above* the first
call site and never got it.  Nothing here is new machinery — it is the existing
doctrine applied to the one stretch of pre-solve that was missed, plus the
abstention logging #1456 asks for.

WHY NOT A DETERMINISTIC WORK ALLOWANCE.  A clock makes the skip decision a
function of machine speed, which is the reproducibility defect #912 exists to
prevent, so a deterministic allowance was the first design: charge each pass the
model's DAG-node count against a shared budget.  **It was built, calibrated over
1610 MINLPLib instances, and falsified.**  The calibration assumed pre-solve
cost is predictable from DAG size — it is, across the size-stratified sample the
block's cost was measured on (2.11-3.82 us per walked node from ``eg_all_s`` to
``densitymod``), but the instances that actually blow a time limit are *small*:

    johnall      5,432 nodes    44 min     (#1455, a leaf budget)
    glider400   38,087 nodes    28.4 s
    truck       47,030 nodes    26.1 s
    arki0014   143,090 nodes    > 240 s

No node allowance catches those without being set so low it disables structure
detection corpus-wide.  The measurement wins (CLAUDE.md §4): the allowance and
its calibration table were deleted rather than shipped with a justification
known to be false.  Reproducibility is therefore traded here exactly as the
later phases trade it, and ``stopped_on`` records that a clock decided, so a
panel can tell a clock-decided run from a deterministic one instead of assuming.

DIVISION OF LABOUR — three layers, none of which subsumes another:

  1. **Leaf budgets** bound ONE pass.  ``term_classifier._DISTRIBUTE_TERM_BUDGET``
     (#1455) and ``binary_multilinear_reform``'s ``_poly_add`` budget + the
     ``is_homogeneous_psd_quadratic`` support restriction (#1458) are
     deterministic and catch the three measured single-pass blowups.  A deadline
     cannot do their job: ``johnall``'s 44 minutes were inside one call, and this
     gate does not interrupt a pass once started.
  2. **This gate** bounds the AGGREGATE.  Six passes of 4 s each are individually
     reasonable and together five times a 5 s limit; only a shared gate sees that.
  3. **The enforced inventory** (``test_1456_presolve_work_inventory.py``) stops a
     new pass from re-opening the class unrecorded.

SOUNDNESS.  Declining a pass is always safe.  Each one is optional structure
*recognition* whose failure path is "return the model unchanged"; skipping it
yields a weaker relaxation — a larger box, fewer cuts — never an unsound one, so
a skipped pass can cost nodes but cannot produce a false bound.  This mirrors
``_deadline_exhausted``'s own docstring, and it is why the gate may be a clock at
all: the worst case is a slower solve, not a wrong answer.

The overrun this leaves is the documented one: at most a single in-flight pass,
because a pass is never interrupted once started.  Wall therefore scales with
``time_limit`` instead of with the model.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["PresolveDeadline", "presolve_deadline_enabled"]

_ENV_FLAG = "DISCOPT_PRESOLVE_DEADLINE"


def presolve_deadline_enabled() -> bool:
    """``DISCOPT_PRESOLVE_DEADLINE=0`` restores the pre-#1456 behaviour.

    This is an opt-*out* for a shipped default, not a stalled graduation
    (CLAUDE.md §5 "Out of scope"): the gate is ON, and the flag exists so the
    default can be A/B'd in one process — which is how the panel in #1456 is
    run, and how the next person who suspects the gate of costing them structure
    can check in one environment variable instead of two checkouts.

    It is deliberately not the inverse.  A default-off deadline would leave the
    defect in place for everyone who has not heard of the flag, and the defect
    is a 20 s solve returning in 44 minutes.
    """
    return os.environ.get(_ENV_FLAG, "1") != "0"


class PresolveDeadline:
    """Phase-entry gate + abstention log for one solve's reformulation block.

    ``deadline_exhausted`` is ``solve_model``'s own closure over the solve
    anchor, so this shares the caller's budget rather than starting a second
    clock of its own.  Pass ``None`` to disable the gate (every pass affordable),
    which is what a caller with no meaningful budget wants;
    ``DISCOPT_PRESOLVE_DEADLINE=0`` does the same for a whole process.

    The flag is read once, at construction, so a single solve cannot change its
    mind halfway through and leave half a reformulation block gated.
    """

    __slots__ = ("_abandoned", "_deadline_exhausted", "_ran", "_skipped", "_stopped_on")

    def __init__(self, deadline_exhausted: Optional[Callable[[], bool]]) -> None:
        if not presolve_deadline_enabled():
            deadline_exhausted = None
        self._deadline_exhausted = deadline_exhausted
        self._ran: list[str] = []
        self._skipped: list[str] = []
        self._abandoned: list[str] = []
        self._stopped_on: Optional[str] = None

    @property
    def ran(self) -> tuple[str, ...]:
        """Passes this gate admitted, in order."""
        return tuple(self._ran)

    @property
    def skipped(self) -> tuple[str, ...]:
        """Passes declined for want of budget, in order.

        #1456 item 3: abstention is logged, not silent — it means weaker
        structure recognition, and a reader comparing two runs of the same model
        needs to be able to see that the difference is an abstention.
        """
        return tuple(self._skipped)

    @property
    def stopped_on(self) -> Optional[str]:
        """``"deadline"`` once a pass was declined, else ``None``.

        Deliberately explicit rather than inferred from ``skipped``: a panel
        asserting run-to-run reproducibility has to know a *clock* made a
        decision here, and the whole point of the falsification recorded in this
        module's docstring is that it does.
        """
        return self._stopped_on

    @property
    def abandoned(self) -> tuple[str, ...]:
        """Passes that started, ran out of budget mid-traversal, and abstained.

        Distinct from :attr:`skipped`, which never started at all.  Both mean
        "structure this solve did not recognise"; only this one means the solve
        also *paid* for the part it walked before giving up.
        """
        return tuple(self._abandoned)

    def abandon_hook(self, pass_name: str) -> Callable[[], bool]:
        """A coarse in-pass check for ``pass_name`` (#1456 item 2).

        A pass-entry gate alone leaves the *first* heavy pass unbounded, which
        is not a corner case: ``truck`` was measured spending **81.5 s** between
        two consecutive entry checks — inside a single pass — against
        ``time_limit=10``.  Item 2 of the issue asks for this check "in the outer
        loop over constraints/terms, not per node", which is where the returned
        callable is meant to be called and nowhere finer.

        Admissible for the same reason the entry gate is: the pass it guards has
        exactly two documented outcomes, "rewritten" and "returned unchanged",
        and abstaining selects the second.  A check that left a pass *partway*
        through its rewrite would be a different thing entirely, and is not what
        this returns — the callers discard their partial work.
        """

        def expired() -> bool:
            if self._deadline_exhausted is None or not self._deadline_exhausted():
                return False
            if pass_name not in self._abandoned:
                self._abandoned.append(pass_name)
                if self._stopped_on is None:
                    self._stopped_on = "deadline"
                logger.debug(
                    "pre-solve pass %r abandoned mid-traversal: time limit spent "
                    "while it ran; the model is left unchanged (#1456)",
                    pass_name,
                )
            return True

        return expired

    @property
    def deterministic(self) -> bool:
        """False once the clock has decided anything.

        A run that never hit the gate is reproducible; one that did is not, and
        saying so is the difference between a measurement and an assumption
        (CLAUDE.md §6 — the attribute a panel would trust is exactly the one
        worth getting right).
        """
        return self._stopped_on is None

    def afford(self, pass_name: str) -> bool:
        """True if ``pass_name`` may run; records the decision either way.

        Checked at pass *entry* only.  A pass already running is never
        interrupted — truncating an in-flight bound-producing op would drop a
        valid bound (docs/dev/baron-gap-plan.md §8), which is a correctness cost,
        where declining to start one is only a strength cost.
        """
        if self._deadline_exhausted is not None and self._deadline_exhausted():
            self._skipped.append(pass_name)
            if self._stopped_on is None:
                self._stopped_on = "deadline"
            logger.debug(
                "pre-solve pass %r skipped: time limit spent before it started "
                "(structure recognition is weaker for it; #1456)",
                pass_name,
            )
            return False
        self._ran.append(pass_name)
        return True
