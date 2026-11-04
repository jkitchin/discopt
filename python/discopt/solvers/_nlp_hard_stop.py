"""Stop a primal-heuristic NLP at the solve's deadline, from inside the solve (#2143).

The heuristics give each sub-NLP a ``max_wall_time`` and poll the solve deadline
*between* solves. Neither stops an NLP that is already running past the deadline,
and POUNCE does not always honour ``max_wall_time``. Measured on GDPlib ``syngas``
under ``gdp_method="hull"`` with a 120 s limit (``scratchpad/stage3/probe_overrun.py``):

* One root multistart start launched at 86.0 s with ``max_wall_time=30`` returned
  ``OPTIMAL`` at 159.8 s. That is 1,984 IPM iterations, 21,587 of its callbacks after
  its own wall cap. The solve ended at 160.1 s.
* In another run, a pump fix-and-solve with ``max_wall_time=3.0`` ran 7.95 s and
  ended 4 s past the deadline.

The stop is ``intermediate``, the per-iteration hook POUNCE and cyipopt both call:
returning ``False`` ends the solve with ``User_Requested_Stop`` (code 5). It is
checked once per IPM iteration, restoration phase included, so a stopped solve
ends at most one iteration after the deadline.

**Scope.** :func:`stop_at_solve_deadline` arms the stop only for its ``with`` block,
at the ROLE-1 deadline of the live ``solve_model``
(:data:`discopt.solver_tuning.ROLE1_DEADLINE`). It never uses a stage's or a
heuristic's own, earlier deadline. That way the stop truncates only work that would
otherwise run past the user's ``time_limit``. A solve whose deadline never binds runs
exactly as it did before: the hook returns ``True`` on every iteration. Outside a
``solve_model``, or for a solve with no ``time_limit``, arming is a no-op. Only primal
heuristics arm it. Node NLPs, which can be bound producers, are never stopped by it.
"""

from __future__ import annotations

import contextlib
import math
import time
from contextvars import ContextVar
from typing import Any, Iterator, Optional

import numpy as np

#: The armed hard stop (``perf_counter`` epoch), or ``None`` when no stop is armed.
NLP_HARD_DEADLINE: ContextVar[Optional[float]] = ContextVar("_NLP_HARD_DEADLINE", default=None)


@contextlib.contextmanager
def stop_at_solve_deadline() -> Iterator[None]:
    """Arm the hard stop at the live solve's ROLE-1 deadline for this block.

    Nested arming keeps the earlier of the two deadlines. A no-op when no
    ``solve_model`` with a finite ``time_limit`` is live.
    """
    from discopt.solver_tuning import ROLE1_DEADLINE

    deadline = ROLE1_DEADLINE.get()
    if deadline is None or not math.isfinite(deadline):
        yield
        return
    current = NLP_HARD_DEADLINE.get()
    token = NLP_HARD_DEADLINE.set(deadline if current is None else min(current, deadline))
    try:
        yield
    finally:
        NLP_HARD_DEADLINE.reset(token)


def armed_deadline() -> Optional[float]:
    """The armed hard-stop deadline, or ``None``."""
    return NLP_HARD_DEADLINE.get()


class HardStopCallbacks:
    """NLP callbacks whose ``intermediate`` hook ends the solve at ``deadline``.

    Every other attribute is forwarded to ``inner``. When ``inner`` has its own
    ``intermediate``, a ``False`` from it still stops the solve. ``fired`` records
    whether THIS wrapper stopped it, so the caller can report ``TIME_LIMIT`` rather
    than a user stop.
    """

    def __init__(self, inner: Any, deadline: float) -> None:
        self._inner = inner
        self._deadline = float(deadline)
        self.fired = False

    def intermediate(self, *args: Any, **kwargs: Any) -> bool:
        inner = getattr(self._inner, "intermediate", None)
        if inner is not None and inner(*args, **kwargs) is False:
            return False
        if time.perf_counter() >= self._deadline:
            self.fired = True
            return False
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def expired_result(x0: np.ndarray):
    """The result of an NLP that is not started because the hard stop has passed."""
    from discopt.solvers import NLPResult, SolveStatus

    return NLPResult(
        status=SolveStatus.TIME_LIMIT,
        x=np.asarray(x0, dtype=np.float64).copy(),
        objective=float("nan"),
        iterations=0,
        wall_time=0.0,
        raw_status=None,
    )


#: Ipopt/POUNCE ``User_Requested_Stop``.
USER_REQUESTED_STOP = 5
