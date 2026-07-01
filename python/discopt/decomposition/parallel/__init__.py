"""Parallel execution for the Decomposition Advisor (design §13, Phase 6).

The independent subproblem blocks of a decomposition run on a pluggable
:class:`CommunicationLayer` (sequential reference or thread pool today; Ray / MPI
/ GPU later), ordered by a :class:`SchedulingGraph` that starts the biggest
blocks first to avoid stragglers. Execution order is a performance concern only —
results are always reduced back into block order, so the answer is deterministic
regardless of backend or schedule.
"""

from __future__ import annotations

from discopt.decomposition.parallel.comm import (
    CommunicationLayer,
    SequentialComm,
    ThreadPoolComm,
    select_backend,
)
from discopt.decomposition.parallel.schedule import (
    ScheduledTask,
    SchedulingGraph,
    build_schedule,
)

__all__ = [
    "CommunicationLayer",
    "ScheduledTask",
    "SchedulingGraph",
    "SequentialComm",
    "ThreadPoolComm",
    "build_schedule",
    "select_backend",
]
