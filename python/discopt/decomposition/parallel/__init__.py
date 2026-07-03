"""Parallel execution for the Decomposition Advisor (design §13, Phase 6).

The independent subproblem blocks of a decomposition run on a pluggable
:class:`CommunicationLayer` (sequential reference or thread pool today; Ray / MPI
/ GPU later), ordered by a :class:`SchedulingGraph` that starts the biggest
blocks first to avoid stragglers. Execution order is a performance concern only —
results are always reduced back into block order, so the answer is deterministic
regardless of backend or schedule.

This layer is consumed by the shipping drivers: the Lagrangian relaxation
subproblem (``solve_lagrangian(..., backend=...)``) and the multicut Benders
recourse (``BendersConfig.backend``) both solve their per-block subproblems
through :func:`select_backend`, so ``backend="threads"`` runs blocks
concurrently while returning bit-identical bounds.
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
