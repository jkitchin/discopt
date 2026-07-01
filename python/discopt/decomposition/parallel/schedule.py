"""Scheduling graph — order block solves for good parallel efficiency (design §13.2).

The subproblems of a decomposition are independent, so the only scheduling
decision that matters for wall-clock is *straggler avoidance*: start the biggest
blocks first, so the critical path is one big block rather than a big block that
happened to be dispatched last. The :class:`SchedulingGraph` records each block's
estimated cost and hands back a deterministic big-first execution order; the
coordinator executes in that order but reduces results back into block order
(:mod:`discopt.decomposition.parallel.comm`), so scheduling never changes the
answer.

This is the dataflow object later, richer schedulers (work-stealing, pipelined
cut aggregation, data-locality placement — design §13.2) refine; the interface
stays the estimated-cost list + an execution order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledTask:
    """A block to solve, with its estimated relative cost."""

    block_id: int
    est_cost: float


@dataclass(frozen=True)
class SchedulingGraph:
    """Execution plan over a decomposition's independent blocks.

    Attributes
    ----------
    tasks : tuple[ScheduledTask, ...]
        One task per subproblem block, in block order.
    has_master : bool
        Whether a coordinating master problem precedes the blocks (Benders /
        Lagrangian) — informational for the coordinator.
    """

    tasks: tuple[ScheduledTask, ...]
    has_master: bool = False

    def execution_order(self) -> list[int]:
        """Block ids ordered big-cost-first (ties by block id, for determinism)."""
        return [t.block_id for t in sorted(self.tasks, key=lambda t: (-t.est_cost, t.block_id))]

    @property
    def num_blocks(self) -> int:
        """Number of scheduled blocks."""
        return len(self.tasks)

    def total_cost(self) -> float:
        """Sum of block costs — the serial (one-worker) cost estimate."""
        return sum(t.est_cost for t in self.tasks)

    def critical_path_cost(self) -> float:
        """Largest single block cost — the lower bound on parallel wall-clock."""
        return max((t.est_cost for t in self.tasks), default=0.0)

    def ideal_speedup(self) -> float:
        """total / critical-path — the load-balance speedup cap (``≥ 1``)."""
        cp = self.critical_path_cost()
        return (self.total_cost() / cp) if cp > 0 else 1.0

    def summary(self) -> str:
        """One-line human-readable plan."""
        return (
            f"SchedulingGraph: {self.num_blocks} blocks, "
            f"total_cost={self.total_cost():.0f}, critical_path={self.critical_path_cost():.0f}, "
            f"ideal_speedup={self.ideal_speedup():.2f}x, master={self.has_master}"
        )


def build_schedule(decomposed) -> SchedulingGraph:
    """Build a :class:`SchedulingGraph` from a decomposition.

    Block cost is estimated by variable count — the cheap proxy the scorer also
    uses. *decomposed* is duck-typed (``.subproblems`` with ``.block_id``/``.size``
    and a ``.master``) to avoid importing the IR here.
    """
    tasks = tuple(
        ScheduledTask(block_id=sp.block_id, est_cost=float(sp.size))
        for sp in decomposed.subproblems
    )
    return SchedulingGraph(tasks=tasks, has_master=getattr(decomposed, "master", None) is not None)


__all__ = ["ScheduledTask", "SchedulingGraph", "build_schedule"]
