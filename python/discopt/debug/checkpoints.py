"""Checkpoint vocabulary for the interactive branch-and-bound debugger.

The debugger reframes pounce's IPM-iteration phases as a B&B *node lifecycle*.
Each solve loop fires the subset of checkpoints it actually reaches; the JSON
``hello`` handshake advertises which are live for the active path.
"""

from __future__ import annotations

from enum import Enum


class Checkpoint(str, Enum):
    """A well-defined pause point in the branch-and-bound search.

    Ordered by their position within a single batch iteration.
    """

    ITER_START = "iter_start"  # top of a batch iteration
    AFTER_SELECT = "after_select"  # open-node boxes/ids exported from the tree
    AFTER_TIGHTEN = "after_tighten"  # per-node FBBT/OBBT applied to boxes
    AFTER_BOUND = "after_bound"  # per-node relaxation solved (sols/bounds/feas)
    BEFORE_IMPORT = "before_import"  # steer point: inject incumbent / branch hint
    AFTER_PROCESS = "after_process"  # prune/branch/fathom applied by the tree
    INCUMBENT_FOUND = "incumbent_found"  # event: a strictly better incumbent
    TERMINATED = "terminated"  # final / limit / infeasible

    def __str__(self) -> str:  # nicer REPL/JSON output
        return self.value


#: Checkpoints the debugger stops at by default. Others fire every iteration but
#: resume immediately unless a breakpoint or ``stop-at`` requests them (matches
#: pounce's default of stopping only at ``iter_start`` and ``terminated``).
DEFAULT_STOP: frozenset[Checkpoint] = frozenset({Checkpoint.ITER_START, Checkpoint.TERMINATED})

#: Friendly aliases accepted by ``stop-at``.
ALIASES: dict[str, Checkpoint] = {
    "start": Checkpoint.ITER_START,
    "sel": Checkpoint.AFTER_SELECT,
    "select": Checkpoint.AFTER_SELECT,
    "tighten": Checkpoint.AFTER_TIGHTEN,
    "bound": Checkpoint.AFTER_BOUND,
    "relax": Checkpoint.AFTER_BOUND,
    "steer": Checkpoint.BEFORE_IMPORT,
    "import": Checkpoint.BEFORE_IMPORT,
    "process": Checkpoint.AFTER_PROCESS,
    "branch": Checkpoint.AFTER_PROCESS,
    "incumbent": Checkpoint.INCUMBENT_FOUND,
    "end": Checkpoint.TERMINATED,
    "done": Checkpoint.TERMINATED,
}


def resolve_checkpoint(name: str) -> Checkpoint:
    """Resolve a checkpoint name or alias, raising ``KeyError`` if unknown."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    return Checkpoint(key)  # raises ValueError -> caller reports it


#: Solver events a breakpoint can fire on. Extended in later phases.
EVENTS: frozenset[str] = frozenset({"new_incumbent"})
