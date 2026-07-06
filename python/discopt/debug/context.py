"""The state snapshot handed to the debugger at each checkpoint.

``DebugContext`` is a superset of :class:`discopt.callbacks.CallbackContext`:
it carries the aggregate tree state that is always available, plus the
per-node batch arrays that exist only at the batch-level checkpoints. It is
built lazily — only when a debugger is actually attached — so a detached solve
pays nothing (see :func:`discopt.debug.fire`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .checkpoints import Checkpoint

if TYPE_CHECKING:
    from .steer import DebugSteer

# Bounds sentinel used by the solver for infeasible / failed relaxations. Kept
# in sync with solver._SENTINEL_THRESHOLD; values at/above this are "no bound".
_SENTINEL_THRESHOLD = 1e19


@dataclass
class DebugContext:
    """Read-mostly view of solver state at a checkpoint.

    Aggregate fields are always populated. Batch fields (``batch_*`` and
    ``result_*``) are present only at ``AFTER_SELECT`` and later within an
    iteration; they are ``None`` at ``ITER_START`` and ``TERMINATED``.
    """

    checkpoint: Checkpoint
    iteration: int
    elapsed: float

    # Aggregate tree state (always available).
    node_count: int
    open_nodes: int
    incumbent_obj: Optional[float]
    best_bound: float
    gap: Optional[float]

    # Per-node batch state (available at AFTER_SELECT .. AFTER_PROCESS).
    batch_lb: Optional[np.ndarray] = None
    batch_ub: Optional[np.ndarray] = None
    batch_ids: Optional[np.ndarray] = None
    result_lbs: Optional[np.ndarray] = None
    result_sols: Optional[np.ndarray] = None
    result_feas: Optional[np.ndarray] = None

    # Optional event tag (e.g. "new_incumbent") and safe-steer handle.
    event: Optional[str] = None
    steer: Optional["DebugSteer"] = None
    model: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_batch(self) -> int:
        """Number of nodes in the current batch (0 if no batch is live)."""
        return 0 if self.batch_ids is None else len(self.batch_ids)

    def metrics(self) -> dict[str, float]:
        """Scalar metrics addressable by conditional breakpoints and ``print``."""
        return {
            "nodes": float(self.node_count),
            "open": float(self.open_nodes),
            "incumbent": (float(self.incumbent_obj) if self.incumbent_obj is not None else np.inf),
            "bound": float(self.best_bound),
            "gap": float(self.gap) if self.gap is not None else np.inf,
            "iter": float(self.iteration),
            "elapsed": float(self.elapsed),
        }

    @classmethod
    def build(
        cls,
        checkpoint: Checkpoint,
        *,
        tree: Any,
        model: Any = None,
        iteration: int = 0,
        elapsed: float = 0.0,
        batch_lb: Any = None,
        batch_ub: Any = None,
        batch_ids: Any = None,
        result_lbs: Any = None,
        result_sols: Any = None,
        result_feas: Any = None,
        event: Optional[str] = None,
        validator: Any = None,
    ) -> "DebugContext":
        """Construct a context by reading aggregate state from the Rust tree.

        Only pure reads (``tree.stats()`` / ``tree.incumbent()``) are performed,
        so building a context never perturbs the search — a no-op debugger is
        bound-neutral. ``validator`` is the solve loop's candidate-validation
        closure (see :data:`discopt.debug.steer.Validator`); the ``inject``
        steer is available only at checkpoints that wire one.
        """
        from .steer import DebugSteer

        stats = tree.stats() if tree is not None else {}
        inc = tree.incumbent() if tree is not None else None
        inc_obj: Optional[float] = None
        if inc is not None:
            _, obj = inc
            if np.isfinite(obj) and obj < _SENTINEL_THRESHOLD:
                inc_obj = float(obj)

        gap = stats.get("gap")
        if gap is not None and not np.isfinite(gap):
            gap = None

        return cls(
            checkpoint=checkpoint,
            iteration=iteration,
            elapsed=elapsed,
            node_count=int(stats.get("total_nodes", 0)),
            open_nodes=int(stats.get("open_nodes", 0)),
            incumbent_obj=inc_obj,
            best_bound=float(stats.get("global_lower_bound", -np.inf)),
            gap=gap,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
            result_lbs=result_lbs,
            result_sols=result_sols,
            result_feas=result_feas,
            event=event,
            steer=DebugSteer(tree, model, validator=validator) if tree is not None else None,
            model=model,
        )

    @classmethod
    def from_rust(cls, state: dict) -> "DebugContext":
        """Build a context from the pure-Rust MILP hook's aggregate state dict.

        The Rust ``solve_milp`` fast-path has no ``PyTreeManager``, so this path
        carries aggregate scalars only (no per-node batch arrays, no steer
        handle — inspection and flow control only).
        """
        cp = Checkpoint(str(state["checkpoint"]))
        inc = state.get("incumbent")
        inc_obj = (
            float(inc)
            if inc is not None and np.isfinite(inc) and float(inc) < _SENTINEL_THRESHOLD
            else None
        )
        gap = state.get("gap")
        gap = float(gap) if gap is not None and np.isfinite(gap) else None
        return cls(
            checkpoint=cp,
            iteration=int(state.get("iteration", 0)),
            elapsed=float(state.get("elapsed", 0.0)),
            node_count=int(state.get("nodes", 0)),
            open_nodes=int(state.get("open_nodes", 0)),
            incumbent_obj=inc_obj,
            best_bound=float(state.get("bound", -np.inf)),
            gap=gap,
        )
