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

# "No bound" cutoff for the DEBUG DISPLAY: an incumbent at or above this
# magnitude is the LP layer's infinity sentinel carried through arithmetic, not
# a number to show a user.
#
# #1401: renamed from ``_SENTINEL_THRESHOLD``, which collided with
# ``solver._SENTINEL_THRESHOLD`` -- same identifier, DIFFERENT value (1e29) and a
# different question ("is this the infeasibility marker?" vs "is this
# effectively infinite?"). The old comment here claimed the two were "kept in
# sync"; they never were, and that claim is exactly the confusion that made the
# first draft of the #1401 fix a no-op. 1e19 is the right value for the question
# this module asks, so the value is unchanged -- only the name and the false
# claim go. Deliberately a module-local constant, not an import of
# ``solver._EFF_INF_BOUND``: the debug layer must not take a dependency on the
# solver module to display a number.
_NO_BOUND_MAGNITUDE = 1e19


@dataclass(frozen=True)
class NodeLineage:
    """Where a set of nodes sits in the tree and how each got there (#2043).

    Arrays are aligned with ``ids``. The root has ``parent == -1``,
    ``branch_var == -1``, ``branch_side == ""`` and ``branch_value`` NaN. A child
    was created by ``x[branch_var] <= branch_value`` (side ``"down"``) or
    ``x[branch_var] >= branch_value`` (side ``"up"``); the record is taken when the
    child is made, so a later tightening of its box does not change it.
    ``status`` is one of ``pending``, ``evaluated``, ``branched``, ``pruned``,
    ``fathomed`` at the moment of the checkpoint; ``pruned`` covers both a bound
    prune and an infeasible relaxation.
    """

    ids: np.ndarray
    parent: np.ndarray
    depth: np.ndarray
    branch_var: np.ndarray
    branch_side: tuple[str, ...]
    branch_value: np.ndarray
    status: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict) -> "NodeLineage":
        """From the dict ``PyTreeManager.node_lineage`` and the Rust MILP hook emit."""
        return cls(
            ids=np.asarray(d["id"], dtype=np.int64),
            parent=np.asarray(d["parent"], dtype=np.int64),
            depth=np.asarray(d["depth"], dtype=np.int64),
            branch_var=np.asarray(d["branch_var"], dtype=np.int64),
            branch_side=tuple(d["branch_side"]),
            branch_value=np.asarray(d["branch_value"], dtype=np.float64),
            status=tuple(d["status"]),
        )

    def __len__(self) -> int:
        return len(self.ids)


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

    # Tree structure (#2043). ``batch_lineage`` is aligned with ``batch_ids`` at
    # AFTER_SELECT .. AFTER_PROCESS (at AFTER_PROCESS its ``status`` says how each
    # node closed); ``created`` lists every child the batch made, including those
    # pruned before they are ever selected, at AFTER_PROCESS only.
    batch_lineage: Optional[NodeLineage] = None
    created: Optional[NodeLineage] = None

    # Optional event tag (e.g. "new_incumbent") and safe-steer handle.
    event: Optional[str] = None
    steer: Optional["DebugSteer"] = None
    model: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_batch(self) -> int:
        """Number of nodes in the current batch (0 if no batch is live)."""
        return 0 if self.batch_ids is None else len(self.batch_ids)

    @property
    def batch_parent(self) -> Optional[np.ndarray]:
        """Parent id of each batch node (-1 for the root), or ``None``."""
        return None if self.batch_lineage is None else self.batch_lineage.parent

    @property
    def batch_depth(self) -> Optional[np.ndarray]:
        """Depth of each batch node (root = 0), or ``None``."""
        return None if self.batch_lineage is None else self.batch_lineage.depth

    @property
    def batch_branch_var(self) -> Optional[np.ndarray]:
        """Branching variable that created each batch node (-1 for the root)."""
        return None if self.batch_lineage is None else self.batch_lineage.branch_var

    @property
    def batch_branch_side(self) -> Optional[tuple[str, ...]]:
        """``"down"``/``"up"`` for each batch node (``""`` for the root)."""
        return None if self.batch_lineage is None else self.batch_lineage.branch_side

    @property
    def batch_branch_value(self) -> Optional[np.ndarray]:
        """Bound the creating branch imposed on each batch node (NaN for the root)."""
        return None if self.batch_lineage is None else self.batch_lineage.branch_value

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
            if np.isfinite(obj) and obj < _NO_BOUND_MAGNITUDE:
                inc_obj = float(obj)

        gap = stats.get("gap")
        if gap is not None and not np.isfinite(gap):
            gap = None

        # #2043: lineage is a pure read of immutable per-node fields (plus status),
        # so, like the reads above, it cannot perturb the search.
        lineage_of = getattr(tree, "node_lineage", None)
        batch_lineage = None
        created = None
        if lineage_of is not None:
            if batch_ids is not None:
                batch_lineage = NodeLineage.from_dict(
                    lineage_of(np.asarray(batch_ids, dtype=np.int64))
                )
            if checkpoint == Checkpoint.AFTER_PROCESS:
                created = NodeLineage.from_dict(lineage_of(tree.last_created()))

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
            batch_lineage=batch_lineage,
            created=created,
            event=event,
            steer=DebugSteer(tree, model, validator=validator) if tree is not None else None,
            model=model,
        )

    @classmethod
    def from_rust(cls, state: dict) -> "DebugContext":
        """Build a context from the pure-Rust MILP hook's state dict.

        The Rust ``solve_milp`` fast-path has no ``PyTreeManager`` (so no steer
        handle — inspection and flow control only), but at the ``after_select``
        checkpoint the hook marshals the batch's per-node boxes/ids, so
        ``print node <i>`` / ``print nodes`` work here too. The Rust hook carries
        no final status, so ``extra["error"]`` is never set and ``"on-error"``
        sessions do not trigger on this path (documented in ``Model.solve``).
        """
        cp = Checkpoint(str(state["checkpoint"]))
        inc = state.get("incumbent")
        inc_obj = (
            float(inc)
            if inc is not None and np.isfinite(inc) and float(inc) < _NO_BOUND_MAGNITUDE
            else None
        )
        gap = state.get("gap")
        gap = float(gap) if gap is not None and np.isfinite(gap) else None

        # Batch boxes are present only at after_select (nested lists from Rust).
        b_lb = state.get("batch_lb")
        b_ub = state.get("batch_ub")
        b_ids = state.get("batch_ids")
        batch_lb = np.asarray(b_lb, dtype=np.float64) if b_lb is not None else None
        batch_ub = np.asarray(b_ub, dtype=np.float64) if b_ub is not None else None
        batch_ids = np.asarray(b_ids, dtype=np.int64) if b_ids is not None else None
        b_lin = state.get("batch_lineage")
        c_lin = state.get("created")
        batch_lineage = NodeLineage.from_dict(b_lin) if b_lin is not None else None
        created = NodeLineage.from_dict(c_lin) if c_lin is not None else None
        if batch_ids is None and batch_lineage is not None:
            batch_ids = batch_lineage.ids

        return cls(
            checkpoint=cp,
            iteration=int(state.get("iteration", 0)),
            elapsed=float(state.get("elapsed", 0.0)),
            node_count=int(state.get("nodes", 0)),
            open_nodes=int(state.get("open_nodes", 0)),
            incumbent_obj=inc_obj,
            best_bound=float(state.get("bound", -np.inf)),
            gap=gap,
            batch_lb=batch_lb,
            batch_ub=batch_ub,
            batch_ids=batch_ids,
            batch_lineage=batch_lineage,
            created=created,
        )
