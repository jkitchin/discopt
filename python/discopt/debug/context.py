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

    def branch_text(self, k: int, names: Optional[list[str]] = None) -> str:
        """The branch that created node ``k`` as text: ``root``, ``x3 <= 0``, ...

        ``names`` maps a flat variable index to a display name; without it (or for
        an index it does not cover) the variable shows as ``x[j]``.
        """
        j = int(self.branch_var[k])
        side = self.branch_side[k]
        if j < 0 or side == "":
            return "root"
        name = names[j] if names is not None and j < len(names) else f"x[{j}]"
        op = "<=" if side == "down" else ">="
        return f"{name} {op} {float(self.branch_value[k]):.6g}"

    def to_dict(self) -> dict[str, list]:
        """Columnar, JSON-ready form (the ``batch``/``created`` pause fields, #2049)."""
        return {
            "ids": [int(v) for v in self.ids],
            "parent": [int(v) for v in self.parent],
            "depth": [int(v) for v in self.depth],
            "branch_var": [int(v) for v in self.branch_var],
            "branch_side": list(self.branch_side),
            "branch_value": [float(v) for v in self.branch_value],
            "status": list(self.status),
        }

    def row(self, k: int, names: Optional[list[str]] = None) -> dict[str, Any]:
        """Node ``k`` as one record, with the creating branch rendered as text."""
        return {
            "id": int(self.ids[k]),
            "parent": int(self.parent[k]),
            "depth": int(self.depth[k]),
            "branch": self.branch_text(k, names),
            "branch_var": int(self.branch_var[k]),
            "branch_side": self.branch_side[k],
            "branch_value": float(self.branch_value[k]),
            "status": self.status[k],
        }


def flat_variable_names(model: Any, n: Optional[int] = None) -> Optional[list[str]]:
    """Display name of each flat variable index of ``model``, or ``None``.

    Indices follow ``model._variables`` flattened in order, the layout the B&B tree
    is built from (``solver._extract_variable_info``). Returns ``None`` when the
    model has no variable list, or when ``n`` (the width of the boxes the tree
    reports) disagrees with the model's size: then the indices belong to some
    other variable space and naming them from this model would mislabel them.
    """
    variables = getattr(model, "_variables", None)
    if not variables:
        return None
    names: list[str] = []
    for v in variables:
        shape = tuple(getattr(v, "shape", ()) or ())
        size = int(np.prod(shape)) if shape else 1
        if size == 1 and len(shape) <= 1:
            names.append(str(v.name))
            continue
        for flat in range(size):
            idx = np.unravel_index(flat, shape)
            names.append(f"{v.name}[{','.join(str(int(i)) for i in idx)}]")
    if n is not None and len(names) != n:
        return None
    return names


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
