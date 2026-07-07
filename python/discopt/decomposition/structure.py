"""Decomposition structure: annotation resolution + auto-detection.

Both Benders decomposition and Lagrangian relaxation key off problem
*structure*:

- **Benders** partitions variables into *complicating* (first-stage / master)
  and *subproblem* (recourse) sets. Fixing the complicating variables leaves a
  tractable subproblem whose dual yields cuts.
- **Lagrangian relaxation** dualizes *coupling* constraints — the linking rows
  whose removal separates the model into independent blocks that can be solved
  in parallel.

This module resolves explicit user annotations (set on the :class:`Model` via
``first_stage`` / ``second_stage`` / ``set_block`` / ``mark_coupling``) and
falls back to auto-detection when none are supplied:

- *Complicating variables* default to the integer/binary variables — the
  canonical Benders split for two-stage (mixed-)integer programs.
- *Coupling constraints* are auto-detected with a guarded **bridge-constraint**
  heuristic: a constraint is coupling when removing it alone disconnects the
  variable-incidence graph. Multi-constraint cuts are not auto-found in v1;
  annotate ``mark_coupling`` for those cases.

Block membership is computed from the **constraint** incidence graph only (the
objective is assumed separable across blocks, the standard Lagrangian setup).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.decomposition.graph import kernels
from discopt.decomposition.graph.base import _BRIDGE_SCAN_BUDGET
from discopt.modeling.core import Model, VarType

# ``_BRIDGE_SCAN_BUDGET`` (the guard for the O(m·(V+E)) bridge scan) is defined
# once in ``graph.base`` and imported here so the two paths cannot drift.


@dataclass(frozen=True)
class DecompositionStructure:
    """Resolved decomposition structure consumed by the decomposition solvers.

    Attributes
    ----------
    blocks : list[list[str]]
        Variable names per block (connected components of the non-coupling
        constraint graph). Deterministic in declared variable order.
    block_of_var : dict[str, int]
        Maps each variable name to its block index.
    block_of_constraint : list[int]
        ``block_of_constraint[i]`` is the block of constraint ``i``, or ``-1``
        if the constraint is coupling or references no variables.
    complicating_vars : list[str]
        Variable names held in the Benders master (first-stage).
    coupling_constraints : list[int]
        Indices (into ``model._constraints``) of the dualized coupling rows.
    is_separable : bool
        True iff there are ≥2 blocks once coupling constraints are removed.
    source : str
        ``"annotated"``, ``"detected"``, or ``"mixed"`` — provenance of the
        partition, for diagnostics.
    """

    blocks: list[list[str]]
    block_of_var: dict[str, int]
    block_of_constraint: list[int]
    complicating_vars: list[str]
    coupling_constraints: list[int]
    is_separable: bool
    source: str
    detection_truncated: bool = False

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def summary(self) -> str:
        lines = [
            f"DecompositionStructure ({self.source})",
            f"  blocks: {self.num_blocks} (separable={self.is_separable})",
            f"  complicating vars: {len(self.complicating_vars)}",
            f"  coupling constraints: {len(self.coupling_constraints)}",
        ]
        return "\n".join(lines)


def _vars_in(expr, name_to_idx: dict[str, int]) -> list[int]:
    """Flat variable indices referenced by *expr* (deduplicated, ordered)."""
    if expr is None:
        return []
    from discopt._jax.gdp_reformulate import _collect_variables

    out: list[int] = []
    for nm in _collect_variables(expr):
        idx = name_to_idx.get(nm)
        if idx is not None:
            out.append(idx)
    return out


def _components(
    n: int,
    cliques: list[list[int]],
) -> tuple[list[int], int]:
    """Union-find connected components over variable-index *cliques*.

    Delegates to :func:`discopt.decomposition.graph.kernels.connected_components`
    (the shared, Rust-mirrored kernel). Returns
    ``(root_block_id_per_var, num_blocks)`` with block ids assigned in ascending
    variable order for determinism.
    """
    return kernels.connected_components(n, cliques)


def _annotated_coupling(model: Model) -> set[int]:
    """Resolve ``mark_coupling`` annotations to constraint indices."""
    keys: set = getattr(model, "_coupling_keys", set())
    if not keys:
        return set()
    out: set[int] = set()
    for i, c in enumerate(model._constraints):
        cname = getattr(c, "name", None)
        if id(c) in keys or (cname is not None and cname in keys):
            out.add(i)
    return out


def _bearing_blocks(n: int, cliques: list[list[int]]) -> int:
    """Number of connected components that contain at least one constraint.

    Delegates to :func:`discopt.decomposition.graph.kernels.bearing_blocks`.
    Counting *constraint-bearing* components (rather than raw variable
    components) avoids spurious singletons: a variable that appears in only one
    constraint becomes isolated when that constraint is dropped, which must not
    be mistaken for a genuine block split.
    """
    return kernels.bearing_blocks(n, cliques)


def _detect_bridge_coupling(
    model: Model,
    constraint_cliques: list[list[int]],
    n: int,
) -> tuple[set[int], bool]:
    """Bridge-constraint heuristic: a constraint whose sole removal disconnects.

    A constraint is coupling when dropping it raises the number of
    constraint-bearing components. Delegates the graph scan to
    :func:`discopt.decomposition.graph.kernels.bridge_cliques_status`, guarded by
    ``_BRIDGE_SCAN_BUDGET``. Returns ``(coupling_indices, truncated)``; a
    ``truncated`` scan (too large) logs a WARNING so "no coupling" is never
    silently confused with "gave up" (S3).
    """
    return kernels.bridge_cliques_status(constraint_cliques, n, _BRIDGE_SCAN_BUDGET)


def detect_decomposition(
    model: Model,
    *,
    complicating: list[str] | None = None,
    coupling: list[int] | None = None,
    dec_file: str | None = None,
) -> DecompositionStructure:
    """Resolve the decomposition structure of *model*.

    Annotations on the model take precedence; explicit ``complicating`` /
    ``coupling`` arguments override both. ``dec_file`` short-circuits to a GCG
    ``.dec`` file (its ``MASTERCONSS`` become the coupling rows). See the module
    docstring for the auto-detection rules.
    """
    if dec_file is not None:
        from discopt.decomposition.graph.export import read_dec

        return read_dec(dec_file, model)

    var_names = [v.name for v in model._variables]
    name_to_idx = {nm: i for i, nm in enumerate(var_names)}
    n = len(var_names)

    constraint_cliques = [
        _vars_in(getattr(c, "body", None), name_to_idx) for c in model._constraints
    ]

    # ── coupling constraints ──
    src_parts = []
    detection_truncated = False
    if coupling is not None:
        coupling_set = set(coupling)
        src_parts.append("annotated")
    else:
        coupling_set = _annotated_coupling(model)
        if coupling_set:
            src_parts.append("annotated")
        else:
            coupling_set, detection_truncated = _detect_bridge_coupling(
                model, constraint_cliques, n
            )
            if coupling_set:
                src_parts.append("detected")

    # ── blocks from non-coupling constraints ──
    block_cliques = [clique for i, clique in enumerate(constraint_cliques) if i not in coupling_set]
    block_of, num_blocks = _components(n, block_cliques)

    blocks: list[list[str]] = [[] for _ in range(num_blocks)]
    for i, nm in enumerate(var_names):
        blocks[block_of[i]].append(nm)
    block_of_var = {nm: block_of[i] for i, nm in enumerate(var_names)}

    def _block_of_constraint(i: int) -> int:
        if i in coupling_set:
            return -1
        clique = constraint_cliques[i]
        return block_of[clique[0]] if clique else -1

    block_of_constraint = [_block_of_constraint(i) for i in range(len(model._constraints))]

    # ── complicating variables ──
    if complicating is not None:
        complicating_vars = list(complicating)
        src_parts.append("annotated")
    else:
        stages = getattr(model, "_decomp_stages", {})
        annotated = [nm for nm, st in stages.items() if st == 1]
        if annotated:
            complicating_vars = annotated
            src_parts.append("annotated")
        else:
            complicating_vars = [
                v.name for v in model._variables if v.var_type in (VarType.BINARY, VarType.INTEGER)
            ]
            if complicating_vars:
                src_parts.append("detected")

    if not src_parts:
        source = "detected"
    elif len(set(src_parts)) == 1:
        source = src_parts[0]
    else:
        source = "mixed"

    return DecompositionStructure(
        blocks=blocks,
        block_of_var=block_of_var,
        block_of_constraint=block_of_constraint,
        complicating_vars=complicating_vars,
        coupling_constraints=sorted(coupling_set),
        is_separable=num_blocks >= 2,
        source=source,
        detection_truncated=detection_truncated,
    )


def flat_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lb, ub)`` of the flat variable vector in declared order."""
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.asarray(v.lb, dtype=np.float64).reshape(-1))
        ubs.append(np.asarray(v.ub, dtype=np.float64).reshape(-1))
    if not lbs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(lbs), np.concatenate(ubs)


def restricted_bounds(
    model: Model,
    fixed: dict[str, float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flat ``(lb, ub)`` with the named variables pinned to *fixed* values.

    Used to form Benders subproblems: pinning ``lb == ub`` fixes a variable so
    the relaxation/presolve treats it as a constant. Indices follow the flat
    vector implied by declared variable order (the convention used by the
    evaluators and the Rust B&B tree).
    """
    lb, ub = flat_bounds(model)
    offset = 0
    for v in model._variables:
        size = int(np.prod(v.shape)) if v.shape else 1
        if v.name in fixed:
            val = np.asarray(fixed[v.name], dtype=np.float64).reshape(-1)
            if val.size == 1:
                val = np.full(size, val.item())
            if val.size != size:
                raise ValueError(f"fixed value for {v.name!r} has size {val.size}, expected {size}")
            lb[offset : offset + size] = val
            ub[offset : offset + size] = val
        offset += size
    return lb, ub


__all__ = [
    "DecompositionStructure",
    "detect_decomposition",
    "flat_bounds",
    "restricted_bounds",
]
