"""Block-separability detection (D5 of issue #53).

A model is *block-separable* if its variables partition into disjoint
groups such that every constraint and the objective use variables from
exactly one group. The partition is the connected components of a graph
whose nodes are variables and whose edges are induced by co-occurrence
inside any single constraint body or the objective expression.

Separability is purely structural — it doesn't change the model — but
surfacing it early enables:

1. **Parallel relaxation evaluation.** Independent blocks have
   independent McCormick / convex envelopes; the relaxation compiler
   can build them concurrently.
2. **Decomposition methods.** Generalized Benders, ADMM, and Lagrangian
   relaxation all key off block structure.
3. **Branching heuristics.** Branching inside one block doesn't change
   bounds in another, so branching rules can specialize per block.

The pass is informational: it stamps the discovered block partition
onto the orchestrator delta as ``structure["separable_blocks"]`` so
downstream consumers (relaxation compiler, branching) can read it
without re-running the analysis.

References
----------
- Tarjan (1972) — connected components in linear time on adjacency lists.
- Achterberg et al. (2020), §3.7 — block decomposition as a presolve
  reduction in the MIP setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import make_python_delta


@dataclass(frozen=True)
class SeparabilityReport:
    """Outcome of one D5 separability scan.

    Attributes
    ----------
    blocks : list[list[str]]
        One inner list per connected component, holding the names of
        the variables in that block. Empty list if the model has no
        constraints and a constant objective.
    constraint_block : list[int]
        ``constraint_block[i]`` is the block index that constraint
        ``i`` belongs to. ``-1`` means the constraint mentions no
        variables (a tautology / pure-constant body) and is therefore
        unassigned.
    objective_block : int
        Block index that the objective expression touches, or ``-1``
        if the objective is constant.
    separable : bool
        True iff there are at least two non-empty blocks. A model
        with a single block is technically separable (trivially),
        but the flag captures the case the consumer cares about.
    """

    blocks: list[list[str]]
    constraint_block: list[int]
    objective_block: int
    separable: bool


def detect_separability(model: Any) -> SeparabilityReport:
    """Compute the variable-block partition of ``model``.

    Each constraint and the objective contribute a clique edge among
    the variables they reference. Connected components of the
    resulting variable graph are the blocks.

    The implementation is union-find for ``O((V + E) · α(V))`` time;
    deterministic in the model's declared variable order.
    """
    from discopt._jax.gdp_reformulate import _collect_variables

    var_names: list[str] = [v.name for v in model._variables]
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(var_names)}
    n = len(var_names)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def _names_in(expr) -> list[int]:
        if expr is None:
            return []
        names = _collect_variables(expr)
        out: list[int] = []
        for nm in names:
            idx = name_to_idx.get(nm)
            if idx is not None:
                out.append(idx)
        return out

    def _link_clique(idxs: list[int]) -> None:
        if len(idxs) <= 1:
            return
        anchor = idxs[0]
        for i in idxs[1:]:
            union(anchor, i)

    constraint_idxs: list[list[int]] = []
    for c in model._constraints:
        body = getattr(c, "body", None)
        idxs = _names_in(body)
        constraint_idxs.append(idxs)
        _link_clique(idxs)

    obj_idxs: list[int] = []
    obj = getattr(model, "_objective", None)
    if obj is not None:
        obj_idxs = _names_in(getattr(obj, "expression", None))
        _link_clique(obj_idxs)

    # Materialize the partition. Roots are encountered in declared
    # variable order, so block ids are deterministic.
    root_to_block: dict[int, int] = {}
    blocks: list[list[str]] = []
    for i in range(n):
        r = find(i)
        if r not in root_to_block:
            root_to_block[r] = len(blocks)
            blocks.append([])
        blocks[root_to_block[r]].append(var_names[i])

    def _block_of(idxs: list[int]) -> int:
        if not idxs:
            return -1
        return root_to_block[find(idxs[0])]

    constraint_block = [_block_of(idxs) for idxs in constraint_idxs]
    objective_block = _block_of(obj_idxs)

    # Drop empty blocks that can arise if the model declares a
    # variable that nothing references — they remain singletons in
    # ``blocks`` but we still report them so callers see every var.
    nonempty = sum(1 for b in blocks if b)
    return SeparabilityReport(
        blocks=blocks,
        constraint_block=constraint_block,
        objective_block=objective_block,
        separable=nonempty >= 2,
    )


class SeparabilityPass:
    """Block-separability detection as a presolve pass.

    Args:
        model: the Python ``Model`` whose constraints + objective
            define the variable-co-occurrence graph.
    """

    name = "separability"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.last_report: SeparabilityReport | None = None

    def run(self, model_repr: Any) -> dict:
        delta = make_python_delta(self.name)
        try:
            report = detect_separability(self.model)
        except Exception:
            return delta
        self.last_report = report
        delta["work_units"] = len(self.model._constraints) + 1
        delta["separable_blocks"] = [list(b) for b in report.blocks]
        delta["separable"] = bool(report.separable)
        delta["constraint_block"] = list(report.constraint_block)
        delta["objective_block"] = int(report.objective_block)
        return delta


__all__ = [
    "SeparabilityPass",
    "SeparabilityReport",
    "detect_separability",
]
