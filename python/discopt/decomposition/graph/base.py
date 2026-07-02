"""Model-induced graph views and their cache.

The Decomposition Advisor analyzes a model through the graphs it induces. This
module builds the two foundational, near-linear ones from a model's expression
DAG and exposes the cheap structural queries on top of them:

- the **variable–constraint incidence** (bipartite) — the substrate every other
  graph is a transform of; and
- the **variable dependency graph** (variables adjacent iff they co-occur in a
  constraint) — the object connectivity / cut analysis runs on.

Graphs are built lazily and memoized per ``(model, structural-version)`` so the
FBBT/OBBT loop — which changes *bounds*, not structure — reuses them for free
across many bound-tightening rounds (see the incremental-update contract in the
design doc). Heavier views (Jacobian/Hessian/KKT, elimination trees) are planned
follow-ups and slot into the same ``GraphKind`` registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from discopt.decomposition.graph import kernels

# Guard for the O(m·(n+E)) bridge scan; above this, skip it and rely on
# annotations. Single source of truth — ``structure.py`` imports this value.
_BRIDGE_SCAN_BUDGET = 200_000


class GraphKind(Enum):
    """The graph representations the advisor can build over a model.

    Only the two foundational graphs are materialized today; the remaining kinds
    are registered so the builder, cache, and exporter have stable identifiers as
    they come online.
    """

    VARIABLE_CONSTRAINT = "variable_constraint"
    DEPENDENCY = "dependency"
    # Planned (see design doc §4): JACOBIAN, HESSIAN, KKT, ELIMINATION_TREE,
    # BLOCK, COUPLING, STAGE, SCENARIO.


def _collect_constraint_cliques(model) -> tuple[list[str], list[list[int]]]:
    """Extract variable names and per-constraint variable-index cliques.

    Shared with ``discopt.decomposition.structure`` — one DAG scan produces the
    canonical incidence both the advisor graphs and ``detect_decomposition``
    consume.
    """
    from discopt._jax.gdp_reformulate import _collect_variables

    var_names = [v.name for v in model._variables]
    name_to_idx = {nm: i for i, nm in enumerate(var_names)}

    cliques: list[list[int]] = []
    for c in model._constraints:
        body = getattr(c, "body", None)
        if body is None:
            cliques.append([])
            continue
        idxs: list[int] = []
        seen: set[int] = set()
        for nm in _collect_variables(body):
            j = name_to_idx.get(nm)
            if j is not None and j not in seen:
                seen.add(j)
                idxs.append(j)
        cliques.append(idxs)
    return var_names, cliques


@dataclass(frozen=True)
class ModelGraph:
    """An immutable view of the graphs a model induces.

    Holds the canonical variable–constraint incidence (as ``var_names`` +
    per-constraint ``constraint_cliques``) and derives the cheap structural
    queries on demand. Cheap to construct, cheap to clone, safe to cache.

    Attributes
    ----------
    var_names : list[str]
        Variable names in declared order; index ``i`` is vertex ``i``.
    constraint_cliques : list[list[int]]
        ``constraint_cliques[c]`` is the sorted-by-appearance list of variable
        indices referenced by constraint ``c`` (empty if it references none).
    """

    var_names: list[str]
    constraint_cliques: list[list[int]]

    @classmethod
    def from_model(cls, model) -> "ModelGraph":
        """Build the incidence view from a discopt ``Model`` (one DAG scan)."""
        var_names, cliques = _collect_constraint_cliques(model)
        return cls(var_names=var_names, constraint_cliques=cliques)

    @property
    def num_vars(self) -> int:
        """Number of variables (bipartite variable vertices)."""
        return len(self.var_names)

    @property
    def num_constraints(self) -> int:
        """Number of constraints (bipartite constraint vertices)."""
        return len(self.constraint_cliques)

    @property
    def num_incidences(self) -> int:
        """Number of variable-in-constraint incidences (bipartite edges)."""
        return sum(len(c) for c in self.constraint_cliques)

    def variable_components(self) -> tuple[list[int], int]:
        """Connected components over variables; ``(block_of_var, num_blocks)``."""
        return kernels.connected_components(self.num_vars, self.constraint_cliques)

    def num_blocks(self) -> int:
        """Number of independent variable blocks (connected components)."""
        return self.variable_components()[1]

    def is_block_diagonal(self) -> bool:
        """True iff the model already splits into ≥2 independent blocks."""
        return self.num_blocks() >= 2

    def dependency_edges(self, max_clique_expand: int = 64) -> list[tuple[int, int]]:
        """Undirected variable–variable edges (see ``kernels.dependency_edges``)."""
        return kernels.dependency_edges(
            self.constraint_cliques, max_clique_expand=max_clique_expand
        )

    def bridge_constraints(self, budget: int = _BRIDGE_SCAN_BUDGET) -> list[int]:
        """Constraint indices whose sole removal disconnects the model."""
        return sorted(kernels.bridge_cliques(self.constraint_cliques, self.num_vars, budget))

    def articulation_variables(self) -> list[int]:
        """Variable indices whose removal disconnects the dependency graph.

        These are prime *complicating variables*: fixing one splits the model,
        the structural signal behind a Benders master/subproblem split.
        """
        pts, _ = kernels.articulation_and_bridges(self.num_vars, self.dependency_edges())
        return pts

    def summary(self) -> str:
        """Human-readable one-block summary of the induced structure."""
        block_of, k = self.variable_components()
        sizes = [0] * k
        for b in block_of:
            sizes[b] += 1
        return "\n".join(
            [
                "ModelGraph (variable-constraint incidence)",
                f"  variables:   {self.num_vars}",
                f"  constraints: {self.num_constraints}",
                f"  incidences:  {self.num_incidences}",
                f"  blocks:      {k}"
                + (f" (sizes {sizes[:8]}{'…' if k > 8 else ''})" if k else ""),
            ]
        )


@dataclass
class GraphCache:
    """Per-model memoization of built graphs, keyed by structural version.

    The advisor bumps ``model_version`` on a *structural* edit (add/drop a
    variable or constraint, change a term's variable set); bound-only edits do
    not invalidate structural graphs. A cache entry stores the version it was
    built at and is discarded on mismatch. Instances are cheap; one may be held
    per advisor session.
    """

    _entries: dict[tuple[int, GraphKind], tuple[int, ModelGraph]] = field(default_factory=dict)

    def get(self, model, kind: GraphKind = GraphKind.VARIABLE_CONSTRAINT) -> ModelGraph:
        """Return the built graph for *model*/*kind*, building/refreshing if stale."""
        version = int(getattr(model, "_structural_version", 0))
        key = (id(model), kind)
        hit = self._entries.get(key)
        if hit is not None and hit[0] == version:
            return hit[1]
        graph = build_graph(model, kind)
        self._entries[key] = (version, graph)
        return graph

    def clear(self) -> None:
        """Drop all cached graphs."""
        self._entries.clear()


def build_graph(model, kind: GraphKind = GraphKind.VARIABLE_CONSTRAINT) -> ModelGraph:
    """Build a ``ModelGraph`` of the requested *kind* from a discopt ``Model``.

    The variable–constraint incidence and the (derived) dependency graph share
    the same ``ModelGraph`` carrier, since the dependency edges are a transform
    of the incidence cliques. Unimplemented kinds raise ``NotImplementedError``.
    """
    if kind in (GraphKind.VARIABLE_CONSTRAINT, GraphKind.DEPENDENCY):
        return ModelGraph.from_model(model)
    name = getattr(kind, "value", kind)
    raise NotImplementedError(f"graph kind {name!r} is not yet implemented")


__all__ = [
    "GraphCache",
    "GraphKind",
    "ModelGraph",
    "build_graph",
]
