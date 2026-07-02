"""Model-induced graph infrastructure for the Decomposition Advisor.

Phase 1 of the advisor (see ``docs/design/decomposition-advisor.md``): the
foundational graph layer. A model induces several graphs — this package builds
the near-linear ones (variable–constraint incidence, variable dependency graph),
exposes the cheap structural detectors on them (connected components,
bridges/articulation), caches them by structural version, and serializes them
for external tooling and partitioners.

The heavy graph kernels are mirrored in Rust
(``crates/discopt-core/src/decomp/``); the pure-Python kernels here are the
portable fallback and the correctness oracle, so this layer works with or
without the compiled extension.
"""

from __future__ import annotations

from discopt.decomposition.graph.base import (
    GraphCache,
    GraphKind,
    ModelGraph,
    build_graph,
)
from discopt.decomposition.graph.export import export_graph

__all__ = [
    "GraphCache",
    "GraphKind",
    "ModelGraph",
    "build_graph",
    "export_graph",
]
