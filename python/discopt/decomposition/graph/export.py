"""Serialize model-induced graphs to interchange formats.

The advisor's ``export_graph`` surface (design doc §14) dumps a graph for
external tooling — graph databases, partitioners, and visualizers. Four formats
cover the immediate needs:

- ``"json"``   — a node-link dict, easy to consume anywhere.
- ``"dot"``    — Graphviz, for quick rendering.
- ``"graphml"``— for graph-analysis tools (Gephi, networkx, igraph).
- ``"metis"``  — the input format for the METIS/KaHyPar partitioners the block
  detector will call in Phase 2.

Exports operate on the **variable dependency graph** (variable vertices, edges =
co-occurrence in a constraint), which is what partitioners and community
detection consume. The bipartite incidence is available via the ``bipartite``
flag for visualization.
"""

from __future__ import annotations

import json
import xml.sax.saxutils as _xml

from discopt.decomposition.graph.base import ModelGraph

_FORMATS = ("json", "dot", "graphml", "metis")


def _simple_edges(graph: ModelGraph) -> list[tuple[int, int]]:
    """De-duplicated undirected dependency edges as ``(min, max)`` pairs."""
    seen: set[tuple[int, int]] = set()
    for a, b in graph.dependency_edges():
        if a == b:
            continue
        seen.add((a, b) if a < b else (b, a))
    return sorted(seen)


def export_graph(graph: ModelGraph, fmt: str = "json") -> str:
    """Serialize *graph*'s variable dependency graph to *fmt*.

    Parameters
    ----------
    graph : ModelGraph
        The model-induced graph to export.
    fmt : str
        One of ``"json"``, ``"dot"``, ``"graphml"``, ``"metis"``.

    Returns
    -------
    str
        The serialized graph.
    """
    fmt = fmt.lower()
    if fmt not in _FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {_FORMATS}")
    edges = _simple_edges(graph)
    names = graph.var_names
    if fmt == "json":
        return _to_json(names, edges)
    if fmt == "dot":
        return _to_dot(names, edges)
    if fmt == "graphml":
        return _to_graphml(names, edges)
    return _to_metis(graph.num_vars, edges)


def _to_json(names: list[str], edges: list[tuple[int, int]]) -> str:
    return json.dumps(
        {
            "directed": False,
            "nodes": [{"id": i, "name": nm} for i, nm in enumerate(names)],
            "edges": [{"source": a, "target": b} for a, b in edges],
        },
        indent=2,
    )


def _to_dot(names: list[str], edges: list[tuple[int, int]]) -> str:
    lines = ["graph dependency {"]
    for i, nm in enumerate(names):
        safe = nm.replace('"', '\\"')
        lines.append(f'  {i} [label="{safe}"];')
    for a, b in edges:
        lines.append(f"  {a} -- {b};")
    lines.append("}")
    return "\n".join(lines)


def _to_graphml(names: list[str], edges: list[tuple[int, int]]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
        '  <graph edgedefault="undirected">',
    ]
    for i, nm in enumerate(names):
        lines.append(f'    <node id="n{i}"><data key="name">{_xml.escape(nm)}</data></node>')
    for e, (a, b) in enumerate(edges):
        lines.append(f'    <edge id="e{e}" source="n{a}" target="n{b}"/>')
    lines.append("  </graph>")
    lines.append("</graphml>")
    return "\n".join(lines)


def _to_metis(n: int, edges: list[tuple[int, int]]) -> str:
    """METIS graph format: 1-indexed adjacency, header ``<#vertices> <#edges>``."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b + 1)  # METIS is 1-indexed
        adj[b].append(a + 1)
    lines = [f"{n} {len(edges)}"]
    for v in range(n):
        lines.append(" ".join(str(x) for x in sorted(adj[v])))
    return "\n".join(lines)


__all__ = ["export_graph"]
