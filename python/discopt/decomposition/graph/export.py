"""Serialize model-induced graphs to interchange formats.

The advisor's ``export_graph`` surface (design doc §14) dumps a graph for
external tooling — graph databases, partitioners, and visualizers. Four formats
cover the immediate needs:

- ``"json"``   — a node-link dict, easy to consume anywhere.
- ``"dot"``    — Graphviz, for quick rendering.
- ``"graphml"``— for graph-analysis tools (Gephi, networkx, igraph).
- ``"metis"``  — the graph input format for the METIS/KaHyPar partitioners; feed
  it to an external partitioner to obtain a bordered-block-diagonal ordering,
  then re-import the result via the GCG-style ``.dec`` reader (:func:`read_dec`).

Exports operate on the **variable dependency graph** (variable vertices, edges =
co-occurrence in a constraint), which is what partitioners consume. The bipartite
incidence is available via the ``bipartite`` flag for visualization.

``.dec`` interchange (:func:`write_dec` / :func:`read_dec`) round-trips a resolved
decomposition to/from the format SCIP and GCG use, so an external partitioner or
GCG run can supply the block/border structure.
"""

from __future__ import annotations

import json
import xml.sax.saxutils as _xml
from typing import TYPE_CHECKING

from discopt.decomposition.graph.base import ModelGraph

if TYPE_CHECKING:
    from discopt.decomposition.structure import DecompositionStructure

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


# ── GCG .dec interchange (T4.2) ───────────────────────────────


def _constraint_id(model, i: int) -> str:
    """Stable identifier for constraint ``i``: its name, else ``c{i}``."""
    name = getattr(model._constraints[i], "name", None)
    return name if name else f"c{i}"


def write_dec(structure, model, path: str) -> None:
    """Write a GCG-style ``.dec`` file for *structure* over *model*.

    Emits ``NBLOCKS``, one ``BLOCK k`` section listing that block's constraint
    identifiers, and a ``MASTERCONSS`` section for the coupling rows — the format
    SCIP/GCG read and write, so a discopt decomposition can be handed to (or taken
    from) that toolchain.
    """
    block_of_constraint = structure.block_of_constraint
    coupling = set(structure.coupling_constraints)
    n_blocks = structure.num_blocks
    per_block: list[list[str]] = [[] for _ in range(n_blocks)]
    master: list[str] = []
    for i in range(len(model._constraints)):
        if i in coupling:
            master.append(_constraint_id(model, i))
            continue
        b = block_of_constraint[i]
        if b is not None and b >= 0:
            per_block[b].append(_constraint_id(model, i))
    lines = ["NBLOCKS", str(n_blocks)]
    for b in range(n_blocks):
        lines.append(f"BLOCK {b + 1}")
        lines.extend(per_block[b])
    lines.append("MASTERCONSS")
    lines.extend(master)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_dec(path: str, model) -> "DecompositionStructure":
    """Read a GCG ``.dec`` file and return a ``DecompositionStructure``.

    The ``MASTERCONSS`` become the coupling rows; blocks are recomputed from the
    non-coupling constraints (so the partition is always internally consistent
    with the model, regardless of how the file grouped them).
    """
    from discopt.decomposition.structure import detect_decomposition

    id_to_index = {_constraint_id(model, i): i for i in range(len(model._constraints))}
    master_ids: list[str] = []
    section = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "NBLOCKS":
                section = "nblocks"
                continue
            if upper.startswith("BLOCK"):
                section = "block"
                continue
            if upper == "MASTERCONSS":
                section = "master"
                continue
            if section == "nblocks":
                section = None  # skip the count line
                continue
            if section == "master":
                master_ids.append(line)
    coupling = [id_to_index[m] for m in master_ids if m in id_to_index]
    return detect_decomposition(model, coupling=coupling)


__all__ = ["export_graph", "read_dec", "write_dec"]
