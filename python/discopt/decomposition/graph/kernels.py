"""Pure-Python graph kernels for the Decomposition Advisor.

These are the reference implementations of the primitives the advisor runs over
the graphs a model induces (see ``docs/design/decomposition-advisor.md``). They
mirror — and are the portable fallback for — the Rust kernels in
``crates/discopt-core/src/decomp/`` (``connected_components``,
``strongly_connected_components``, ``articulation_and_bridges``). Keeping a
tested Python copy means the graph layer works with or without the compiled
extension, and gives the Rust port a bit-for-bit oracle.

Two input conventions appear here:

- *clique lists* — ``cliques[i]`` is the set of variable indices touched by
  constraint ``i``. This is the natural output of scanning a model's expression
  DAG and the form the existing ``discopt.decomposition.structure`` layer uses.
- *edge lists* — ``(u, v)`` pairs over an explicit vertex set, for the
  articulation/bridge kernel which needs a true graph topology.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def connected_components(n: int, cliques: list[list[int]]) -> tuple[list[int], int]:
    """Union-find connected components over variable-index *cliques*.

    Two variables share a component iff a chain of shared constraints connects
    them. Returns ``(block_of_var, num_blocks)`` with block ids assigned in
    ascending variable order (a component's id is the order its
    smallest-indexed member is first seen), matching the block-ordering
    convention of the decomposition structure layer. ``O(n + E·α)``.
    """
    parent = list(range(n))
    size = [1] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        # Union by size (matches the Rust kernel). The final block ids are
        # relabelled by ascending first-seen member, so the chosen root does not
        # affect the output — this is a pure worst-case speedup.
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if size[ri] < size[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        size[ri] += size[rj]

    for clique in cliques:
        if len(clique) <= 1:
            continue
        anchor = clique[0]
        for i in clique[1:]:
            union(anchor, i)

    root_to_block: dict[int, int] = {}
    block_of = [0] * n
    for i in range(n):
        r = find(i)
        if r not in root_to_block:
            root_to_block[r] = len(root_to_block)
        block_of[i] = root_to_block[r]
    return block_of, len(root_to_block)


def bearing_blocks(n: int, cliques: list[list[int]]) -> int:
    """Number of connected components that contain at least one constraint.

    Counting *constraint-bearing* components (rather than raw variable
    components) avoids spurious singletons: a variable appearing in only one
    constraint becomes isolated when that constraint is dropped, which must not
    be mistaken for a genuine block split.
    """
    block_of, _ = connected_components(n, cliques)
    bearing = {block_of[c[0]] for c in cliques if c}
    return len(bearing)


def bridge_cliques_status(cliques: list[list[int]], n: int, budget: int) -> tuple[set[int], bool]:
    """Bridge-constraint scan with an explicit truncation flag.

    Returns ``(coupling_indices, truncated)`` where ``truncated`` is True iff the
    ``O(m·(n+E))`` scan was skipped because its estimated cost exceeded *budget*
    (so ``set()`` means "no coupling found", not "gave up" — S3). A truncated
    scan is logged at WARNING so it is never silent.
    """
    nontrivial = [c for c in cliques if len(c) >= 2]
    base = bearing_blocks(n, cliques)
    if base >= 2:
        # Already separable; no single coupling clique is needed.
        return set(), False
    est = len(nontrivial) * (n + sum(len(c) for c in nontrivial))
    if est > budget:
        logger.warning(
            "bridge-constraint scan skipped: estimated cost %d exceeds budget %d "
            "(%d constraints, %d vars); coupling not auto-detected — annotate with "
            "model.mark_coupling(...) or install the hypergraph detector.",
            est,
            budget,
            len(nontrivial),
            n,
        )
        return set(), True
    coupling: set[int] = set()
    for i, clique in enumerate(cliques):
        if len(clique) < 2:
            continue
        without = [c for j, c in enumerate(cliques) if j != i]
        if bearing_blocks(n, without) > base:
            coupling.add(i)
    return coupling, False


def bridge_cliques(cliques: list[list[int]], n: int, budget: int) -> set[int]:
    """Indices of cliques whose sole removal raises the bearing-component count.

    Thin wrapper over :func:`bridge_cliques_status` that drops the truncation
    flag (kept for backward compatibility). Prefer the ``_status`` variant when
    the caller needs to distinguish "no coupling" from "scan too large".
    """
    coupling, _ = bridge_cliques_status(cliques, n, budget)
    return coupling


def dependency_edges(
    cliques: list[list[int]], max_clique_expand: int = 64
) -> list[tuple[int, int]]:
    """Variable–variable edges induced by co-occurrence in a constraint.

    Each clique of size ``k`` contributes its pairwise edges. To avoid the
    ``O(k²)`` blow-up of a dense coupling row, cliques wider than
    *max_clique_expand* are expanded as a **star** around their first member
    (which preserves connectivity for component analysis) rather than a full
    clique. Returns an undirected edge list; de-duplication is left to the graph
    builder.
    """
    edges: list[tuple[int, int]] = []
    for clique in cliques:
        k = len(clique)
        if k < 2:
            continue
        if k > max_clique_expand:
            anchor = clique[0]
            for v in clique[1:]:
                edges.append((anchor, v))
        else:
            for a in range(k):
                for b in range(a + 1, k):
                    edges.append((clique[a], clique[b]))
    return edges


def _rust_kernels():
    """The compiled Rust kernels module, or ``None`` if the extension is absent.

    Probed once. The Python reference kernels are the fallback and the
    correctness oracle, so the graph layer works with or without the extension.
    """
    global _RUST
    if _RUST is _UNSET:
        try:
            import discopt._rust as _r

            _RUST = _r if hasattr(_r, "decomp_articulation_and_bridges") else None
        except Exception:
            _RUST = None
    return _RUST


_UNSET = object()
_RUST = _UNSET


def articulation_and_bridges(
    n: int, edges: list[tuple[int, int]]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Articulation points and bridges of an undirected simple graph.

    Dispatches to the Rust kernel (``crates/discopt-core/src/decomp/mincut.rs``)
    when the compiled extension is available, and to the pure-Python reference
    otherwise; both are bit-for-bit equivalent (articulation points ascending,
    bridges sorted ``(min, max)``). Input edges are de-duplicated into a simple
    graph first. Iterative DFS (safe for deep graphs), ``O(n + E)``.
    """
    rust = _rust_kernels()
    if rust is not None:
        pts, bridges = rust.decomp_articulation_and_bridges(n, [(int(a), int(b)) for a, b in edges])
        return [int(p) for p in pts], sorted((int(a), int(b)) for a, b in bridges)
    return _articulation_and_bridges_py(n, edges)


def _articulation_and_bridges_py(
    n: int, edges: list[tuple[int, int]]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Pure-Python reference for :func:`articulation_and_bridges` (and its oracle)."""
    # Build a simple undirected adjacency.
    adj: list[list[int]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for a, b in edges:
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        adj[a].append(b)
        adj[b].append(a)

    disc = [-1] * n
    low = [0] * n
    is_art = [False] * n
    bridges: list[tuple[int, int]] = []
    timer = 0

    for s in range(n):
        if disc[s] != -1:
            continue
        disc[s] = low[s] = timer
        timer += 1
        stack: list[list[int]] = [[s, -1, 0]]  # [vertex, parent, next-child-idx]
        root_children = 0
        while stack:
            u, parent, ci = stack[-1]
            nbrs = adj[u]
            if ci < len(nbrs):
                stack[-1][2] += 1
                w = nbrs[ci]
                if w == parent:
                    continue
                if disc[w] == -1:
                    if parent == -1:
                        root_children += 1
                    disc[w] = low[w] = timer
                    timer += 1
                    stack.append([w, u, 0])
                elif disc[w] < low[u]:
                    low[u] = disc[w]
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if p != s and low[u] >= disc[p]:
                        is_art[p] = True
                    if low[u] > disc[p]:
                        bridges.append((p, u) if p < u else (u, p))
        if root_children > 1:
            is_art[s] = True

    articulation_points = [v for v in range(n) if is_art[v]]
    bridges.sort()
    return articulation_points, bridges


__all__ = [
    "articulation_and_bridges",
    "bearing_blocks",
    "bridge_cliques",
    "connected_components",
    "dependency_edges",
]
