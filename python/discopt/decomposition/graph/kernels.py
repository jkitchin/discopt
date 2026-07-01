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


def connected_components(n: int, cliques: list[list[int]]) -> tuple[list[int], int]:
    """Union-find connected components over variable-index *cliques*.

    Two variables share a component iff a chain of shared constraints connects
    them. Returns ``(block_of_var, num_blocks)`` with block ids assigned in
    ascending variable order (a component's id is the order its
    smallest-indexed member is first seen), matching the block-ordering
    convention of the decomposition structure layer. ``O(n + E·α)``.
    """
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


def bridge_cliques(cliques: list[list[int]], n: int, budget: int) -> set[int]:
    """Indices of cliques whose sole removal raises the bearing-component count.

    This is the graph core of the "bridge-constraint" coupling heuristic: a
    constraint is coupling when dropping it alone disconnects the model. Guarded
    by *budget* (an estimate of the scan cost); returns ``set()`` when the model
    is already separable, when nothing qualifies, or when the scan would exceed
    the budget. ``O(m · (n + E))`` when it runs.
    """
    nontrivial = [c for c in cliques if len(c) >= 2]
    base = bearing_blocks(n, cliques)
    if base >= 2:
        # Already separable; no single coupling clique is needed.
        return set()
    est = len(nontrivial) * (n + sum(len(c) for c in nontrivial))
    if est > budget:
        return set()
    coupling: set[int] = set()
    for i, clique in enumerate(cliques):
        if len(clique) < 2:
            continue
        without = [c for j, c in enumerate(cliques) if j != i]
        if bearing_blocks(n, without) > base:
            coupling.add(i)
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


def articulation_and_bridges(
    n: int, edges: list[tuple[int, int]]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Articulation points and bridges of an undirected simple graph.

    Mirrors ``crates/discopt-core/src/decomp/mincut.rs``. Input edges are
    de-duplicated into a simple graph first. Returns
    ``(articulation_points, bridges)`` where bridges are ``(min, max)`` pairs.
    Iterative DFS (safe for deep graphs), ``O(n + E)``.
    """
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
