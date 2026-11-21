"""Optimal complicating-feature separators by integer programming.

A decomposition needs a set of *complicating* features whose removal splits the
model into independent subproblems: variables for Benders (fix them in a
master), constraints for Lagrangian relaxation (dualize them). The bridge and
integer-projection detectors find only the cases where one constraint, or the
integer variables, happen to do it. This module solves for the separator
directly, after Kalamaris & Maravelias (2026, arXiv:2611.23472, §4): on a
projection of the element graph, choose a vertex set ``R`` so that the rest
splits into at least ``|C|`` components of balanced weight, trading imbalance
against ``epsilon * weight(R)``.

**Soundness does not depend on the program's optimality.** A separator is used
only after it is lifted back to element columns/rows and the split is
re-verified on :class:`~discopt.decomposition.graph.element.ElementGraph`
(:attr:`Separator.blocks`); a lifted separator that does not split the model
raises :class:`SeparatorError`. Benders and Lagrangian bounds are valid for
*any* partition, so a budget-limited, sub-optimal separator costs speed, never
correctness.

**Coarsening.** The program is solved on a reduced graph: heavy-edge matching
(Karypis & Kumar 1998) that only merges vertices of the same *type* -- the same
array for variables, the same referenced-array signature for constraint rows
-- following the paper's index coarsening, which never merges different types.
Measured on a 150-column capacity-expansion LP: unrestricted matching merged the
design variables (hubs) with operational ones and returned 14-column separators
where the 6 design columns suffice; type-preserving matching returns the 6. Any
coarsening is safe: a vertex set that separates the reduced graph separates the
original once lifted, because merging only adds adjacency. Same-type elements
that share no row are merged through shared neighbours (two hops); see
:func:`coarsen_by_matching` for the measurement.

**Symmetry.** Components are interchangeable, which makes the program hard
(the paper reports hours for large ``|C|``). Vertex ``v`` may only join a
component ``c <= v``, the representative break; measured 0.71 s vs 2.39 s for
the paper's size ordering at ``|C| = 3`` with identical separators.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import scipy.sparse as sp

from discopt.decomposition.graph.element import CoarseGraph, ElementGraph

Kind = Literal["variables", "constraints"]


class SeparatorError(RuntimeError):
    """A lifted separator does not split the model it was computed for."""


@dataclass(frozen=True)
class Separator:
    """A verified complicating-feature separator.

    Attributes
    ----------
    kind : {"variables", "constraints"}
    num_components : int
        The ``|C|`` the program was asked for.
    columns, rows : tuple[int, ...]
        The separator, lifted to element columns (``kind="variables"``) or
        element rows (``kind="constraints"``); the other is empty.
    blocks : int
        Constraint-bearing blocks of the element graph once the separator is
        removed -- measured, not taken from the program. ``>= num_components``.
    block_columns : tuple[tuple[int, ...], ...]
        Columns of each of those blocks.
    weight : int
        Number of separator elements.
    imbalance : float
        ``sum_{c<c'} |S_c - S_c'| / (|C| - 1)`` over the measured blocks'
        column counts (the paper's balance measure, recomputed after lifting).
    status : str
        The program's solver status (``optimal``, or ``iteration_limit`` for a
        node-budget incumbent).
    seconds : float
    """

    kind: Kind
    num_components: int
    columns: tuple[int, ...]
    rows: tuple[int, ...]
    blocks: int
    block_columns: tuple[tuple[int, ...], ...]
    weight: int
    imbalance: float
    status: str
    seconds: float


# ── projections ─────────────────────────────────────────────────


def _projection(graph: ElementGraph, kind: Kind):
    """``(n, edges, type_of, active)`` of the variable or constraint projection.

    ``edges`` maps ``(a, b)`` (``a < b``) to the number of shared rows (variable
    projection) or shared columns (constraint projection). ``active`` marks the
    vertices that take part: a column no row touches, or a row that touches no
    column, would otherwise count as a free one-vertex "component".
    """
    edges: dict[tuple[int, int], int] = {}
    if kind == "variables":
        n = graph.num_cols
        active = np.zeros(n, dtype=bool)
        for clique in graph.row_cliques:
            cl = clique.tolist()
            active[cl] = True
            for a, b in itertools.combinations(cl, 2):
                edges[(a, b)] = edges.get((a, b), 0) + 1
        type_of = np.asarray(graph.col_var, dtype=np.int64)
        return n, edges, type_of, active
    n = graph.num_rows
    rows_of_col: dict[int, list[int]] = {}
    for i, clique in enumerate(graph.row_cliques):
        for j in clique.tolist():
            rows_of_col.setdefault(j, []).append(i)
    for rs in rows_of_col.values():
        for a, b in itertools.combinations(rs, 2):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    sig: dict[frozenset, int] = {}
    type_of = np.array(
        [
            sig.setdefault(frozenset(int(graph.col_var[j]) for j in clique.tolist()), len(sig))
            for clique in graph.row_cliques
        ],
        dtype=np.int64,
    )
    active = np.array([clique.size > 0 for clique in graph.row_cliques], dtype=bool)
    return n, edges, type_of, active


# ── coarsening ──────────────────────────────────────────────────


def coarsen_by_matching(
    n: int,
    edges: dict[tuple[int, int], int],
    type_of: np.ndarray,
    target: int,
    pinned: Sequence[int] = (),
    hub_degree: int = 64,
) -> tuple[np.ndarray, int, dict[tuple[int, int], int], np.ndarray]:
    """Type-preserving heavy-edge matching until ``<= target`` vertices.

    Returns ``(group, n_coarse, coarse_edges, coarse_weight)``: ``group[v]`` is
    the coarse vertex of original vertex ``v``. Vertices in *pinned* are only
    merged with other pinned vertices (a forced-separator vertex must not drag
    a free one into the separator). A vertex with no unmatched same-type
    neighbour is matched to the same-type vertex it shares the most neighbour
    weight with (two hops away): elements of one array rarely share a row
    (``Q[s, t]`` and ``Q[s, t+1]`` do not), but share neighbours, and the
    shared weight ranks a time step (shares ``K`` and ``I``) above another
    scenario (shares only ``K``). Measured on a 774-column capacity LP: the
    one-hop rule stalled at 395 vertices and the design separator cost 4.9 s;
    two-hop reaches 54 vertices and finds it in 0.09 s (3,078 columns: still 54
    vertices, 0.09 s). Deterministic: vertices are visited by ascending degree
    then index, partners by descending (edge or shared) weight, ascending coarse
    weight, then index. Stops early when a round merges nothing.

    The two-hop search does not pass through a *hub*, an intermediate of more
    than *hub_degree* neighbours: a design variable ``K[g, p]`` is adjacent to
    every operational element of every scenario, so walking through it is
    quadratic in the model and only ever proposes the cross-scenario merges the
    ranking exists to avoid. Measured at 12,294 columns: 11.3 M candidate
    checks and 13.2 s of the separator search's 13.4 s without the cap.
    """
    group = np.arange(n, dtype=np.int64)
    cur_n = n
    cur_edges = dict(edges)
    weight = np.ones(n, dtype=np.int64)
    kind = np.asarray(type_of, dtype=np.int64).copy()
    pin = np.zeros(n, dtype=bool)
    pin[np.asarray(list(pinned), dtype=np.int64)] = True
    while cur_n > target:
        adj: list[dict[int, int]] = [{} for _ in range(cur_n)]
        for (a, b), wt in cur_edges.items():
            adj[a][b] = wt
            adj[b][a] = wt
        match = [-1] * cur_n
        for v in sorted(range(cur_n), key=lambda u: (len(adj[u]), u)):
            if match[v] != -1:
                continue
            match[v] = v

            def free(u: int, v: int = v) -> bool:
                return match[u] == -1 and u != v and kind[u] == kind[v] and pin[u] == pin[v]

            score = {u: wt for u, wt in adj[v].items() if free(u)}
            if not score:  # no same-type neighbour: look two hops away
                for x, w_vx in adj[v].items():
                    if len(adj[x]) > hub_degree:
                        continue  # never look through a hub (see docstring)
                    for u, w_xu in adj[x].items():
                        if free(u):
                            score[u] = score.get(u, 0) + min(w_vx, w_xu)
            if score:
                u = min(score, key=lambda k: (-score[k], weight[k], k))
                match[v], match[u] = u, v
        new_id: dict[int, int] = {}
        nid = np.empty(cur_n, dtype=np.int64)
        for v in range(cur_n):
            r = min(v, match[v])
            if r not in new_id:
                new_id[r] = len(new_id)
            nid[v] = new_id[r]
        m = len(new_id)
        if m == cur_n:
            break  # no same-type pair within two hops: no further progress
        new_edges: dict[tuple[int, int], int] = {}
        for (a, b), wt in cur_edges.items():
            x, y = int(nid[a]), int(nid[b])
            if x != y:
                key = (x, y) if x < y else (y, x)
                new_edges[key] = new_edges.get(key, 0) + wt
        new_w = np.zeros(m, dtype=np.int64)
        np.add.at(new_w, nid, weight)
        new_kind = np.empty(m, dtype=np.int64)
        new_kind[nid] = kind
        new_pin = np.zeros(m, dtype=bool)
        new_pin[nid] = pin
        group = nid[group]
        cur_n, cur_edges, weight, kind, pin = m, new_edges, new_w, new_kind, new_pin
    return group, cur_n, cur_edges, weight


# ── the partitioning program ────────────────────────────────────


def _solve_partition(
    nv: int,
    edges: Sequence[tuple[int, int]],
    weight: np.ndarray,
    num_components: int,
    epsilon: float,
    forced: Sequence[int],
    node_limit: int,
    time_limit: float,
) -> tuple[list[int] | None, str]:
    """Kalamaris & Maravelias eqs. (30)-(36), (40) with ``gamma = 0``.

    Variables ``R_v`` (separator), ``X_{v,c}`` (assignment), ``S_c`` (component
    weight), ``D_{c,c'}`` (pairwise weight gap). Returns the separator's
    coarse vertices, or ``None`` when no incumbent was found.
    """
    from discopt.solvers import SolveStatus, milp_highs
    from discopt.solvers.lp_backend import get_milp_solver

    C = num_components
    pairs = list(itertools.combinations(range(C), 2))
    nX = nv * C
    off_X, off_S, off_D = nv, nv + nX, nv + nX + C
    N = off_D + len(pairs)

    def ix(v: int, c: int) -> int:
        return off_X + v * C + c

    cost = np.zeros(N)
    cost[:nv] = epsilon * weight
    cost[off_D:] = 1.0 / (C - 1)

    ri: list[int] = []
    ci: list[int] = []
    vi: list[float] = []
    rhs: list[float] = []

    def add(entries, b):
        r = len(rhs)
        for j, a in entries:
            ri.append(r)
            ci.append(j)
            vi.append(a)
        rhs.append(b)

    for a, b in edges:  # (30): adjacent non-separator vertices share a component
        for c in range(C):
            add([(ix(a, c), 1.0), (ix(b, c), -1.0), (a, -1.0), (b, -1.0)], 0.0)
            add([(ix(b, c), 1.0), (ix(a, c), -1.0), (a, -1.0), (b, -1.0)], 0.0)
    for c in range(C):  # (33): every component is non-empty
        add([(off_S + c, -1.0)], -1.0)
    for k, (c, d) in enumerate(pairs):  # (35), (36)
        add([(off_S + c, 1.0), (off_S + d, -1.0), (off_D + k, -1.0)], 0.0)
        add([(off_S + d, 1.0), (off_S + c, -1.0), (off_D + k, -1.0)], 0.0)
    A_ub = sp.csr_matrix((vi, (ri, ci)), shape=(len(rhs), N))
    b_ub = np.asarray(rhs, dtype=np.float64)

    er: list[int] = []
    ec: list[int] = []
    ev: list[float] = []
    eb: list[float] = []
    for v in range(nv):  # (31): assigned to exactly one component or separated
        q = len(eb)
        er += [q] * (C + 1)
        ec += [v] + [ix(v, c) for c in range(C)]
        ev += [1.0] * (C + 1)
        eb.append(1.0)
    for c in range(C):  # (32): component weight
        q = len(eb)
        er += [q] * (nv + 1)
        ec += [off_S + c] + [ix(v, c) for v in range(nv)]
        ev += [1.0] + [-float(w) for w in weight]
        eb.append(0.0)
    A_eq = sp.csr_matrix((ev, (er, ec)), shape=(len(eb), N))
    b_eq = np.asarray(eb, dtype=np.float64)

    bounds = [(0.0, 1.0)] * (nv + nX) + [(0.0, math.inf)] * (C + len(pairs))
    for v in forced:
        bounds[v] = (1.0, 1.0)
    for v in range(nv):  # representative symmetry break: v joins only c <= v
        for c in range(v + 1, C):
            bounds[ix(v, c)] = (0.0, 0.0)
    integrality = np.array([1] * (nv + nX) + [0] * (C + len(pairs)), dtype=np.int32)

    kwargs = dict(
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=integrality,
        time_limit=time_limit,
        gap_tolerance=1e-6,
    )
    res = None
    if _partition_engine() == "highs":
        # A partition heuristic needs no certified bound: skip HiGHS's
        # presolve-free cross-check solve (measured to halve the time).
        res = milp_highs.solve_milp(
            cost, confirm_bound_from=math.inf, max_nodes=node_limit, **kwargs
        )
        if res.status == SolveStatus.ERROR and _engine_falls_back():
            res = None  # as the pure-MILP route: retry an ``error`` on the Rust engine
    if res is None:
        res = get_milp_solver(backend="simplex")(cost, max_nodes=node_limit, **kwargs)
    status = getattr(res.status, "name", str(res.status)).lower()
    if res.x is None:
        return None, status
    x = np.asarray(res.x, dtype=np.float64)
    return [v for v in range(nv) if x[v] > 0.5], status


def _partition_engine() -> str:
    """``"highs"`` or ``"rust"``: the pure-MILP engine (``DISCOPT_LP_ENGINE``).

    The partitioning program is a standalone pure MILP, so it follows the
    pure-MILP seam rather than the decomposition-master selector (the in-house
    simplex B&B, chosen for masters). Measured on the 150-column
    capacity-expansion LP at ``|C| = 3`` (78 coarse vertices): HiGHS optimal in
    0.71 s; the simplex B&B 5.7-6.4 s, and at a 2 s limit only a degenerate
    incumbent (147 of 150 columns in the separator).
    """
    from discopt._optional_highs import highs_available
    from discopt.solver import _lp_engine

    engine = _lp_engine()
    if engine == "rust":
        return "rust"
    if engine == "highs":
        return "highs"  # explicit: a missing highspy raises at the call
    return "highs" if highs_available() else "rust"


def _engine_falls_back() -> bool:
    from discopt.solver import _lp_engine

    return _lp_engine() == "auto"


# ── driver ──────────────────────────────────────────────────────


def find_separators(
    graph: ElementGraph,
    kind: Kind,
    *,
    max_components: int = 4,
    epsilon: float = 2.0,
    target: int = 60,
    node_limit: int = 200,
    time_limit: float = 30.0,
    forced_columns: Sequence[int] = (),
    max_fraction: float = 0.5,
    skip_covered: bool = True,
    coarse: CoarseGraph | None = None,
) -> list[Separator]:
    """Separators for ``|C| = 2 .. max_components``, verified and de-duplicated.

    *forced_columns* (``kind="variables"`` only) must be in every separator --
    e.g. the integer columns, which a Benders driver requires in the master.
    Each ``|C|`` gets *node_limit* branch-and-bound nodes, a deterministic
    budget, so the answer does not depend on machine load; *time_limit* is a
    per-``|C|`` wall-clock backstop sized well above the node budget. A program
    with no incumbent contributes nothing. Every returned separator has been
    lifted and its split measured on *graph*.

    With *skip_covered* a ``|C|`` is not solved when an earlier separator is
    already verified to leave at least ``|C|`` blocks: that question is
    answered, and each further program costs 2-3 s of HiGHS root work on a large
    graph. Measured on the capacity-expansion LP family (whole call, coarsening
    included): 3.78 s -> 0.58 s at 3,078 columns and 2.54 s -> 0.14 s at 774,
    each still returning the design separator; at 150 columns 1.02 s -> 0.89 s,
    returning the ``|C| = 2`` split (14 columns, 7 blocks) but no longer the
    6-column, 3-block one ``|C| = 3`` finds. ``skip_covered=False`` runs every
    ``|C|``.

    *coarse* replaces the automatic coarsening with the user's own: a
    :class:`~discopt.decomposition.graph.element.CoarseGraph` from
    ``graph.coarsen(var_axes, con_axes)`` -- the paper's index coarsening, where
    the modeller names the axes to merge ("over time, keep scenarios"). Its
    super-columns (``kind="variables"``) or super-rows (``kind="constraints"``)
    are the program's vertices as given; *target* is then unused. A super-vertex
    holding any forced column is forced whole. It must have been built from
    *graph* (checked by size; a mismatch raises ``ValueError``).

    A separator holding more than *max_fraction* of the active elements is
    dropped: a time-limited incumbent can isolate a few vertices by putting
    almost everything else in the separator (measured: 135 of 147 rows), which
    is a valid split and a useless decomposition.
    """
    if kind not in ("variables", "constraints"):
        raise ValueError(f"kind must be 'variables' or 'constraints', not {kind!r}")
    if forced_columns and kind != "variables":
        raise ValueError("forced_columns applies to kind='variables' only")
    n, edges, type_of, active = _projection(graph, kind)
    forced = np.asarray(list(forced_columns), dtype=np.int64)
    # Inactive vertices are dropped from the program (they cannot form a
    # component); they never belong to a separator either.
    keep = np.flatnonzero(active)
    local = np.full(n, -1, dtype=np.int64)
    local[keep] = np.arange(keep.size)
    k_edges = {
        (int(local[a]), int(local[b])): w
        for (a, b), w in edges.items()
        if local[a] >= 0 and local[b] >= 0
    }
    k_forced = [int(local[v]) for v in forced if local[v] >= 0]
    if coarse is None:
        group, nv, c_edges, c_weight = coarsen_by_matching(
            int(keep.size), k_edges, type_of[keep], target, pinned=k_forced
        )
    else:
        owner = coarse.col_of if kind == "variables" else coarse.row_of
        expected = graph.num_cols if kind == "variables" else graph.num_rows
        if owner.size != expected:
            raise ValueError(
                f"the CoarseGraph maps {owner.size} {kind} elements but the graph has "
                f"{expected}; it was built from a different graph"
            )
        group, nv, c_edges, c_weight = _quotient(np.asarray(owner)[keep], k_edges)
    c_forced = sorted({int(group[v]) for v in k_forced})

    out: list[Separator] = []
    seen: set[tuple[int, ...]] = set()
    covered = 1  # the most blocks any separator so far is verified to leave
    for C in range(2, max_components + 1):
        if C > nv - len(c_forced):
            break
        if skip_covered and C <= covered:
            continue
        t0 = time.perf_counter()
        R, status = _solve_partition(
            nv, list(c_edges), c_weight, C, epsilon, c_forced, node_limit, time_limit
        )
        seconds = time.perf_counter() - t0
        if R is None:
            continue
        members = keep[np.isin(group, R)]
        lifted = tuple(sorted(int(v) for v in members))
        if lifted in seen or len(lifted) > max_fraction * keep.size:
            continue
        seen.add(lifted)
        found = _verify(graph, kind, C, lifted, status, seconds)
        covered = max(covered, found.blocks)
        out.append(found)
    return out


def _quotient(
    owner: np.ndarray, edges: dict[tuple[int, int], int]
) -> tuple[np.ndarray, int, dict[tuple[int, int], int], np.ndarray]:
    """Collapse vertices by a given grouping (*owner*), like one coarsening round.

    Returns ``(group, n_groups, edges, weight)`` in the shape
    :func:`coarsen_by_matching` returns, with groups renumbered densely in order
    of first appearance.
    """
    remap: dict[int, int] = {}
    group = np.array([remap.setdefault(int(o), len(remap)) for o in owner], dtype=np.int64)
    out: dict[tuple[int, int], int] = {}
    for (a, b), w in edges.items():
        x, y = int(group[a]), int(group[b])
        if x != y:
            key = (x, y) if x < y else (y, x)
            out[key] = out.get(key, 0) + w
    weight = np.bincount(group, minlength=len(remap)).astype(np.int64)
    return group, len(remap), out, weight


def _verify(
    graph: ElementGraph, kind: Kind, C: int, lifted: tuple[int, ...], status: str, seconds: float
) -> Separator:
    """Measure the lifted separator's split on the element graph; refuse a non-split."""
    if kind == "variables":
        block_of, _ = graph.components(drop_cols=lifted)
        dropped_rows: set[int] = set()
    else:
        block_of, _ = graph.components(drop_rows=lifted)
        dropped_rows = set(lifted)
    bearing: dict[int, None] = {}
    for i, clique in enumerate(graph.row_cliques):
        if i in dropped_rows:
            continue
        kept = [b for b in block_of[clique].tolist() if b >= 0]
        if kept:
            bearing.setdefault(kept[0], None)
    cols: dict[int, list[int]] = {b: [] for b in bearing}
    for j, b in enumerate(block_of.tolist()):
        if b in cols:
            cols[b].append(j)
    blocks = sorted((tuple(c) for c in cols.values()), key=lambda c: c[0])
    if len(blocks) < C:
        raise SeparatorError(
            f"the {kind} separator for |C|={C} ({len(lifted)} elements) leaves "
            f"{len(blocks)} constraint-bearing block(s) on the element graph; the "
            "partitioning program and the graph disagree, which is a defect."
        )
    sizes = [len(b) for b in blocks]
    imbalance = sum(abs(a - b) for a, b in itertools.combinations(sizes, 2)) / max(
        1, len(sizes) - 1
    )
    return Separator(
        kind=kind,
        num_components=C,
        columns=lifted if kind == "variables" else (),
        rows=lifted if kind == "constraints" else (),
        blocks=len(blocks),
        block_columns=tuple(blocks),
        weight=len(lifted),
        imbalance=float(imbalance),
        status=status,
        seconds=seconds,
    )


__all__ = ["Separator", "SeparatorError", "coarsen_by_matching", "find_separators"]
