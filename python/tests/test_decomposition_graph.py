"""Tests for the Decomposition Advisor graph infrastructure (Phase 1).

Covers the pure-Python graph kernels (mirrors of the Rust
``crates/discopt-core/src/decomp/`` kernels), the ``ModelGraph`` views built
from a discopt ``Model``, the version-keyed ``GraphCache``, and ``export_graph``.
"""

import discopt.modeling as dm
import pytest
from discopt.decomposition.graph import (
    GraphCache,
    GraphKind,
    ModelGraph,
    build_graph,
    export_graph,
    kernels,
)

# ── kernels: connected components ─────────────────────────────


def test_cc_two_disjoint_blocks():
    # cliques: {0,1}, {2,3} -> two blocks
    block_of, k = kernels.connected_components(4, [[0, 1], [2, 3]])
    assert k == 2
    assert block_of[0] == block_of[1]
    assert block_of[2] == block_of[3]
    assert block_of[0] != block_of[2]
    # deterministic ascending labels
    assert block_of[0] == 0
    assert block_of[2] == 1


def test_cc_single_block_chain():
    _, k = kernels.connected_components(4, [[0, 1], [1, 2], [2, 3]])
    assert k == 1


def test_cc_isolated_vars():
    block_of, k = kernels.connected_components(3, [])
    assert k == 3
    assert block_of == [0, 1, 2]


def test_bearing_blocks_ignores_singletons():
    # var 2 appears in no constraint; only one constraint-bearing block.
    assert kernels.bearing_blocks(3, [[0, 1]]) == 1


# ── kernels: bridge detection ─────────────────────────────────


def test_bridge_clique_found():
    # {0,1}, {1,2}, {2,3}: removing the middle clique splits {0,1} from {2,3}.
    coupling = kernels.bridge_cliques([[0, 1], [1, 2], [2, 3]], 4, budget=10**9)
    assert 1 in coupling


def test_bridge_already_separable_returns_empty():
    # two disjoint blocks: no coupling clique needed.
    assert kernels.bridge_cliques([[0, 1], [2, 3]], 4, budget=10**9) == set()


def test_bridge_budget_guard():
    # a tiny budget forces the scan to bail out.
    assert kernels.bridge_cliques([[0, 1], [1, 2], [2, 3]], 4, budget=1) == set()


# ── kernels: articulation & bridges (mirror of mincut.rs) ─────


def test_articulation_path():
    # 0-1-2-3: internal vertices 1,2 are cut vertices; all edges are bridges.
    pts, bridges = kernels.articulation_and_bridges(4, [(0, 1), (1, 2), (2, 3)])
    assert pts == [1, 2]
    assert bridges == [(0, 1), (1, 2), (2, 3)]


def test_articulation_cycle_has_no_cuts():
    pts, bridges = kernels.articulation_and_bridges(3, [(0, 1), (1, 2), (2, 0)])
    assert pts == []
    assert bridges == []


def test_articulation_two_triangles_bridge():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    pts, bridges = kernels.articulation_and_bridges(6, edges)
    assert pts == [2, 3]
    assert bridges == [(2, 3)]


def test_articulation_rust_matches_python_reference():
    # The public kernel dispatches to Rust when the compiled extension is
    # present; it must be bit-for-bit equivalent to the pure-Python reference.
    graphs = [
        (4, [(0, 1), (1, 2), (2, 3)]),  # path
        (3, [(0, 1), (1, 2), (2, 0)]),  # triangle
        (6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]),  # bridge
        (4, [(0, 1), (0, 2), (0, 3)]),  # star
        (5, [(0, 1), (2, 3), (3, 4), (4, 2)]),  # disconnected
        (5, [(0, 1), (1, 0), (1, 2), (2, 3)]),  # duplicate edge
    ]
    for n, edges in graphs:
        assert kernels.articulation_and_bridges(n, edges) == kernels._articulation_and_bridges_py(
            n, edges
        )


def test_articulation_uses_rust_when_available():
    # In this environment the compiled extension is installed, so the fast path
    # is live. (No-op assertion elsewhere; the equivalence test covers behavior.)
    rust = kernels._rust_kernels()
    if rust is not None:
        pts, bridges = rust.decomp_articulation_and_bridges(4, [(0, 1), (1, 2), (2, 3)])
        assert list(pts) == [1, 2]


def test_rust_kernels_reject_out_of_range_vertices():
    # An edge referencing a vertex >= n must raise a clean ValueError, not panic
    # across the PyO3 boundary (PanicException). Guards the bindings' validation.
    rust = kernels._rust_kernels()
    if rust is None:
        return
    for fn in (
        rust.decomp_connected_components,
        rust.decomp_strongly_connected_components,
        rust.decomp_articulation_and_bridges,
    ):
        with pytest.raises(ValueError):
            fn(3, [(0, 5)])  # vertex 5 is out of range for n=3


def _normalize_scc_labels(comp):
    """Canonicalize SCC labels to a partition signature (label-value independent).

    Two labelings are equivalent iff they induce the same partition. We compare
    the *partition* (which vertices share a component) rather than raw ids so a
    genuine divergence is caught while a benign relabeling is not — but the Rust
    and Python kernels also share the exact id-assignment convention, so the raw
    lists match too (asserted separately).
    """
    canon: dict[int, int] = {}
    out = []
    for c in comp:
        if c not in canon:
            canon[c] = len(canon)
        out.append(canon[c])
    return out


def test_cc_rust_matches_python_reference():
    # The Rust ``decomp_connected_components`` (edge-list) must be bit-for-bit
    # equivalent to the pure-Python edge-list reference. #391 item 4: the kernel
    # is a future speed path for structure.py's block detection; a silent labeling
    # divergence would change Benders/Lagrangian block detection.
    rust = kernels._rust_kernels()
    if rust is None:
        pytest.skip("Rust extension not built")
    graphs = [
        (4, [(0, 1), (1, 2), (2, 3)]),  # single chain
        (4, [(0, 1), (2, 3)]),  # two blocks
        (3, []),  # all isolated
        (5, [(0, 1), (1, 2), (2, 0), (3, 4)]),  # triangle + edge
        (6, [(0, 1), (1, 0), (2, 3)]),  # duplicate edge + a block + isolated
        (5, [(4, 0), (0, 2)]),  # first-seen ordering not vertex 0
    ]
    for n, edges in graphs:
        got = [int(x) for x in rust.decomp_connected_components(n, edges)]
        ref, _ = kernels._connected_components_edges_py(n, edges)
        assert got == ref, f"CC divergence on n={n}, edges={edges}: {got} != {ref}"


def test_scc_rust_matches_python_reference():
    # The Rust ``decomp_strongly_connected_components`` must be bit-for-bit
    # equivalent to the pure-Python Tarjan reference (same finalize-order id
    # convention). #391 item 4: SCC underpins stage/dual-dependency ordering; a
    # divergence would mis-detect mutually-dependent stages.
    rust = kernels._rust_kernels()
    if rust is None:
        pytest.skip("Rust extension not built")
    graphs = [
        (3, [(0, 1), (1, 2), (2, 0)]),  # one cycle
        (3, [(0, 1), (1, 2)]),  # DAG: three SCCs
        (4, [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]),  # two cycles, one-way link
        (5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]),  # nested cycles
        (4, []),  # all singletons
        (5, [(0, 0), (0, 1), (1, 2), (2, 1)]),  # self-loop + cycle
    ]
    for n, arcs in graphs:
        got = [int(x) for x in rust.decomp_strongly_connected_components(n, arcs)]
        ref, _ = kernels._strongly_connected_components_py(n, arcs)
        # Same partition …
        assert _normalize_scc_labels(got) == _normalize_scc_labels(ref), (
            f"SCC partition divergence on n={n}, arcs={arcs}: {got} vs {ref}"
        )
        # … and the exact shared id convention (finalize order).
        assert got == ref, f"SCC label divergence on n={n}, arcs={arcs}: {got} != {ref}"


def test_cc_scc_rust_matches_python_reference_randomized():
    # Randomized parity over many small graphs (mirrors the articulation parity
    # test's intent for CC/SCC). #391 item 4.
    import random

    rust = kernels._rust_kernels()
    if rust is None:
        pytest.skip("Rust extension not built")
    rng = random.Random(391)
    for _ in range(200):
        n = rng.randint(1, 9)
        m = rng.randint(0, 2 * n)
        undirected = [
            (rng.randrange(n), rng.randrange(n)) for _ in range(m)
        ]  # CC: undirected edges
        cc_got = [int(x) for x in rust.decomp_connected_components(n, undirected)]
        cc_ref, _ = kernels._connected_components_edges_py(n, undirected)
        assert cc_got == cc_ref, f"CC divergence n={n}, edges={undirected}"

        directed = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]  # SCC: arcs
        scc_got = [int(x) for x in rust.decomp_strongly_connected_components(n, directed)]
        scc_ref, _ = kernels._strongly_connected_components_py(n, directed)
        assert scc_got == scc_ref, f"SCC divergence n={n}, arcs={directed}"


def test_dependency_edges_star_expansion_for_wide_cliques():
    # a width-5 clique with max_clique_expand=3 becomes a star (4 edges),
    # not a full clique (10 edges).
    edges = kernels.dependency_edges([[0, 1, 2, 3, 4]], max_clique_expand=3)
    assert len(edges) == 4
    assert all(a == 0 for a, _ in edges)


# ── ModelGraph views ──────────────────────────────────────────


def _two_block_model():
    m = dm.Model("twoblock")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    u = m.continuous("u", lb=0, ub=1)
    v = m.continuous("v", lb=0, ub=1)
    m.subject_to(x + y <= 1)
    m.subject_to(u + v <= 1)
    m.minimize(x + y + u + v)
    return m


def test_modelgraph_from_model_counts():
    m = _two_block_model()
    g = ModelGraph.from_model(m)
    assert g.num_vars == 4
    assert g.num_constraints == 2
    assert g.num_incidences == 4


def test_modelgraph_block_diagonal_detection():
    m = _two_block_model()
    g = build_graph(m)
    assert g.is_block_diagonal()
    assert g.num_blocks() == 2


def test_modelgraph_coupled_is_not_block_diagonal():
    m = dm.Model("coupled")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    u = m.continuous("u", lb=0, ub=1)
    v = m.continuous("v", lb=0, ub=1)
    m.subject_to(x + y <= 1)
    m.subject_to(u + v <= 1)
    m.subject_to(x + u <= 1)  # coupling row
    m.minimize(x + y + u + v)
    g = build_graph(m)
    assert not g.is_block_diagonal()
    assert g.num_blocks() == 1
    # the coupling row is index 2; removing it re-separates the model.
    assert 2 in g.bridge_constraints()


def test_modelgraph_summary_is_string():
    g = build_graph(_two_block_model())
    s = g.summary()
    assert "ModelGraph" in s
    assert "blocks:" in s


def test_build_graph_unimplemented_kind_raises():
    m = _two_block_model()
    # VARIABLE_CONSTRAINT and DEPENDENCY are implemented; an unknown kind is not.
    assert build_graph(m, GraphKind.VARIABLE_CONSTRAINT).num_vars == 4
    with pytest.raises(NotImplementedError):
        build_graph(m, kind="jacobian")  # type: ignore[arg-type]


# ── GraphCache ─────────────────────────────────────────────────


def test_graph_cache_reuses_until_version_bumps():
    m = _two_block_model()
    cache = GraphCache()
    g1 = cache.get(m)
    g2 = cache.get(m)
    assert g1 is g2  # same object reused
    # a structural edit bumps the version -> rebuild
    m._structural_version = int(getattr(m, "_structural_version", 0)) + 1
    g3 = cache.get(m)
    assert g3 is not g1
    assert g3.num_vars == g1.num_vars


def test_graph_cache_clear():
    m = _two_block_model()
    cache = GraphCache()
    g1 = cache.get(m)
    cache.clear()
    g2 = cache.get(m)
    assert g1 is not g2


# ── export ─────────────────────────────────────────────────────


def test_export_json_roundtrips():
    import json

    m = _two_block_model()
    g = build_graph(m)
    payload = json.loads(export_graph(g, "json"))
    assert len(payload["nodes"]) == 4
    assert payload["directed"] is False
    # x+y and u+v each induce one edge
    assert len(payload["edges"]) == 2


def test_export_dot_and_graphml_and_metis_produce_text():
    g = build_graph(_two_block_model())
    dot = export_graph(g, "dot")
    assert dot.startswith("graph dependency {")
    gml = export_graph(g, "graphml")
    assert "<graphml" in gml
    metis = export_graph(g, "metis")
    # header line: "<n> <#edges>"
    assert metis.splitlines()[0] == "4 2"


def test_export_unknown_format_raises():
    g = build_graph(_two_block_model())
    with pytest.raises(ValueError):
        export_graph(g, "nonsense")
