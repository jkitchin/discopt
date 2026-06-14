"""Clique cuts for the self-hosted MILP B&B (roadmap Phase 3).

Conflict-graph 2-clique edges from the Rust presolve are greedily merged into
larger cliques, giving ``sum_{j in C} x_j <= 1``. Unlike pairwise edges (which
are usually redundant with the constraints they came from), a merged clique of
size >= 3 separates even the symmetric IPM analytic center — partly overcoming
the interior-point cut limitation that cover cuts hit. Tests: separation
validity (cuts are true cliques and never exclude a feasible point), the
symmetric-center bite, edge extraction, and an end-to-end node reduction.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._jax.cover_cuts import separate_clique_cuts


class TestSeparation:
    def test_merges_triangle(self):
        cuts = separate_clique_cuts([(0, 1), (0, 2), (1, 2)], np.array([0.5, 0.5, 0.5]))
        assert cuts == [(frozenset({0, 1, 2}), 1.0)]  # merged, and violated (1.5>1)

    def test_path_stays_pairwise(self):
        # 0-1-2-3 has no triangle: only pairwise cliques.
        cuts = separate_clique_cuts([(0, 1), (1, 2), (2, 3)], np.array([0.6, 0.6, 0.1, 0.1]))
        assert cuts == [(frozenset({0, 1}), 1.0)]

    def test_symmetric_center_violated_for_triangle(self):
        # The interior-point limitation that defeats cover cuts is overcome for
        # cliques of size >= 3: 0.5*3 = 1.5 > 1.
        assert separate_clique_cuts([(0, 1), (0, 2), (1, 2)], np.full(3, 0.5))

    def test_validity_exhaustive(self):
        rng = np.random.default_rng(1)
        edge_set = set
        for _ in range(150):
            n = int(rng.integers(3, 6))
            poss = [(i, j) for i in range(n) for j in range(i + 1, n)]
            k = int(rng.integers(1, len(poss) + 1))
            edges = [poss[t] for t in rng.choice(len(poss), size=k, replace=False)]
            x = rng.random(n)
            cuts = separate_clique_cuts(edges, x)
            es = edge_set(frozenset(e) for e in edges)
            feas = [
                np.array(b, float)
                for b in itertools.product([0, 1], repeat=n)
                if all(not (b[i] and b[j]) for i, j in edges)
            ]
            for clique, rhs in cuts:
                # Every pair in the cut is a genuine conflict edge.
                assert all(frozenset((a, b)) in es for a in clique for b in clique if a < b)
                for xv in feas:
                    assert (
                        sum(xv[t] for t in clique) <= rhs + 1e-9
                    )  # never excludes a feasible point


class TestExtraction:
    def test_extract_clique_edges(self):
        m = dm.Model("conf")
        x = [m.binary(f"x{i}") for i in range(4)]
        m.minimize(-(x[0] + x[1] + x[2] + x[3]))
        m.subject_to(x[0] + x[1] <= 1)
        m.subject_to(x[1] + x[2] <= 1)
        m.subject_to(x[2] + x[3] <= 1)
        edges = S._extract_clique_edges(m)
        assert {frozenset(e) for e in edges} >= {
            frozenset({0, 1}),
            frozenset({1, 2}),
            frozenset({2, 3}),
        }


def _triangle_set_packing():
    m = dm.Model("tri")
    x = [m.binary(f"x{i}") for i in range(3)]
    m.minimize(-(x[0] + x[1] + x[2]))
    m.subject_to(x[0] + x[1] <= 1)
    m.subject_to(x[0] + x[2] <= 1)
    m.subject_to(x[1] + x[2] <= 1)
    return m


class TestEndToEnd:
    def test_clique_cut_reduces_nodes_and_preserves_optimum(self, monkeypatch):
        pytest.importorskip("pounce")
        r_with = _triangle_set_packing().solve(use_highs_milp=False, time_limit=60)
        monkeypatch.setattr(S, "_root_cover_cut_loop", lambda ld, *a, **k: (ld, 0))
        r_without = _triangle_set_packing().solve(use_highs_milp=False, time_limit=60)

        assert r_with.status == "optimal" and r_without.status == "optimal"
        assert abs(r_with.objective - r_without.objective) < 1e-4
        assert abs(r_with.objective - (-1.0)) < 1e-4  # at most one of the three
        assert r_with.node_count <= r_without.node_count
