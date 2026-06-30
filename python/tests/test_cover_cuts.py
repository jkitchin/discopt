"""Knapsack cover cuts for the self-hosted MILP B&B (roadmap Phase 3).

Covers (1) separation soundness — a cover cut never excludes a feasible 0/1
point — and (2) that root cover cuts tighten the relaxation enough to cut the
B&B node count while preserving the optimum. Also documents the interior-point
limitation: cover cuts separate a vertex sharply but a symmetric IPM
analytic-center weakly (motivating the Phase-2 crossover).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._jax.cover_cuts import has_binary_knapsack_rows, separate_cover_cuts


# ---------------------------------------------------------------------------
# Separation: validity (exhaustive) + violation
# ---------------------------------------------------------------------------
class TestSeparationSoundness:
    def test_never_excludes_a_feasible_point(self):
        rng = np.random.default_rng(0)
        for _ in range(150):
            n = int(rng.integers(3, 7))
            a = rng.integers(1, 6, n).astype(float)
            b = float(rng.integers(int(a.min()), int(a.sum())))
            A, bb = a.reshape(1, n), np.array([b])
            x = rng.random(n)
            cuts = separate_cover_cuts(A, bb, x, np.ones(n, bool))
            feas = [
                np.array(bits, float)
                for bits in itertools.product([0, 1], repeat=n)
                if A @ np.array(bits, float) <= bb + 1e-9
            ]
            for cover, rhs in cuts:
                for xv in feas:
                    assert sum(xv[j] for j in cover) <= rhs + 1e-9  # valid
                assert sum(x[j] for j in cover) > rhs + 1e-6  # violated by x*

    def test_finds_violated_cover_at_vertex(self):
        # x* = [1, 0.8, 0, 0] on 5*sum x <= 9: pair {0,1} is a violated cover.
        cuts = separate_cover_cuts(
            np.array([[5.0, 5.0, 5.0, 5.0]]),
            np.array([9.0]),
            np.array([1.0, 0.8, 0.0, 0.0]),
            np.ones(4, bool),
        )
        assert any(cover == frozenset({0, 1}) and rhs == 1.0 for cover, rhs in cuts)

    def test_symmetric_center_not_separated(self):
        # The symmetric analytic center [0.45]*4 violates no cover (0.9 < 1):
        # the documented interior-point limitation.
        cuts = separate_cover_cuts(
            np.array([[5.0, 5.0, 5.0, 5.0]]),
            np.array([9.0]),
            np.full(4, 0.45),
            np.ones(4, bool),
        )
        assert cuts == []

    def test_precheck_rejects_non_knapsack(self):
        assert has_binary_knapsack_rows(np.array([[2.0, 3.0]]), np.array([6.0]), np.ones(2, bool))
        # negative coefficient
        assert not has_binary_knapsack_rows(
            np.array([[2.0, -3.0]]), np.array([6.0]), np.ones(2, bool)
        )
        # continuous variable in the row
        assert not has_binary_knapsack_rows(
            np.array([[2.0, 3.0]]), np.array([6.0]), np.array([True, False])
        )


# ---------------------------------------------------------------------------
# End-to-end: cuts reduce nodes and preserve the optimum
# ---------------------------------------------------------------------------
def _distinct_knapsack():
    """Unique LP optimum (distinct values) -> IPM hits the vertex [1,0.8,0,0],
    which cover cuts separate."""
    m = dm.Model("distinct")
    xs = [m.binary(f"x{i}") for i in range(4)]
    m.minimize(-(10 * xs[0] + 9 * xs[1] + 8 * xs[2] + 1 * xs[3]))
    m.subject_to(5 * xs[0] + 5 * xs[1] + 5 * xs[2] + 5 * xs[3] <= 9)
    return m


class TestEndToEnd:
    def test_cuts_reduce_nodes_and_preserve_optimum(self, monkeypatch):
        pytest.importorskip("pounce")
        r_with = _distinct_knapsack().solve(time_limit=60)
        nodes_with = r_with.node_count

        monkeypatch.setattr(S, "_root_cover_cut_loop", lambda ld, *a, **k: (ld, 0))
        r_without = _distinct_knapsack().solve(time_limit=60)

        assert r_with.status == "optimal" and r_without.status == "optimal"
        assert abs(r_with.objective - r_without.objective) < 1e-4  # same optimum
        assert abs(r_with.objective - (-10.0)) < 1e-4
        # Cuts must not increase the node count (they tighten the relaxation).
        assert nodes_with <= r_without.node_count

    def test_noop_on_non_knapsack(self, monkeypatch):
        pytest.importorskip("pounce")
        seen = {"cuts": 0}
        orig = S._root_cover_cut_loop

        def spy(*a, **k):
            ld, n = orig(*a, **k)
            seen["cuts"] += n
            return ld, n

        monkeypatch.setattr(S, "_root_cover_cut_loop", spy)
        m = dm.Model("eq")
        a = m.integer("a", lb=0, ub=5)
        b = m.integer("b", lb=0, ub=5)
        m.minimize(a + b)
        m.subject_to(a + b >= 3)
        r = m.solve(time_limit=60)
        assert r.status == "optimal"
        assert seen["cuts"] == 0  # no binary-knapsack rows -> no cuts
