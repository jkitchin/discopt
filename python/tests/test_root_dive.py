"""Root fractional diving for the self-hosted MILP B&B (roadmap Phase 3).

Diving fixes the most-fractional integer at each step and re-solves the LP,
producing an early incumbent that front-loads pruning / reduced-cost fixing.
It complements the snap-fix-resolve purification (which only fires on
near-integral points). Tests: the dive finds a valid integer-feasible point,
and end to end it preserves the optimum and never worsens the node count.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._jax.problem_classifier import extract_lp_data
from discopt.solver import _root_dive


def _knapsack():
    m = dm.Model("dive_knap")
    xs = [m.binary(f"x{i}") for i in range(5)]
    vals, wts = [10, 9, 8, 5, 3], [6, 5, 4, 3, 2]
    m.minimize(-sum(v * x for v, x in zip(vals, xs)))
    m.subject_to(sum(w * x for w, x in zip(wts, xs)) <= 9)
    return m


class TestRootDive:
    def test_finds_integer_feasible_incumbent(self):
        m = _knapsack()
        ld = extract_lp_data(m)
        dive = _root_dive(ld, 5, list(range(5)), time.perf_counter(), 30.0)
        assert dive is not None
        obj, x = dive
        # All integer columns are exactly integral.
        assert np.allclose(x, np.round(x), atol=1e-6)
        # Feasible: weight <= 9.
        assert np.dot([6, 5, 4, 3, 2], x) <= 9 + 1e-6
        # Objective equals -value of the chosen items (min sense).
        assert abs(obj - (-np.dot([10, 9, 8, 5, 3], x))) < 1e-6

    def test_no_integers_returns_none(self):
        m = dm.Model("cont")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x)
        ld = extract_lp_data(m)
        assert _root_dive(ld, 1, [], time.perf_counter(), 30.0) is None


class TestEndToEnd:
    def test_optimum_preserved_and_nodes_not_worse(self, monkeypatch):
        pytest.importorskip("pounce")
        r_with = _knapsack().solve(use_highs_milp=False, time_limit=60)
        nodes_with = r_with.node_count

        monkeypatch.setattr(S, "_root_dive", lambda *a, **k: None)
        r_without = _knapsack().solve(use_highs_milp=False, time_limit=60)

        assert r_with.status == "optimal" and r_without.status == "optimal"
        assert abs(r_with.objective - r_without.objective) < 1e-4
        assert abs(r_with.objective - (-17.0)) < 1e-4  # items {1,2}: 9+8
        assert nodes_with <= r_without.node_count  # an early incumbent can't hurt

    def test_dive_incumbent_is_feasible_via_solve(self):
        pytest.importorskip("pounce")
        # The injected dive incumbent must be a genuine feasible solution.
        r = _knapsack().solve(use_highs_milp=False, time_limit=60)
        assert r.status == "optimal"
        x = np.array([float(r.x[f"x{i}"]) for i in range(5)])
        assert np.allclose(x, np.round(x), atol=1e-5)
        assert np.dot([6, 5, 4, 3, 2], x) <= 9 + 1e-6
