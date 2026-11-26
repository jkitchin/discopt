"""#2357: the native MILP engine stops on the criterion its caller judges it by.

``Model.solve`` judges a published pair with ``_gap_criterion``: the absolute
tolerance (1e-6 when omitted) or the unfloored relative test
``(ub - lb) / max(|ub|, |lb|, 1e-10)``. The Rust engine was given no absolute
tolerance when the caller omitted one, and so stopped on its floored relative
test ``(ub - lb) / max(|ub|, 1)``. On an objective below 1 that closes at an
absolute gap up to ``gap_tolerance``, which the caller then refused
(``certificate/published_pair_refused``), publishing "feasible" with a solve that
had stopped early. On MINLPLib neos17 (|ub| = 0.15) that was 7,711 nodes, absolute
gap 9.8e-5, relative 6.5e-4.

The models are 30-item knapsacks with the objective scaled by 1e-5 (|obj| ~ 0.008),
so the root's floored gap is already below 1e-4. Before the fix both stopped at
the root and were refused; seed 3's published incumbent was also not optimal.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _knapsack(seed: int) -> dm.Model:
    rng = np.random.default_rng(seed)
    n = 30
    w = rng.integers(20, 60, n)
    v = rng.integers(20, 60, n)
    m = dm.Model("k")
    x = m.binary("x", shape=(n,))
    cap = int(w.sum() // 2)
    m.subject_to(sum(int(w[i]) * x[i] for i in range(n)) <= cap)
    m.subject_to(sum(int(w[(i + 3) % n]) * x[i] for i in range(n)) <= cap)
    m.minimize(-1e-5 * sum(int(v[i]) * x[i] for i in range(n)))
    return m


@pytest.mark.parametrize("seed", [0, 3])
def test_rust_engine_certifies_small_objective(seed, monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    r = _knapsack(seed).solve(threads=1, deterministic=True, time_limit=60)
    stats = r.solver_stats or {}
    assert stats.get("certificate/published_pair_refused") is None, stats
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    # The certificate holds under the caller's own rule, not the engine's floor.
    gap = r.objective - r.bound
    assert gap <= 1e-6 or gap / max(abs(r.objective), abs(r.bound)) <= 1e-4
    # The engine searched past the root instead of stopping on the floored test.
    assert r.node_count > 1


def test_rust_engine_matches_highs_optimum(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    rust = _knapsack(3).solve(threads=1, deterministic=True, time_limit=60)
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "highs")
    highs = _knapsack(3).solve(threads=1, deterministic=True, time_limit=60)
    assert highs.status == "optimal"
    assert abs(rust.objective - highs.objective) <= 1e-9
