"""Regression for issue #287: per-node OBBT must not starve the root primal.

The per-node OBBT lever (`_PER_NODE_OBBT_BUDGET_FRAC = 0.6`) ran on the ROOT node
before the root primal heuristic, spending up to 60% of the time budget on bound
tightening *before any incumbent existed*. On nonconvex models with
functionally-dependent intermediates this pushed time-to-first-incumbent past the
limit (kall_congruentcircles_c72: first incumbent at ~10.8 s on an 8 s budget ->
`hard_timeout`, no incumbent, under the benchmark's hard kill).

Fix: skip per-node OBBT on the root iteration (it is redundant there with the
global root OBBT, and cutoff-driven OBBT wants an incumbent anyway). It still runs
on branched nodes, where it does its work (welded-beam / nvs05 fathoming).

Measured before/after (kall, time_limit=8): first incumbent 10.8 s -> 8.6 s, wall
11.0 s -> 8.7 s — i.e. it now returns under a 10 s hard budget.

This test guards the property robustly (no tight timing): a dependent-var nonconvex
instance still reaches its optimum incumbent within a generous budget — which would
fail if the root primal were starved badly enough to miss it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_st_e31_reaches_optimum_incumbent():
    """st_e31 (functionally-dependent nonconvex, per-node-OBBT-eligible, opt -2.0)
    must surface its optimal incumbent within a generous budget — the root primal
    is not starved by per-node OBBT."""
    path = os.path.join(_DATA, "st_e31.nl")
    if not os.path.exists(path):
        pytest.skip("st_e31 instance unavailable")
    r = dm.from_nl(path).solve(time_limit=30, gap_tolerance=1e-4)
    assert r.objective is not None  # an incumbent was found (not hard_timeout)
    assert r.objective == pytest.approx(-2.0, abs=1e-2)
