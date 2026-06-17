"""Densification guard for the LP-form McCormick relaxer (autocorr HANG, Issue #20).

The matrix-form MILP backends (warm-started Rust simplex, POUNCE) materialize the
lifted relaxation as a DENSE ``(m, n+m)`` array plus a dense ``m×m`` slack
identity. A binary-multilinear lift explodes the row count -- autocorr_bern's
degree-4 objective lifts to ~3.7k cols x ~85k rows, a ~7.5e9-cell (~60 GB) dense
allocation that thrashes swap forever (a hang, not a bound). ``solve_at_node``
caps the densified size (``_MAX_RELAX_DENSE_CELLS``) and declines above it.

Declining is SOUND: it only forgoes a node's LP underestimator; the spatial B&B
keeps the rigorous alphaBB/interval bound and still converges to the same global
optimum, just with a weaker per-node bound (more nodes). These tests pin both the
graceful-degradation (same certified optimum) and the direct gate behavior.
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import discopt
import discopt._jax.mccormick_lp as mc
import numpy as np
import pytest


def _small_bilinear():
    m = discopt.Model("bil")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 0.5)
    m.minimize(x * y - x - y)
    return m


def test_densify_guard_degrades_to_same_optimum(monkeypatch):
    """Forcing the cap absurdly low disables the LP bound on every node, yet the
    solve still reaches the SAME global optimum and stays certified -- proving the
    alphaBB fallback is sound and the guard never changes the answer."""
    res_normal = _small_bilinear().solve(time_limit=20.0)
    assert res_normal.status in ("optimal", "feasible")

    # Cap = 1 cell: any lifted relaxation with a single constraint row trips it.
    monkeypatch.setattr(mc, "_MAX_RELAX_DENSE_CELLS", 1.0)
    res_gated = _small_bilinear().solve(time_limit=20.0)
    assert res_gated.status in ("optimal", "feasible")
    # Same optimum, to the solver's global tolerance.
    assert res_gated.objective == pytest.approx(res_normal.objective, abs=1e-4, rel=1e-4)
    # The guard must never *certify* a different value than the ungated path.
    if getattr(res_normal, "gap_certified", False):
        assert getattr(res_gated, "gap_certified", False)


def test_solve_at_node_declines_oversize(monkeypatch):
    """Directly: above the cap, ``solve_at_node`` returns no bound (never an
    unsound one and never a false ``infeasible`` that would fathom the node)."""
    m = _small_bilinear()
    relaxer = mc.MccormickLPRelaxer(m)

    lb = np.array([-2.0, -2.0], dtype=np.float64)
    ub = np.array([2.0, 2.0], dtype=np.float64)

    # Below cap (default): a real LP bound (or a clean status), never oversize.
    ok = relaxer.solve_at_node(lb, ub, time_limit=5.0)
    assert ok.status != "skipped_oversize"

    # Above cap: declined with no bound, and crucially NOT "infeasible".
    monkeypatch.setattr(mc, "_MAX_RELAX_DENSE_CELLS", 1.0)
    gated = relaxer.solve_at_node(lb, ub, time_limit=5.0)
    assert gated.status == "skipped_oversize"
    assert gated.lower_bound is None
    assert gated.status != "infeasible"  # a declined node must never be fathomed
