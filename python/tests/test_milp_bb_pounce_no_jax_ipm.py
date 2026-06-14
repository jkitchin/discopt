"""Regression: the POUNCE MILP-B&B path must not fall back to the pure-JAX IPM.

Phase 8 (POUNCE-only stack) routes every LP/MILP solve through POUNCE. Two
leftovers used the pure-JAX ``lp_ipm_solve`` even in ``prefer_pounce`` mode:

* the per-node relaxation solve (``_solve_node_lp_pounce``) handed POUNCE the
  slack-expanded *standard* form, on which its IPM stalls; and
* the root fractional-dive heuristic (``_root_dive``).

Both are now POUNCE-native (the node solve uses the inequality form; the dive is
skipped in POUNCE mode). This test pins that a POUNCE-mode MILP solve never
calls the JAX IPM, by making any such call fail loudly.
"""

from __future__ import annotations

import discopt._jax.lp_ipm as _lp_ipm
import numpy as np
import pytest
from discopt.solvers import SolveStatus

POUNCE_AVAILABLE = pytest.importorskip(
    "discopt.solvers.lp_pounce"
).POUNCE_AVAILABLE  # type: ignore[attr-defined]

pytestmark = pytest.mark.skipif(not POUNCE_AVAILABLE, reason="POUNCE not installed")


def test_pounce_milp_bb_avoids_jax_ipm(monkeypatch):
    from discopt.solvers.milp_pounce import solve_milp

    def _boom(*args, **kwargs):
        raise AssertionError("JAX IPM (lp_ipm_solve) must not run in POUNCE MILP mode")

    monkeypatch.setattr(_lp_ipm, "lp_ipm_solve", _boom)
    monkeypatch.setattr(_lp_ipm, "lp_ipm_solve_batch", _boom)

    # Small 0/1 knapsack-style MILP whose LP relaxation is fractional, so the
    # B&B opens child nodes (exercises the node-solve path, not just the root).
    # max 5a + 4b + 3c  s.t. 2a + 3b + c <= 4,  a,b,c in {0,1}
    # -> as a minimization: min -(5a+4b+3c).
    c = np.array([-5.0, -4.0, -3.0])
    A_ub = np.array([[2.0, 3.0, 1.0]])
    b_ub = np.array([4.0])
    bounds = [(0.0, 1.0)] * 3
    integrality = np.array([1, 1, 1])

    r = solve_milp(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        integrality=integrality,
        time_limit=30,
        gap_tolerance=1e-4,
    )

    assert r.status == SolveStatus.OPTIMAL
    # Optimum: a=1, c=1 (2+1=3<=4) -> 5+3=8; adding b violates (2+3=5>4).
    assert r.objective is not None
    assert abs(float(r.objective) - (-8.0)) <= 1e-4
