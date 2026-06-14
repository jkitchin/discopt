"""Regression: POUNCE LP/MILP wrappers snap tiny floating-point bound
inversions instead of failing.

The relaxation / bound-tightening machinery can emit a variable bound where
``lb`` exceeds ``ub`` by a rounding-scale amount (e.g. ``lb = 30 + 1.8e-11``,
``ub = 30``). POUNCE's interior-point method strictly validates bounds and
rejects such a problem as ``Invalid_Problem_Definition`` (raw status ``-11``,
unmapped -> ``ERROR``), whereas presolve-based solvers (HiGHS) silently snap the
inversion. This regression pins the snap so POUNCE-only environments do not
spuriously fail (root cause of the AMP gas-network stall in issue #15).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus

POUNCE_AVAILABLE = pytest.importorskip(
    "discopt.solvers.lp_pounce"
).POUNCE_AVAILABLE  # type: ignore[attr-defined]

pytestmark = pytest.mark.skipif(not POUNCE_AVAILABLE, reason="POUNCE not installed")


def _tiny_inverted_bounds_lp():
    # min x + y, with x in [0, 2], y bound *inverted* by ~1.8e-11 around 1.0.
    c = np.array([1.0, 1.0])
    bounds = [(0.0, 2.0), (1.0 + 1.8e-11, 1.0)]
    return c, bounds


def test_lp_snaps_tiny_inverted_bound():
    from discopt.solvers.lp_pounce import solve_lp

    c, bounds = _tiny_inverted_bounds_lp()
    r = solve_lp(c=c, bounds=bounds)
    assert r.status == SolveStatus.OPTIMAL
    # y snapped to ~1.0, x at its lower bound 0 -> objective ~1.0.
    assert r.x is not None
    assert abs(float(r.x[1]) - 1.0) <= 1e-6
    assert abs(float(r.objective) - 1.0) <= 1e-5


def test_lp_does_not_mask_genuine_inversion():
    """A large inversion (well beyond snap tolerance) is real infeasibility and
    must NOT be silently snapped into a wrong optimum."""
    from discopt.solvers.lp_pounce import solve_lp

    c = np.array([1.0, 1.0])
    bounds = [(0.0, 2.0), (5.0, 1.0)]  # lb=5 > ub=1 by 4.0
    r = solve_lp(c=c, bounds=bounds)
    assert r.status != SolveStatus.OPTIMAL


def test_milp_snaps_tiny_inverted_bound():
    from discopt.solvers.milp_pounce import solve_milp

    c, bounds = _tiny_inverted_bounds_lp()
    integrality = np.array([1, 0])  # x integer, y continuous
    r = solve_milp(c=c, bounds=bounds, integrality=integrality, time_limit=10)
    assert r.status == SolveStatus.OPTIMAL
    assert r.x is not None
    assert abs(float(r.x[1]) - 1.0) <= 1e-6


def test_nlp_snaps_tiny_inverted_bound():
    """The NLP path must also snap tiny inversions: AMP integer-fixed
    subproblems can arrive with a continuous bound tightened to lb=ub+1e-11,
    which POUNCE's IPM otherwise rejects as Invalid_Problem_Definition,
    blocking incumbent recovery (issue #15 gas network)."""
    import discopt.modeling as dm
    from discopt.solver import _BoundOverrideEvaluator, _extract_variable_info, _make_evaluator
    from discopt.solvers.nlp_pounce import solve_nlp

    m = dm.Model("nlp_snap")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0))
    m.subject_to(x + y >= 1.0)

    ev = _make_evaluator(m)
    _, lb, ub, _, _ = _extract_variable_info(m)
    lb2 = lb.copy()
    ub2 = ub.copy()
    # Fix x ~= 2 with a tiny lb > ub inversion (the failure mode from AMP).
    lb2[0] = 2.0 + 1.8e-11
    ub2[0] = 2.0
    be = _BoundOverrideEvaluator(ev, lb2, ub2)

    r = solve_nlp(be, np.array([2.0, 3.0]))
    assert r.status == SolveStatus.OPTIMAL  # not ERROR / Invalid_Problem_Definition
    assert r.x is not None
    assert abs(float(r.x[1]) - 3.0) <= 1e-4
