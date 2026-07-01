"""Regression: the POUNCE MILP-B&B path seeds a finite, sound root bound.

The pure-JAX LP-IPM was retired entirely in #370 (``lp_ipm.py`` deleted), so the
old "must not fall back to the JAX IPM" guard is now guaranteed by construction —
the module no longer exists to call. What remains worth pinning is that a POUNCE
MILP solve returns a finite, sound lower bound even under a tiny time budget
(issue #15 gas network: an unseeded bound left AMP's LB stuck at -inf / None).
"""

from __future__ import annotations

import pytest

POUNCE_AVAILABLE = pytest.importorskip("discopt.solvers.lp_pounce").POUNCE_AVAILABLE  # type: ignore[attr-defined]

pytestmark = pytest.mark.skipif(not POUNCE_AVAILABLE, reason="POUNCE not installed")

def test_pounce_milp_seeds_finite_root_bound():
    """A POUNCE MILP solve must return a finite, sound lower bound even when it
    cannot finish in the time budget — the root LP relaxation bound is seeded so
    the result is never bound=None/-inf (which left AMP's LB stuck at -inf and
    its reported bound at None on the issue #15 gas network)."""
    import numpy as np
    from discopt.solvers.milp_pounce import solve_milp

    c = np.array([-5.0, -4.0, -3.0])
    A_ub = np.array([[2.0, 3.0, 1.0]])
    b_ub = np.array([4.0])
    bounds = [(0.0, 1.0)] * 3
    integrality = np.array([1, 1, 1])

    # Tiny budget: the bounding loop exits before closing the tree, so the
    # reported bound comes from the seeded root LP relaxation.
    r = solve_milp(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        integrality=integrality,
        time_limit=1e-6,
        gap_tolerance=1e-4,
    )
    assert r.bound is not None
    assert np.isfinite(r.bound)
    # A valid lower bound never exceeds the true optimum (-8).
    assert float(r.bound) <= -8.0 + 1e-4
