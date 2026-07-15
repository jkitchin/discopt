"""Convex-subexpression claimer (issue #358, hybrid handler #1).

A convex polynomial subexpression (e.g. a convex quadratic objective) is lifted to
a single aux and relaxed with supporting-hyperplane gradient cuts, instead of being
McCormick-decomposed term-by-term (which injects bilinear envelope slack on the
cross terms). With LP-point separation the convex relaxation becomes *exact*.

Soundness rests entirely on the curvature gate: only a node the detector certifies
CONVEX/CONCAVE (DCP or the interval-Hessian PSD certificate) is lifted, so the
gradient cut is always a valid global under-/over-estimator. The lift is now
DEFAULT-ON — the former ``DISCOPT_CONVEX_CLAIMER`` gate was removed with the #632
federation cutover (the whole convex objective is lifted unconditionally and
``_separate_convex`` tightens it); soundness still rests on the curvature gate.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer

pytestmark = [pytest.mark.claim_boundary]


def _convex_qp():
    m = dm.Model("cvxq")
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=0, ub=20)
    m.minimize(3 * x**2 + 2 * y**2 + x * y - 40 * x - 30 * y)
    return m


# The unconstrained (and box-interior) minimiser of 3x²+2y²+xy−40x−30y solves
# 6x+y=40, x+4y=30 → x≈5.652, y≈6.087, f≈−204.35 (the exact convex minimum).
_CVXQP_MIN = -204.35


def test_convex_objective_lift_is_tight_and_sound():
    """The composite-convex OA lift + LP-point (Kelley) separation reaches close to
    the exact convex minimum and is far tighter than the unseparated relaxation.

    The lift is now DEFAULT-ON (the former ``DISCOPT_CONVEX_CLAIMER`` gate was
    removed with the federation cutover): the whole convex objective is lifted and
    ``_separate_convex`` tightens it. Tightness is therefore measured separated (the
    OA/Kelley rounds) vs unseparated, not flag-on vs flag-off. Closing the last ~0.4
    to the exact minimum is tracked in #640."""
    m = _convex_qp()
    lb, ub = np.array([0.0, 0.0]), np.array([20.0, 20.0])
    on = float(MccormickLPRelaxer(m).solve_at_node(lb, ub, separate=True).lower_bound)
    off = float(MccormickLPRelaxer(m).solve_at_node(lb, ub, separate=False).lower_bound)
    # Sound: a valid lower bound never exceeds the true convex minimum.
    assert on <= _CVXQP_MIN + 1e-2
    assert off <= _CVXQP_MIN + 1e-2
    # Tight: the separated OA lift is within ~0.4 of the exact convex minimum...
    assert on == pytest.approx(_CVXQP_MIN, abs=0.5)
    # ...and materially tighter than the unseparated relaxation (~-350).
    assert on > off + 10.0


def test_nonconvex_sum_is_not_claimed():
    """Soundness of the curvature gate: a sum with an indefinite Hessian must NOT be
    lifted as convex — a gradient cut on a nonconvex objective would be an *unsound*
    over-estimator (a lower bound above the true minimum). The composite-convex lift
    (now default-on) must therefore abstain on ``x*y - x**2 - 5x`` (Hessian
    ``[[-2,1],[1,0]]``, indefinite), leaving a sound bound."""
    m = dm.Model("noncvx")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x * y - x**2 - 5 * x)
    r = MccormickLPRelaxer(m).solve_at_node(
        np.array([0.0, 0.0]), np.array([10.0, 10.0]), separate=True
    )
    assert r.status == "optimal"
    bound = float(r.lower_bound)
    # The true minimum over [0,10]^2 is -150 (corner x=10, y=0). A valid lower bound
    # never exceeds it; had the gate wrongly lifted this indefinite objective as
    # convex, the gradient cut would push the bound ABOVE -150 (unsound).
    assert np.isfinite(bound)
    assert bound <= -150.0 + 1e-6, (
        f"unsound bound {bound} > true min -150 (gate lifted a nonconvex objective)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
