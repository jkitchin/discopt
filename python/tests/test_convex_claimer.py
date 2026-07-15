"""Convex-subexpression claimer (issue #358, hybrid handler #1).

A convex polynomial subexpression (e.g. a convex quadratic objective) is lifted to
a single aux and relaxed with supporting-hyperplane gradient cuts, instead of being
McCormick-decomposed term-by-term (which injects bilinear envelope slack on the
cross terms). With LP-point separation the convex relaxation becomes *exact*.

Soundness rests entirely on the curvature gate: only a node the detector certifies
CONVEX/CONCAVE (DCP or the interval-Hessian PSD certificate) is lifted, so the
gradient cut is always a valid global under-/over-estimator. The claimer is gated
by ``DISCOPT_CONVEX_CLAIMER`` (default off) while it is validated.
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


def _node_bound(monkeypatch, model, lb, ub, claimer: bool) -> float:
    # Use monkeypatch (auto-reverting) rather than a raw os.environ write: the
    # claim flag is read fresh per build, so a leaked value would silently flip
    # later tests in the same xdist worker (issue #632; enforced by the autouse
    # _guard_discopt_env_leaks fixture in conftest.py).
    monkeypatch.setenv("DISCOPT_CONVEX_CLAIMER", "1" if claimer else "0")
    relaxer = MccormickLPRelaxer(model)
    r = relaxer.solve_at_node(np.asarray(lb, float), np.asarray(ub, float), separate=True)
    assert r.status == "optimal"
    return float(r.lower_bound)


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


def test_default_off_leaves_relaxation_unchanged(monkeypatch):
    """With the flag unset the claimer must not fire (term-by-term bound)."""
    m = _convex_qp()
    monkeypatch.delenv("DISCOPT_CONVEX_CLAIMER", raising=False)
    relaxer = MccormickLPRelaxer(m)
    default = float(relaxer.solve_at_node(np.array([0.0, 0.0]), np.array([20.0, 20.0])).lower_bound)
    off = _node_bound(monkeypatch, m, [0, 0], [20, 20], claimer=False)
    assert default == pytest.approx(off, abs=1e-6)


def test_nonconvex_sum_is_not_claimed(monkeypatch):
    """Soundness gate: a sum with an indefinite Hessian must NOT be lifted as
    convex — its gradient cut would be an unsound over-estimator. The claimer
    abstains (curvature UNKNOWN) and the bound stays the term-by-term value, so
    enabling the flag cannot change a non-convex relaxation."""
    m = dm.Model("noncvx")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    # x*y - x**2 has Hessian [[-2,1],[1,0]] — indefinite (not convex/concave).
    m.minimize(x * y - x**2 - 5 * x)
    on = _node_bound(monkeypatch, m, [0, 0], [10, 10], claimer=True)
    off = _node_bound(monkeypatch, m, [0, 0], [10, 10], claimer=False)
    # The claimer abstained → identical relaxation, and still a sound lower bound.
    assert on == pytest.approx(off, abs=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
