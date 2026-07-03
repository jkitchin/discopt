"""Lifted univariate squares ``s = x**2`` must be cut tightly by tangent
separation, not left under-cut by a sparse static envelope (#114).

A lifted square is convex, so its exact convex underestimator is the family of
supporting tangents ``s >= 2t*x - t**2``. The *static* envelope places tangents
only at the box endpoints (and 0), so deep inside a wide box the LP underestimates
``x**2`` by ~the whole box width: on a square over ``x in [0, 100]`` the relaxed
``s - 2x`` is driven to ~-100 against a true minimum of -1. That uselessly loose
(but still valid) dual bound is what made the spatial B&B time out on ex9_2_6
(MPEC: root LP -406 / surfaced -201.5 vs true optimum -1.0) where BARON certifies
in 0.03s.

``MccormickLPRelaxer._separate_univariate_square`` adds, after each LP solve, the
EXACT supporting tangent at the current LP point for every lifted square whose
``s`` sits below the parabola, and re-solves. Each tangent is a global
underestimator of a convex function, so no feasible point is ever cut -- the bound
stays a rigorous lower bound at every round. These tests pin (a) the separator
sharply tightens the bound, (b) it never produces an UNSOUND (above-optimum)
bound, and (c) the gap survives end-to-end on the ex9_2_6 MPEC.
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


def _square_model(ub: float = 100.0):
    """Minimize the lifted square ``z = x**2 - 2x`` over ``x in [0, ub]``.

    True optimum is -1 at x=1; any valid lower bound must be <= -1. The static
    square envelope (tangents only at {0, ub}) under-cuts ``x**2`` badly in the
    interior, so without separation the relaxation bound collapses to ~-ub.
    """
    m = discopt.Model("square_undercut")
    x = m.continuous("x", lb=0.0, ub=ub)
    m.minimize(x * x - 2.0 * x)
    return m


def _solve_root_bound(model, square_separate: bool, ub: float = 100.0):
    prev = os.environ.get("DISCOPT_SQUARE_SEPARATE")
    os.environ["DISCOPT_SQUARE_SEPARATE"] = "1" if square_separate else "0"
    try:
        relaxer = mc.MccormickLPRelaxer(model)
        # Exercise the separating (cold) path directly. Since cert:T1.3 the
        # incremental fast path returns *before* the per-node separation chain and
        # instead inherits cuts captured once into the root pool; a unit test of the
        # separator itself must run the path that actually separates.
        relaxer._inc = None
        lb = np.array([0.0], dtype=np.float64)
        hi = np.array([ub], dtype=np.float64)
        res = relaxer.solve_at_node(lb, hi, time_limit=10.0)
        return res
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_SQUARE_SEPARATE", None)
        else:
            os.environ["DISCOPT_SQUARE_SEPARATE"] = prev


def test_square_separation_tightens_root_bound():
    """The separator must turn the static ~-100 under-cut into a near-exact -1,
    while NEVER fabricating an above-optimum (unsound) bound."""
    m = _square_model(ub=100.0)

    off = _solve_root_bound(m, square_separate=False)
    on = _solve_root_bound(m, square_separate=True)

    assert off.status == "optimal" and off.lower_bound is not None
    assert on.status == "optimal" and on.lower_bound is not None

    # Without separation the convex square is badly under-cut: the bound is far
    # below the true optimum (-1).
    assert off.lower_bound <= -25.0, (
        f"static envelope unexpectedly tight ({off.lower_bound}); the test no "
        f"longer exercises the under-cut it guards"
    )
    # With separation the bound is near the true optimum.
    assert on.lower_bound >= -2.0, (
        f"square separation failed to tighten the bound: {on.lower_bound}"
    )
    # Soundness (both paths): a lower bound for a minimization can never exceed
    # the true optimum (-1). A bound above -1 would be invalid.
    assert off.lower_bound <= -1.0 + 1e-6
    assert on.lower_bound <= -1.0 + 1e-6
    # And the separated bound is strictly tighter.
    assert on.lower_bound > off.lower_bound + 1.0


def test_square_separation_never_cuts_a_feasible_point():
    """At several boxes the separated bound stays a valid (<= optimum) lower
    bound -- the cut is a supporting tangent, so it can never exclude the true
    minimizer."""
    for ub in (5.0, 20.0, 100.0, 500.0):
        m = _square_model(ub=ub)
        on = _solve_root_bound(m, square_separate=True, ub=ub)
        assert on.status == "optimal" and on.lower_bound is not None
        # x**2 - 2x has its unconstrained min -1 at x=1, inside every box here.
        assert on.lower_bound <= -1.0 + 1e-6, (
            f"ub={ub}: separated bound {on.lower_bound} exceeds the true optimum -1 "
            f"(an UNSOUND dual bound)"
        )


@pytest.mark.slow
def test_ex9_2_6_surfaced_bound_is_tight_and_sound():
    """End-to-end MPEC (the KKT system of a QP with complementarity x*y=0 and a
    quadratic objective). The square envelope was the dominant root looseness; the
    surfaced fallback bound must now be near the true optimum -1.0 (not the old
    -201.5) AND remain a valid lower bound (<= incumbent)."""
    m = _ex9_2_6_model()
    r = m.solve(time_limit=30.0)
    assert r.objective is not None
    # discopt finds the true incumbent -1.0 via local search.
    assert r.objective <= -1.0 + 1e-3, f"missed the -1.0 incumbent: {r.objective}"
    # The surfaced dual bound must be a valid lower bound (<= incumbent) ...
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-4, (
            f"surfaced an invalid (above-incumbent) bound {r.bound} > {r.objective}"
        )
        # ... and the square-separation + fallback routing keeps it tight: far
        # better than the old static -201.5 (regression guard).
        assert r.bound >= -10.0, f"surfaced bound {r.bound} is still in the old loose ~-201 regime"


def _ex9_2_6_model():
    """ex9_2_6 (GLOBALLib): KKT conditions of a quadratic program, an MPEC with
    six complementarity products and a quadratic objective. Variables positive;
    x6..x17 bounded to [0, 200]."""
    m = discopt.Model("ex9_2_6")
    x = {i: m.continuous(f"x{i}", lb=0.0, ub=(200.0 if i >= 6 else 1.0e20)) for i in range(2, 18)}
    objvar = m.continuous("objvar", lb=-1.0e20, ub=1.0e20)
    # e1: quadratic objective.
    m.subject_to(
        x[2] * x[2] - 2 * x[2] + x[3] * x[3] - 2 * x[3] + x[4] * x[4] + x[5] * x[5] - objvar == 0.0
    )
    m.subject_to(-x[4] + x[6] == -0.5)
    m.subject_to(-x[5] + x[7] == -0.5)
    m.subject_to(x[4] + x[8] == 1.5)
    m.subject_to(x[5] + x[9] == 1.5)
    # complementarity products x_a * x_b == 0
    m.subject_to(x[6] * x[12] == 0.0)
    m.subject_to(x[7] * x[13] == 0.0)
    m.subject_to(x[8] * x[14] == 0.0)
    m.subject_to(x[9] * x[15] == 0.0)
    m.subject_to(x[10] * x[16] == 0.0)
    m.subject_to(x[11] * x[17] == 0.0)
    m.subject_to(-2 * x[2] + 2 * x[4] - x[12] + x[14] == 0.0)
    m.subject_to(-2 * x[3] + 2 * x[5] - x[13] + x[15] == 0.0)
    m.minimize(objvar)
    return m
