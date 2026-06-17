"""Unbounded McCormick relaxation must not fabricate a finite lower bound (#24).

A McCormick/RLT envelope for a product/power term is only a valid relaxation when
its variables have FINITE bounds. Over a box where a nonlinear-term variable is
still unbounded (e.g. the root box before FBBT could bound it), the lifted aux is
effectively free and the relaxation is genuinely UNBOUNDED -- it carries no valid
finite lower bound. HiGHS correctly reports "unbounded"; the fast warm-started
Rust simplex instead mis-handles the unbounded ray and fabricates a finite
"optimal" (on himmel16's root relaxation the simplex returns 0.0 / with RLT cuts
-0.6749 where HiGHS returns "unbounded"). Trusting that finite value as a lower
bound is a too-high dual bound: it fathoms feasible nodes and certifies a
suboptimal incumbent -- a false-"optimal", the worst failure class.

himmel16 (the "largest small hexagon" with only pairwise-diameter constraints)
admits a self-intersecting doubly-traced equilateral triangle of area 0.866, so
its true optimum is objvar = -0.866; discopt previously certified -0.6749 (the
*convex* hexagon optimum) as global. These tests pin the two soundness guards
that close that gap: ``MccormickLPRelaxer.solve_at_node`` cross-checks an
unbounded relaxation with HiGHS, and ``_root_relaxation_lower_bound`` requires an
OPTIMAL relaxation solve before surfacing a fallback bound.
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


def _bilinear_obj():
    """Minimize a bilinear objective; the unbounded box makes x*y unbounded."""
    m = discopt.Model("bil_unbounded")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 0.5)
    m.minimize(x * y - x - y)
    return m


def test_solve_at_node_unbounded_box_returns_no_finite_bound():
    """Directly: over an UNBOUNDED box the bilinear relaxation is unbounded, so
    ``solve_at_node`` must NOT return a finite ``lower_bound`` -- even though the
    fast simplex fabricates a finite "optimal" there. A fabricated finite bound
    would be a too-high dual bound and certify a suboptimal incumbent."""
    relaxer = mc.MccormickLPRelaxer(_bilinear_obj())

    # Bounded box: a real finite LP bound is fine (relaxation is bounded).
    lb_f = np.array([-2.0, -2.0], dtype=np.float64)
    ub_f = np.array([2.0, 2.0], dtype=np.float64)
    bounded = relaxer.solve_at_node(lb_f, ub_f, time_limit=5.0)
    assert bounded.status != "skipped_oversize"

    # Unbounded box: the McCormick envelope of x*y is free -> the LP is unbounded.
    # The guard must decline (no finite lower bound), never fabricate one.
    lb_u = np.array([-np.inf, -np.inf], dtype=np.float64)
    ub_u = np.array([np.inf, np.inf], dtype=np.float64)
    unb = relaxer.solve_at_node(lb_u, ub_u, time_limit=5.0)
    assert unb.lower_bound is None, (
        f"unbounded relaxation fabricated a finite bound {unb.lower_bound} (status={unb.status})"
    )


def test_has_unbounded_nonlinear_col_flags_free_bilinear_var():
    """The gate fires iff a nonlinear-participating column is non-finite."""
    relaxer = mc.MccormickLPRelaxer(_bilinear_obj())
    assert relaxer._nonlinear_cols  # x*y registers both columns

    class _Milp:
        def __init__(self, bounds):
            self._bounds = bounds

    finite = _Milp([(-2.0, 2.0), (-2.0, 2.0)])
    assert not relaxer._has_unbounded_nonlinear_col(finite)

    free = _Milp([(-np.inf, 2.0), (-2.0, 2.0)])
    assert relaxer._has_unbounded_nonlinear_col(free)


def test_himmel16_no_false_certification():
    """End-to-end: the largest-small-hexagon model with only pairwise-diameter
    constraints has a feasible self-intersecting solution at objvar=-0.866. The
    solver must EITHER report a bound <= the incumbent (sound), OR not certify --
    it must never certify a value above -0.866."""
    m = _himmel16_model()
    r = m.solve(time_limit=40.0)
    # discopt finds the true optimum (-0.866) via local search.
    assert r.objective is not None
    assert r.objective <= -0.86 + 1e-3, f"missed the -0.866 incumbent: {r.objective}"
    # Soundness: a certified bound can never sit above the incumbent it certifies.
    if getattr(r, "gap_certified", False):
        assert r.bound is not None and r.bound <= r.objective + 1e-4, (
            f"certified an invalid (above-incumbent) bound: "
            f"bound={r.bound} > objective={r.objective}"
        )
    # A reported finite bound must be a valid lower bound (<= incumbent).
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= r.objective + 1e-4, (
            f"surfaced an invalid lower bound {r.bound} > incumbent {r.objective}"
        )


def _himmel16_model():
    """himmel16 (GLOBALLib): maximize hexagon area (min objvar=-area) with all
    pairwise vertex distances <= 1. Vertices (x_i, x_{i+6}), i=1..6; x1=x7=x8=0
    fixed; coordinates otherwise free. The shoelace area terms are bilinear."""
    m = discopt.Model("himmel16")
    x = {i: m.continuous(f"x{i}", lb=-1e20, ub=1e20) for i in range(1, 19)}
    objvar = m.continuous("objvar", lb=-1e20, ub=1e20)
    # x1, x7, x8 fixed at 0
    for i in (1, 7, 8):
        m.subject_to(x[i] == 0.0)
    # pairwise diameter constraints e1..e15: sqr(xa-xb)+sqr(xc-xd) <= 1
    pairs = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 5),
        (3, 6),
        (4, 5),
        (4, 6),
        (5, 6),
    ]
    for a, b in pairs:
        m.subject_to((x[a] - x[b]) ** 2 + (x[a + 6] - x[b + 6]) ** 2 <= 1.0)
    # shoelace area pieces x13..x18 (e17..e22)
    m.subject_to(-0.5 * (x[1] * x[8] - x[7] * x[2]) + x[13] == 0.0)
    m.subject_to(-0.5 * (x[2] * x[9] - x[8] * x[3]) + x[14] == 0.0)
    m.subject_to(-0.5 * (x[3] * x[10] - x[9] * x[4]) + x[15] == 0.0)
    m.subject_to(-0.5 * (x[4] * x[11] - x[10] * x[5]) + x[16] == 0.0)
    m.subject_to(-0.5 * (x[5] * x[12] - x[11] * x[6]) + x[17] == 0.0)
    m.subject_to(-0.5 * (x[6] * x[7] - x[12] * x[1]) + x[18] == 0.0)
    # e16: objvar = -(x13+...+x18)
    m.subject_to(-(x[13] + x[14] + x[15] + x[16] + x[17] + x[18]) - objvar == 0.0)
    m.minimize(objvar)
    return m
