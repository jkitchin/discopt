"""Tests for log-concave relaxation (item 6) and Proposition-9 gating (item 7).

Item 6 (``g_concave_overestimator_cut``): positive log-concave intermediates
are G-concave, so the detector recognizes them and the negation-reduction
overestimator cut relaxes ``t ≤ φ(x)`` soundly. The guard is the mirror of
item 3 — no feasible ``(x, t)`` with ``t ≤ φ(x)`` is ever separated.

Item 7 (``transformation_adds_value``): Proposition 9 — the transformation is
skipped where the recursive factorable relaxation already captures the
G-convexity (convex/affine outer, or concave outer with the convex ``exp``
transform), and applied where the outer composition is curvature-unknown.
"""

from __future__ import annotations

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
from discopt._jax.convexity import Curvature
from discopt._jax.convexity.g_convex_cut import (
    g_concave_overestimator_cut,
    g_convex_supporting_cut,
)
from discopt._jax.convexity.g_convexity import certify_g_convex
from discopt._jax.convexity.g_prop9 import transformation_adds_value
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import Model

# ──────────────────────────────────────────────────────────────────────
# Item 6 — log-concave overestimator cut
# ──────────────────────────────────────────────────────────────────────


def _feasible_never_cut_concave(phi, m, cut, *, box_lb, box_ub, seed=0, n=3000):
    """No feasible ``(x, t)`` with ``t ≤ φ(x)``, ``x`` in box, is separated."""
    f = compile_expression(phi, m)
    rng = np.random.default_rng(seed)
    worst = -np.inf
    for _ in range(n):
        x = rng.uniform(box_lb, box_ub)
        pv = float(f(jnp.asarray(x)))
        t = pv - rng.uniform(0.0, 1.0)  # any t ≤ φ(x) is feasible
        worst = max(worst, cut.violation(x, t))
    assert worst <= 1e-7, f"overestimator cut separates a feasible point ({worst})"


class TestLogConcaveCut:
    def test_gaussian_is_g_concave(self):
        # exp(-(x²+y²)) is log-concave, neither convex nor concave on this box.
        m = Model("t")
        x = m.continuous("x", lb=0.45, ub=0.55)
        y = m.continuous("y", lb=0.45, ub=0.55)
        cert = certify_g_convex(dm.exp(-(x**2 + y**2)), m)
        assert cert is not None and cert.kind == "g_concave"

    def test_overestimator_cut_is_sound(self):
        m = Model("t")
        x = m.continuous("x", lb=0.45, ub=0.55)
        y = m.continuous("y", lb=0.45, ub=0.55)
        phi = dm.exp(-(x**2 + y**2))
        cut = g_concave_overestimator_cut(phi, m)
        assert cut is not None
        _feasible_never_cut_concave(phi, m, cut, box_lb=[0.45, 0.45], box_ub=[0.55, 0.55])

    def test_overestimator_separates_infeasible_point(self):
        m = Model("t")
        x = m.continuous("x", lb=0.45, ub=0.55)
        y = m.continuous("y", lb=0.45, ub=0.55)
        phi = dm.exp(-(x**2 + y**2))
        f = compile_expression(phi, m)
        cut = g_concave_overestimator_cut(phi, m, x0=np.array([0.5, 0.5]))
        xv = np.array([0.5, 0.5])
        t_bad = float(f(jnp.asarray(xv))) + 0.3  # t well ABOVE φ(x): infeasible
        assert cut.violation(xv, t_bad) > 0.0

    def test_none_when_not_g_concave(self):
        # A g_convex (not g_concave) body yields no overestimator cut here.
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        assert g_concave_overestimator_cut(dm.log(x**2 + y**2), m) is None


# ──────────────────────────────────────────────────────────────────────
# Item 7 — Proposition-9 gate
# ──────────────────────────────────────────────────────────────────────


class TestProp9Gate:
    def test_convex_outer_is_redundant(self):
        assert transformation_adds_value(Curvature.CONVEX) is False

    def test_affine_outer_is_redundant(self):
        assert transformation_adds_value(Curvature.AFFINE) is False

    def test_concave_outer_with_convex_transform_is_redundant(self):
        # G=exp is convex ⇒ concave outer is captured by recursion.
        assert transformation_adds_value(Curvature.CONCAVE, transform_is_convex=True) is False

    def test_concave_outer_with_nonconvex_transform_adds_value(self):
        assert transformation_adds_value(Curvature.CONCAVE, transform_is_convex=False) is True

    def test_unknown_outer_adds_value(self):
        assert transformation_adds_value(Curvature.UNKNOWN) is True

    def test_no_outer_context_adds_value(self):
        # A bare intermediate (no wrapping f) always proceeds.
        assert transformation_adds_value(None) is True

    def test_gate_suppresses_cut(self):
        # Same G-convex body: with a convex outer context the cut is gated off;
        # with unknown outer (or none) it fires.
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        phi = dm.log(x**2 + y**2)
        assert g_convex_supporting_cut(phi, m, outer_curvature=Curvature.CONVEX) is None
        assert g_convex_supporting_cut(phi, m, outer_curvature=Curvature.UNKNOWN) is not None
        assert g_convex_supporting_cut(phi, m, outer_curvature=None) is not None
