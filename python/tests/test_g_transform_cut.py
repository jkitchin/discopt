"""Tests for the G-convexity transform (item 2) and supporting cut (item 3).

Item 2 (``g_transform``): the constant-``ρ`` least convexifying transform
``G(t)=exp(ρt)`` — its calculus (apply/inverse/deriv), the secant concave
overestimator dominating ``exp`` on an interval, and the composition rules.

Item 3 (``g_convex_cut``): the transformation supporting cut. The central
guard is **soundness** — a valid cut never shaves a feasible ``(x,t)`` with
``φ(x) ≤ t`` — plus **usefulness**: it does separate points violating the
intermediate relation.
"""

from __future__ import annotations

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.convexity.g_convex_cut import g_convex_supporting_cut
from discopt._jax.convexity.g_convexity import GConvexCertificate
from discopt._jax.convexity.g_transform import (
    ExpTransform,
    compose_affine_inner,
    compose_increasing_outer,
    least_convexifying_transform,
)
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import Model

# ──────────────────────────────────────────────────────────────────────
# Item 2 — the transform
# ──────────────────────────────────────────────────────────────────────


class TestExpTransform:
    def test_identity_for_rho_zero(self):
        T = ExpTransform(0.0)
        assert T.apply(3.7) == 3.7
        assert T.inverse(3.7) == 3.7
        assert T.deriv(10.0) == 1.0
        sec = T.concave_overestimator(-1.0, 2.0)
        # affine G ⇒ secant coincides with G (slope 1, intercept 0)
        assert sec.b == pytest.approx(1.0)
        assert sec.a == pytest.approx(0.0)

    def test_exp_apply_inverse_roundtrip(self):
        T = ExpTransform(1.3)
        for t in (-2.0, 0.0, 0.5, 3.0):
            assert T.inverse(T.apply(t)) == pytest.approx(t, abs=1e-12)

    def test_convex_and_increasing(self):
        T = ExpTransform(0.7)
        assert T.is_convex and T.is_increasing
        # numeric second difference > 0 (convex), first difference > 0 (incr)
        ts = np.linspace(-1, 1, 9)
        vals = np.array([T.apply(t) for t in ts])
        assert np.all(np.diff(vals) > 0)
        assert np.all(np.diff(vals, 2) > 0)

    def test_secant_dominates_exp_on_interval(self):
        T = ExpTransform(1.0)
        sec = T.concave_overestimator(-0.5, 1.5)
        for t in np.linspace(-0.5, 1.5, 50):
            assert sec(t) >= T.apply(t) - 1e-12  # chord ≥ convex G

    def test_degenerate_interval(self):
        T = ExpTransform(2.0)
        sec = T.concave_overestimator(0.4, 0.4)
        assert sec.b == 0.0
        assert sec(0.4) == pytest.approx(T.apply(0.4))

    def test_rejects_negative_rho(self):
        with pytest.raises(ValueError):
            ExpTransform(-0.1)

    def test_least_convexifying_transform_from_cert(self):
        T = least_convexifying_transform(GConvexCertificate("g_convex", 1.5))
        assert isinstance(T, ExpTransform) and T.rho == 1.5


class TestComposition:
    def test_affine_inner_preserves_transform(self):
        T = ExpTransform(0.9)
        assert compose_affine_inner(T) is T  # ρ unchanged (identity on transform)

    def test_increasing_outer_composition(self):
        # φ is G*-convex with G*=exp(ρ·); outer f(u)=u+5 (increasing),
        # f⁻¹(s)=s-5. Then G*∘f⁻¹ applied at s equals exp(ρ(s-5)).
        base = ExpTransform(1.0)
        comp = compose_increasing_outer(base, f_inverse=lambda s: s - 5.0)
        assert comp.apply(7.0) == pytest.approx(np.exp(1.0 * (7.0 - 5.0)))
        assert comp.is_increasing


# ──────────────────────────────────────────────────────────────────────
# Item 3 — the supporting cut
# ──────────────────────────────────────────────────────────────────────


def _feasible_never_cut(phi, m, cut, *, box_lb, box_ub, seed=0, n=3000):
    """No feasible ``(x, t)`` (``φ(x) ≤ t``, ``x`` in box) is separated."""
    f = compile_expression(phi, m)
    rng = np.random.default_rng(seed)
    worst = -np.inf
    for _ in range(n):
        x = rng.uniform(box_lb, box_ub)
        pv = float(f(jnp.asarray(x)))
        t = pv + rng.uniform(0.0, 1.0)  # any t ≥ φ(x) is feasible
        worst = max(worst, cut.violation(x, t))
    assert worst <= 1e-7, f"cut separates a feasible point (violation={worst})"


class TestSupportingCut:
    def test_none_when_not_g_convex(self):
        m = Model("t")
        c = m.continuous("c", lb=1.0, ub=2.0)
        d = m.continuous("d", lb=1.0, ub=2.0)
        assert g_convex_supporting_cut(c * d, m) is None

    def test_cut_sound_g_convex_not_convex(self):
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        phi = dm.log(x**2 + y**2)
        cut = g_convex_supporting_cut(phi, m)
        assert cut is not None and cut.rho > 0.0
        _feasible_never_cut(phi, m, cut, box_lb=[1.4, 1.6], box_ub=[1.5, 1.7])

    def test_cut_separates_infeasible_point(self):
        # A point with t well below φ(x) must be separated (violation > 0),
        # else the cut is vacuous.
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        phi = dm.log(x**2 + y**2)
        f = compile_expression(phi, m)
        cut = g_convex_supporting_cut(phi, m, x0=np.array([1.45, 1.65]))
        xv = np.array([1.45, 1.65])
        t_bad = float(f(jnp.asarray(xv))) - 0.3  # t far below φ(x): infeasible
        assert cut.violation(xv, t_bad) > 0.0

    def test_rho_zero_is_ordinary_tangent(self):
        # Convex φ ⇒ ρ=0 ⇒ cut is φ(x0)+∇φ(x0)·(x−x0) ≤ t (standard OA cut).
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        phi = x**2
        cut = g_convex_supporting_cut(phi, m, x0=np.array([0.5]))
        assert cut.rho == 0.0
        # tangent of x^2 at 0.5: 0.25 + 1.0*(x-0.5) = x - 0.25 ≤ t
        # ⇒ x_coeff=1, t_coeff=-1, rhs≈-0.25 (+safety)
        assert cut.x_coeffs[0] == pytest.approx(1.0, abs=1e-6)
        assert cut.t_coeff == pytest.approx(-1.0, abs=1e-6)
        _feasible_never_cut(phi, m, cut, box_lb=[-1.0], box_ub=[1.0])

    def test_box_override(self):
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=3.0)
        y = m.continuous("y", lb=1.0, ub=3.0)
        from discopt._jax.convexity.interval import Interval

        tight = {x: Interval.from_bounds(1.4, 1.5), y: Interval.from_bounds(1.6, 1.7)}
        assert g_convex_supporting_cut(dm.log(x**2 + y**2), m) is None
        cut = g_convex_supporting_cut(dm.log(x**2 + y**2), m, box=tight)
        assert cut is not None
        _feasible_never_cut(dm.log(x**2 + y**2), m, cut, box_lb=[1.4, 1.6], box_ub=[1.5, 1.7])
