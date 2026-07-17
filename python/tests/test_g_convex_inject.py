"""Soundness tests for the flag-gated G-convexity cut injector (#181).

The injector adds linear cuts to the model to expose the transformed convex
shape of a G-convex constraint body. Its one non-negotiable property is
**soundness**: an injected cut must never remove a point that is feasible for
the original constraint. These tests fuzz thousands of feasible points against
every injected cut, and also confirm the cuts are non-vacuous (they separate
relaxation points the original box does not) and that the flag gates the whole
thing off by default.
"""

from __future__ import annotations

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
from discopt._jax.convexity.g_convex_inject import (
    g_convex_cuts_enabled,
    inject_g_convex_cuts,
)
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import Model


def _cut_bodies(m):
    return [
        compile_expression(c.body, m)
        for c in m._constraints
        if c.name and c.name.startswith("gconv_cut")
    ]


def _feasible_never_removed(m, body_expr, cut_fns, *, box_lb, box_ub, seed=0, n=60000):
    """No point feasible for ``body_expr ≤ 0`` violates any injected cut.

    The fuzz is fully vectorized: the ``n`` sample points are drawn as one
    ``(n, d)`` block and every compiled body/cut is evaluated with ``jax.vmap``
    in a single dispatch. ``rng.uniform`` fills the block row-major from the
    same bit stream, so these are the *identical* points the old per-point loop
    drew — same samples, same tolerance, same soundness assertion, ~1000× fewer
    JAX dispatches.
    """
    fbody = compile_expression(body_expr, m)
    rng = np.random.default_rng(seed)
    x = rng.uniform(box_lb, box_ub, size=(n, len(box_lb)))
    feasible = np.asarray(jax.vmap(fbody)(jnp.asarray(x))) <= 0.0  # feasible for the original
    checked = int(feasible.sum())
    worst = -np.inf
    if checked:
        xf = jnp.asarray(x[feasible])
        for fc in cut_fns:
            worst = max(worst, float(jnp.max(jax.vmap(fc)(xf))))  # cut body ≤ 0 form
    assert worst <= 1e-7, f"injected cut removed a feasible point (residual={worst})"
    return checked


class TestInjectorGating:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("DISCOPT_G_CONVEX_CUTS", raising=False)
        assert g_convex_cuts_enabled() is False

    def test_enabled_by_flag(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_G_CONVEX_CUTS", "1")
        assert g_convex_cuts_enabled() is True


class TestInjectorSoundness:
    def _model(self, rhs_const):
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        body = dm.log(x**2 + y**2) - rhs_const  # G-convex body, not convex
        m.subject_to(body <= 0, name="c0")
        m.minimize(x + y)
        return m, body

    def test_cuts_injected_and_sound(self):
        m, body = self._model(1.6)  # feasible region intersects the box
        n = inject_g_convex_cuts(m)
        assert n >= 1
        checked = _feasible_never_removed(
            m, body, _cut_bodies(m), box_lb=[1.4, 1.6], box_ub=[1.5, 1.7]
        )
        assert checked > 1000  # the fuzz actually exercised feasible points

    def test_cuts_are_non_vacuous(self):
        m, body = self._model(1.6)
        inject_g_convex_cuts(m)
        fbody = compile_expression(body, m)
        cut_fns = _cut_bodies(m)
        rng = np.random.default_rng(3)
        x = rng.uniform([1.4, 1.6], [1.5, 1.7], size=(40000, 2))
        infeasible = np.asarray(jax.vmap(fbody)(jnp.asarray(x))) > 0.0  # infeasible for original
        cut_max = np.full(x.shape[0], -np.inf)  # max cut residual per point
        for fc in cut_fns:
            cut_max = np.maximum(cut_max, np.asarray(jax.vmap(fc)(jnp.asarray(x))))
        separated = int(np.count_nonzero(infeasible & (cut_max > 1e-7)))
        assert separated > 0, "cuts are valid but vacuous — they separate nothing"

    def test_ge_constraint_g_concave_body(self):
        # exp(-(x²+y²)) ≥ 0.2 : body ≥ 0 with a G-concave body → -body G-convex.
        m = Model("t")
        x = m.continuous("x", lb=0.45, ub=0.55)
        y = m.continuous("y", lb=0.45, ub=0.55)
        body = dm.exp(-(x**2 + y**2)) - 0.2
        m.subject_to(body >= 0, name="c0")
        m.minimize(x + y)
        inject_g_convex_cuts(m)
        # feasible for body ≥ 0 ⟺ -body ≤ 0; check no such point is removed.
        neg = -body
        _feasible_never_removed(m, neg, _cut_bodies(m), box_lb=[0.45, 0.45], box_ub=[0.55, 0.55])

    def test_convex_body_not_injected(self):
        # An ordinarily-convex body (ρ=0) is left to the existing OA path.
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.subject_to(x**2 - 0.5 <= 0, name="c0")
        m.minimize(x)
        assert inject_g_convex_cuts(m) == 0

    def test_array_variable_model_skipped(self):
        # Non-scalar variables are out of scope for this slice → no cut, no crash.
        m = Model("t")
        x = m.continuous("x", shape=(2,), lb=1.4, ub=1.5)
        m.subject_to(dm.log(x[0] ** 2 + x[1] ** 2) - 1.6 <= 0, name="c0")
        m.minimize(x[0] + x[1])
        assert inject_g_convex_cuts(m) == 0

    def test_unbounded_variable_skipped(self):
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y")  # unbounded
        m.subject_to(dm.log(x**2 + y**2) - 1.6 <= 0, name="c0")
        m.minimize(x + y)
        assert inject_g_convex_cuts(m) == 0
