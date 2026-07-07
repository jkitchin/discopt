"""Soundness regression tests for ``solver._compute_alphabb_bound``.

The alphaBB underestimator ``L(x) = f(x) - sum_i alpha_i (x_i-lb_i)(ub_i-x_i)``
is a valid lower bound on ``f`` over ``[node_lb, node_ub]`` ONLY when

  (a) x stays inside the box (there the perturbation is >= 0 so L <= f), AND
  (b) L is CONVEX over the WHOLE box (else its supporting hyperplane can sit
      ABOVE ``min_box f``).

Two false-optimal classes are pinned here:

* The clip-mismatch (historical): the optimizer was pushed OUTSIDE the true box
  on big-M / unbounded nodes, turning the perturbation negative and yielding a
  spurious ~1e17 "lower bound". The box-limit / abstain guards close it.

* **C-17**: ``alpha`` used to come from ``estimate_alpha`` — the Hessian SAMPLED
  at a fixed set of interior points and inflated x1.5 — and convexity was checked
  ONLY at the box center. A negative-curvature band narrower than the sample
  spacing left ``alpha`` too small; the center Hessian still passed the gate, so
  ``L`` was nonconvex over the box and its "lower bound" EXCEEDED ``min_box f`` ->
  the node holding the optimum was fathomed -> wrong answer certified optimal.
  The fix routes the node bound through ``rigorous_alpha`` (a SOUND interval
  Hessian + interval-Gershgorin eigenvalue bound), which makes ``L`` provably
  convex over the box or ABSTAINS (``+inf`` -> no bound). No sampled alpha, no
  center-only PSD fast path.

A lower bound must NEVER exceed the true minimum of ``f`` over the box. These
tests build REAL models (so alpha is derived rigorously, as in production) and
pin that invariant.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.solver import _compute_alphabb_bound

pytestmark = pytest.mark.smoke


def _build(expr_fn, lb, ub, names=("x",)):
    """Build (evaluator, model, internal_expr) for a minimize model.

    ``expr_fn`` maps the created scalar Variables (in ``names`` order) to the
    objective expression. Returns the pieces ``_compute_alphabb_bound`` needs.
    """
    m = Model()
    lb = np.atleast_1d(np.asarray(lb, dtype=np.float64))
    ub = np.atleast_1d(np.asarray(ub, dtype=np.float64))
    xs = [m.continuous(nm, lb=float(lb[i]), ub=float(ub[i])) for i, nm in enumerate(names)]
    obj = expr_fn(*xs)
    m.minimize(obj)
    ev = NLPEvaluator(m)
    internal_expr = m._objective.expression  # minimize sense: no negation
    return ev, m, internal_expr


def _dense_min(f, lb, ub, n=2_000_001):
    grid = np.linspace(lb, ub, n)
    return float(np.min(f(grid)))


# ──────────────────────────────────────────────────────────────────────
# C-17: the sampled-alpha + center-only-PSD false optimal.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("s", [0.006, 0.003, 0.0015])
def test_c17_spike_bound_is_sound(s):
    """``f(x) = 1/2 x^2 - B exp(-(x-a)^2 / 2 s^2)`` on [-2, 2], B=4, a=1.

    A negative-curvature spike narrower than the sampled-Hessian spacing made
    ``estimate_alpha`` return 0, the center Hessian (=1) passed the old PSD gate,
    and the returned "bound" was 0.0 while ``min_box f = -3.5`` -> false optimal.
    The rigorous path must return a bound <= the true box minimum (or abstain).
    """
    B, a = 4.0, 1.0
    LB, UB = -2.0, 2.0
    ev, m, expr = _build(
        lambda x: 0.5 * x * x - B * dm.exp(-((x - a) ** 2) / (2.0 * s * s)),
        LB,
        UB,
    )
    true_min = _dense_min(lambda g: 0.5 * g**2 - B * np.exp(-((g - a) ** 2) / (2 * s * s)), LB, UB)
    bound = _compute_alphabb_bound(ev, m, expr, np.array([LB]), np.array([UB]))
    # Sound: never above the true box minimum. (May abstain to -inf.)
    assert bound <= true_min + 1e-6, f"unsound alphaBB bound {bound} > true min {true_min}"


def test_c17_sampled_alpha_would_have_been_unsound():
    """Documents the bug: the OLD sampled-alpha + center-only gate DOES exceed the
    true box minimum on the spike, so the fix is load-bearing (not vacuous)."""
    from discopt._jax.alphabb import estimate_alpha

    B, a, s = 4.0, 1.0, 0.006
    LB, UB = -2.0, 2.0
    ev, m, expr = _build(
        lambda x: 0.5 * x * x - B * dm.exp(-((x - a) ** 2) / (2.0 * s * s)),
        LB,
        UB,
    )
    true_min = _dense_min(lambda g: 0.5 * g**2 - B * np.exp(-((g - a) ** 2) / (2 * s * s)), LB, UB)
    alpha_sampled = np.asarray(estimate_alpha(ev._obj_fn, np.array([LB]), np.array([UB])))
    # The sampled alpha collapses to 0 (spike falls between all sample points)...
    assert float(alpha_sampled[0]) < 1e-8
    # ...and the rigorous alpha needed to convexify the box is enormous & finite.
    from discopt._jax.alphabb import rigorous_alpha

    alpha_rig = np.asarray(rigorous_alpha(expr, m))
    assert np.all(np.isfinite(alpha_rig))
    assert float(alpha_rig[0]) > float(alpha_sampled[0]) + 1.0
    # The now-current bound stays sound where the sampled one would not.
    bound = _compute_alphabb_bound(ev, m, expr, np.array([LB]), np.array([UB]))
    assert bound <= true_min + 1e-6


# ──────────────────────────────────────────────────────────────────────
# Big-M / unbounded-box abstain (historical clip-mismatch class).
# ──────────────────────────────────────────────────────────────────────


def test_infinite_bound_abstains():
    """A genuinely unbounded variable yields -inf (abstain), never a finite lie."""
    ev, m, expr = _build(lambda x: x * x, -10.0, 1.0e19)
    bound = _compute_alphabb_bound(ev, m, expr, np.array([0.0]), np.array([np.inf]))
    assert bound == -np.inf


def test_bound_above_box_limit_abstains():
    """|bound| > 1e8 (big-M territory) abstains rather than risk corruption."""
    ev, m, expr = _build(lambda x: x, 0.0, 1.0e19)
    assert _compute_alphabb_bound(ev, m, expr, np.array([0.0]), np.array([1.0e9])) == -np.inf


def test_multivariable_box_with_one_huge_dimension_abstains():
    """A single unbounded dimension is enough to abstain."""
    ev, m, expr = _build(lambda x, y: x + y, [10.0, 100.0], [20.0, 1.0e19], names=("x", "y"))
    bound = _compute_alphabb_bound(ev, m, expr, np.array([10.0, 100.0]), np.array([20.0, 1.0e19]))
    assert bound == -np.inf


# ──────────────────────────────────────────────────────────────────────
# In-box soundness on finite boxes.
# ──────────────────────────────────────────────────────────────────────


def test_in_box_bound_is_sound_on_finite_nonconvex():
    """On a finite box alphaBB still produces a valid (<= true min) lower bound."""
    # f(x) = -x^2 is concave on [1, 5]; true min over the box is -25 (at x=5).
    ev, m, expr = _build(lambda x: -(x * x), 1.0, 5.0)
    bound = _compute_alphabb_bound(ev, m, expr, np.array([1.0]), np.array([5.0]))
    true_min = -25.0
    assert np.isfinite(bound)
    assert bound <= true_min + 1e-6, f"unsound: {bound} > {true_min}"


def test_convex_objective_bound_recovers_minimum():
    """On a convex objective rigorous alpha is 0, so L == f and the bound is the
    (near-)exact box minimum -- a useful, tight, and sound bound."""
    ev, m, expr = _build(lambda x: (x - 3.0) ** 2 + 1.0, 0.0, 10.0)
    bound = _compute_alphabb_bound(ev, m, expr, np.array([0.0]), np.array([10.0]))
    assert np.isfinite(bound)
    assert bound <= 1.0 + 1e-6  # minimum at x=3 is 1.0
    assert bound >= 1.0 - 1e-3  # and it is tight (alpha=0 -> L == f)


def test_random_box_panel_never_exceeds_true_min():
    """Differential-bound sweep: over many random sub-boxes of a nonconvex
    objective, the alphaBB bound is NEVER above the dense-grid box minimum."""
    B, a, s = 4.0, 1.0, 0.05
    ev, m, expr = _build(
        lambda x: 0.5 * x * x - B * dm.exp(-((x - a) ** 2) / (2.0 * s * s)),
        -2.0,
        2.0,
    )

    def f(g):
        return 0.5 * g**2 - B * np.exp(-((g - a) ** 2) / (2 * s * s))

    rng = np.random.RandomState(20260703)
    worst = -np.inf
    for _ in range(100):
        c0, c1 = np.sort(rng.uniform(-2.0, 2.0, size=2))
        if c1 - c0 < 1e-3:
            continue
        bound = _compute_alphabb_bound(ev, m, expr, np.array([c0]), np.array([c1]))
        if bound == -np.inf:
            continue
        true_min = _dense_min(f, c0, c1, n=200_001)
        margin = bound - true_min
        worst = max(worst, margin)
        assert margin <= 1e-6, f"unsound on [{c0},{c1}]: bound {bound} > true min {true_min}"
