"""Soundness regression tests for ``solver._compute_alphabb_bound``.

The alphaBB underestimator L(x) = f(x) - sum_i alpha_i (x_i-lb_i)(ub_i-x_i) is a
valid lower bound on f ONLY for x inside [node_lb, node_ub]; there the
perturbation term is >= 0 so L <= f. A previous implementation minimized L over
a domain CLIPPED to [-1e4, 1e4] while the perturbation kept using the unclipped
node bounds. On any node with a bound outside that range (e.g. a big-M ~1e19 on
an unbounded variable) the optimizer was pushed OUTSIDE the true box, the
perturbation went NEGATIVE, and L turned into an over-estimator — producing a
spurious ~1e17 "lower bound" that falsely certified suboptimal incumbents as
global (regression: gms60/prob07 reported obj=160488 glob=True; true opt
154990).

A lower bound must NEVER exceed the true minimum of f over the box. These tests
pin that invariant for the large/unbounded-box cases that triggered the bug, and
the in-box soundness that alphaBB must still deliver.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from discopt.solver import _compute_alphabb_bound


class _StubEvaluator:
    """Evaluator exposing objective + finite-difference gradient/Hessian, matching
    the real ``_make_evaluator`` interface the alphaBB bound now relies on."""

    def __init__(self, f):
        self._f = f

    def evaluate_objective(self, x):
        return float(self._f(np.asarray(x, dtype=np.float64)))

    def evaluate_gradient(self, x):
        x = np.asarray(x, dtype=np.float64)
        h = 1e-6
        g = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += h
            xm = x.copy()
            xm[i] -= h
            g[i] = (self._f(xp) - self._f(xm)) / (2.0 * h)
        return g

    def evaluate_hessian(self, x):
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        h = 1e-4
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                xpp = x.copy()
                xpp[i] += h
                xpp[j] += h
                xpm = x.copy()
                xpm[i] += h
                xpm[j] -= h
                xmp = x.copy()
                xmp[i] -= h
                xmp[j] += h
                xmm = x.copy()
                xmm[i] -= h
                xmm[j] -= h
                num = self._f(xpp) - self._f(xpm) - self._f(xmp) + self._f(xmm)
                H[i, j] = num / (4.0 * h * h)
        return 0.5 * (H + H.T)


def test_large_lower_bound_does_not_fabricate_positive_bound():
    """The exact prob07 trigger: node_lb >> 1e4 with a huge node_ub.

    f(x) = x is linear, so its true minimum over the box is node_lb. The old
    clip-mismatch returned ~+9e17 here; the fix must return a SOUND bound, i.e.
    <= the true minimum (it abstains to -inf because the box is unbounded).
    """
    ev = _StubEvaluator(lambda x: x[0])
    node_lb = np.array([1.0e5])
    node_ub = np.array([1.0e19])
    alpha = np.array([1.0e-6])

    bound = _compute_alphabb_bound(ev, node_lb, node_ub, alpha)
    true_min = 1.0e5
    assert bound <= true_min + 1e-6, f"unsound alphaBB bound {bound} > true min {true_min}"
    assert bound == -np.inf  # abstains on the oversized box


def test_infinite_bound_abstains():
    """A genuinely unbounded variable yields -inf (abstain), never a finite lie."""
    ev = _StubEvaluator(lambda x: x[0] ** 2)
    node_lb = np.array([0.0])
    node_ub = np.array([np.inf])
    alpha = np.array([1.0])
    assert _compute_alphabb_bound(ev, node_lb, node_ub, alpha) == -np.inf


def test_bound_above_box_limit_abstains():
    """|bound| > 1e8 (big-M territory) abstains rather than risk corruption."""
    ev = _StubEvaluator(lambda x: x[0])
    node_lb = np.array([0.0])
    node_ub = np.array([1.0e9])
    alpha = np.array([1.0])
    assert _compute_alphabb_bound(ev, node_lb, node_ub, np.array([1e-3])) == -np.inf
    # ...but a box just under the limit is honored (and stays sound).
    node_ub_ok = np.array([1.0e7])
    b = _compute_alphabb_bound(ev, node_lb, node_ub_ok, alpha)
    assert b <= 0.0 + 1e-6  # true min of f=x over [0, 1e7] is 0


def test_in_box_bound_is_sound_on_finite_nonconvex():
    """On a finite box alphaBB still produces a valid (<= true min) lower bound."""
    # f(x) = -x^2 is concave on [1, 5]; true min over the box is -25 (at x=5).
    ev = _StubEvaluator(lambda x: -(x[0] ** 2))
    node_lb = np.array([1.0])
    node_ub = np.array([5.0])
    alpha = np.array([1.5])  # >= 1 convexifies (Hessian of -x^2 is -2)

    bound = _compute_alphabb_bound(ev, node_lb, node_ub, alpha)
    true_min = -25.0
    assert np.isfinite(bound)
    assert bound <= true_min + 1e-6, f"unsound: {bound} > {true_min}"


def test_zero_alpha_recovers_objective_minimum():
    """alpha = 0 makes L == f, so the bound is exactly min f over the box."""
    ev = _StubEvaluator(lambda x: (x[0] - 3.0) ** 2 + 1.0)
    node_lb = np.array([0.0])
    node_ub = np.array([10.0])
    alpha = np.array([0.0])

    bound = _compute_alphabb_bound(ev, node_lb, node_ub, alpha)
    assert abs(bound - 1.0) < 1e-4  # minimum at x=3 is 1.0


def test_multivariable_box_with_one_huge_dimension_abstains():
    """A single unbounded dimension is enough to abstain (mirrors lifted prob07)."""
    ev = _StubEvaluator(lambda x: x[0] + x[1])
    node_lb = np.array([10.0, 100.0])
    node_ub = np.array([20.0, 1.0e19])  # second dim is big-M
    alpha = np.array([1e-3, 1e-3])
    assert _compute_alphabb_bound(ev, node_lb, node_ub, alpha) == -np.inf
