"""#1728: ``u @ Q @ u`` with constant semidefinite ``Q`` is convex/concave.

Also the soundness hole found beside it: a constant operand of ``@`` kept the
other side's curvature whatever the weights' signs, so ``[-1, -1] @ (x**2)``
(concave) classified CONVEX -- a false convexity proof.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.convexity.rules import classify_expr

N = 4
RNG = np.random.default_rng(0)
_L = RNG.normal(size=(N, N))
PSD = _L @ _L.T + 1e-3 * np.eye(N)
SINGULAR_PSD = np.outer(np.ones(N), np.ones(N))  # (sum w)^2: singular, PSD
INDEF = np.diag([1.0, -1.0, 1.0, 1.0])


def _w():
    m = dm.Model()
    return m, m.continuous("w", shape=(N,), lb=-1, ub=1)


@pytest.mark.parametrize(
    "Q, want",
    [
        (PSD, Curvature.CONVEX),
        (SINGULAR_PSD, Curvature.CONVEX),
        (-PSD, Curvature.CONCAVE),
        (INDEF, Curvature.UNKNOWN),
        (np.triu(PSD) * 2 - np.diag(np.diag(PSD)), Curvature.CONVEX),  # asymmetric, same form
    ],
)
@pytest.mark.parametrize("order", ["left", "right"])
def test_quadratic_form_curvature(Q, want, order):
    m, w = _w()
    expr = (w @ Q) @ w if order == "left" else w @ (Q @ w)
    assert classify_expr(expr, m) == want


def test_quadratic_form_needs_the_same_operand():
    m = dm.Model()
    u = m.continuous("u", shape=(N,), lb=-1, ub=1)
    v = m.continuous("v", shape=(N,), lb=-1, ub=1)
    assert classify_expr(u @ PSD @ v, m) == Curvature.UNKNOWN


@pytest.mark.parametrize(
    "weights, want",
    [
        (np.full(N, -1.0), Curvature.CONCAVE),
        (np.full(N, 2.0), Curvature.CONVEX),
        (np.array([1.0, -1.0, 1.0, 1.0]), Curvature.UNKNOWN),
    ],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_constant_weights_respect_sign(weights, want, side):
    m, w = _w()
    expr = weights @ (w**2) if side == "left" else (w**2) @ weights
    assert classify_expr(expr, m) == want


def test_mixed_sign_weights_on_affine_stay_affine():
    m, w = _w()
    assert classify_expr(INDEF @ w, m) == Curvature.AFFINE


def test_portfolio_with_norm_row_takes_convex_route():
    rng = np.random.default_rng(0)
    n = 6
    mu = 0.05 + 0.1 * rng.random(n)
    L = rng.normal(scale=0.05, size=(n, n))
    Sigma = L @ L.T + 1e-3 * np.eye(n)
    m = dm.Model("p")
    w = m.continuous("w", shape=(n,), lb=0, ub=1)
    m.subject_to(dm.sum(w) == 1)
    m.subject_to(mu @ w >= 0.08)
    m.subject_to(dm.norm(w) <= 0.6)
    m.minimize(w @ Sigma @ w)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert "convex" in (r.algorithm_route or "")
    assert r.objective == pytest.approx(0.0010307078, rel=1e-6)
