"""Tests for the opt-in Gauss-Newton objective Hessian (issue #98).

The Gauss-Newton Hessian ``2 Jᵀ J`` of a sum-of-squares objective sidesteps the
super-linear dense ``jax.hessian`` compile. These tests pin down both the
sum-of-squares *detection* and the numerical contract: gradient/objective are
unchanged, the Hessian is PSD, it equals the exact Hessian at a zero-residual
point and on quadratic objectives, constraint curvature stays exact, and the
evaluator falls back to the dense Hessian whenever GN does not apply.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.least_squares import extract_residuals
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model

# ─────────────────────────────────────────────────────────────
# Residual extraction (sum-of-squares detection)
# ─────────────────────────────────────────────────────────────


def test_extract_scalar_sum_of_squares():
    m = Model("d")
    x = m.continuous("x", shape=(3,))
    expr = (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + x[2] ** 2
    residuals = extract_residuals(expr)
    assert residuals is not None
    assert len(residuals) == 3


def test_extract_vectorized_square():
    m = Model("d")
    x = m.continuous("x", shape=(3,))
    A = np.arange(9.0).reshape(3, 3)
    b = np.ones(3)
    residuals = extract_residuals(dm.sum((A @ x - b) ** 2))
    assert residuals is not None
    assert len(residuals) == 1  # one array-valued residual (raveled downstream)


def test_extract_product_form_square():
    # sum((Ax-b) * (Ax-b)) with structurally-identical (distinct object) factors
    m = Model("d")
    x = m.continuous("x", shape=(3,))
    A = np.arange(9.0).reshape(3, 3)
    b = np.ones(3)
    residuals = extract_residuals(dm.sum((A @ x - b) * (A @ x - b)))
    assert residuals is not None
    assert len(residuals) == 1


def test_extract_nonnegative_weighted():
    m = Model("d")
    x = m.continuous("x", shape=(2,))
    residuals = extract_residuals(4.0 * (x[0] - 1.0) ** 2 + x[1] ** 2)
    assert residuals is not None
    assert len(residuals) == 2


def test_extract_division_by_positive_constant():
    # Variance-weighted least squares: (...)**2 / sigma with sigma > 0 is a
    # sum of squares with residual sqrt(1/sigma)·base (issue #100).
    m = Model("d")
    x = m.continuous("x", shape=(2,))
    expr = (x[0] - 1.0) ** 2 / 1e-3 + (x[1] - 2.0) ** 2 / 4.0
    residuals = extract_residuals(expr)
    assert residuals is not None
    assert len(residuals) == 2


def test_extract_division_of_vectorized_square():
    m = Model("d")
    x = m.continuous("x", shape=(3,))
    A = np.arange(9.0).reshape(3, 3)
    b = np.ones(3)
    residuals = extract_residuals(dm.sum((A @ x - b) ** 2) / 2.0)
    assert residuals is not None
    assert len(residuals) == 1


def test_division_by_constant_matches_equivalent_weight():
    # g / c and (1/c) * g must produce the *same* Gauss-Newton Hessian, since
    # the residual scaling sqrt(1/c) is identical for both forms.
    m_div = Model("div")
    xd = m_div.continuous("x", shape=(2,), lb=-5, ub=5)
    m_div.minimize((xd[0] - 1.0) ** 2 / 4.0 + (xd[1] - 2.0) ** 2 / 0.25)
    ev_div = NLPEvaluator(m_div, gauss_newton=True)
    assert ev_div.is_gauss_newton

    m_mul = Model("mul")
    xm = m_mul.continuous("x", shape=(2,), lb=-5, ub=5)
    m_mul.minimize(0.25 * (xm[0] - 1.0) ** 2 + 4.0 * (xm[1] - 2.0) ** 2)
    ev_mul = NLPEvaluator(m_mul, gauss_newton=True)

    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 2)
    lam = np.array([], dtype=np.float64)
    H_div = ev_div.evaluate_lagrangian_hessian(x, 1.0, lam)
    H_mul = ev_mul.evaluate_lagrangian_hessian(x, 1.0, lam)
    np.testing.assert_allclose(H_div, H_mul, atol=1e-12)
    # Closed form: H = diag(2/4, 2/0.25) = diag(0.5, 8).
    np.testing.assert_allclose(np.diag(H_div), [0.5, 8.0], atol=1e-10)


@pytest.mark.parametrize(
    "make_expr",
    [
        lambda x: dm.exp(x[0]) + x[1] ** 2,  # transcendental term, not SoS
        lambda x: x[0] ** 2 - x[1] ** 2,  # difference of squares
        lambda x: -2.0 * x[0] ** 2,  # negative weight
        lambda x: x[0] ** 3 + x[1] ** 2,  # cubic term
        lambda x: x[0] * x[1],  # bilinear, not a square (distinct factors)
        lambda x: x[0] ** 2 / (-2.0),  # division by a negative constant
        lambda x: x[0] ** 2 / (x[1] + 3.0),  # non-constant denominator
    ],
)
def test_extract_rejects_non_sum_of_squares(make_expr):
    m = Model("d")
    x = m.continuous("x", shape=(2,))
    assert extract_residuals(make_expr(x)) is None


# ─────────────────────────────────────────────────────────────
# Gauss-Newton Hessian numerical contract
# ─────────────────────────────────────────────────────────────


def _bilinear_ls(nt=8, nl=5, nc=3):
    """sum over scalar bilinear terms of (Σ_c C[t,c] S[l,c] - 1)² — the issue's
    stress objective, at a small size."""
    m = Model("bilinear")
    C = m.continuous("C", shape=(nt, nc), lb=0)
    S = m.continuous("S", shape=(nl, nc), lb=0)
    D = np.ones((nt, nl))
    terms = []
    for ti in range(nt):
        for li in range(nl):
            pred = C[ti, 0] * S[li, 0]
            for ci in range(1, nc):
                pred = pred + C[ti, ci] * S[li, ci]
            terms.append((pred - float(D[ti, li])) ** 2)
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    m.minimize(total)
    return m, nt, nl, nc


def test_gauss_newton_active_flag():
    m, *_ = _bilinear_ls()
    assert NLPEvaluator(m, gauss_newton=True).is_gauss_newton is True
    assert NLPEvaluator(m, gauss_newton=False).is_gauss_newton is False


def test_gradient_and_objective_unchanged():
    m, nt, nl, nc = _bilinear_ls()
    ev_full = NLPEvaluator(m, gauss_newton=False)
    ev_gn = NLPEvaluator(m, gauss_newton=True)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.2, 1.0, ev_full.n_variables)
    assert np.allclose(ev_full.evaluate_gradient(x), ev_gn.evaluate_gradient(x))
    assert np.isclose(ev_full.evaluate_objective(x), ev_gn.evaluate_objective(x))


def test_gauss_newton_hessian_is_psd():
    m, *_ = _bilinear_ls()
    ev_gn = NLPEvaluator(m, gauss_newton=True)
    rng = np.random.default_rng(2)
    x = rng.uniform(0.2, 1.0, ev_gn.n_variables)
    eigmin = float(np.linalg.eigvalsh(ev_gn.evaluate_hessian(x))[0])
    assert eigmin > -1e-8


def test_gauss_newton_matches_full_at_zero_residual():
    m, nt, nl, nc = _bilinear_ls()
    ev_full = NLPEvaluator(m, gauss_newton=False)
    ev_gn = NLPEvaluator(m, gauss_newton=True)
    # Construct x with all residuals exactly zero: C[:,0]=S[:,0]=1, rest 0 -> pred=1=D.
    n = ev_full.n_variables
    x0 = np.zeros(n)
    C0 = x0[: nt * nc].reshape(nt, nc)
    S0 = x0[nt * nc :].reshape(nl, nc)
    C0[:, 0] = 1.0
    S0[:, 0] = 1.0
    assert np.isclose(ev_full.evaluate_objective(x0), 0.0)
    Hf = ev_full.evaluate_hessian(x0)
    Hg = ev_gn.evaluate_hessian(x0)
    assert np.allclose(Hf, Hg, atol=1e-8)


def test_gauss_newton_differs_away_from_solution():
    # Sanity check that GN is genuinely dropping the second-order term, so the
    # zero-residual agreement above is meaningful.
    m, *_ = _bilinear_ls()
    ev_full = NLPEvaluator(m, gauss_newton=False)
    ev_gn = NLPEvaluator(m, gauss_newton=True)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 1.0, ev_full.n_variables)
    Hf = ev_full.evaluate_hessian(x)
    Hg = ev_gn.evaluate_hessian(x)
    assert np.linalg.norm(Hf - Hg) > 1e-6


def test_lagrangian_keeps_exact_constraint_curvature():
    # Quadratic (sum-of-squares) objective => GN objective Hessian is exact, and
    # the constraint curvature must be kept exact, so the Lagrangian Hessian
    # equals the dense reference exactly even for a nonlinear constraint.
    m = Model("lh")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    m.minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] + 1) ** 2)
    m.subject_to(x[0] * x[1] + x[2] ** 2 == 1.0)
    ev_full = NLPEvaluator(m, gauss_newton=False)
    ev_gn = NLPEvaluator(m, gauss_newton=True)
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, ev_full.n_variables)
    lam = rng.uniform(-1, 1, ev_full.n_constraints)
    Hf = ev_full.evaluate_lagrangian_hessian(x, 1.0, lam)
    Hg = ev_gn.evaluate_lagrangian_hessian(x, 1.0, lam)
    assert np.allclose(Hf, Hg, atol=1e-8)


def test_hessian_values_coo_consistent_under_gauss_newton():
    m, *_ = _bilinear_ls()
    ev_gn = NLPEvaluator(m, gauss_newton=True)
    rng = np.random.default_rng(4)
    x = rng.uniform(0.2, 1.0, ev_gn.n_variables)
    lam = np.array([], dtype=np.float64)
    rows, cols = ev_gn.hessian_structure()
    vals = ev_gn.evaluate_hessian_values(x, 1.0, lam)
    H = ev_gn.evaluate_lagrangian_hessian(x, 1.0, lam)
    assert np.allclose(vals, H[rows, cols], atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Fallbacks: GN silently declines and uses the exact dense Hessian
# ─────────────────────────────────────────────────────────────


def test_fallback_when_not_sum_of_squares():
    m = Model("nl")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    m.minimize(dm.exp(x[0]) + x[1] ** 2)
    ev = NLPEvaluator(m, gauss_newton=True)
    assert ev.is_gauss_newton is False
    # Hessian still correct: d²/dx0² exp(x0) = exp(x0), d²/dx1² = 2.
    H = ev.evaluate_hessian(np.zeros(2))
    assert np.isclose(H[0, 0], 1.0) and np.isclose(H[1, 1], 2.0)


def test_fallback_when_maximize():
    m = Model("mx")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    m.maximize(-((x[0]) ** 2) - (x[1]) ** 2)  # concave, maximized -> not min SoS
    ev = NLPEvaluator(m, gauss_newton=True)
    assert ev.is_gauss_newton is False


# ─────────────────────────────────────────────────────────────
# End-to-end plumbing through Model.solve
# ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_solve_gauss_newton_matches_full_nonlinear_ls():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 12)
    y = 2.0 * np.exp(1.3 * t) + 1e-3 * rng.standard_normal(12)

    def build():
        m = Model("exp")
        p = m.continuous("p", lb=0.1, ub=5)
        q = m.continuous("q", lb=0.1, ub=3)
        expr = (p * dm.exp(q * float(t[0])) - float(y[0])) ** 2
        for i in range(1, len(t)):
            expr = expr + (p * dm.exp(q * float(t[i])) - float(y[i])) ** 2
        m.minimize(expr)
        return m, p, q

    sols = {}
    for gn in (False, True):
        m, p, q = build()
        r = m.solve(gauss_newton=gn, time_limit=120, skip_convex_check=True)
        assert getattr(m, "_gauss_newton_hessian") is gn
        assert r.status == "optimal"
        sols[gn] = (r.value(p), r.value(q))
    assert np.allclose(sols[False], sols[True], atol=1e-3)
