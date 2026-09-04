"""Tests for the opt-in Gauss-Newton objective Hessian (issue #98).

The Gauss-Newton Hessian ``2 Jᵀ J`` of a sum-of-squares objective sidesteps the
super-linear dense ``jax.hessian`` compile. These tests pin down both the
sum-of-squares *detection* and the numerical contract: gradient/objective are
unchanged, the Hessian is PSD, it equals the exact Hessian at a zero-residual
point and on quadratic objectives, constraint curvature stays exact, and the
evaluator falls back to the dense Hessian whenever GN does not apply.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.least_squares import extract_residuals
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Constant, Model

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
# Constant additive terms (the builtin ``sum()`` trap)
# ─────────────────────────────────────────────────────────────


def test_extract_survives_builtin_sum():
    """``sum(r**2 for ...)`` is THE natural spelling and it must be recognized.

    Builtin ``sum()`` seeds its accumulator with int ``0``, so this builds
    ``0 + r₀² + r₁² + r₂²``. That leading zero used to make the whole objective
    unrecognized, silently disabling Gauss-Newton on both backends with only an
    INFO log — the flag was set, accepted, and did nothing.
    """
    m = Model("d")
    x = m.continuous("x", shape=(3,))
    residuals = extract_residuals(sum((x[i] - float(i)) ** 2 for i in range(3)))
    assert residuals is not None
    assert len(residuals) == 3


def test_extract_ignores_constant_terms():
    """``∇²(f + c) = ∇²f``, so a constant term of any sign contributes nothing."""
    m = Model("d")
    x = m.continuous("x", shape=(2,))
    for offset in (0.0, 7.5, -3.25):
        residuals = extract_residuals(x[0] ** 2 + x[1] ** 2 + offset)
        assert residuals is not None, f"offset {offset} rejected"
        assert len(residuals) == 2


def test_constant_offset_does_not_change_the_hessian():
    """The dropped constant must be genuinely curvature-free, not assumed so."""
    m1 = Model("plain")
    a = m1.continuous("a", shape=(2,), lb=-3, ub=3)
    m1.minimize((a[0] - 1.0) ** 2 + (a[1] + 2.0) ** 2)

    m2 = Model("offset")
    b = m2.continuous("b", shape=(2,), lb=-3, ub=3)
    m2.minimize(sum((b[i] - c) ** 2 for i, c in enumerate((1.0, -2.0))) + 12.0)

    ev1 = NLPEvaluator(m1, gauss_newton=True)
    ev2 = NLPEvaluator(m2, gauss_newton=True)
    assert ev1.is_gauss_newton and ev2.is_gauss_newton
    pt = np.array([0.3, -0.7])
    lam = np.zeros(0)
    np.testing.assert_allclose(
        ev1.evaluate_lagrangian_hessian(pt, 1.0, lam),
        ev2.evaluate_lagrangian_hessian(pt, 1.0, lam),
        rtol=0,
        atol=1e-12,
    )
    # ...and the objective still differs by exactly the dropped constant.
    assert ev2.evaluate_objective(pt) - ev1.evaluate_objective(pt) == pytest.approx(12.0)


def test_bare_constant_objective_declines():
    """``[]`` is falsy, so a constant objective uses the exact Hessian (which is 0)."""
    m = Model("c")
    m.continuous("x", lb=-1, ub=1)
    m.minimize(Constant(4.0))
    assert extract_residuals(Constant(4.0)) == []
    assert NLPEvaluator(m, gauss_newton=True).is_gauss_newton is False


def test_declining_gauss_newton_warns_not_whispers(caplog):
    """An explicitly-set option that does nothing must be audible."""
    m = Model("nl")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    m.minimize(dm.exp(x[0]) + x[1] ** 2)
    with caplog.at_level(logging.WARNING, logger="discopt.nlp"):
        assert NLPEvaluator(m, gauss_newton=True).is_gauss_newton is False
    assert any(
        r.levelno >= logging.WARNING and "gauss_newton" in r.message for r in caplog.records
    ), f"expected a WARNING, got {[(r.levelname, r.message) for r in caplog.records]}"


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

    def sse(pv, qv):
        """Oracle written outside the system: plain numpy on the returned point."""
        return float(np.sum((pv * np.exp(qv * t) - y) ** 2))

    sols = {}
    for gn in (False, True):
        m, p, q = build()
        r = m.solve(gauss_newton=gn, time_limit=120, skip_convex_check=True)
        assert getattr(m, "_gauss_newton_hessian") is gn

        # #1039: this used to assert ``r.status == "optimal"``. Both arms return
        # ``feasible`` with ``node_count == 0`` and ``bound is None``, and raising
        # the budget 120 -> 600 changes nothing (probe_gn.py), so the root
        # relaxation structurally yields no dual bound for a sum-of-squares-of-
        # exponentials objective -- it is not a budget miss. Demanding
        # certification the solver cannot deliver on this instance made the test
        # fail for a reason unrelated to what it tests.
        #
        # The status assertion is therefore SOUNDNESS-shaped rather than
        # completeness-shaped: an honest ``feasible`` is accepted, a claim of
        # ``optimal`` is not accepted on faith.
        assert r.status in ("optimal", "feasible"), f"gn={gn}: unexpected status {r.status}"
        if r.status == "optimal":
            assert r.bound is not None, f"gn={gn}: 'optimal' with no dual bound"
            assert r.bound <= r.objective + 1e-9, (
                f"gn={gn}: certificate inverted, bound {r.bound} > objective {r.objective}"
            )

        pv, qv = float(r.value(p)), float(r.value(q))
        # The reported objective must be the one attained AT the returned point.
        # (#1039 found a case where it was not -- see test_gp_corpus.py's
        # test_bb_reported_objective_is_attained_by_its_own_incumbent.)
        assert r.objective == pytest.approx(sse(pv, qv), rel=1e-9), (
            f"gn={gn}: reported objective {r.objective!r} is not attained by its "
            f"own point (p={pv!r}, q={qv!r}, true SSE {sse(pv, qv)!r})"
        )
        # Both backends must actually recover the generating parameters (2.0, 1.3),
        # which is the substantive claim and is much stronger than a status check.
        assert sse(pv, qv) < 1e-4, f"gn={gn}: poor fit, SSE {sse(pv, qv)}"
        sols[gn] = (pv, qv)

    # The point of the test: Gauss-Newton and the exact Hessian land on the same
    # least-squares solution. Measured agreement is ~1e-12, far inside this.
    assert np.allclose(sols[False], sols[True], atol=1e-3)


# ─────────────────────────────────────────────────────────────
# Independent-oracle checks (added by the #75 review sweep)
#
# Everything above compares GN against the *exact* Hessian or against the other
# backend. Both are discopt code. These two compare against an oracle written
# outside the system -- a hand-derived Jacobian in numpy -- so a shared mistake
# in the two backends cannot make them pass.
# ─────────────────────────────────────────────────────────────

_TS = np.array([0.0, 0.4, 0.9, 1.5, 2.2, 3.0])
_YS = np.array([2.9, 2.1, 1.5, 1.0, 0.75, 0.62])


def _exp_fit_model(gauss_newton=True):
    """min Σ (a·e^(−b·t) + c − y)², written with builtin sum()."""
    m = Model("expfit")
    a = m.continuous("a", lb=0.1, ub=10.0)
    b = m.continuous("b", lb=0.05, ub=5.0)
    c = m.continuous("c", lb=-5.0, ub=5.0)
    m.minimize(sum((a * dm.exp(-b * t) + c - y) ** 2 for t, y in zip(_TS, _YS)))
    m._gauss_newton_hessian = gauss_newton
    return m


def _analytic_jacobian(x):
    """∂r/∂(a,b,c) for r_i = a·e^(−b·t_i) + c − y_i, derived by hand."""
    a, b, _c = x
    e = np.exp(-b * _TS)
    return np.column_stack([e, -a * _TS * e, np.ones_like(_TS)])


@pytest.mark.parametrize("backend", ["jax", "tape"])
def test_gauss_newton_matches_a_hand_derived_jacobian(backend):
    """GN objective Hessian must equal 2·JᵀJ for a J computed outside discopt.

    This is also the variable-ORDER test on the tape arm: ∂r/∂a, ∂r/∂b and ∂r/∂c
    differ by orders of magnitude here, so a column permutation between the main
    tape and the auxiliary residual tape could not pass.
    """
    if backend == "tape":
        pounce = pytest.importorskip("pounce")
        assert pounce is not None
        from discopt._tape_nlp_evaluator import TapeNLPEvaluator

        ev = TapeNLPEvaluator(_exp_fit_model())
    else:
        ev = NLPEvaluator(_exp_fit_model(), gauss_newton=True)
    assert ev.is_gauss_newton, f"{backend}: GN did not activate on a sum-of-squares objective"

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(5):
        x = np.array([rng.uniform(0.5, 4.0), rng.uniform(0.1, 2.0), rng.uniform(-2.0, 2.0)])
        got = np.asarray(ev.evaluate_lagrangian_hessian(x, 1.0, np.zeros(0)))
        J = _analytic_jacobian(x)
        np.testing.assert_allclose(got, 2.0 * (J.T @ J), rtol=1e-7, atol=1e-7)
        checked += 1
    assert checked == 5, "the oracle comparison did not run"


def test_tape_hessian_structure_is_lower_triangular():
    """`hessian_structure` must stay lower-triangular on every objective shape.

    `_widen_for_gauss_newton` unions with `if a >= b` and
    `evaluate_lagrangian_hessian` mirrors with `lower + tril(lower, -1).T`. An
    upper-triangle entry would double-count every off-diagonal GN value and leave
    the dense Hessian asymmetric -- silently. The evaluator now raises on it;
    this pins that the ordinary shapes do not trip it.
    """
    pytest.importorskip("pounce")
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    checked = 0
    for gn in (True, False):
        ev = TapeNLPEvaluator(_exp_fit_model(gauss_newton=gn))
        rows, cols = ev.hessian_structure()
        assert rows.size, "empty structure would make this vacuous"
        assert np.all(rows >= cols), f"upper-triangle entries: {rows[rows < cols]}"
        H = np.asarray(ev.evaluate_lagrangian_hessian(np.array([2.0, 0.7, 0.3]), 1.0, np.zeros(0)))
        np.testing.assert_allclose(H, H.T, rtol=0, atol=1e-12)
        checked += 1
    assert checked == 2
