"""#2062: a Taylor-form test of ``f''`` certifies the PFR rows #2054 left uncut.

The natural interval Hessian of ``(exp(t) - 1) / (exp(0.75 t) - 1)`` loses the
correlation between numerator and denominator, so #2054's bisection needed
2048-8192 pieces (3-10 s) to certify the PFR by-product row ``c`` and volume row
``v``. A Taylor expansion of ``f''`` about each piece's midpoint with a Lagrange
remainder (third-order-and-up interval Taylor jets, ``interval_jet.py``) removes
that loss at order ``w**K``: both rows certify inside the default 128-piece
budget. ``DISCOPT_OA_CONVEXITY_TAYLOR`` gates it.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import Curvature
from discopt._relax.convexity.certificate import (
    _oa_convexity_taylor_enabled,
    certify_convex,
    certify_convex_subdivided,
)
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_jet import (
    JetAbstain,
    interval_jet,
    second_derivative_enclosure,
)
from discopt._relax.convexity.rules import classify_oa_cut_convexity

KAP, TH = 0.25, (0.25, 4.0)
FLAG = "DISCOPT_OA_CONVEXITY_TAYLOR"


def c_pfr(th, ops):
    """PFR: C made per B (the issue's ``c``)."""
    e = ops.exp((1 - KAP) * th) - 1
    return (1 - KAP) * (ops.exp(th) - 1) / e - 1


def v_pfr(th, ops):
    """PFR: theta times A fed per B (the issue's ``v``)."""
    e = ops.exp((1 - KAP) * th) - 1
    return (1 - KAP) * th * ops.exp(th) / e


ROWS = [(c_pfr, 50.0), (v_pfr, 20.0)]


def _row_model(f, coef=50.0, lb=TH[0], ub=TH[1]):
    m = dm.Model("row")
    th = m.continuous("theta", lb=lb, ub=ub)
    w = m.continuous("W", lb=0, ub=1e4)
    m.subject_to(coef * f(th, dm) - w <= 0)
    m.minimize(w)
    return m, th


def _numerically_convex(f, lb=TH[0], ub=TH[1]) -> bool:
    g = np.linspace(lb, ub, 4001)
    y = f(g, np)
    return bool((y[2:] - 2 * y[1:-1] + y[:-2] >= -1e-12).all())


def _fd2(f, t: float) -> float:
    """``f''(t)`` by Richardson-extrapolated central differences (~1e-9 rel)."""

    def d2(h):
        return (f(t + h, np) - 2 * f(t, np) + f(t - h, np)) / (h * h)

    h = 1e-3 * max(1.0, abs(t))
    return float((16 * d2(h / 2) - d2(h)) / 15)


# ── the flag ──────────────────────────────────────────────────────────────


def test_flag_defaults_off_with_an_opt_in(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not _oa_convexity_taylor_enabled()
    for on in ("1", "true", "on"):
        monkeypatch.setenv(FLAG, on)
        assert _oa_convexity_taylor_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not _oa_convexity_taylor_enabled()


# ── the jets are correct ──────────────────────────────────────────────────


def _binom(p: float, k: int) -> float:
    """Generalised binomial coefficient ``p choose k``."""
    out = 1.0
    for i in range(k):
        out *= (p - i) / (i + 1)
    return out


def _pow_coef(p: float):
    """Normalised Taylor coefficients of ``t**p``: ``binom(p, k) t**(p - k)``."""
    return lambda t, k: 0.0 if _binom(p, k) == 0.0 else _binom(p, k) * t ** (p - k)


ATOMS = [
    # (expression builder, k-th normalised Taylor coefficient at t, domain)
    (lambda t: dm.exp(2 * t), lambda t, k: 2.0**k * math.exp(2 * t) / math.factorial(k), (-1, 2)),
    (
        lambda t: dm.log(t),
        lambda t, k: math.log(t) if k == 0 else (-1) ** (k + 1) / (k * t**k),
        (0.2, 5),
    ),
    (lambda t: dm.sqrt(t), lambda t, k: _pow_coef(0.5)(t, k), (0.2, 5)),
    (lambda t: t**-3, _pow_coef(-3), (0.3, 3)),
    (lambda t: t**5, _pow_coef(5), (-2, 2)),
    (lambda t: t**1.7, _pow_coef(1.7), (0.2, 3)),
    (lambda t: 1 / (1 + t), lambda t, k: (-1) ** k / (1 + t) ** (k + 1), (0, 3)),
    (
        lambda t: dm.sin(3 * t),
        lambda t, k: 3.0**k * math.sin(3 * t + k * math.pi / 2) / math.factorial(k),
        (-3, 3),
    ),
    (lambda t: dm.cos(t), lambda t, k: math.cos(t + k * math.pi / 2) / math.factorial(k), (-3, 3)),
]


@pytest.mark.parametrize(
    "build, coef, dom",
    ATOMS,
    ids=["exp", "log", "sqrt", "inv_cube", "pow5", "pow1_7", "recip", "sin", "cos"],
)
def test_point_jets_enclose_the_closed_form_coefficients(build, coef, dom):
    m = dm.Model("atom")
    t = m.continuous("t", lb=dom[0], ub=dom[1])
    expr = build(t)
    checked = 0
    for x in np.linspace(dom[0], dom[1], 9):
        jet = interval_jet(expr, t, {t: Interval.point(float(x))}, 6)
        for k, iv_k in enumerate(jet):
            true = coef(float(x), k)
            tol = 1e-12 * max(1.0, abs(true))
            assert iv_k.lo - tol <= true <= iv_k.hi + tol, (x, k, iv_k, true)
            checked += 1
    assert checked == 9 * 7


@pytest.mark.parametrize("f, coef", ROWS, ids=["c", "v"])
def test_point_jets_match_finite_differences_on_the_reactor_rows(f, coef):
    m, th = _row_model(f, coef)
    expr = f(th, dm)
    checked = 0
    for x in np.linspace(*TH, 21):
        a2 = interval_jet(expr, th, {th: Interval.point(float(x))}, 2)[2]
        ref = _fd2(f, float(x)) / 2.0
        assert abs(0.5 * (a2.lo + a2.hi) - ref) <= 1e-6 * max(1.0, abs(ref))
        checked += 1
    assert checked == 21


# ── the enclosures are sound ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "f, lb, ub",
    [
        (c_pfr, *TH),
        (v_pfr, *TH),
        (lambda t, ops: t**3 - 2 * t, -1.0, 1.0),
        (lambda t, ops: ops.sin(t) * ops.exp(-t / 3), -2.0, 6.0),
        (lambda t, ops: 1 / ops.exp(ops.exp(t)), -2.0, 2.0),
        (lambda t, ops: ops.log(1 + t * t) / (2 + ops.cos(t)), -3.0, 3.0),
    ],
    ids=["c", "v", "cubic", "damped_sin", "exp_neg_exp", "log_over_cos"],
)
@pytest.mark.parametrize("order", [1, 2, 4])
def test_enclosure_contains_sampled_second_derivatives(f, lb, ub, order):
    m = dm.Model("s")
    t = m.continuous("t", lb=lb, ub=ub)
    expr = f(t, dm)
    rng = np.random.default_rng(2062)
    edges = np.sort(np.concatenate([[lb, ub], rng.uniform(lb, ub, 15)]))
    lo, hi = edges[:-1], edges[1:]
    enc_lo, enc_hi = second_derivative_enclosure(expr, t, {}, lo, hi, taylor_order=order)
    checked = 0
    for a, b, el, eh in zip(lo, hi, enc_lo, enc_hi):
        assert np.isfinite(el) and np.isfinite(eh)
        for x in np.linspace(a, b, 7):
            true = _fd2(f, float(x))
            tol = 1e-6 * max(1.0, abs(true))
            assert el - tol <= true <= eh + tol, (a, b, x, el, eh, true)
            checked += 1
    assert checked == 7 * lo.size


def test_lanes_are_independent():
    """One batched walk returns exactly what one walk per piece returns.

    Regression: the first batched version read the split variable's box with
    ``ravel()[0]`` and evaluated every lane on lane 0's interval.
    """
    m = dm.Model("lanes")
    t = m.continuous("t", lb=0.25, ub=4.0)
    expr = c_pfr(t, dm)
    edges = np.linspace(0.25, 4.0, 13)
    lo, hi = edges[:-1], edges[1:]
    blo, bhi = second_derivative_enclosure(expr, t, {}, lo, hi)
    for i in range(lo.size):
        slo, shi = second_derivative_enclosure(expr, t, {}, lo[i : i + 1], hi[i : i + 1])
        assert slo[0] == blo[i] and shi[0] == bhi[i]
    assert len(set(blo.tolist())) == lo.size


def test_a_lane_undefined_on_its_piece_is_nan_and_the_others_survive():
    m = dm.Model("pole")
    t = m.continuous("t", lb=-1.0, ub=3.0)
    lo = np.array([-1.0, -0.5, 1.0, 2.0])
    hi = np.array([-0.5, 0.5, 2.0, 3.0])
    for expr in (1 / t, dm.log(t), t**0.5):
        enc_lo, enc_hi = second_derivative_enclosure(expr, t, {}, lo, hi)
        assert np.isnan(enc_lo[1]) and np.isnan(enc_hi[1])
        assert np.all(np.isfinite(enc_lo[2:])) and np.all(enc_hi[2:] < 0 + 1e300)
    lo1, _ = second_derivative_enclosure(1 / t, t, {}, lo, hi)
    assert np.isfinite(lo1[0])  # 1/t is defined on [-1, -0.5]


def test_atoms_outside_the_table_abstain():
    m = dm.Model("abs")
    t = m.continuous("t", lb=-1.0, ub=1.0)
    with pytest.raises(JetAbstain):
        interval_jet(abs(t), t, {}, 3)
    with pytest.raises(JetAbstain):
        interval_jet(dm.tan(t), t, {}, 3)


# ── the issue's rows ──────────────────────────────────────────────────────


@pytest.mark.parametrize("f, coef", ROWS, ids=["c", "v"])
def test_pfr_rows_certify_only_with_the_flag(f, coef, monkeypatch):
    assert _numerically_convex(f)
    m, _ = _row_model(f, coef)
    body = m._constraints[0].body
    assert certify_convex(body, m) is None  # premise: whole box abstains
    monkeypatch.setenv(FLAG, "0")
    assert certify_convex_subdivided(body, m) is None  # premise of #2062
    assert classify_oa_cut_convexity(m).constraint_mask == [False]

    monkeypatch.setenv(FLAG, "1")
    assert certify_convex_subdivided(body, m) == Curvature.CONVEX
    on = classify_oa_cut_convexity(m)
    assert on.constraint_mask == [True] and on.subdivided == 1


@pytest.mark.parametrize("f, coef", ROWS, ids=["c", "v"])
def test_negated_pfr_row_is_concave(f, coef, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m = dm.Model("neg")
    th = m.continuous("theta", lb=TH[0], ub=TH[1])
    w = m.continuous("W", lb=-1e4, ub=1e4)
    m.subject_to(-coef * f(th, dm) - w <= 0)
    assert certify_convex_subdivided(m._constraints[0].body, m) == Curvature.CONCAVE
    assert classify_oa_cut_convexity(m).constraint_mask == [False]


@pytest.mark.parametrize(
    "f, lb, ub",
    [
        (lambda t, ops: t**3, -1.0, 1.0),
        (lambda t, ops: 1 / ops.exp(ops.exp(t)), -2.0, 2.0),
        (lambda t, ops: ops.sin(t), 0.5, 5.5),
        # Convex on all but a sliver near the right end.
        (lambda t, ops: t**4 - 0.05 * ops.exp(3 * t), -1.0, 1.32),
    ],
    ids=["cube", "exp_neg_exp", "sin", "sliver"],
)
def test_nonconvex_rows_are_never_certified(f, lb, ub, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert not _numerically_convex(f, lb, ub)
    m, _ = _row_model(f, 50.0, lb, ub)
    assert certify_convex_subdivided(m._constraints[0].body, m, max_pieces=4096) != Curvature.CONVEX
    assert classify_oa_cut_convexity(m).constraint_mask == [False]


def test_budget_and_deadline_still_abstain(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m, _ = _row_model(c_pfr)
    body = m._constraints[0].body
    assert certify_convex_subdivided(body, m, max_pieces=4) is None
    assert certify_convex_subdivided(body, m, deadline=0.0) is None


def test_bivariate_rows_keep_the_interval_hessian_route(monkeypatch):
    """The Taylor form is univariate; a two-variable curvature support is #2054's."""
    monkeypatch.setenv(FLAG, "1")
    m = dm.Model("wide")
    xs = [m.continuous(f"x{i}", lb=0.25, ub=4.0) for i in range(2)]
    m.subject_to(sum((1 - KAP) / (dm.exp((1 - KAP) * x) - 1) for x in xs) <= 10)
    body = m._constraints[0].body
    assert certify_convex_subdivided(body, m, max_dims=2, max_pieces=1024) == Curvature.CONVEX
