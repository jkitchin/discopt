"""Unit tests for the LR-2 H-LOG log-space product envelope math.

Task ``cert:LR-2``. The H-LOG lift bounds a positive product ``t = coef·∏xᵢ^{aᵢ}``
via ``zᵢ = ln xᵢ`` (concave), ``s = Σaᵢzᵢ`` (linear), ``t = coef·exp(s)`` (convex).
These tests assert the emitted ln/exp envelope *lines* (as built in
``milp_relaxation``) never cut a feasible point, for both signs of ``coef`` — the
soundness property the additive rows must satisfy.
"""

from __future__ import annotations

import math

import numpy as np
from discopt._jax.milp_relaxation import _tangent_points, _univariate_grad, _univariate_value


def test_ln_concave_envelope_lines_are_sound():
    lo, hi = 3.0, 9.0
    rng = np.random.default_rng(0)
    worst = -1e9
    for x in np.concatenate([[lo, hi], rng.uniform(lo, hi, 20000)]):
        z = math.log(x)
        # tangents overestimate: z <= ln(a) + (x-a)/a
        for a in _tangent_points("log", lo, hi):
            slope = _univariate_grad("log", a)
            tangent = _univariate_value("log", a) + slope * (x - a)
            worst = max(worst, z - tangent)  # want <= 0
        # secant underestimates: z >= secant
        m = (math.log(hi) - math.log(lo)) / (hi - lo)
        secant = math.log(lo) + m * (x - lo)
        worst = max(worst, secant - z)
    assert worst <= 1e-9, f"ln envelope cut a feasible point by {worst}"


def test_exp_convex_envelope_lines_are_sound():
    slo, shi = 3.0, 6.0
    rng = np.random.default_rng(1)
    worst = -1e9
    for s in np.concatenate([[slo, shi], rng.uniform(slo, shi, 20000)]):
        t = math.exp(s)
        # tangents underestimate: exp(s) >= exp(a) + exp(a)(s-a)
        for a in _tangent_points("exp", slo, shi):
            e = _univariate_value("exp", a)
            tangent = e + e * (s - a)
            worst = max(worst, tangent - t)  # want <= 0
        # secant overestimates
        E_lo = math.exp(slo)
        m = (math.exp(shi) - math.exp(slo)) / (shi - slo)
        secant = E_lo + m * (s - slo)
        worst = max(worst, t - secant)
    assert worst <= 1e-9, f"exp envelope cut a feasible point by {worst}"


def _hlog_exp_rows_sound(coef: float, slo: float, shi: float) -> float:
    """Replicate the emitted H-LOG exp rows and return the worst feasible-cut."""
    rng = np.random.default_rng(2)
    worst = -1e9
    tps = _tangent_points("exp", slo, shi)
    E_lo = math.exp(slo)
    M = (math.exp(shi) - math.exp(slo)) / (shi - slo)
    for s in np.concatenate([[slo, shi], rng.uniform(slo, shi, 20000)]):
        t = coef * math.exp(s)  # a feasible (s, t) point
        for a in tps:
            e = math.exp(a)
            rhs = coef * (e - e * a)
            lhs = t - coef * e * s
            viol = (rhs - lhs) if coef >= 0.0 else (lhs - rhs)
            worst = max(worst, viol)
        rhs = coef * (E_lo - M * slo)
        lhs = t - coef * M * s
        viol = (lhs - rhs) if coef >= 0.0 else (rhs - lhs)
        worst = max(worst, viol)
    return worst


def test_hlog_product_envelope_sound_positive_coef():
    assert _hlog_exp_rows_sound(1.0, 3.0, 6.0) <= 1e-7


def test_hlog_product_envelope_sound_negative_coef():
    # nvs09's product enters the objective with -1; the sense must flip correctly.
    assert _hlog_exp_rows_sound(-1.0, 3.0, 6.0) <= 1e-7
