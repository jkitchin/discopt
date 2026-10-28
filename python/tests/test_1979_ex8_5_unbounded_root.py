"""#1979: ``ex8_5_*`` had no dual bound -- four general deductions give it one.

The six phase-equilibrium NLPs (``x_i log x_i`` Gibbs terms, a cubic equation of state
in the compressibility ``Z``, ``x/(Z - x)`` and ``log(Z - x)``) returned
``bound=None``: every column was free, and the relaxation had no row on the
``x log x``, reciprocal and ``log`` auxiliaries. Each piece below is general and
gated; together they bound all six (measured: ``None`` -> finite, every bound below
MINLPLib's best known value):

* ``DISCOPT_FBBT_SIGN_DOMAIN`` -- the affine domain rows of ``log`` / ``sqrt`` /
  non-integer powers (``log(Z - x) => Z - x >= 0``) for the relaxation and OBBT, and
  the structural FBBT domain pass (``fbbt.rs`` ``impose_function_domains``);
* ``DISCOPT_POLE_ENVELOPE`` -- a 1-D envelope keeps its tangent half at a closed-edge
  pole, and a negative power on a base touching 0 gets its one-sided enclosure;
* ``DISCOPT_POLY_ROOT_BOUND`` -- Cauchy's root bound for an equality polynomial in one
  variable (the cubic in ``Z``);
* ``DISCOPT_ENTROPY_ATOM`` -- the ``x log x`` atom now fires on ``[0, hi]`` too.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.function_domains import affine_domain_rows
from discopt._relax.nonlinear_bound_tightening import (
    UnivariatePolynomialRootBoundRule,
    build_flat_variable_metadata,
)
from discopt._relax.uniform_relax import _neg_pow_touching_zero
from discopt.tightening import fbbt_box

EX8_5_3 = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "ex8_5_3.nl")
EX8_5_3_BEST = -0.0041354299  # MINLPLib =best=
BUNDLE = (
    "DISCOPT_FBBT_SIGN_DOMAIN",
    "DISCOPT_POLE_ENVELOPE",
    "DISCOPT_POLY_ROOT_BOUND",
    "DISCOPT_ENTROPY_ATOM",
)


def _cubic_model():
    """``z^3 - z^2 (1 + b) + a z - a b == 0`` with ``a in [0, 3]``, ``b in [0, 0.5]``."""
    m = dm.Model("cubic_eos")
    z = m.continuous("z", lb=0.0)
    a = m.continuous("a", lb=0.0, ub=3.0)
    b = m.continuous("b", lb=0.0, ub=0.5)
    m.subject_to(z**3 - z**2 * (1 + b) + a * z - a * b == 0)
    m.minimize(z)
    return m


@pytest.mark.unit
def test_poly_root_bound_is_cauchys_bound_and_off_by_default(monkeypatch):
    m = _cubic_model()
    lb, ub = np.array([0.0, 0.0, 0.0]), np.array([np.inf, 3.0, 0.5])
    meta = build_flat_variable_metadata(m)
    monkeypatch.setenv("DISCOPT_POLY_ROOT_BOUND", "0")
    _, u0 = UnivariatePolynomialRootBoundRule().tighten(m, lb, ub, meta)
    assert u0[0] == np.inf, "control: OFF leaves z open"
    monkeypatch.setenv("DISCOPT_POLY_ROOT_BOUND", "1")
    _, u1 = UnivariatePolynomialRootBoundRule().tighten(m, lb, ub, meta)
    # max |a_k| / |a_3| over the box = max(1.5, 3, 1.5) = 3  ->  |z| <= 4.
    assert 4.0 <= u1[0] <= 4.0 + 1e-12, u1[0]
    # Sound: sample the cubic's real roots over the coefficient box; all lie inside.
    checked = 0
    for av in np.linspace(0.0, 3.0, 7):
        for bv in np.linspace(0.0, 0.5, 6):
            for r in np.roots([1.0, -(1.0 + bv), av, -av * bv]):
                if abs(r.imag) < 1e-9:
                    assert abs(r.real) <= u1[0] + 1e-9, (av, bv, r)
                    checked += 1
    assert checked > 0


@pytest.mark.unit
def test_poly_root_bound_declines_a_vanishing_leading_coefficient(monkeypatch):
    monkeypatch.setenv("DISCOPT_POLY_ROOT_BOUND", "1")
    m = dm.Model("vanishing")
    z = m.continuous("z")
    a = m.continuous("a", lb=-1.0, ub=1.0)
    m.subject_to(a * z**2 + z - 1 == 0)  # a may be 0: the equation can be linear
    m.minimize(z)
    lb, ub = np.array([-np.inf, -1.0]), np.array([np.inf, 1.0])
    nl, nu = UnivariatePolynomialRootBoundRule().tighten(m, lb, ub, build_flat_variable_metadata(m))
    assert nl[0] == -np.inf and nu[0] == np.inf


@pytest.mark.unit
def test_affine_domain_rows_of_a_log_argument(monkeypatch):
    m = dm.Model("logdiff")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-dm.log(x - y) + x * dm.log(x))  # log(x) is single-variable: FBBT's job
    rows = affine_domain_rows(m)
    assert rows == [({0: 1.0, 1: -1.0}, 0.0, 0.0)], rows


@pytest.mark.unit
def test_negative_power_touching_zero_is_one_sided():
    # 1/t on [0, 4] is [1/4, inf); on [-2, 0]: 1/t is (-inf, -1/2], 1/t^2 is [1/4, inf).
    lo, hi = _neg_pow_touching_zero(0.0, 4.0, -1.0)
    assert hi == math.inf and lo <= 0.25 and lo > 0.2499999
    lo, hi = _neg_pow_touching_zero(-2.0, 0.0, -1.0)
    assert lo == -math.inf and hi >= -0.5 and hi < -0.4999999
    lo, hi = _neg_pow_touching_zero(-2.0, 0.0, -2.0)
    assert hi == math.inf and lo <= 0.25
    assert _neg_pow_touching_zero(-1.0, 2.0, -1.0) is None  # straddles: no claim
    assert _neg_pow_touching_zero(-2.0, 0.0, -0.5) is None  # no real value there


@pytest.mark.unit
def test_fbbt_domain_pass_reaches_log_inside_a_product(monkeypatch):
    """``x * log(x)`` with ``x`` free: the backward walk never divides the product,
    so only the structural domain pass gives ``x >= 0``."""
    m = dm.Model("xlogx")
    x = m.continuous("x")
    y = m.continuous("y")
    m.subject_to(x * dm.log(x) - y == 0)
    m.minimize(y)
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", "0")
    assert fbbt_box(m).lb[0] < -1e19
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", "1")
    assert fbbt_box(m).lb[0] == 0.0


def _solve_ex8_5_3(monkeypatch, on: bool, **budget):
    for f in BUNDLE:
        monkeypatch.setenv(f, "1" if on else "0")
    return dm.from_nl(EX8_5_3).solve(**(budget or {"time_limit": 12}))


@pytest.mark.slow
@pytest.mark.correctness
def test_ex8_5_3_has_no_bound_without_the_bundle(monkeypatch):
    r = _solve_ex8_5_3(monkeypatch, on=False)
    assert r.bound is None, r.bound


@pytest.mark.slow
@pytest.mark.correctness
def test_ex8_5_3_gets_a_sound_bound_with_the_bundle(monkeypatch):
    # A node budget, not a wall-clock one: the tightness bar needs ~200 nodes, and a
    # 12 s budget reached 217 on a laptop but 61 on a CI runner (bound -0.134, a
    # test of the runner). Measured 2026-10-28: certifies optimal at 219 nodes,
    # bound -0.0041359, ~15 s locally, identical over 5 runs (2 deterministic).
    r = _solve_ex8_5_3(monkeypatch, on=True, max_nodes=300, time_limit=300)
    assert r.bound is not None, "the bundle produced no dual bound"
    assert r.bound <= EX8_5_3_BEST + 1e-9, (r.bound, EX8_5_3_BEST)
    assert r.bound > -0.01, (r.bound, r.node_count)
