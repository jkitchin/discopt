"""#1981: root LPs unbounded along lifted columns only (hhfair, launch, like).

* ``launch`` -- a posynomial objective. ``x**-0.1959`` over ``[5, inf)`` was enclosed
  as ``[-5e-324, ...]``: the outward-rounded ``exp(e*log t)`` stepped below 0, so the
  factor read as sign-indefinite, its product with a ``[214, inf)`` factor enclosed
  ``(-inf, inf)``, and the product had no row. A power of a strictly positive base is
  positive, exactly; clamping it (``DISCOPT_POLE_ENVELOPE``) gives the root a bound
  and the solve certifies (measured: 5 nodes, 11 s; was uncertified at 30 s).
* ``hhfair`` -- ``x15 * (x12 - 0.2559 x5) == 1``: the factor's BOX interval straddles
  0, but the model's linear row ``x12 >= 0.2585 x5`` with ``x5 >= 100`` keeps it at
  ``>= 0.26``. The bilinear-equality rule now takes the factor's range by an LP over
  the linear rows when the box cannot decide (``DISCOPT_FBBT_SIGN_DOMAIN``), and
  reads an indexed sum (how presolve writes the factor) as affine. Bound: None ->
  -109.68 (best known -87.159).
* ``like`` -- a Gaussian-mixture likelihood with ``sigma >= 0``. It is unbounded below:
  a component centred on a data point with ``sigma -> 0`` drives the objective down by
  ``ln 10`` per decade of ``sigma``. No relaxation may bound it; nothing to fix.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import nonlinear_bound_tightening as N
from discopt._relax.uniform_relax import _pow_bounds

LAUNCH = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "launch.nl")
LAUNCH_OPT = 2257.7975580  # MINLPLib =opt=


@pytest.mark.unit
def test_power_of_a_positive_base_is_not_enclosed_below_zero(monkeypatch):
    monkeypatch.setenv("DISCOPT_POLE_ENVELOPE", "0")
    lo_off, _ = _pow_bounds(5.0, np.inf, -0.1959)
    monkeypatch.setenv("DISCOPT_POLE_ENVELOPE", "1")
    lo_on, hi_on = _pow_bounds(5.0, np.inf, -0.1959)
    assert lo_off < 0.0, "control: the rounded enclosure dips below 0 off"
    assert lo_on == 0.0 and hi_on >= 5.0**-0.1959


def _hhfair_like_row():
    """``x * (y - 0.2559 z) == 1`` with ``y >= 0.2585 z``, ``z in [100, 200]``."""
    m = dm.Model("lp_sign")
    x = m.continuous("x")
    y = m.continuous("y", lb=0.0, ub=1e4)
    z = m.continuous("z", lb=100.0, ub=200.0)
    m.subject_to(x * (y - 0.255905 * z) == 1)
    m.subject_to(y - 0.25846405 * z >= 0)
    m.minimize(x)
    return m


@pytest.mark.unit
def test_bilinear_equality_takes_the_factor_sign_from_the_linear_rows(monkeypatch):
    m = _hhfair_like_row()
    lb, ub = np.array([-np.inf, 0.0, 100.0]), np.array([np.inf, 1e4, 200.0])
    meta = N.build_flat_variable_metadata(m)
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", "0")
    nl, nu = N.BilinearProductEqualityRule().tighten(m, lb, ub, meta)
    assert nl[0] == -np.inf and nu[0] == np.inf, "control: the box straddles 0, no bound"
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", "1")
    nl, nu = N.BilinearProductEqualityRule().tighten(m, lb, ub, meta)
    # F = y - 0.255905 z >= (0.25846405 - 0.255905) * 100 = 0.2559: x in (0, 1/0.2559].
    assert 0.0 < nl[0] and nu[0] <= 1.0 / 0.2559 * (1 + 1e-6), (nl[0], nu[0])
    # Sound: feasible points (y on the row, z sampled) keep x inside the bound.
    for zv in np.linspace(100.0, 200.0, 11):
        for yv in (0.25846405 * zv, 0.3 * zv, 1e3):
            xv = 1.0 / (yv - 0.255905 * zv)
            assert nl[0] <= xv <= nu[0], (zv, yv, xv)


@pytest.mark.unit
def test_indexed_sum_reads_as_affine_only_under_the_flag(monkeypatch):
    m = dm.Model("sumover")
    a = m.continuous("a", lb=0, ub=1)
    b = m.continuous("b", lb=0, ub=1)
    from discopt.modeling.core import SumOverExpression

    s = SumOverExpression([a, 2 * b])
    meta = N.build_flat_variable_metadata(m)
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", "0")
    assert not N._accumulate_affine(s, 1.0, meta, {}, [0.0])
    monkeypatch.setenv("DISCOPT_FBBT_SIGN_DOMAIN", "1")
    coeffs: dict = {}
    assert N._accumulate_affine(s, 1.0, meta, coeffs, [0.0])
    assert coeffs == {0: 1.0, 1: 2.0}


@pytest.mark.slow
@pytest.mark.correctness
def test_launch_certifies_with_the_envelope_repairs(monkeypatch):
    for f in (
        "DISCOPT_FBBT_SIGN_DOMAIN",
        "DISCOPT_POLE_ENVELOPE",
        "DISCOPT_POLY_ROOT_BOUND",
        "DISCOPT_ENTROPY_ATOM",
    ):
        monkeypatch.setenv(f, "1")
    r = dm.from_nl(LAUNCH).solve(time_limit=60)
    assert r.root_bound is not None, "the root LP is still unbounded"
    assert r.bound <= LAUNCH_OPT * (1 + 1e-6), (r.bound, LAUNCH_OPT)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound, r.objective)
    assert r.objective == pytest.approx(LAUNCH_OPT, rel=1e-6)
