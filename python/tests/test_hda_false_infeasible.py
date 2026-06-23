"""Regression for the hda false-infeasible (factorable-reform bound bugs).

`hda` (MINLPLib, minimization, known optimum -5964.53) was returned as
``infeasible``. Two distinct factorable-reformulation bugs combined to seed a
nonlinear-bound-tightening box that read as empty:

1. ``_Lifter.fractional_power`` computed the induced aux box as the *unordered*
   ``(lo**p, hi**p)``. For a NEGATIVE power ``base**p`` is decreasing, so
   ``lo**p > hi**p`` and the aux was created with ``lb > ub`` (e.g.
   ``p = -1.5`` over ``base in [1e-4, 0.83]`` gave ``[1e6, 1.31]``).
2. The affine FBBT rules certified a hard infeasibility on a ``new_lb > new_ub``
   crossover even when it was below the feasibility tolerance — a floating-point
   degenerate point (``[68930.0596…73, 68930.0596…72]``, ~1e-11 wide).

Both are fixed: fractional_power orders the endpoints, and the affine rules snap
a sub-tolerance crossover to a degenerate point instead of pruning.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


def test_fractional_power_negative_exponent_orders_bounds():
    """A negative fractional power induces a *decreasing* map, so the aux box must
    be ordered ``[min(lo**p, hi**p), max(...)]`` — never the inverted raw pair."""
    from discopt._jax.factorable_reform import _Lifter

    m = dm.Model("frac")
    base = m.continuous("base", lb=1e-4, ub=0.8334)
    d = _Lifter(m).fractional_power(base, -1.5)
    assert d is not None
    lb = float(np.asarray(d.lb).reshape(-1)[0])
    ub = float(np.asarray(d.ub).reshape(-1)[0])
    assert lb <= ub, f"aux box inverted: [{lb}, {ub}]"
    assert lb == pytest.approx(0.8334**-1.5, rel=1e-9)  # hi**p is the minimum
    assert ub == pytest.approx((1e-4) ** -1.5, rel=1e-9)  # lo**p is the maximum


def test_fractional_power_positive_exponent_still_correct():
    """The common ``p > 0`` (increasing) case is unchanged: ``[lo**p, hi**p]``."""
    from discopt._jax.factorable_reform import _Lifter

    m = dm.Model("frac_pos")
    base = m.continuous("base", lb=1.0, ub=2.667)
    d = _Lifter(m).fractional_power(base, 4.3333333)
    assert d is not None
    lb = float(np.asarray(d.lb).reshape(-1)[0])
    ub = float(np.asarray(d.ub).reshape(-1)[0])
    assert lb <= ub
    assert lb == pytest.approx(1.0**4.3333333, rel=1e-9)
    assert ub == pytest.approx(2.667**4.3333333, rel=1e-9)


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_hda_not_false_infeasible():
    """hda (known optimum -5964.53) must never be returned as ``infeasible``."""
    path = os.path.join(_DATA, "hda.nl")
    if not os.path.exists(path):
        pytest.skip("hda instance unavailable")
    r = dm.from_nl(path).solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status != "infeasible", "hda wrongly certified infeasible (known opt -5964.53)"
    # A dual bound, when reported, must be a valid lower bound (<= the optimum).
    if r.bound is not None:
        assert r.bound <= -5964.534 + 1e-2 * (1 + 5964.534)
