"""#1596: the convex fast path must not certify a bound above the true optimum.

The continuous convex routes (the QP route and the single-NLP convex route) used
to publish ``bound := objective`` with ``gap_certified=True`` on the strength of
the backend's ``optimal`` and relative KKT residuals. An interior-point backend
stops with the box complementarity unconverged at a level its scaled test
accepts: ``min (x - 1e6)**2 + y`` with ``y >= 0.01`` came back at
``y = 0.01005`` and was certified at 0.01005 against a true optimum of 0.01.

The bound is now a dual bound that charges the complementarity (the rigorous
tangent bound of the Lagrangian, Newton-refined support points, or the
Lagrangian value at the backend's multipliers with the full stationarity
residual checked), and the certificate is withheld when none closes the gap.
Every test here runs with ``DISCOPT_RECENTRE=0`` so that recentring (#1594)
cannot mask the witness by moving it to a well-scaled origin.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _no_recentre(monkeypatch):
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")


def _assert_sound(r, opt: float, maximize: bool = False, tol: float = 1e-6) -> None:
    """Certified => the bound is on the right side of the true optimum."""
    if r.gap_certified:
        assert r.bound is not None
        if maximize:
            assert r.bound >= opt - tol, (r.status, r.objective, r.bound, opt)
        else:
            assert r.bound <= opt + tol, (r.status, r.objective, r.bound, opt)
    if r.objective is not None and r.status in ("optimal", "feasible"):
        # The incumbent is a feasible point, so it can never beat the optimum.
        if maximize:
            assert r.objective <= opt + tol
        else:
            assert r.objective >= opt - tol


def _offset_witness(k: float, c: float):
    m = dm.Model("w1596")
    x = m.continuous("x", lb=0.0, ub=2 * c)
    y = m.continuous("y", lb=1e-2, ub=1e6)
    m.minimize(k * (x - c) ** 2 + y)
    return m


@pytest.mark.parametrize("k", [1.0, 1e8])
def test_issue_witnesses_are_not_falsely_certified(k):
    """The two witnesses from the issue (on main: certified bound 0.01005)."""
    r = _offset_witness(k, 1e6).solve(time_limit=60)
    _assert_sound(r, 0.01)
    # The honest outcome here is a certified optimum: the dual bound closes.
    assert r.status == "optimal" and r.gap_certified
    assert r.bound == pytest.approx(0.01, abs=1e-6)


@pytest.mark.parametrize("k", [1.0, 1e4])
@pytest.mark.parametrize("c", [1e3, 1e5, 1e7, 1000001.37])
def test_offset_grid(k, c):
    """The class, not the instance: offsets across magnitudes, including an
    offset that is not on a power of two."""
    r = _offset_witness(k, c).solve(time_limit=60)
    _assert_sound(r, 0.01)


def test_nlp_route_quartic():
    m = dm.Model("quartic")
    x = m.continuous("x", lb=0.0, ub=2e6)
    y = m.continuous("y", lb=1e-2, ub=1e6)
    m.minimize((x - 1e6) ** 4 + y)
    _assert_sound(m.solve(time_limit=60), 0.01)


def test_nlp_route_square_plus_exp():
    m = dm.Model("sqexp")
    x = m.continuous("x", lb=0.0, ub=2e6)
    y = m.continuous("y", lb=1e-2, ub=1e6)
    m.minimize((x - 1e6) ** 2 + dm.exp(y))
    _assert_sound(m.solve(time_limit=60), math.exp(0.01))


@pytest.mark.parametrize("c", [1e3, 1e6])
def test_qp_route_with_a_row(c):
    """A row the IPM props up with a spurious multiplier: certified only if right."""
    m = dm.Model("qpcon")
    x = m.continuous("x", lb=0.0, ub=2 * c)
    y = m.continuous("y", lb=1e-2, ub=1e6)
    m.subject_to(x + y >= c)
    m.minimize((x - c) ** 2 + y)
    _assert_sound(m.solve(time_limit=60), 0.01)


def test_maximize_sense():
    m = dm.Model("max")
    x = m.continuous("x", lb=0.0, ub=2e6)
    y = m.continuous("y", lb=1e-2, ub=1e6)
    m.maximize(-((x - 1e6) ** 2) - y)
    r = m.solve(time_limit=60)
    _assert_sound(r, -0.01, maximize=True)


@pytest.mark.parametrize("s", [1.0, 1e5, 1e8])
def test_interior_optimum_still_certifies_at_every_scale(s):
    """The tangent bound at an IPM point is loose by (residual slope) x (box width);
    the Newton-refined support point keeps an interior optimum certified."""
    m = dm.Model("box")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(s * (x - 3) ** 2)
    r = m.solve(time_limit=30)
    _assert_sound(r, 0.0)
    assert r.status == "optimal" and r.gap_certified


def test_no_witness_certificate_charges_box_complementarity():
    """Unit test of the certificate itself on the witness geometry: at a point
    with ``y`` above its bound by 5e-5 (so the box multiplier ``z_y = 1`` leaves a
    complementarity gap of 5e-5), the objective is not a bound; the certificate's
    bound must sit at or below the true optimum."""
    from discopt.solver import _convex_nlp_certificate, _make_evaluator

    m = _offset_witness(1.0, 1e6)
    ev = _make_evaluator(m)
    x = np.array([1e6, 0.01005])
    lb = np.array([0.0, 0.01])
    ub = np.array([2e6, 1e6])
    empty = np.zeros(0)
    cert = _convex_nlp_certificate(
        ev,
        x,
        empty,
        lb,
        ub,
        empty,
        empty,
        float(ev.evaluate_objective(x)),
        gap_tolerance=1e-4,
    )
    # Withheld (``None`` / ``bound is None``) is honest; a bound must be valid.
    assert cert is None or cert.bound is None or cert.bound <= 0.01 + 1e-12, cert
