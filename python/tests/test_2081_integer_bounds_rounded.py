"""#2081: a binary's tightened bound was left non-integral and later inverted.

On a small MINLP with two ``m.complementarity`` pairs, per-node OBBT warned
``root OBBT envelope build (build_milp_relaxation) failed (ValueError: Interval
lo > hi: lo=1.0, hi=0.9999999999998366)`` and stopped early. The box it was
handed had a GDP selector binary at ``[0, 1 - 1.6e-13]`` and its complement at
``[1.6e-13, 1]``; OBBT rounded the tightened ``lb`` to 1 but not the incoming
``ub``, inverting the interval.

The non-integral bound came from the nonlinear tightening pass, whose
``_apply_integrality`` clipped a BINARY to ``[0, 1]`` but (unlike an INTEGER)
never rounded it. Three layers now round:

* ``nonlinear_bound_tightening._apply_integrality`` rounds binaries;
* the node FBBT (``solver._tighten_node_bounds_with_status``) rounds every
  integer column of the box it returns (the linear-row sweep is
  integrality-blind too), reporting a box with no integer as infeasible;
* ``obbt_tighten_root`` rounds the incoming integer bounds, not only the ones
  it tightens.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import discopt.solver as solver
import numpy as np
import pytest
from discopt._relax.nonlinear_bound_tightening import _apply_integrality
from discopt._relax.obbt import obbt_tighten_root
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.modeling.core import VarType


@pytest.mark.smoke
def test_apply_integrality_rounds_binaries():
    assert _apply_integrality(0.0, 1 - 1.6e-13, VarType.BINARY) == (0.0, 1.0)
    assert _apply_integrality(1.6e-13, 1.0, VarType.BINARY) == (0.0, 1.0)
    # A genuine fractional tightening fixes the binary.
    assert _apply_integrality(0.0, 0.5, VarType.BINARY) == (0.0, 0.0)
    assert _apply_integrality(0.4, 1.0, VarType.BINARY) == (1.0, 1.0)
    # Never outside [0, 1]; continuous columns untouched.
    assert _apply_integrality(-3.0, 7.0, VarType.BINARY) == (0.0, 1.0)
    assert _apply_integrality(0.25, 0.75, VarType.CONTINUOUS) == (0.25, 0.75)
    # Round-off inside the acceptance tolerance never cuts 0 or 1.
    assert _apply_integrality(0.0, 1 - 1e-7, VarType.BINARY) == (0.0, 1.0)


@pytest.mark.smoke
def test_apply_integrality_rounds_integers_with_the_acceptance_tolerance():
    # A rule's round-off within the acceptance tolerance (1e-5) never cuts an
    # integer value the acceptance test takes: ``ub = 3 - 5e-8`` used to floor
    # to 2 with the old ``1e-9`` tolerance and cut ``k = 3``.
    assert _apply_integrality(0.0, 3 - 5e-8, VarType.INTEGER) == (0.0, 3.0)
    assert _apply_integrality(2 + 5e-8, 9.0, VarType.INTEGER) == (2.0, 9.0)
    # A genuine fractional tightening still rounds inward.
    assert _apply_integrality(0.4, 3.6, VarType.INTEGER) == (1.0, 3.0)


@pytest.mark.smoke
def test_node_fbbt_returns_integral_integer_bounds():
    m = dm.Model("fbbt_round")
    y = m.binary("y")
    z = m.continuous("z", lb=1, ub=2)
    w = m.binary("w")
    k = m.integer("k", lb=0, ub=10)
    m.subject_to(0.1 * y + 0.2 * z <= 0.3)
    m.subject_to(2 * w <= 1)  # the linear row leaves w at [0, 0.5]
    m.subject_to(3 * k <= 10)  # and k at [0, 3.33]
    m.minimize(z**2 - y - w - k)
    ev = make_evaluator(m)
    cl, cu = solver._infer_constraint_bounds(m, ev)
    lb0 = np.array([0.0, 1.0, 0.0, 0.0])
    ub0 = np.array([1.0, 2.0, 1.0, 10.0])
    lb, ub, infeasible = solver._tighten_node_bounds_with_status(ev, lb0, ub0, cl, cu)
    assert infeasible is False
    is_int = np.array([True, False, True, True])
    assert np.all(lb[is_int] == np.round(lb[is_int]))
    assert np.all(ub[is_int] == np.round(ub[is_int]))
    assert ub[2] == 0.0 and ub[3] == 3.0
    assert ub[1] == pytest.approx(1.5)  # the continuous column is not rounded


@pytest.mark.smoke
def test_round_integer_bounds_reports_an_empty_integer_box():
    m = dm.Model("empty")
    m.binary("y")
    m.continuous("x", lb=0, ub=1)
    lb, ub, crossed = solver._round_integer_bounds(m, np.array([0.3, 0.2]), np.array([0.7, 0.8]))
    assert crossed
    lb, ub, crossed = solver._round_integer_bounds(
        m, np.array([1.6e-13, 0.2]), np.array([1 - 1.6e-13, 0.8])
    )
    assert not crossed and list(lb) == [0.0, 0.2] and list(ub) == [1.0, 0.8]


@pytest.mark.smoke
def test_obbt_rounds_incoming_integer_bounds(caplog):
    # A binary handed in at a non-integral bound must come back integral, with
    # no inverted-interval failure from the envelope build.
    m = dm.Model("obbt_in")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    b = m.binary("b")
    m.subject_to(x * y >= 1)
    m.subject_to(x <= 4 * b)
    m.minimize(x + y + b)
    caplog.set_level(logging.WARNING, logger="discopt")
    res = obbt_tighten_root(
        m, np.array([0.0, 0.0, 1.6e-13]), np.array([4.0, 4.0, 1 - 1.6e-13]), rounds=2
    )
    assert not any("Interval lo > hi" in r.getMessage() for r in caplog.records)
    assert not res.infeasible
    assert res.lb[2] == np.round(res.lb[2]) and res.ub[2] == np.round(res.ub[2])
    assert res.lb[2] <= res.ub[2]


@pytest.mark.smoke
def test_obbt_incoming_rounding_uses_the_acceptance_tolerance():
    # An incoming binary ub of ``1 - 5e-7`` is within the acceptance tolerance
    # of 1: with the pass's ``eps=1e-7`` it floored to 0 and cut ``b = 1``.
    m = dm.Model("obbt_in_tol")
    x = m.continuous("x", lb=0, ub=4)
    b = m.binary("b")
    m.subject_to(x * x >= 1)
    m.minimize(x + b)
    res = obbt_tighten_root(m, np.array([0.0, 0.0]), np.array([4.0, 1 - 5e-7]), rounds=1)
    assert not res.infeasible
    assert res.ub[1] == 1.0


@pytest.mark.smoke
def test_obbt_reports_a_binary_with_no_integer_as_infeasible():
    m = dm.Model("obbt_empty")
    x = m.continuous("x", lb=0, ub=4)
    b = m.binary("b")
    m.subject_to(x * x >= 1)
    m.minimize(x + b)
    res = obbt_tighten_root(m, np.array([0.0, 0.3]), np.array([4.0, 0.7]))
    assert res.infeasible


@pytest.mark.slow
@pytest.mark.correctness
def test_flash_heater_complementarity_no_inverted_interval(caplog):
    """The issue's reproducer (book Vol. II, Ch. II.24, exercise 3)."""
    ant_a = np.array([6.90565, 6.95464])
    ant_b = np.array([1211.033, 1344.800])
    ant_c = np.array([220.790, 219.482])
    p, zf = 760.0, np.array([0.5, 0.5])
    cp, dh, tf, ch = 0.15, 32.0, 355.0, 0.01

    m = dm.Model("flash_heater")
    t = m.continuous("T", lb=355, ub=400)
    liq, vap = m.continuous("L", lb=0, ub=1), m.continuous("V", lb=0, ub=1)
    x = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(2)]
    y = [m.continuous(f"y{i}", lb=0, ub=1) for i in range(2)]
    beta = m.continuous("beta", lb=0, ub=5)
    s_l, s_v = m.continuous("sL", lb=0, ub=5), m.continuous("sV", lb=0, ub=5)
    h = m.binary("h")
    for i in range(2):
        k = dm.exp(np.log(10) * (ant_a[i] - ant_b[i] / (t - 273.15 + ant_c[i]))) / p
        m.subject_to(zf[i] == liq * x[i] + vap * y[i])
        m.subject_to(y[i] == beta * k * x[i])
    m.subject_to(x[0] + x[1] == 1)
    m.subject_to(y[0] + y[1] == 1)
    m.subject_to(beta == 1 - s_l + s_v)
    m.subject_to(t - tf <= 45 * h)
    m.complementarity(liq, s_l)
    m.complementarity(vap, s_v)
    m.maximize(0.68 * vap * y[0] - ch * (cp * (t - tf) + dh * vap) - 0.03 * h)

    caplog.set_level(logging.WARNING, logger="discopt")
    r = m.solve(gap_tolerance=1e-3)
    inverted = [rec for rec in caplog.records if "Interval lo > hi" in rec.getMessage()]
    assert inverted == []
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(0.0, abs=1e-6)
