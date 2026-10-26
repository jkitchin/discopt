"""#1982: presolve FBBT's sign / domain backward rules (``DISCOPT_FBBT_SIGN_DOMAIN``).

Two deductions SCIP's presolve makes and discopt's FBBT did not (found by #1975's
census of root LPs with a negative-cost recession ray):

* a product whose factor interval *touches* zero: ``x * y == 1``, ``y >= 0`` forces
  ``x >= 0`` (the in-tree propagator's ``reverse_div``, now shared);
* a non-integer power: ``y == x ** 0.75`` is real-valued only on ``x >= 0``.

``wallfix`` is the probe: its root LP is unbounded along ``objvar -> -inf`` because
``objvar * x2 == 1`` with ``x2 >= 0`` left ``objvar`` free.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import mccormick_lp as ML
from discopt.tightening import fbbt_box

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_FBBT_SIGN_DOMAIN"
WALLFIX = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "wallfix.nl")
WALLFIX_OPT = 1.0000046580  # MINLPLib =opt=


def _box(m):
    bt = fbbt_box(m)
    return np.asarray(bt.lb, float), np.asarray(bt.ub, float)


def _product_model():
    m = dm.Model("prod")
    x = m.continuous("x")
    y = m.continuous("y", lb=0.0)
    m.subject_to(x * y == 1)
    m.minimize(x)
    return m


def _power_model():
    m = dm.Model("pow")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(y == x**0.75)
    m.minimize(y)
    return m


@pytest.mark.parametrize("build", [_product_model, _power_model], ids=["product", "power"])
def test_flag_off_is_the_pre_1982_control(monkeypatch, build):
    monkeypatch.setenv(FLAG, "0")
    lb, _ = _box(build())
    assert lb[0] < 0, f"OFF must keep x's negative side: lb={lb[0]}"


def test_zero_touching_product_factor_fixes_the_sign(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    lb, ub = _box(_product_model())
    # x > 0, and the unbounded partner is read as infinite: no 1/9.999e19 bound.
    assert lb[0] == 0.0, lb
    assert ub[0] >= 9.999e19


def test_non_integer_power_base_is_nonnegative(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    lb, _ = _box(_power_model())
    assert lb[0] == 0.0, lb


def test_integer_power_keeps_its_negative_branch(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m = dm.Model("cube")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-200.0, ub=200.0)
    m.subject_to(y == x**3)
    m.minimize(y)
    lb, _ = _box(m)
    assert lb[0] == -5.0, lb


def _wallfix_root(monkeypatch, arm: str):
    monkeypatch.setenv(FLAG, arm)
    m = dm.from_nl(WALLFIX)
    lb, ub = _box(m)
    return lb, ub, ML.MccormickLPRelaxer(m).solve_at_node(lb, ub)


def test_wallfix_root_lp_is_unbounded_without_the_rule(monkeypatch):
    lb, _, res = _wallfix_root(monkeypatch, "0")
    assert lb[0] < -1e19, "control: objvar must stay free OFF"
    assert res.lower_bound is None and res.status == "unbounded", res


def test_wallfix_root_lp_is_bounded_and_sound_with_the_rule(monkeypatch):
    lb, _, res = _wallfix_root(monkeypatch, "1")
    assert lb[0] == 0.0, lb[0]
    assert res.status == "optimal" and res.lower_bound is not None, res
    # A valid relaxation bound never exceeds the true optimum.
    assert res.lower_bound <= WALLFIX_OPT + 1e-9, res.lower_bound


def test_wallfix_optimum_survives_the_tightened_box(monkeypatch):
    """Feasible-point check: MINLPLib's optimum stays inside the ON box."""
    lb, ub, _ = _wallfix_root(monkeypatch, "1")
    monkeypatch.setenv(FLAG, "1")
    r = dm.from_nl(WALLFIX).solve(time_limit=3)
    assert r.x is not None and r.objective == pytest.approx(WALLFIX_OPT, rel=1e-5), r.objective
    m = dm.from_nl(WALLFIX)
    flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
    assert flat.shape == lb.shape
    assert np.all(lb <= flat) and np.all(flat <= ub), (lb, flat, ub)
