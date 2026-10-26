"""#1982: presolve FBBT's sign / domain backward rules (``DISCOPT_FBBT_SIGN_DOMAIN``).

#1975's census asked for two deductions SCIP's presolve makes:

* a non-integer power: ``y == x ** 0.75`` is real-valued only on ``x >= 0`` -- shipped;
* a product whose factor interval *touches* zero (``x * y == 1``, ``y >= 0`` forces
  ``x >= 0``) -- **declined**: it is the retired ``DISCOPT_FBBT_ONE_SIDED_DIV``
  (#1616; over its 29 firing MINLPLib instances it lost the gancns and waternd1
  certificates and cost +11 % wall). Built again here, its only measured gain was
  ``wallfix``'s root bound, which does not answer the audit row's reopen condition.
  These tests pin that the sign/domain flag does not revive it, so ``wallfix``'s
  root LP stays unbounded in both arms.
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


def test_zero_touching_product_factor_is_not_divided(monkeypatch):
    """The retired #1616 rule stays retired: ON leaves the product box as OFF does."""
    boxes = []
    for arm in ("0", "1"):
        monkeypatch.setenv(FLAG, arm)
        boxes.append(_box(_product_model()))
    assert np.array_equal(boxes[0][0], boxes[1][0]), boxes
    assert np.array_equal(boxes[0][1], boxes[1][1]), boxes
    assert boxes[1][0][0] < -1e19


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


@pytest.mark.parametrize("arm", ["0", "1"])
def test_wallfix_root_lp_stays_unbounded(monkeypatch, arm):
    """Only the declined product rule bounds wallfix's root; neither arm has it."""
    lb, _, res = _wallfix_root(monkeypatch, arm)
    assert lb[0] < -1e19, "objvar must stay free: the #1616 rule is retired"
    assert res.lower_bound is None and res.status == "unbounded", res
