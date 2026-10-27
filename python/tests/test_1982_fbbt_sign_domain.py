"""#1982: presolve FBBT's sign / domain backward rules (``DISCOPT_FBBT_SIGN_DOMAIN``).

#1975's census asked for two deductions SCIP's presolve makes:

* a non-integer power: ``y == x ** 0.75`` is real-valued only on ``x >= 0`` -- shipped;
* a product whose factor interval *touches* zero (``x * y == 1``, ``y >= 0`` forces
  ``x >= 0``) -- **declined**: it is the retired ``DISCOPT_FBBT_ONE_SIDED_DIV``
  (#1616; over its 29 firing MINLPLib instances it lost the gancns and waternd1
  certificates and cost +11 % wall). Built again here, its only measured gain was
  ``wallfix``'s root bound, which does not answer the audit row's reopen condition.
  These tests pin that the sign/domain flag does not revive it: both of its arms
  leave a product box alike.

#1990 then brought the product deduction back on its own panel, as
``DISCOPT_FBBT_EXTENDED_MUL`` (default ON, ``presolve/fbbt.rs``), so a product box
now *is* tightened -- by that rule, identically in both arms of this one.
``wallfix``'s root, unbounded before, bounds at 0 (optimum 1.0000047).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

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


def test_flag_off_is_the_pre_1982_control(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    lb, _ = _box(_power_model())
    assert lb[0] < 0, f"OFF must keep x's negative side: lb={lb[0]}"


def test_zero_touching_product_factor_is_not_divided_by_this_flag(monkeypatch):
    """Both arms leave the product box alike; the tightening that remains is
    ``DISCOPT_FBBT_EXTENDED_MUL``'s (#1990): ``x * y == 1`` with ``y >= 0`` gives
    ``x >= 0`` -- exactly 0, since ``y``'s ``9.999e19`` is unbounded, not the
    ``1 / 9.999e19 = 1.0001e-20`` that would cut ``x = 1e-25, y = 1e25``."""
    boxes = []
    for arm in ("0", "1"):
        monkeypatch.setenv(FLAG, arm)
        boxes.append(_box(_product_model()))
    assert np.array_equal(boxes[0][0], boxes[1][0]), boxes
    assert np.array_equal(boxes[0][1], boxes[1][1]), boxes
    assert boxes[1][0][0] == 0.0, boxes


def test_without_extended_mul_the_product_box_is_untouched():
    """``DISCOPT_FBBT_EXTENDED_MUL`` is read once per process in Rust: a subprocess."""
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
        "from test_1982_fbbt_sign_domain import _box, _product_model\n"
        "print(json.dumps(float(_box(_product_model())[0][0])))\n"
    )
    env = dict(os.environ, DISCOPT_FBBT_EXTENDED_MUL="0", **{FLAG: "1"})
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) < -1e19


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
def test_wallfix_root_is_bounded_by_the_product_rule_alone(monkeypatch, arm):
    """Only the product deduction bounds wallfix's root (``objvar >= 0``), so both
    arms of this flag do, and the bound stays below the optimum 1.0000046580."""
    lb, _, res = _wallfix_root(monkeypatch, arm)
    assert lb[0] == 0.0, lb[0]
    assert res.status == "optimal" and res.lower_bound is not None, res
    assert 0.0 <= res.lower_bound <= 1.0000046580 + 1e-9, res
