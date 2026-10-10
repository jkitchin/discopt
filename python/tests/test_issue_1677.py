"""Regression tests for #1677: a complemented-binary indicator in ``if_then``.

``m.if_then(1 - z, [...])`` ("if z == 0 then ...") reached ``solve()`` with the
indicator stored as a ``BinaryOp``; the search found the optimum but the model
verifier (``verify_point``) raised ``TypeError: expected a variable or an indexed
variable, got BinaryOp``. ``if_then`` now normalises an affine 0/1 function of one
binary column to that column plus an active value, and refuses any other
expression at the call.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import _IndicatorConstraint

_TL = 30


def _solve(m):
    r = m.solve(time_limit=_TL)
    assert r.status == "optimal", (r.status, getattr(r, "error", None))
    assert r.gap_certified, "the incumbent must be certified on the declared model"
    return r


def _model(indicator_kind, sense, force=None):
    """x in [0, 10], z binary; rows ``x <= 3`` active when the indicator is 1."""
    m = dm.Model(f"ind_{indicator_kind}_{sense}")
    x = m.continuous("x", lb=0, ub=10)
    zs = m.binary("zs", shape=(2,))
    z = zs[1] if indicator_kind == "1-z[i]" else m.binary("z")
    ind = {
        "z": z,
        "1-z": 1 - z,
        "1-z[i]": 1 - z,
        "-(z-1)": -(z - 1),
        "(2-2z)/2": (2 - 2 * z) / 2,
    }[indicator_kind]
    m.if_then(ind, [x <= 3])
    if force is not None:
        m.subject_to(z == force)
    (m.maximize if sense == "max" else m.minimize)(x if sense == "max" else -x)
    return m, x, z


def _zval(r, z):
    """The solved value of ``z`` (a scalar binary or ``zs[1]``)."""
    if isinstance(z, dm.Variable):
        return float(np.asarray(r.value(z)).reshape(()))
    return float(np.asarray(r.value(z.base)).ravel()[1])


@pytest.mark.parametrize("sense", ["max", "min"])
@pytest.mark.parametrize("kind", ["1-z", "1-z[i]", "-(z-1)", "(2-2z)/2"])
def test_complemented_indicator_free_binary_reaches_x_ub(kind, sense):
    """Maximise x: pick z = 1 so the complemented indicator is 0 and x = 10."""
    m, x, z = _model(kind, sense)
    r = _solve(m)
    assert r.value(x) == pytest.approx(10.0, abs=1e-6)
    assert _zval(r, z) == pytest.approx(1.0, abs=1e-6)
    assert abs(r.objective) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_complemented_indicator_forced_active(sense):
    """z fixed to 0: the complement is 1, so x <= 3 is enforced."""
    m, x, z = _model("1-z", sense, force=0)
    r = _solve(m)
    assert r.value(x) == pytest.approx(3.0, abs=1e-6)
    assert abs(r.objective) == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_plain_indicator_forced_active_and_free(sense):
    """The un-complemented control: z free -> 10 with z = 0; z = 1 -> 3."""
    m, x, z = _model("z", sense)
    r = _solve(m)
    assert r.value(x) == pytest.approx(10.0, abs=1e-6)
    assert _zval(r, z) == pytest.approx(0.0, abs=1e-6)
    m, x, z = _model("z", sense, force=1)
    r = _solve(m)
    assert r.value(x) == pytest.approx(3.0, abs=1e-6)


def test_complement_is_stored_as_the_column_with_active_value_zero():
    m = dm.Model("store")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.if_then(1 - z, [x <= 3])
    (ic,) = [c for c in m._constraints if isinstance(c, _IndicatorConstraint)]
    assert ic.indicator is z
    assert ic.active_value == 0


def test_negated_boolean_var_indicator():
    m = dm.Model("notb")
    x = m.continuous("x", lb=0, ub=10)
    b = m.boolean("b")
    m.if_then(~b, [x <= 3])
    m.subject_to(b.variable == 0)
    m.maximize(x)
    r = _solve(m)
    assert r.value(x) == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda z, w, c: 2 * z, id="2z"),
        pytest.param(lambda z, w, c: z + 1, id="z+1"),
        pytest.param(lambda z, w, c: 1 - z - w, id="two-binaries"),
        pytest.param(lambda z, w, c: 1 - c, id="continuous"),
        pytest.param(lambda z, w, c: z * w, id="product"),
    ],
)
def test_non_indicator_expressions_are_refused_at_call_time(build):
    m = dm.Model("bad")
    x = m.continuous("x", lb=0, ub=10)
    z, w = m.binary("z"), m.binary("w")
    c = m.continuous("c", lb=0, ub=1)
    with pytest.raises(TypeError, match="if_then: the indicator expression"):
        m.if_then(build(z, w, c), [x <= 3])
