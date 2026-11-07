"""Exporters must write every finite bound below the declared-no-bound box.

The ``.nl`` writers (Python and Rust) and the GAMS writer treated ``|b| >= 1e18``
as "no bound", and the LP/MPS writers ``|b| >= 1e19``, while the open-bound rule
everywhere else is the ``DEFAULT_VARIABLE_BOUND`` box (9.999e19). So
``maximize x`` with ``x <= 1e18`` solved to ``1e18`` directly and came back
``unbounded`` after a ``.nl`` round trip: the exported file was a different model.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.constants import DEFAULT_VARIABLE_BOUND, bound_is_open
from discopt.modeling.core import from_nl

_LARGE = [1e18, 5e19, 9.99e19]


def test_bound_is_open_rule():
    assert not bound_is_open(1e18) and not bound_is_open(-9.99e19)
    assert bound_is_open(DEFAULT_VARIABLE_BOUND) and bound_is_open(-DEFAULT_VARIABLE_BOUND)
    assert bound_is_open(float("inf")) and bound_is_open(float("-inf")) and bound_is_open(1e20)


@pytest.mark.parametrize("kind", ["continuous", "integer"])
@pytest.mark.parametrize("ub", _LARGE)
def test_nl_round_trip_keeps_large_upper_bound(tmp_path, kind, ub):
    m = dm.Model()
    x = getattr(m, kind)("x", lb=0, ub=ub)
    m.maximize(x)
    p = tmp_path / "b.nl"
    m.to_nl(str(p))
    back = from_nl(str(p))
    assert float(back._variables[0].ub) == ub
    if kind == "integer":
        # An integer bound >= 1e15 is refused (status ``error``) by the pure-MILP
        # route whether or not it went through a file (``READBACK_LIMIT``), so the
        # read-back bound is the check here.
        return
    r = back.solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(ub, rel=1e-9)


@pytest.mark.parametrize("lb", [-v for v in _LARGE])
def test_nl_round_trip_keeps_large_lower_bound(tmp_path, lb):
    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=0)
    m.minimize(x)
    p = tmp_path / "b.nl"
    m.to_nl(str(p))
    r = from_nl(str(p)).solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(lb, rel=1e-9)


def test_default_box_still_exported_as_open(tmp_path):
    # Control: a variable declared without bounds stays unbounded in the file.
    m = dm.Model()
    x = m.continuous("x")
    m.maximize(x)
    p = tmp_path / "b.nl"
    m.to_nl(str(p))
    b_section = p.read_text().split("\nb\n", 1)[1].splitlines()
    assert b_section[0] == "3", b_section[0]


@pytest.mark.parametrize("ub", _LARGE)
def test_text_writers_emit_large_bound(ub):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=ub)
    y = m.continuous("y", lb=-ub, ub=0)
    m.maximize(x + y)
    gams = m.to_gams()
    assert f"x.up = {ub};" in gams and f"y.lo = {-ub};" in gams, gams
    lp = m.to_lp()
    assert "Inf" not in lp.split("Bounds", 1)[1], lp
    mps = m.to_mps()
    assert " UP BND  x" in mps and " LO BND  y" in mps, mps
