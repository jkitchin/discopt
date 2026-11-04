"""#2143: ``DISCOPT_ROOT_WARMSTART_FIRST`` -- the root poses its warm-start sub-NLP first.

The root already poses one ``subnlp`` from the modeler's start (GAMS ``x.l``, else
the caller's ``initial_solution``), but only after the disjunct-configuration
constructor, both feasibility pumps, integer local search and diving. Measured on
GDPlib ``syngas`` under hull, warm-started from the big-M optimum: no incumbent at
120 s, although ``subnlp`` from that start returns 4669.0235 in 1.5 s. ON poses the
same sub-NLP before every other root heuristic, once.

Default OFF; the legacy order is pinned under ``=0``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke

_FLAG = "DISCOPT_ROOT_WARMSTART_FIRST"
_START = (1.2, 1.1, 1.0)


def _model():
    m = dm.Model("ws_first")
    x = m.continuous("x", lb=-3, ub=3)
    z = m.continuous("z", lb=-3, ub=3)
    y = m.binary("y")
    m.subject_to(x * x + z * z <= 4 + 3 * y)
    m.subject_to(x * z >= 0.5)
    m.minimize(-x - z + 1.5 * y + 0.2 * x * x * z)
    return m, {x: _START[0], z: _START[1], y: _START[2]}


def _solve(monkeypatch, flag: str):
    """The order of the root's sub-NLP / pump calls, and the result."""
    monkeypatch.setenv(_FLAG, flag)
    events: list[str] = []
    real_sub, real_pump = PH.subnlp, PH.feasibility_pump

    def sub(model, x, *a, **k):
        is_start = np.allclose(np.asarray(x, dtype=float)[:3], _START)
        events.append("warm_start" if is_start else "subnlp")
        return real_sub(model, x, *a, **k)

    def pump(*a, **k):
        events.append("pump")
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "subnlp", sub)
    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m, start = _model()
    r = m.solve(time_limit=30, initial_solution=start)
    return events, r


def test_flag_is_default_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert S._root_warm_start_first_enabled() is False
    monkeypatch.setenv(_FLAG, "1")
    assert S._root_warm_start_first_enabled() is True


def test_legacy_poses_the_warm_start_after_the_pumps(monkeypatch):
    events, r = _solve(monkeypatch, "0")
    assert events.count("warm_start") == 1
    assert events.index("pump") < events.index("warm_start")
    assert r.status == "optimal"


def test_on_poses_the_warm_start_first_and_once(monkeypatch):
    events, r = _solve(monkeypatch, "1")
    assert events[0] == "warm_start"
    assert events.count("warm_start") == 1
    assert "pump" in events  # the other root heuristics still run after it
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-2.3989357997, abs=1e-6)
