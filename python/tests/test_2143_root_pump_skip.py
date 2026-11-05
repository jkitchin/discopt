"""#2143: ``DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT`` -- no root pumps once an incumbent exists.

At the end of the root batch the LP-vertex feasibility pump and the NLP-relaxation
pump (a root NLP multistart seeding a second pump) run whether or not an incumbent
already exists. Both are primal only. Measured on GDPlib ``syngas`` under hull (the
other #2143 flags ON, 120 s): they cost 15.0 s plus a 23.9 s multistart after LOA had
already found the optimum, and the tree reached 7 nodes. With them skipped it reached
29.

ON skips both pumps when the tree holds an incumbent; with none they still run.
Default OFF; the legacy path is pinned under ``=0``.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import pytest
from discopt import solver as S
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke

_FLAG = "DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT"
_SKIP_LOG = "Feasibility pump (root): skipped, an incumbent exists"


def _model():
    m = dm.Model("pump_skip")
    x = m.continuous("x", lb=-3, ub=3)
    z = m.continuous("z", lb=-3, ub=3)
    y = m.binary("y")
    m.subject_to(x * x + z * z <= 4 + 3 * y)
    m.subject_to(x * z >= 0.5)
    m.minimize(-x - z + 1.5 * y + 0.2 * x * x * z)
    return m, {x: 1.2, z: 1.1, y: 1.0}


def _solve(monkeypatch, caplog, flag: str, *, start: bool):
    """Root pump calls, whether the skip was logged, and the result.

    ``start=True`` poses the caller's start first (``DISCOPT_ROOT_WARMSTART_FIRST``),
    so the tree holds an incumbent before the pumps; ``start=False`` gives none.
    """
    monkeypatch.setenv(_FLAG, flag)
    monkeypatch.setenv("DISCOPT_ROOT_WARMSTART_FIRST", "1" if start else "0")
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m, x0 = _model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = m.solve(time_limit=30, initial_solution=x0 if start else None)
    skipped = any(_SKIP_LOG in rec.getMessage() for rec in caplog.records)
    caplog.clear()
    return len(calls), skipped, r


def test_flag_is_default_on(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert S._root_pump_skip_enabled() is True  # graduated 2026-11-05 (#2143)
    monkeypatch.setenv(_FLAG, "0")
    assert S._root_pump_skip_enabled() is False
    monkeypatch.setenv(_FLAG, "1")
    assert S._root_pump_skip_enabled() is True


def test_legacy_runs_the_pumps_with_an_incumbent(monkeypatch, caplog):
    n_pump, skipped, r = _solve(monkeypatch, caplog, "0", start=True)
    assert not skipped
    assert n_pump >= 1
    assert r.status == "optimal"


def test_on_skips_the_pumps_with_an_incumbent(monkeypatch, caplog):
    n_off, _, r_off = _solve(monkeypatch, caplog, "0", start=True)
    n_on, skipped, r_on = _solve(monkeypatch, caplog, "1", start=True)
    assert skipped
    assert n_off >= 1 and n_on == 0
    assert r_on.status == r_off.status == "optimal"
    assert r_on.objective == pytest.approx(r_off.objective, abs=1e-6)


def test_on_runs_the_pumps_without_an_incumbent(monkeypatch, caplog):
    """MINLPLib ``ex1221`` reaches both root pumps with no incumbent: ON runs them."""
    from pathlib import Path

    from discopt.modeling.core import from_nl

    monkeypatch.setenv(_FLAG, "1")
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m = from_nl(str(Path(__file__).parent / "data" / "minlplib_nl" / "ex1221.nl"))
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = m.solve(time_limit=30)
    assert not any(_SKIP_LOG in rec.getMessage() for rec in caplog.records)
    assert len(calls) == 2  # the LP-vertex pump and the NLP-relaxation pump
    assert r.status == "optimal"
