"""#2042: ``Model.solve(debug=...)`` must not be silently bypassed by the native kernel.

A continuous nonconvex model takes the native spatial kernel by default, and the
kernel fires no debugger checkpoint -- not ``iter_start``, not ``terminated``. An
attached session therefore recorded nothing while the solve returned ``optimal``
(measured: 105 nodes, 0 checkpoints). The callbacks already route such a solve to
the Python tree; an attached debugger now does too.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt import debug
from discopt.debug.engine import Control
from discopt.debug.session import DebugSession

KERNEL = "DISCOPT_NATIVE_SPATIAL_KERNEL"


class _Count:
    def __init__(self):
        self.n = 0
        self.kinds: set[str] = set()

    def interact(self, ctx, session):
        self.n += 1
        self.kinds.add(str(ctx.checkpoint))
        return Control.STEPI


def _ncnlp():
    m = dm.Model("ncnlp")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize(x**4 - 3 * x**2 + y**2 + x * y)
    return m


def test_plain_solve_still_takes_the_kernel(monkeypatch):
    """The model is on the kernel's route, so the debug test below exercises the fix."""
    monkeypatch.delenv(KERNEL, raising=False)
    r = _ncnlp().solve()
    assert r.status == "optimal"
    assert "native-spatial" in (r.algorithm_route or ""), r.algorithm_route


def test_attached_debugger_sees_checkpoints_on_kernel_route(monkeypatch):
    monkeypatch.delenv(KERNEL, raising=False)
    plain = _ncnlp().solve()
    c = _Count()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = _ncnlp().solve(debug=DebugSession(c))
    assert c.n > 0, "debugger never called"
    assert {"iter_start", "terminated"} <= c.kinds, c.kinds
    # The route change is announced (#1678), since node counts differ from a plain solve.
    assert any("debug" in str(w.message) and "native" in str(w.message) for w in caught)
    assert "Python tree" in (r.algorithm_route or ""), r.algorithm_route
    assert r.status == plain.status == "optimal"
    assert r.objective == pytest.approx(plain.objective, abs=1e-6)


def test_instruments_this_solve_only_at_the_outermost_solve():
    """Nested heuristic sub-solves fire nothing, so they keep the kernel."""
    assert not debug.instruments_this_solve()
    with debug.attach(DebugSession(_Count())):
        assert debug.instruments_this_solve()
        debug._TLS.depth = 2
        try:
            assert not debug.instruments_this_solve()
        finally:
            debug._TLS.depth = 0
    assert not debug.instruments_this_solve()
