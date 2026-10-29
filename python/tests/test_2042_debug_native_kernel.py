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


def _market_split():
    """A 2x10 market split: a pure MILP that branches (239 nodes on the Rust engine)."""
    import numpy as np

    rng = np.random.default_rng(1)
    a = rng.integers(0, 100, size=(2, 10))
    b = a.sum(axis=1) // 2
    m = dm.Model("market_split")
    x = [m.binary(f"x{i}") for i in range(10)]
    sp = [m.continuous(f"sp{j}", lb=0, ub=1000) for j in range(2)]
    sm = [m.continuous(f"sm{j}", lb=0, ub=1000) for j in range(2)]
    for j in range(2):
        m.subject_to(sum(int(a[j, i]) * x[i] for i in range(10)) + sp[j] - sm[j] == int(b[j]))
    m.minimize(sum(sp) + sum(sm))
    return m


def test_attached_debugger_skips_highs_on_a_pure_milp(monkeypatch):
    """HiGHS fires no checkpoint either; the Rust MILP engine does (#2042)."""
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    plain = _market_split().solve()
    assert "highs-milp" in (plain.algorithm_route or ""), plain.algorithm_route
    c = _Count()
    r = _market_split().solve(debug=DebugSession(c))
    assert c.n > 0, "debugger never called"
    assert "after_select" in c.kinds, c.kinds
    assert "native-milp" in (r.algorithm_route or ""), r.algorithm_route
    assert r.objective == pytest.approx(plain.objective, abs=1e-6)


def test_route_without_a_tree_warns_instead_of_staying_silent():
    """An LP has no tree to step through; the session sees nothing, and is told so."""
    m = dm.Model("lp")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + 2 * y <= 5)
    m.maximize(x + y)
    c = _Count()
    with pytest.warns(UserWarning, match=r"debug: this solve fired no debugger checkpoints"):
        r = m.solve(debug=DebugSession(c))
    assert r.status == "optimal" and c.n == 0


def test_no_silence_warning_when_checkpoints_fired():
    c = _Count()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ncnlp().solve(debug=DebugSession(c))
    assert c.n > 0
    assert not [w for w in caught if "fired no debugger checkpoints" in str(w.message)]
