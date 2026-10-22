"""Issue #1874 (2): the native MILP engine left most of the caller's budget unused.

``_solve_milp_simplex`` runs the Rust engine twice -- run 1 on a <= 10 s stall slice,
then a re-entry on the remainder. Measured on MINLPLib ``tln4`` (optimum 8.3):
at ``time_limit=30`` the re-entry was skipped (the #917 withheld reserve was never
plumbed here) and 8.9 came back at 9.9 s; at ``time_limit=120`` both runs stopped on
the 100k-node default and 8.7 came back at 24.3 s. With
``DISCOPT_MILP_ENGINE_FULL_BUDGET`` the same solve certifies 8.3 in 73.6 s.

The 100k default was a de-facto memory guard (the pool kept every node's vectors), so
the flag rests on ``NodePool::release_storage`` plus an explicit open-frontier cap,
both checked here through the engine call itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import _rust
from discopt import solver as SV
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_MILP_ENGINE_FULL_BUDGET"


def _milp_route_model() -> Model:
    """Integer-bilinear, small ranges: linearised and served by the native MILP engine."""
    m = Model("budget")
    x = [m.integer(f"x{i}", lb=0, ub=5) for i in range(4)]
    m.subject_to(x[0] + x[1] + x[2] + x[3] <= 9)
    m.subject_to(x[0] * x[1] - x[2] * x[3] <= 4)
    m.minimize(x[0] * x[2] - 3 * x[0] * x[1] + (x[3] - 2) ** 2 + x[1])
    return m


def _spy(monkeypatch, script=None):
    """Record every main engine call (``max_nodes`` > 1000; auxiliary sub-MILPs pass
    smaller caps). ``script(call_no, real_out, args, kwargs)`` may replace an answer."""
    real = _rust.solve_milp_csc_py
    calls: list[dict] = []

    def spy(*a, **kw):
        out = real(*a, **kw)
        if a[12] <= 1000:
            return out
        calls.append({"max_nodes": a[12], **kw})
        return script(len(calls), out, a, kw) if script else out

    monkeypatch.setattr(_rust, "solve_milp_csc_py", spy)
    return calls


def _worse_feasible_first(n, out, a, kw):
    """Run 1 reports ``feasible`` at its own point, so the re-entry runs."""
    return ("feasible",) + tuple(out[1:]) if n == 1 else out


def test_flag_on_default_cap_lets_the_clock_and_the_frontier_govern(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    calls = _spy(monkeypatch, _worse_feasible_first)
    r = _milp_route_model().solve(time_limit=20)
    assert (r.algorithm_route or "").startswith("native-milp"), r.algorithm_route
    assert len(calls) == 2, "the re-entry did not run -- the probe measured nothing"
    for c in calls:
        assert c["max_nodes"] == SV._MILP_ENGINE_UNCAPPED_NODES
        assert isinstance(c["max_open_nodes"], int) and c["max_open_nodes"] >= 1000


def test_explicit_max_nodes_is_one_total_over_both_runs(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    seen = {}

    def script(n, out, a, kw):
        if n == 1:
            seen["nodes1"] = int(out[4])
            return ("feasible",) + tuple(out[1:])
        return out

    calls = _spy(monkeypatch, script)
    _milp_route_model().solve(time_limit=20, max_nodes=5000)
    assert calls[0]["max_nodes"] == 5000
    assert len(calls) == 2, "the re-entry did not run -- the probe measured nothing"
    assert calls[1]["max_nodes"] == 5000 - seen["nodes1"], (calls[1]["max_nodes"], seen)


def test_run1_node_limit_with_a_verified_incumbent_is_kept(monkeypatch):
    """Run 1 hits a node budget holding a real point: under the flag the point is
    reported (uncertified, ``termination="node_limit"``) instead of deferring."""
    monkeypatch.setenv(FLAG, "1")

    def script(n, out, a, kw):
        assert np.isfinite(out[2]), "precondition: the real engine found an incumbent"
        return ("node_limit",) + tuple(out[1:])

    calls = _spy(monkeypatch, script)
    r = _milp_route_model().solve(time_limit=20, max_nodes=5000)
    assert calls, "the engine never ran -- the probe measured nothing"
    assert (r.algorithm_route or "").startswith("native-milp"), r.algorithm_route
    assert r.status == "feasible" and not r.gap_certified
    assert r.termination == "node_limit"
    assert r.objective is not None


def test_incumbent_reclaims_the_withheld_reserve(monkeypatch):
    """With an incumbent from run 1 the #917 extension goes to the re-entry: run 2's
    clock exceeds what was left of the (reduced) ``time_limit``. The extension is
    injected at the real call site, which must forward it."""
    monkeypatch.setenv(FLAG, "1")
    calls = _spy(monkeypatch, _worse_feasible_first)
    real_simplex = SV._solve_milp_simplex
    seen = {}
    ext = 4.0

    def with_extension(*a, **kw):
        seen["forwarded"] = kw.get("incumbent_time_extension")
        seen["time_limit"] = a[1]
        kw["incumbent_time_extension"] = ext
        return real_simplex(*a, **kw)

    monkeypatch.setattr(SV, "_solve_milp_simplex", with_extension)
    _milp_route_model().solve(time_limit=6)
    assert "forwarded" in seen, "the engine route was not taken -- the probe measured nothing"
    assert seen["forwarded"] is not None, "the call site does not forward the extension"
    assert len(calls) == 2, "the re-entry did not run -- the probe measured nothing"
    assert calls[1]["time_limit_s"] > seen["time_limit"] - calls[0]["time_limit_s"] + 0.5 * ext


def test_flag_off_is_the_legacy_budget(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    calls = _spy(monkeypatch, _worse_feasible_first)
    _milp_route_model().solve(time_limit=20)
    assert calls and all(c["max_nodes"] == 100_000 for c in calls)
    assert all("max_open_nodes" not in c for c in calls)


def test_flag_defaults_on(monkeypatch):
    """Graduated under CLAUDE.md section 5 (panel in the gate's docstring)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert SV._milp_engine_full_budget_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not SV._milp_engine_full_budget_enabled()
