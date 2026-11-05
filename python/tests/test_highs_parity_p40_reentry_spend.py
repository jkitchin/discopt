"""highs-parity P4.0: ``DISCOPT_MILP_REENTRY_SPEND`` re-enters with run 1's leftover.

``_solve_milp_simplex`` gives run 1 a stall slice of half the remaining time (below the
10 s cap) and re-enters on an uncertified ``feasible`` only when MORE than that slice
remains. A run 1 that spends its slice therefore leaves less than the slice behind, and
the re-entry never runs: measured on the MIPLIB-38 rows at ``time_limit=20``, 20 of 24
returned at 10.1-11.1 s (``docs/dev/data/highs-parity/p4.1``). The flag re-enters
whenever at least ``_MILP_REENTRY_MIN_S`` remains.

The engine is scripted around REAL calls (as in ``test_issue_1874_milp_reentry_node_limit``):
run 1 sleeps through its whole slice and claims a worse but genuine incumbent, so the
leftover is below the slice by construction, independent of machine speed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from discopt import _rust
from discopt import solver as _solver
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_MILP_REENTRY_SPEND"


@pytest.fixture(autouse=True)
def _pins(monkeypatch):
    # The native route, not HiGHS (the pure-MILP default).
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    # Keep the test short: the slice is ~1.4 s, so the floor must sit below the
    # ~1.4 s the scripted run 1 leaves behind.
    monkeypatch.setattr(_solver, "_MILP_REENTRY_MIN_S", 0.2, raising=False)


def _milp_route_model() -> Model:
    """A pure MILP: routed to the native engine, with no #844 reserve withheld (the
    lifted MINLP path reclaims one for its re-entry, which hides the gate)."""
    m = Model("p40")
    x = [m.integer(f"x{i}", lb=0, ub=7) for i in range(5)]
    m.subject_to(3 * x[0] + 5 * x[1] + 4 * x[2] + 7 * x[3] + 2 * x[4] <= 23)
    m.subject_to(x[0] + x[1] >= 2)
    m.maximize(4 * x[0] + 7 * x[1] + 5 * x[2] + 9 * x[3] + 2 * x[4])
    return m


WORSE = np.array([2.0, 0, 0, 0, 0])  # feasible, objective 8 < the optimum 32


def _script(monkeypatch):
    real = _rust.solve_milp_csc_py
    seen = {"n": 0, "slices": [], "true": None, "worse": None}

    def scripted(*a, **kw):
        if a[12] <= 1000:  # auxiliary sub-MILPs keep their real answer
            return real(*a, **kw)
        seen["n"] += 1
        seen["slices"].append(kw.get("time_limit_s"))
        out = real(*a, **kw)
        if seen["n"] > 1:
            return out
        status, x, obj, bound, nodes, iters = out
        assert np.isfinite(obj), "precondition: the real engine found an incumbent"
        seen["true"] = obj
        b = list(a)
        k = WORSE.size
        xl = np.array(b[7], dtype=np.float64)
        xu = np.array(b[8], dtype=np.float64)
        xl[:k], xu[:k] = WORSE, WORSE
        b[7], b[8] = np.ascontiguousarray(xl), np.ascontiguousarray(xu)
        w = real(*b, **kw)
        assert w[0] == "optimal" and w[2] > obj + 1e-6, w[:3]
        seen["worse"] = w[2]
        time.sleep(float(kw["time_limit_s"]))  # run 1 spends its whole slice
        return ("feasible", w[1], w[2], bound, nodes, iters)

    monkeypatch.setattr(_rust, "solve_milp_csc_py", scripted)
    return seen


def test_legacy_condition_discards_the_leftover(monkeypatch):
    """Flag OFF: the leftover is smaller than run 1's slice, so no re-entry."""
    monkeypatch.delenv(FLAG, raising=False)
    seen = _script(monkeypatch)
    r = _milp_route_model().solve(time_limit=3)
    assert (r.algorithm_route or "").startswith("native-milp"), r.algorithm_route
    assert seen["n"] == 1, seen
    assert r.objective == pytest.approx(8.0)  # run 1's scripted point (maximize)


def test_flag_spends_the_leftover(monkeypatch):
    """Flag ON: the re-entry runs on the leftover and its better point is adopted."""
    monkeypatch.setenv(FLAG, "1")
    seen = _script(monkeypatch)
    r = _milp_route_model().solve(time_limit=3)
    assert (r.algorithm_route or "").startswith("native-milp"), r.algorithm_route
    assert seen["n"] == 2, "the re-entry did not run -- the probe measured nothing"
    # The re-entry got less than run 1's slice: exactly the case the legacy gate skips.
    assert seen["slices"][1] < seen["slices"][0], seen["slices"]
    assert r.objective == pytest.approx(32.0)  # the re-entry's better point
