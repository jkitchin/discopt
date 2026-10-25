"""#1935: parent-basis warm start for the native spatial kernel's node LPs.

``DISCOPT_NATIVE_WARM_START`` selects how a child's node LP starts: ``0`` cold
(the opt-out), ``1`` from the parent's optimal basis, ``2`` (the default since
#1944's graduation panel) from that basis with a dual-infeasible start repaired
by bound flips. The start can move which
optimal vertex a degenerate node LP lands on, never what the search may
conclude, so every mode must certify the same optimum. The counters are
asserted, not inferred, so "the warm start never engaged" cannot read as a pass
(CLAUDE.md §6).
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.smoke


def _haverly_q(case: int):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "discopt_benchmarks", "scripts"))
    from _pooling import pool18

    return pool18.build_pool(pool18.haverly(case), "q")[0]


def test_warm_start_mode_parsing(monkeypatch):
    from discopt.solver import _native_warm_start_mode

    monkeypatch.delenv("DISCOPT_NATIVE_WARM_START", raising=False)
    assert _native_warm_start_mode() == 2
    for raw, mode in (("", 2), ("0", 0), ("1", 1), ("2", 2), (" 2 ", 2)):
        monkeypatch.setenv("DISCOPT_NATIVE_WARM_START", raw)
        assert _native_warm_start_mode() == mode, raw
    # A typo must not silently measure the cold arm twice.
    for bad in ("3", "on", "true", "-1"):
        monkeypatch.setenv("DISCOPT_NATIVE_WARM_START", bad)
        with pytest.raises(ValueError, match="DISCOPT_NATIVE_WARM_START"):
            _native_warm_start_mode()


@pytest.mark.parametrize("case", [1, 2])
def test_every_warm_mode_certifies_the_cold_optimum(monkeypatch, case):
    results = {}
    for mode in ("0", "1", "2"):
        monkeypatch.setenv("DISCOPT_NATIVE_WARM_START", mode)
        r = _haverly_q(case).solve(time_limit=60)
        assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
        assert r.status == "optimal" and r.gap_certified, (mode, r.status)
        results[mode] = r
    stats = {m: r.solver_stats for m, r in results.items()}
    assert results["0"].node_count > 3, "too few nodes to exercise inheritance"
    assert stats["0"]["tree/warm_starts"] == 0
    assert stats["1"]["tree/warm_starts"] > 0
    assert stats["2"]["tree/warm_starts"] > 0
    ref = results["0"].objective
    for m, r in results.items():
        assert r.objective == pytest.approx(ref, rel=1e-6, abs=1e-6), m
        # Each certified bound is valid against every arm's incumbent (maximize).
        for other in results.values():
            assert r.bound >= other.objective - 1e-6 * (1 + abs(ref)), (m, r.bound)
