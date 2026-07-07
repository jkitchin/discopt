"""G2 — the effort governor: unit + integration tests.

Covers the policy state machine (throttle-after-k-misses, gap gating,
hit-resets-streak, default-OFF inertness) and one end-to-end assertion that
turning the governor on does not change the certified objective on a
RENS-heavy easy instance (fac2) — the heuristic-policy regime (certified
objective unchanged; node_count may shift).
"""

from __future__ import annotations

import pytest
from discopt.heuristic_governor import (
    EXPENSIVE_SOURCES,
    K_DISABLE,
    HeuristicGovernor,
    governor,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    yield


def test_default_off_is_inert(monkeypatch):
    """With the flag unset, every source is allowed and record() is a no-op."""
    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    g = HeuristicGovernor()
    # even a closed gap on an expensive source is allowed when OFF
    assert g.allowed("rens", gap_open=False) is True
    # record() does nothing when OFF
    for _ in range(10):
        g.record("rens", improved=False)
    assert g.allowed("rens", gap_open=True) is True
    assert g.snapshot() == {}  # no stats accumulated while OFF


def test_throttle_after_k_consecutive_misses(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "1")
    g = HeuristicGovernor()
    assert g.allowed("rens", gap_open=True) is True
    for _ in range(K_DISABLE):
        g.record("rens", improved=False)
    # disabled for the rest of the process after K consecutive misses
    assert g.allowed("rens", gap_open=True) is False
    snap = g.snapshot()["rens"]
    assert snap["disabled"] is True
    assert snap["consecutive_misses"] == K_DISABLE
    assert snap["throttled_events"] >= 1


def test_hit_resets_miss_streak_and_reenables(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "1")
    g = HeuristicGovernor()
    for _ in range(K_DISABLE):
        g.record("rens", improved=False)
    assert g.allowed("rens", gap_open=True) is False
    # a genuine improvement clears the streak and re-enables the source
    g.record("rens", improved=True)
    assert g.allowed("rens", gap_open=True) is True
    assert g.snapshot()["rens"]["consecutive_misses"] == 0


def test_expensive_source_requires_open_gap(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "1")
    g = HeuristicGovernor()
    for src in EXPENSIVE_SOURCES:
        assert g.allowed(src, gap_open=True) is True
        assert g.allowed(src, gap_open=False) is False


def test_ungoverned_sources_are_never_throttled(monkeypatch):
    """rins / local_branching are load-bearing on the convex class (cvxnonsep):
    the governor must never throttle them, no matter how many misses accrue."""
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "1")
    g = HeuristicGovernor()
    for src in ("rins", "lbranch", "enumerate", "feasibility_pump"):
        for _ in range(K_DISABLE + 5):
            g.record(src, improved=False)
        # never disabled, never gap-gated, never a throttle event
        assert g.allowed(src, gap_open=False) is True
        assert g.snapshot() == {}  # ungoverned sources accrue no stats


def test_non_expensive_source_ignores_gap(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "1")
    g = HeuristicGovernor()
    # a source not governed is never gap-gated
    assert g.allowed("feasibility_pump", gap_open=False) is True


def test_singleton_is_shared():
    assert governor() is governor()


@pytest.mark.slow
def test_governor_cert_neutral_on_fac2(monkeypatch):
    """End-to-end: the governor does not change the certified objective.

    fac2 is RENS-heavy (RENS is 33 % of its solve wall at a 0 % incumbent hit
    rate); it is exactly where the governor throttles. Heuristic-policy regime:
    the certified objective must be identical OFF vs ON (node_count may shift).
    """
    from pathlib import Path

    from discopt.modeling.core import from_nl

    data = Path(__file__).parent / "data" / "minlplib_nl" / "fac2.nl"
    if not data.exists():
        pytest.skip("fac2.nl not in the vendored corpus")

    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    governor().reset()
    off = from_nl(str(data)).solve(time_limit=60, gap_tolerance=1e-4)

    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "1")
    governor().reset()
    # prime the miss streak so the once-per-solve RENS call is throttled on this
    # instance (models a warm governor mid-benchmark; RENS has a 0% hit rate)
    for _ in range(K_DISABLE):
        governor().record("rens", improved=False)
    on = from_nl(str(data)).solve(time_limit=60, gap_tolerance=1e-4)

    assert off.objective is not None and on.objective is not None
    assert abs(off.objective - on.objective) <= 1e-6 * (abs(off.objective) + 1.0)
    # the governor must have actually throttled something (the firing proof)
    assert governor().any_throttled()

    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    governor().reset()


def test_env_truthy_parsing(monkeypatch):
    from discopt.heuristic_governor import _governor_enabled

    for v in ("1", "true", "on", "YES", "True"):
        monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", v)
        assert _governor_enabled() is True
    for v in ("0", "off", "false", "", "no"):
        monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", v)
        assert _governor_enabled() is False
    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    assert _governor_enabled() is False
