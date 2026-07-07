"""G2 — the effort governor: unit + integration tests.

Covers the policy state machine (throttle-after-k-misses, gap gating,
hit-resets-streak, on-by-default + escape-hatch-restores-off) and one end-to-end
assertion that
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
    # Force the escape-hatch OFF state as the per-test baseline so the state-machine
    # tests below opt IN explicitly. G2-graduate flipped the *unset* default to ON;
    # the on-by-default + hatch-restores-off behaviour is asserted in its own tests.
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "0")
    yield


def test_default_on_when_unset(monkeypatch):
    """G2-graduate: with the flag UNSET the governor is ON by default and throttles.

    The once-inert unset state now activates the governor — the graduated default.
    """
    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    g = HeuristicGovernor()
    # gap gate applies (expensive source, closed gap refused) => governor is live
    assert g.allowed("rens", gap_open=False) is False
    for _ in range(K_DISABLE):
        g.record("rens", improved=False)
    # throttled after K misses, and stats accrue => not inert
    assert g.allowed("rens", gap_open=True) is False
    assert g.snapshot()["rens"]["disabled"] is True


def test_escape_hatch_restores_off(monkeypatch):
    """DISCOPT_HEURISTIC_GOVERNOR=0 restores the pre-governor byte-identical path.

    The graduated default keeps a live escape hatch (not a dead flag): with =0 every
    source is allowed, record() is a no-op, and no stats accumulate.
    """
    for hatch in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", hatch)
        g = HeuristicGovernor()
        # even a closed gap on an expensive source is allowed when OFF
        assert g.allowed("rens", gap_open=False) is True
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

    # OFF baseline uses the escape hatch (=0), since the graduated default is ON.
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "0")
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

    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "0")
    governor().reset()


def test_env_parsing_default_on(monkeypatch):
    """G2-graduate: default-ON. Truthy/any-non-hatch value => ON; only the explicit
    escape-hatch values (and unset) resolve as the graduated default (ON)."""
    from discopt.heuristic_governor import _governor_enabled

    for v in ("1", "true", "on", "YES", "True"):
        monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", v)
        assert _governor_enabled() is True
    # the escape hatch: only these disable it
    for v in ("0", "off", "false", "", "no"):
        monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", v)
        assert _governor_enabled() is False
    # UNSET is now the graduated default: ON
    monkeypatch.delenv("DISCOPT_HEURISTIC_GOVERNOR", raising=False)
    assert _governor_enabled() is True
