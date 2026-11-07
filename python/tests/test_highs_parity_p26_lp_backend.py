"""highs-parity D1: a pure LP defaults to the Rust route; MILP stays on HiGHS.

Owner decision 2026-11-05 on the P2.6 gate (3 rounds, 480 solves, 0 incorrect; Rust
certifies all 25 L LPs, 1.07x the HiGHS route's wall). ``DISCOPT_LP_BACKEND`` selects the
pure-LP engine; unset, an explicitly set ``DISCOPT_LP_MILP_BACKEND`` still decides (A/B
runs keep their meaning), else ``rust``. MILP reads only ``DISCOPT_LP_MILP_BACKEND``.
Before D1 the LP default was HiGHS, so ``test_lp_defaults_to_the_rust_route`` fails there.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import pytest


def _lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.subject_to(x[0] + 2 * x[1] - x[2] <= 4)
    m.subject_to(x[0] + x[1] + x[2] >= 2)
    m.minimize(3 * x[0] + x[1] + 2 * x[2])
    return m


def _milp():
    m = dm.Model("milp")
    x = m.integer("x", shape=(2,), lb=0, ub=5)
    m.subject_to(2 * x[0] + 3 * x[1] >= 7)
    m.minimize(x[0] + x[1])
    return m


def _on_highs(r) -> bool:
    return (r.solver_stats or {}).get("route/lp_milp_backend") == 1.0


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_BACKEND", raising=False)
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)


def test_lp_defaults_to_the_rust_route():
    assert S._lp_backend() == "rust"
    r = _lp().solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert not _on_highs(r)
    assert r.algorithm_route.startswith("rust-lp: discopt LP solve, Rust simplex")
    assert r.objective == pytest.approx(2.0)


def test_lp_backend_highs_opts_out(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_BACKEND", "highs")
    r = _lp().solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified and _on_highs(r)


def test_joint_variable_still_decides_an_lp_when_lp_backend_is_unset(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    assert S._lp_backend() == "highs"
    assert _on_highs(_lp().solve(time_limit=30))
    monkeypatch.setenv("DISCOPT_LP_BACKEND", "rust")  # the specific variable wins
    assert S._lp_backend() == "rust"


def test_milp_default_is_unchanged():
    r = _milp().solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified and _on_highs(r)


def test_explicit_keyword_decides_the_lp(monkeypatch):
    assert _on_highs(_lp().solve(time_limit=30, milp_backend="highs"))


def test_typo_is_refused(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_BACKEND", "hihgs")
    with pytest.raises(ValueError, match="DISCOPT_LP_BACKEND"):
        S._lp_backend()
