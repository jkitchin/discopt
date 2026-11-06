"""#2182: ``DISCOPT_GDP_CONFIG_ROOT_SKI`` applies the ski-rental rule to the root share.

Under ``DISCOPT_GDP_CONFIG_DEFER`` (default ON) the GDP-config constructor's root
call gets a fixed ``_GDP_CONFIG_DEFER_ROOT_S`` = 1.5 s and the rest of its grant is
held back. With this flag ON the root share is ``min(1.5 s, time the solve spent
before the constructor)``; the hold-back is unchanged, so a solve that runs long
enough still spends the whole grant. Default ON since its §5 panel
(``docs/dev/data/2182-root-ski/``); ``=0`` opts out (BC: it changes which
incumbents are found and when, never a bound).
"""

from __future__ import annotations

import time
import warnings

import discopt.modeling as dm
import discopt.solver as SV
import pytest
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke


def test_default_on_and_opt_out(monkeypatch):
    """Graduated on its §5 panel: unset is ON, ``=0`` (and its spellings) opt out,
    empty is not off."""
    monkeypatch.delenv("DISCOPT_GDP_CONFIG_ROOT_SKI", raising=False)
    assert SV._gdp_config_root_ski_enabled()
    for on in ("", "1", "true", "yes", "on"):
        monkeypatch.setenv("DISCOPT_GDP_CONFIG_ROOT_SKI", on)
        assert SV._gdp_config_root_ski_enabled(), on
    for off in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("DISCOPT_GDP_CONFIG_ROOT_SKI", off)
        assert not SV._gdp_config_root_ski_enabled(), off


def test_root_share_is_capped_by_the_time_already_spent(monkeypatch):
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_ROOT_SKI", "0")
    # OFF: the fixed 1.5 s share whatever ran before.
    assert SV._gdp_config_root_until(100.0, 50.0, 49.8) == 50.0 + SV._GDP_CONFIG_DEFER_ROOT_S
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_ROOT_SKI", "1")
    # ON: 0.2 s spent before the constructor -> a 0.2 s root share.
    assert SV._gdp_config_root_until(100.0, 50.0, 49.8) == pytest.approx(50.2)
    # More than the share spent before: the fixed share again (unchanged).
    assert SV._gdp_config_root_until(100.0, 50.0, 45.0) == 50.0 + SV._GDP_CONFIG_DEFER_ROOT_S
    # The long-root rule still buys the whole grant now.
    assert SV._gdp_config_root_until(108.6, 100.0, 91.4) == 108.6
    # Never past the full grant, never before ``now``.
    assert SV._gdp_config_root_until(50.1, 50.0, 49.0) == 50.1
    assert SV._gdp_config_root_until(100.0, 50.0, 50.0) == 50.0


def test_defer_off_or_deterministic_keeps_the_single_grant(monkeypatch):
    from discopt.solver_tuning import SolverTuning

    monkeypatch.setenv("DISCOPT_GDP_CONFIG_ROOT_SKI", "1")
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "0")
    assert SV._gdp_config_root_until(100.0, 50.0, 49.8) == 100.0
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    monkeypatch.setattr(SV, "_tuning", lambda: SolverTuning(deterministic=True))
    assert SV._gdp_config_root_until(100.0, 50.0, 49.8) == 100.0


def test_a_short_root_share_keeps_the_rest_held_back():
    """The hold-back reads the shorter root deadline as any other: a root call
    stopped by it leaves the rest of the grant for the deferred call."""
    import numpy as np

    held = SV._gdp_config_hold_back(np.zeros(2), 115.0, 100.2, 100.2)
    assert held is not None and held["remaining"] == pytest.approx(14.8)


def _small_gdp():
    """Two two-way disjunctions over a convex-in-each-disjunct nonlinear row: four
    configurations, a root that is cheap next to the 1.5 s share."""
    m = dm.Model("root_ski")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x <= 1], [x >= 2.5]], name="dx")
    m.either_or([[y <= 1], [y >= 2.5]], name="dy")
    m.subject_to(x * y >= 1.5)
    m.minimize((x - 1.8) ** 2 + (y - 1.7) ** 2 + 0.1 * x * y)
    return m


def _root_grants(monkeypatch, on: bool) -> tuple[list[float], object]:
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    monkeypatch.setenv("DISCOPT_GDP_LOA_PROPOSER", "0")
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_ROOT_SKI", "1" if on else "0")
    grants: list[float] = []
    real = PH.one_hot_config_subnlp

    def spy(*a, **k):
        grants.append(k["deadline"] - time.perf_counter())
        return real(*a, **k)

    monkeypatch.setattr(PH, "one_hot_config_subnlp", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _small_gdp().solve(time_limit=40)
    return grants, r


def test_end_to_end_the_root_call_gets_the_shorter_share(monkeypatch):
    """Fails before #2182: the root call's grant was the fixed 1.5 s."""
    off, r_off = _root_grants(monkeypatch, on=False)
    on, r_on = _root_grants(monkeypatch, on=True)
    assert off and on, (off, on)
    assert off[0] == pytest.approx(SV._GDP_CONFIG_DEFER_ROOT_S, abs=0.05)
    assert on[0] < off[0] - 0.2, (on, off)
    # Same certified answer: the schedule moves incumbents, never the bound.
    assert r_off.status == r_on.status == "optimal"
    assert r_on.objective == pytest.approx(r_off.objective, rel=1e-4, abs=1e-6)
