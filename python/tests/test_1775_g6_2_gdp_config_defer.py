"""#1775 G6.2: ``DISCOPT_GDP_CONFIG_DEFER`` splits the GDP-config constructor's grant.

The root disjunct-configuration constructor always spent its full 15 s grant on
the GDP parity panel, finding its final point in 0.03-5 s. With the flag on, the
root call gets ``_GDP_CONFIG_DEFER_ROOT_S`` and the rest is spent in one more call
from the same seed once the search has run for as long as that rest. Default ON
since its §5 panel; ``=0`` opts out.
(BC: it changes which incumbents are found, never a bound).
"""

from __future__ import annotations

import time
import warnings

import discopt.modeling as dm
import discopt.solver as SV
import numpy as np
import pytest
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke


def test_default_on_and_opt_out(monkeypatch):
    """Graduated: unset is ON, ``=0`` (and its spellings) opt out, empty is not off."""
    monkeypatch.delenv("DISCOPT_GDP_CONFIG_DEFER", raising=False)
    assert SV._gdp_config_defer_enabled()
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "")
    assert SV._gdp_config_defer_enabled()
    for off in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", off)
        assert not SV._gdp_config_defer_enabled(), off


def test_opt_out_and_root_until(monkeypatch):
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "0")
    assert not SV._gdp_config_defer_enabled()
    assert SV._gdp_config_root_until(100.0, 50.0, 45.0) == 100.0
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    assert SV._gdp_config_root_until(100.0, 50.0, 45.0) == 50.0 + SV._GDP_CONFIG_DEFER_ROOT_S
    assert SV._gdp_config_root_until(50.5, 50.0, 49.9) == 50.5  # never past the full grant


def test_a_long_root_spends_the_whole_grant_now(monkeypatch):
    """Ski rental counts the time already spent: the gdp_col case (the solve ran
    ~43 s before the constructor, its 7 s rest would never have been spent)."""
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    assert SV._gdp_config_root_until(108.6, 100.0, 57.0) == 108.6
    assert SV._gdp_config_root_until(108.6, 100.0, 91.4) == 108.6  # equal: buy now
    assert SV._gdp_config_root_until(108.6, 100.0, 92.0) == 101.5


def test_hold_back(monkeypatch):
    seed = np.array([1.0, 2.0])
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "0")
    assert SV._gdp_config_hold_back(seed, 115.0, 101.5, 101.6) is None  # opted out
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    held = SV._gdp_config_hold_back(seed, 115.0, 101.5, 101.6)
    assert held is not None and held["remaining"] == pytest.approx(13.4)
    assert held["ready_at"] == pytest.approx(101.6 + 13.4)  # ski rental
    seed[0] = 9.0
    assert held["seed"][0] == 1.0  # a copy
    # Stopped before its deadline: it ran out of plans, nothing to repeat.
    assert SV._gdp_config_hold_back(seed, 115.0, 101.5, 101.0) is None
    assert SV._gdp_config_hold_back(seed, 115.0, 101.5, 116.0) is None  # nothing left


def test_due():
    held = {"ready_at": 10.0}
    assert not SV._gdp_config_deferred_due(None, 3, 99.0)
    assert not SV._gdp_config_deferred_due(held, 0, 99.0)  # never at the root
    assert not SV._gdp_config_deferred_due(held, 3, 9.9)
    assert SV._gdp_config_deferred_due(held, 3, 10.0)


def _spatial_gdp():
    """Nonconvex: two disjuncts, a bilinear objective, needs branching."""
    m = dm.Model("defer_spatial")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 2], [x + y >= 5, x <= 3.5]], name="d")
    m.subject_to(x * y >= 1)
    m.minimize(-x * y + 0.3 * x**2 - y)
    return m


def _convex_gdp():
    """Convex with a nonlinear row: routes to NLP-B&B. The target lies between the
    disjuncts, so the root relaxation is fractional and the tree branches."""
    m = dm.Model("defer_convex")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 1], [x + y >= 3], [x - y >= 2]], name="d")
    m.subject_to(x**2 + y**2 <= 12)
    m.minimize((x - 1.6) ** 2 + (y - 0.4) ** 2)
    return m


@pytest.mark.parametrize("build", [_spatial_gdp, _convex_gdp])
def test_the_deferred_call_runs_with_the_rest_of_the_grant(build, monkeypatch):
    """Both routes: the root call is short, a later iteration spends the rest."""
    calls: list[tuple[float, float]] = []
    real = PH.one_hot_config_subnlp

    def spy(*a, **k):
        t = time.perf_counter()
        calls.append((t, k["deadline"] - t))
        return real(*a, **k)

    monkeypatch.setattr(PH, "one_hot_config_subnlp", spy)
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    # A zero root share makes the root call stop on its deadline, so the rest is
    # held back; making it due at iteration 1 tests the call, not ski-rental timing.
    monkeypatch.setattr(SV, "_GDP_CONFIG_DEFER_ROOT_S", 0.0)
    monkeypatch.setattr(SV, "_gdp_config_deferred_due", lambda h, it, now: h is not None and it > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = build().solve(time_limit=40)
    assert r.status == "optimal"
    assert len(calls) == 2, calls
    root_grant, rest = calls[0][1], calls[1][1]
    assert root_grant <= 0.01
    assert rest > 1.0  # 0.15 x 40 s, less the root, capped by the solve deadline


@pytest.mark.parametrize("build", [_spatial_gdp, _convex_gdp])
def test_opt_out_is_one_full_grant(build, monkeypatch):
    calls: list[float] = []
    real = PH.one_hot_config_subnlp

    def spy(*a, **k):
        calls.append(k["deadline"] - time.perf_counter())
        return real(*a, **k)

    monkeypatch.setattr(PH, "one_hot_config_subnlp", spy)
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build().solve(time_limit=40)
    assert len(calls) == 1 and calls[0] > 5.0  # min(0.15 x 40, 15) = 6 s


def test_deterministic_turns_the_split_off(monkeypatch):
    """The held-back call fires on a clock (#912), so ``deterministic`` keeps the
    single grant whatever the variable says."""
    from discopt.solver_tuning import SolverTuning

    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    assert SV._gdp_config_defer_enabled()
    monkeypatch.setattr(SV, "_tuning", lambda: SolverTuning(deterministic=True))
    assert not SV._gdp_config_defer_enabled()
    assert SV._gdp_config_root_until(100.0, 50.0, 45.0) == 100.0
