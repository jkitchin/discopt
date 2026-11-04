"""A raising GDP disjunct-configuration constructor is reported at WARNING, and the solve goes on.

Every call site of ``one_hot_config_subnlp`` catches its exceptions, which is right:
it is a primal heuristic, and a crash in it must not kill a solve whose bound is
still certifiable. But the catch logged at DEBUG, so a constructor that raised read
exactly like one that searched and found nothing (CLAUDE.md §7). It was found when an
instrument's own exception, raised from a hook at the NLP-B&B root call, was
swallowed and the solve silently ran on for its whole time limit.

Each test injects a raising constructor at one call site and asserts both halves:
the WARNING carries the traceback, and the solve still returns its certified optimum.
"""

from __future__ import annotations

import logging
import warnings

import discopt.modeling as dm
import discopt.solver as SV
import pytest
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke


class _Injected(RuntimeError):
    pass


def _spatial_gdp():
    """Nonconvex: two disjuncts, a bilinear objective, needs branching."""
    m = dm.Model("raise_spatial")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 2], [x + y >= 5, x <= 3.5]], name="d")
    m.subject_to(x * y >= 1)
    m.minimize(-x * y + 0.3 * x**2 - y)
    return m


def _convex_gdp():
    """Convex with a nonlinear row: routes to NLP-B&B, and the root is fractional."""
    m = dm.Model("raise_convex")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 1], [x + y >= 3], [x - y >= 2]], name="d")
    m.subject_to(x**2 + y**2 <= 12)
    m.minimize((x - 1.6) ** 2 + (y - 0.4) ** 2)
    return m


def _solve(build, monkeypatch, caplog, raise_on_call: int):
    """Solve with the constructor raising on its ``raise_on_call``-th call (1-based)."""
    calls = {"n": 0}
    real = PH.one_hot_config_subnlp

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == raise_on_call:
            raise _Injected("injected constructor failure")
        return real(*a, **k)

    monkeypatch.setattr(PH, "one_hot_config_subnlp", flaky)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = build().solve(time_limit=40)
    assert calls["n"] >= raise_on_call, f"the injected failure never fired ({calls['n']} calls)"
    hits = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "GDP config subNLP" in rec.getMessage()
        and rec.exc_info is not None
        and rec.exc_info[0] is _Injected
    ]
    return r, hits


def _assert_reported_and_solved(r, hits, site):
    assert len(hits) == 1, f"expected one WARNING with the traceback, got {len(hits)}"
    assert site in hits[0].getMessage()
    assert r.status == "optimal" and r.gap_certified


@pytest.mark.parametrize(
    "build, site",
    [(_spatial_gdp, "spatial root"), (_convex_gdp, "nlp-bb root")],
)
def test_root_call_raise_is_a_warning(build, site, monkeypatch, caplog):
    # The Python trees are what own these two call sites (#2015 routes the spatial
    # model to the native kernel by default; that site is tested below).
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    r, hits = _solve(build, monkeypatch, caplog, raise_on_call=1)
    _assert_reported_and_solved(r, hits, site)


@pytest.mark.parametrize(
    "build, site",
    [(_spatial_gdp, "spatial deferred"), (_convex_gdp, "nlp-bb deferred")],
)
def test_deferred_call_raise_is_a_warning(build, site, monkeypatch, caplog):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_DEFER", "1")
    # A zero root share holds the whole grant back, and making it due at iteration
    # 1 reaches the deferred call without depending on wall-clock timing.
    monkeypatch.setattr(SV, "_GDP_CONFIG_DEFER_ROOT_S", 0.0)
    monkeypatch.setattr(SV, "_gdp_config_deferred_due", lambda h, it, now: h is not None and it > 0)
    r, hits = _solve(build, monkeypatch, caplog, raise_on_call=2)
    _assert_reported_and_solved(r, hits, site)


def test_native_seed_raise_is_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")
    # The native seed reaches the constructor only above its enumeration cap (the
    # GDPlib big-M models carry 20-138 free binaries against a cap of 10); this
    # model has 2, so lower the cap to take that branch.
    monkeypatch.setattr(SV, "_NATIVE_SEED_MAX_FREE_INTEGERS", 0)
    r, hits = _solve(_spatial_gdp, monkeypatch, caplog, raise_on_call=1)
    _assert_reported_and_solved(r, hits, "native seed")
