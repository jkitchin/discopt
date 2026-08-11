"""#844: no-incumbent fallback to the LP-per-node spatial engine (default ON).

A class of pure-integer MINLPs returns NO incumbent from the default NLP-per-node
path while the LP-per-node engine finds one. #826 concluded "no tractable general
primal cracks this family" after testing the NLP-based `feasibility_pump`, `rens`
and `diving`; the LP-based variants inside `lp_spatial_bb` were never tried because
they sat behind an opt-in flag.

Graduation panel — load-gated idle machine, 60 s budget, interleaved off/on, 2 reps
(byte-identical across reps), fallback off -> on:

    tln4         no incumbent -> 19.6  (opt 8.3)     1.00x
    tln5         no incumbent -> 32.8  (opt 10.3)    1.00x
    tln6         no incumbent -> 50.4  (opt 15.3)    1.00x
    ball_mk2_30  no incumbent -> no incumbent        0.65x  (engine declines)
    nvs04/nvs06/nvs09/nvs15   byte-identical, still certified

gains=3, lost_incumbents=0, cert_regressions=0, overshoots=0, unsound=0.

It is a FALLBACK, not a route, and that ordering is the whole safety argument.
Routing in-scope models *instead of* the default path was built and panelled first
(72 instances) and FAILED: 3 gains but 4 certification regressions (nvs04
0.72 -> 6.3e8, nvs06, nvs09, nvs15). The panel split exactly along one line -- every
gain was an instance the default path leaves empty, every regression one it already
certifies -- so as a fallback the gains are kept and the regressions cannot occur.

The wall overshoot that kept this opt-in (1.38-1.67x) was a stale
``model._solve_deadline`` from the primary solve making `IncrementalMcCormickLP`
decline to build, which degraded the engine to the per-node cold build. See
`_lp_spatial_fallback_enabled` and `test_844_lp_spatial_deadline.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling.core import _lp_spatial_fallback_enabled  # noqa: E402


@pytest.fixture
def fb(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("DISCOPT_LP_SPATIAL_FALLBACK", raising=False)
        else:
            monkeypatch.setenv("DISCOPT_LP_SPATIAL_FALLBACK", value)

    return _set


def _pure_integer_min() -> dm.Model:
    """In scope for the engine: all-integer, MINIMIZE. Optimum 6 at x=2, y=0."""
    m = dm.Model("pure_int_min")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(3 * x + 5 * y)
    m.subject_to(x + y >= 2)
    return m


def _mixed() -> dm.Model:
    """Has a continuous variable: out of the fallback's scope while the #860 mixed
    flag is off (its default), in scope when it is on."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=0, ub=5)
    c = m.continuous("c", lb=0, ub=5)
    m.minimize(3 * x + c)
    m.subject_to(x + c >= 2)
    return m


def _pure_continuous() -> dm.Model:
    """Out of the engine's scope under EITHER flag: no integer variable at all."""
    m = dm.Model("pure_cont")
    a = m.continuous("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=5)
    m.minimize(3 * a + b)
    m.subject_to(a + b >= 2)
    return m


def test_flag_defaults_to_on(fb):
    """Default-ON after the graduation panel (3 gains, 0 lost incumbents, 0
    certification regressions, 0 overshoots). ``=0`` remains the opt-out."""
    fb(None)
    assert _lp_spatial_fallback_enabled() is True


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("off", False)])
def test_flag_parses(fb, value, expected):
    fb(value)
    assert _lp_spatial_fallback_enabled() is expected


@pytest.mark.parametrize("flag", ["0", "1"])
def test_solve_with_an_incumbent_is_unchanged(fb, flag):
    """The fallback must not touch a solve that already found an incumbent — this is
    the property that makes the four nvs certification regressions impossible."""
    fb(flag)
    r = _pure_integer_min().solve(time_limit=60)
    assert r.objective == pytest.approx(6.0, abs=1e-6)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent"


@pytest.mark.parametrize("flag", ["0", "1"])
def test_out_of_scope_model_is_unaffected(fb, flag):
    """A model with no integer variable can never reach the engine under either
    flag, so its result and its full time budget are untouched. Optimum 2.0."""
    fb(flag)
    r = _pure_continuous().solve(time_limit=60)
    assert r.objective == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("mixed", ["0", "1"])
def test_mixed_model_result_is_unchanged_by_the_860_flag(fb, monkeypatch, flag, mixed):
    """#860: turning the mixed flag on may change *how* a mixed model is solved (the
    default path hands 35% of its budget to the fallback) but never *what* it
    returns. Optimum is 2.0 at x=0, c=2; the dual bound must never cross it."""
    fb(flag)
    monkeypatch.setenv("DISCOPT_LP_SPATIAL_MIXED", mixed)
    r = _mixed().solve(time_limit=60)
    assert r.objective == pytest.approx(2.0, abs=1e-4)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent"


def test_860_mixed_flag_defaults_to_off_and_gates_the_reserve(monkeypatch):
    """The #860 widening reaches the default path only through this flag, which is off
    until its graduation panel.

    Updated with the flag's scope, not loosened. This test previously asserted
    ``_is_in_scope(m) is True`` with the comment "engine: widened unconditionally",
    because the flag then governed only the fallback's 35% budget reserve while the
    engine gate was always widened. The #860 review moved the engine gate behind the
    same flag — the widening is sound but not net-positive on the default path (see
    ``_lp_spatial_mixed_fallback_enabled``, and ``test_860_mixed_gate_is_opt_in``) —
    so the *default* is now the conservative gate at both entry points. The
    substantive assertion, that ``mixed=False`` declines a mixed model, is unchanged
    and is now also what the default does.
    """
    from discopt._relax.lp_spatial_bb import _is_in_scope
    from discopt.modeling.core import _lp_spatial_mixed_fallback_enabled

    monkeypatch.delenv("DISCOPT_LP_SPATIAL_MIXED", raising=False)
    assert _lp_spatial_mixed_fallback_enabled() is False
    monkeypatch.setenv("DISCOPT_LP_SPATIAL_MIXED", "1")
    assert _lp_spatial_mixed_fallback_enabled() is True

    m = _mixed()
    assert _is_in_scope(m, mixed=True) is True  # capability, on request
    assert _is_in_scope(m, mixed=False) is False  # pre-#860 gate
    assert _is_in_scope(m) is False  # default is now the conservative gate


def test_fallback_never_breaks_a_solve(fb):
    """Even enabled, a fallback failure must surface as a warning and keep the
    default-path result — never raise. (A bare ``except`` here previously turned a
    hard TypeError into an invisible no-op.)"""
    fb("1")
    r = _pure_integer_min().solve(time_limit=60)
    assert r.objective is not None
