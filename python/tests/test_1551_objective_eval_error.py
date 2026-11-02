"""#1551: no certificate finer than the objective's own float resolution.

``min 6y^2 + By + C`` with ``y`` integer in ``[c, c+3]``, ``B = -(12c+12)`` and
``C = 6c^2+12c`` is the shifted quadratic of #1543 written out in expanded form.
At ``c = 2345678.9`` each term is ~3.3e13 (ulp ~4e-3), so float64 cannot
evaluate the objective to better than ~1e-2 anywhere in the box. The OA
auto-route returned ``objective = bound = -5.9375`` as ``gap_certified=True``
while the exact optimum of the function as written is ``-5.93832`` -- a lower
bound 8.2e-4 above the optimum, past the 5.9e-4 tolerance it was certified at.

The guard measures the objective's evaluation error at the incumbent as the width
of an outward-rounded interval enclosure, and withdraws a certificate (and its
bound) whose pair no longer closes the requested gap once widened by that error.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.solver import (
    _objective_evaluation_error,
    _withhold_unresolved_objective_certificate,
    solve_model,
)

BAD_C = 2345678.9


def _expanded(c: float, sense: str = "min"):
    B = -(12 * c + 12)
    C = 6 * c * c + 12 * c
    m = dm.Model("exp")
    y = m.integer("y", lb=c, ub=c + 3)
    f = 6 * y * y + B * y + C
    if sense == "min":
        m.minimize(f)
    else:
        m.maximize(-f)
    opt = min(F(6) * k * k + F(B) * k + F(C) for k in range(math.ceil(c), math.floor(c + 3) + 1))
    return m, y, opt


@pytest.mark.parametrize("sense", ["min", "max"])
def test_unresolvable_objective_is_not_certified_when_recentred(sense):
    """Default path (recentring ON). The recentred objective is collected exactly
    at the solve door (#2097: its shape is unknown to the cheap inference, and it
    used to be left as written), so the inner solve no longer certifies the noisy
    -5.9375; it returns a bound computed from exact coefficients. What #1551
    requires still holds: no certificate, and any bound published is valid
    against the exact optimum (-5.93832; the false one was -5.9375, above it)."""
    m, _, opt = _expanded(BAD_C, sense)
    r = m.solve(time_limit=30)
    assert r.x is not None and r.objective is not None
    assert not r.gap_certified
    assert r.status != "optimal"
    if r.bound is not None:
        # ``max(-f)`` reports an upper bound on -f.
        assert F(r.bound) <= opt if sense == "min" else F(r.bound) >= -opt


@pytest.mark.parametrize("sense", ["min", "max"])
def test_unresolvable_objective_is_not_certified(sense, monkeypatch):
    # Recentring OFF: the solve sees the objective as written and the guard is what
    # withholds the certificate -- the mechanism this test pins.
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")
    m, _, opt = _expanded(BAD_C, sense)
    r = m.solve(time_limit=30)
    assert r.x is not None and r.objective is not None
    assert not r.gap_certified
    assert r.status != "optimal"
    assert r.bound is None and not r.bound_valid
    stats = r.solver_stats or {}
    assert stats.get("certificate/objective_unresolved") == 1.0
    # The measured error is what made the certificate unsupportable.
    assert stats["certificate/objective_eval_error"] > 1e-4 * abs(float(opt))


def test_direct_solve_model_call_is_guarded():
    """The ``solve_model`` wrapper applies the guard without ``Model.solve``."""
    m, _, _ = _expanded(BAD_C)
    r = solve_model(m, time_limit=30)
    assert not r.gap_certified
    assert r.bound is None


def test_enclosure_contains_the_exact_value():
    """The measured error is a rigorous enclosure, checked in exact arithmetic."""
    m, y, _ = _expanded(BAD_C)
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval

    checked = 0
    for k in range(math.ceil(BAD_C), math.floor(BAD_C + 3) + 1):
        B = -(12 * BAD_C + 12)
        C = 6 * BAD_C * BAD_C + 12 * BAD_C
        exact = F(6) * k * k + F(B) * k + F(C)
        enc = evaluate_interval(m._objective.expression, m, {y: Interval.point(float(k))})
        assert F(float(enc.lo)) <= exact <= F(float(enc.hi))
        err, reason = _objective_evaluation_error(m, {"y": np.array(float(k))})
        assert reason == "" and err == pytest.approx(float(enc.hi - enc.lo))
        checked += 1
    assert checked == 3


def test_well_scaled_model_keeps_its_certificate():
    """The same function at a small offset resolves to the ulp and stays certified."""
    m, _, opt = _expanded(0.9)
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert r.bound is not None and F(r.bound) <= opt + F(1e-6)
    stats = r.solver_stats or {}
    assert stats["certificate/objective_eval_error"] < 1e-12
    assert "certificate/objective_unresolved" not in stats


def _certified(obj: float, bound: float, y: float) -> SolveResult:
    return SolveResult(
        status="optimal",
        objective=obj,
        bound=bound,
        gap=0.0,
        x={"y": np.array(y)},
        gap_certified=True,
    )


def test_guard_unit():
    m, _, _ = _expanded(BAD_C)
    y0 = float(math.ceil(BAD_C) + 1)

    bad = _certified(-5.9375, -5.9375, y0)
    _withhold_unresolved_objective_certificate(bad, m, 1e-4, 1e-6)
    assert not bad.gap_certified and bad.status == "feasible" and bad.bound is None

    # A loose caller tolerance absorbs the same error: left alone.
    loose = _certified(-5.9375, -5.9375, y0)
    _withhold_unresolved_objective_certificate(loose, m, 1e-1, 1e-1)
    assert loose.gap_certified and loose.bound == -5.9375

    # Uncertified results are never touched (downgrade-only).
    unc = _certified(-5.9375, -5.9375, y0)
    unc.gap_certified = False
    unc.status = "feasible"
    _withhold_unresolved_objective_certificate(unc, m, 1e-4, 1e-6)
    assert unc.bound == -5.9375


def test_unmeasurable_point_is_not_measured():
    m, _, _ = _expanded(BAD_C)
    assert _objective_evaluation_error(m, {}) == (None, "the point carries no value for 'y'")
    assert _objective_evaluation_error(dm.Model("empty"), {}) == (None, "no objective")


def test_not_applicable_keeps_the_certificate_and_says_why():
    """A model the check does not apply to keeps its certificate, visibly."""
    m, _, _ = _expanded(BAD_C)
    r = _certified(-5.9375, -5.9375, float(math.ceil(BAD_C) + 1))
    r.x = {}
    _withhold_unresolved_objective_certificate(r, m, 1e-4, 1e-6)
    assert r.gap_certified and r.bound == -5.9375
    assert r.solver_stats["certificate/objective_eval_error_skipped"] == (
        "the point carries no value for 'y'"
    )


def test_a_failed_measurement_fails_closed(monkeypatch):
    """A measurement that raises withdraws the certificate (review on #1556)."""
    import discopt._relax.convexity.interval_eval as ie

    def boom(*a, **k):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(ie, "evaluate_interval", boom)
    m, _, _ = _expanded(0.9)
    r = _certified(1.0, 1.0, 1.0)
    r.root_bound, r.root_gap = 1.0, 0.0
    _withhold_unresolved_objective_certificate(r, m, 1e-4, 1e-6)
    assert not r.gap_certified and r.status == "feasible"
    assert r.bound is None and not r.bound_valid
    assert r.root_bound is None and r.root_gap is None
    assert r.solver_stats["certificate/objective_eval_error_skipped"].startswith("RecursionError")
    assert r.solver_stats["certificate/objective_unresolved"] == 1.0


@pytest.mark.parametrize("n_terms", [2000])
def test_loop_built_objective_is_measured(n_terms, monkeypatch):
    """``f = f + term`` in a loop is as deep as it has terms; the guard must still
    measure it. On the first revision of #1556 this raised ``RecursionError``, the
    guard skipped, and the #1551 false certificate came back (bound +8.17e-4)."""
    # Pinned OFF: this tests the #1551 guard on the route that reaches a
    # certificate. Recentring (default ON, #1537) moves ``y`` and rewrites
    # ``(6*y)*y`` as ``(6*z + c1)*(z + c2)``, which the quadratic extractor does not
    # expand, so the recentred objective is not proven convex: the solve takes
    # spatial B&B, stops uncertified at the node limit with a valid bound, and never
    # reaches the certificate this guard checks. No false certificate either way,
    # and no certificate in either arm: the ON arm is measured by the next test.
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")
    m, _, opt = _expanded(BAD_C)
    f = m._objective.expression
    for i in range(n_terms):
        f = f + 1e-3 * m.continuous(f"z{i}", lb=0, ub=1)
    m.minimize(f)
    r = m.solve(time_limit=60)
    stats = r.solver_stats or {}
    assert "certificate/objective_eval_error_skipped" not in stats
    assert not r.gap_certified
    assert r.bound is None or F(r.bound) <= opt
    assert stats.get("certificate/objective_unresolved") == 1.0


@pytest.mark.slow
def test_loop_built_objective_under_recentring_is_not_certified_either(monkeypatch):
    """The recentred arm of the test above, measured rather than pinned (#1537).

    The model is uncertifiable by construction (#1551: the objective cannot be
    evaluated to its own gap tolerance), and the OFF arm above asserts it is NOT
    certified. Under recentring it reaches the same verdict by a different route:
    ``(6*z + c1)*(z + c2)`` is not recognised as quadratic, so the solve takes
    spatial B&B and stops at a limit (measured: the 100000-node limit at ~22 s).
    That is a cost in time, not a lost certificate. The bound it keeps must still
    be valid."""
    monkeypatch.setenv("DISCOPT_RECENTRE", "1")
    m, _, opt = _expanded(BAD_C)
    f = m._objective.expression
    for i in range(2000):
        f = f + 1e-3 * m.continuous(f"z{i}", lb=0, ub=1)
    m.minimize(f)
    r = m.solve(time_limit=60)
    stats = r.solver_stats or {}
    assert stats.get("recentre/variables_moved", 0) > 0
    assert not r.gap_certified
    assert r.bound is None or F(r.bound) <= opt


@pytest.mark.parametrize("recentre", ["0", "1"])
def test_convex_kernel_return_is_guarded(monkeypatch, recentre):
    """The convex-kernel fast path returns from ``Model.solve`` before
    ``solve_model``; its result must pass the guard too.

    Both recentring arms (#1537). With it ON (the default) ``y`` is moved, so the
    kernel is handed the recentred model, not ``m``, and the fake must answer in
    that model's coordinates (``y - shift``); the guard has to fire on the
    mapped-back result all the same."""
    import discopt.solvers._convex_kernel as ck
    from discopt.modeling import _recentre

    monkeypatch.setenv("DISCOPT_RECENTRE", recentre)
    m, y, _ = _expanded(BAD_C)
    y0 = float(math.ceil(BAD_C) + 1)
    shift = 0.0
    if recentre == "1":
        shift = float(_recentre.plan_shifts(m, _recentre.recentre_ratio())[id(y)])
        assert shift != 0.0
    calls = []

    def fake(model, **kwargs):
        calls.append(model)
        return _certified(-5.9375, -5.9375, y0 - shift)

    monkeypatch.setattr(ck, "try_convex_solve", fake)
    r = m.solve(time_limit=30)
    assert len(calls) == 1
    assert (calls[0] is m) == (recentre == "0")
    assert not r.gap_certified and r.bound is None
    assert r.solver_stats["certificate/objective_unresolved"] == 1.0
