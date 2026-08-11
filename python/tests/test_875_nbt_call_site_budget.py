"""#875: every ``tighten_nonlinear_bounds`` call site must carry a budget.

``tighten_nonlinear_bounds`` grew a ``deadline`` parameter, and three of its four
call sites pass one — the root declared-box pass, the periodic/domain pass, and
AMP. ``_apply_nonlinear_tightening_with_status`` did not, so it ran the pass to
completion on every invocation.

Measured on ``watercontamination0202`` (106,711 vars / 107,209 rows), the instance
#875 is about: that unbudgeted site cost ~23 s per call and fired 3x inside a 30 s
``time_limit`` — about 70 s of a 126 s profile, the single largest remaining overrun
after the sparse-linearizer fix, and the reason a ``deadline`` on
``tighten_nonlinear_bounds`` alone did not bound the solve. End to end:

======================================  ==========  ==========
stage                                   T=30        T=60
======================================  ==========  ==========
before #878                             579.3 s     620.8 s
#878 (sparse linearizer + nbt poll)      88.5 s     118.0 s
plus this call site                      47.2 s      76.3 s
======================================  ==========  ==========

This test pins the wiring rather than the wall-clock, so it cannot rot into a
machine-speed assertion: it asserts the deadline actually *arrives* at
``tighten_nonlinear_bounds``, and that it equals the solve's absolute deadline
rather than a fresh per-call fraction (a fraction recomputed per call is how the
convexity classifier's budget used to multiply across model objects, and this
helper runs per node as well as at the root).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as solver_mod


def _tiny_nonlinear_model():
    m = dm.Model("nbt_budget")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x + y)
    m.subject_to(x * y >= 1.0)
    return m


def test_call_site_forwards_the_solve_deadline(monkeypatch):
    """The deadline reaches ``tighten_nonlinear_bounds``, and it is the absolute one."""
    seen: list[float | None] = []
    real = solver_mod._apply_nonlinear_tightening_with_status

    import discopt._relax.nonlinear_bound_tightening as nbt

    orig = nbt.tighten_nonlinear_bounds

    def spy(model, lb, ub, *args, **kwargs):
        seen.append(kwargs.get("deadline", "MISSING"))  # type: ignore[arg-type]
        return orig(model, lb, ub, *args, **kwargs)

    monkeypatch.setattr(nbt, "tighten_nonlinear_bounds", spy)

    m = _tiny_nonlinear_model()
    sentinel = 12345.0
    m._solve_deadline = sentinel
    lb = np.array([0.5, 0.5])
    ub = np.array([3.0, 3.0])
    real(m, lb, ub)

    assert seen, "tighten_nonlinear_bounds was never called — the test is vacuous"
    assert "MISSING" not in seen, (
        "_apply_nonlinear_tightening_with_status called tighten_nonlinear_bounds with no "
        "deadline= argument; this call site is unbudgeted (#875)"
    )
    assert seen[0] == sentinel, (
        f"expected the solve's absolute deadline {sentinel!r}, got {seen[0]!r} — a per-call "
        "fraction here compounds across the per-node invocations"
    )


def test_absent_solve_deadline_leaves_the_pass_unbounded(monkeypatch):
    """No ``time_limit`` must keep today's behavior: an unbounded pass, not a zero budget.

    Guards the direction of the fix: reading a missing attribute as ``0.0`` would
    make every untimed solve skip tightening entirely.
    """
    seen: list[object] = []
    import discopt._relax.nonlinear_bound_tightening as nbt

    orig = nbt.tighten_nonlinear_bounds

    def spy(model, lb, ub, *args, **kwargs):
        seen.append(kwargs.get("deadline", "MISSING"))
        return orig(model, lb, ub, *args, **kwargs)

    monkeypatch.setattr(nbt, "tighten_nonlinear_bounds", spy)

    m = _tiny_nonlinear_model()
    assert not hasattr(m, "_solve_deadline")
    solver_mod._apply_nonlinear_tightening_with_status(
        m, np.array([0.5, 0.5]), np.array([3.0, 3.0])
    )
    assert seen, "vacuous: tighten_nonlinear_bounds never called"
    assert seen[0] is None, f"expected deadline=None for an untimed solve, got {seen[0]!r}"


def test_tightening_still_tightens_under_a_generous_budget():
    """The budget must not cost the tightening it is meant to bound.

    ``x*y >= 1`` on ``[0.5,3]^2`` is tightenable; with a far-future deadline the pass
    must still return a box no wider than it was given (soundness is one-directional
    here: tightening may only shrink).
    """
    m = _tiny_nonlinear_model()
    import time

    m._solve_deadline = time.perf_counter() + 3600.0
    lb = np.array([0.5, 0.5])
    ub = np.array([3.0, 3.0])
    new_lb, new_ub, infeasible = solver_mod._apply_nonlinear_tightening_with_status(m, lb, ub)
    assert not infeasible
    assert np.all(new_lb >= lb - 1e-12), "tightening widened a lower bound"
    assert np.all(new_ub <= ub + 1e-12), "tightening widened an upper bound"
    assert new_lb.shape == lb.shape and new_ub.shape == ub.shape


def test_expired_budget_is_sound_and_does_not_widen():
    """An already-expired deadline must return a box that is still valid, never wider."""
    m = _tiny_nonlinear_model()
    m._solve_deadline = 1.0  # far in the past for perf_counter
    lb = np.array([0.5, 0.5])
    ub = np.array([3.0, 3.0])
    new_lb, new_ub, infeasible = solver_mod._apply_nonlinear_tightening_with_status(m, lb, ub)
    assert not infeasible, "an expired budget must not manufacture an infeasibility"
    assert np.all(new_lb >= lb - 1e-12) and np.all(new_ub <= ub + 1e-12)


@pytest.mark.parametrize("attr", ["_apply_nonlinear_tightening_with_status"])
def test_no_other_unbudgeted_call_site_regresses(attr):
    """Source-level guard: no ``tighten_nonlinear_bounds(`` call in solver.py may omit
    ``deadline``. Cheap, and it catches a new call site added without a budget."""
    import inspect
    import re

    src = inspect.getsource(solver_mod)
    calls = [m.start() for m in re.finditer(r"tighten_nonlinear_bounds\(", src)]
    assert calls, "vacuous: no call sites found in solver.py"
    checked = 0
    for pos in calls:
        window = src[pos : pos + 400]
        # the import line itself is not a call
        if window.startswith("tighten_nonlinear_bounds(") and "deadline" not in window:
            raise AssertionError(
                f"a tighten_nonlinear_bounds call in solver.py has no deadline= within 400 "
                f"chars: ...{window[:120]}..."
            )
        checked += 1
    assert checked == len(calls)
