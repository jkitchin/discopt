"""Issue #1422: ``SolveResult.wall_time`` must include the convex-kernel attempt.

#911 made a DECLINED convex-kernel attempt's wall count against the budget handed
to ``solve_model``; it never made that wall count in what the solve *reports*. The
two halves have to move together, because the deduction is exactly what makes
``solve_model``'s clock start *after* the attempt is over: on an
eligible-but-uncertifiable model the attempt is granted
``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`` -- i.e. the whole budget for any
``time_limit <= 120`` -- so ``solve_model`` gets ~0 s, takes its #654 deadline
short-circuit, and times a wall from an anchor set after the budget was gone.

Measured on ``clay0303hfsg`` (vendored corpus) at ``time_limit=8`` before the fix::

    status=time_limit  nodes=0  reported_wall=0.061 s  TRUE wall=8.528 s
    convex-kernel attempt=8.051 s        unaccounted=8.466 s

-- an 8.5-second solve reported as 61 milliseconds. #1422 filed the same signature
on ``ball_mk2_30`` (``status='time_limit' nodes=0 t=0.010s`` against an 8 s budget)
as a *false status*; it is not. The status is honest -- the budget really was spent
-- and the WALL was the lie. Every consumer that buckets by time reads this field,
including this repo's own benchmark runner (median time over solved instances,
total wall over all of them).
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import pytest
from discopt.modeling.core import SolveResult


def _tiny_model():
    """A model whose solve is fast and route-agnostic; the attempt is faked."""
    m = dm.Model("wall_accounting")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    m.minimize(x + y)
    m.subject_to(x + y >= 2.0)
    return m


_FAKE_ATTEMPT_S = 0.40
_FAKE_RUST_S = 0.25


@pytest.fixture
def _declining_attempt(monkeypatch):
    """Stand in for a convex-kernel attempt that burns wall and then DECLINES.

    Patches the attempt *and* its published clocks together, which is how the real
    pair behaves: ``try_convex_solve`` sets both in a ``finally`` covering every
    exit. What is under test is the caller -- whether ``Model.solve`` bills the
    published wall to the result -- so the clock is an input here, not the subject.
    """
    import discopt.solvers._convex_kernel as ck

    def _fake_try(model, *, time_limit=3600.0, gap_tolerance=1e-4):
        time.sleep(_FAKE_ATTEMPT_S)
        return None  # declined: the caller keeps the default path

    monkeypatch.setattr(ck, "try_convex_solve", _fake_try)
    monkeypatch.setattr(ck, "last_attempt_seconds", lambda: _FAKE_ATTEMPT_S)
    monkeypatch.setattr(ck, "last_attempt_rust_seconds", lambda: _FAKE_RUST_S)
    return ck


def test_declined_attempt_is_billed_to_wall_time(_declining_attempt):
    """The attempt's wall appears in ``wall_time`` -- the #1422 defect itself.

    Asserted against the attempt, not against the caller's own stopwatch: on a
    model this small the first solve in a process also pays a one-off evaluator /
    JIT build that ``Model.solve`` performs OUTSIDE ``solve_model``'s clock and
    that no budget accounts for either. Measured, so it is not left as a suspicion:
    ~0.37 s on this model's first solve, and 0.007-0.032 s per instance across the
    66-instance vendored corpus at an 8 s budget -- i.e. two orders of magnitude
    below the convex-kernel gap this file exists for, and a separate question from
    it (that work is not deducted from the budget, so reporting it and budgeting it
    do not move together the way #911 and this fix do). The end-to-end
    wall-vs-report check lives in the slow test below, where the budget is 8 s and
    the fixed overhead is noise.
    """
    res = _tiny_model().solve(time_limit=20.0)

    assert res.wall_time >= _FAKE_ATTEMPT_S, (
        f"the {_FAKE_ATTEMPT_S}s convex-kernel attempt is missing from the reported "
        f"wall ({res.wall_time}s) -- the #1422 under-report"
    )


def test_the_rust_python_partition_survives_the_charge(_declining_attempt):
    """``rust_time`` and ``python_time`` are documented as partitioning ``wall_time``.

    Billing the attempt to ``wall_time`` alone would break that invariant by the
    whole attempt, so the charge is split: the native tree is Rust, the spec build
    / convexity classification and the #779 incumbent verification are Python.
    """
    res = _tiny_model().solve(time_limit=20.0)
    assert res.rust_time + res.python_time == pytest.approx(res.wall_time, rel=1e-6, abs=1e-6)
    assert res.rust_time >= _FAKE_RUST_S
    assert res.python_time >= _FAKE_ATTEMPT_S - _FAKE_RUST_S


def test_a_flag_off_solve_is_billed_exactly_nothing(monkeypatch):
    """``DISCOPT_CONVEX_KERNEL=0`` must leave the reported wall bit-unchanged.

    ``last_attempt_seconds`` is documented as *exactly* 0.0 when the flag is off,
    and the charge is gated on ``> 0.0``, so a flag-off solve adds a literal zero.
    Guards the same property #911 relies on for the budget deduction.
    """
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    t0 = time.perf_counter()
    res = _tiny_model().solve(time_limit=20.0)
    true_wall = time.perf_counter() - t0
    assert isinstance(res, SolveResult)
    assert res.wall_time <= true_wall + 1e-3


@pytest.mark.slow
def test_a_real_declining_attempt_is_billed():
    """End-to-end on the shape #1422 reported, with the real kernel in the loop.

    ``ball_mk2_30``'s MINLPLib form -- 30 integers in [-1,1], a LINEAR objective and
    the single CONVEX ball row -- is convex-kernel eligible (the ``x**2`` form the
    ``.nl`` ``^`` opcode produces builds a spec; the algebraically identical
    ``x*x`` form does not), and at an 8 s budget the kernel cannot certify it, so
    the attempt consumes the budget and declines. Reconstructed rather than
    vendored: the named instance is a gate probe, the shape is the subject
    (CLAUDE.md §2).
    """
    n = 30
    m = dm.Model("ball_mk2_30_shape")
    xs = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(-sum(xs))
    m.subject_to(sum(x**2 - 0.995825 * x for x in xs) <= 0.0)

    from discopt.solvers._convex_kernel import build_convex_spec, last_attempt_seconds

    assert build_convex_spec(m) is not None, (
        "the reconstruction is no longer convex-kernel eligible, so this test would "
        "measure nothing -- repro drifted"
    )

    t0 = time.perf_counter()
    res = m.solve(time_limit=8.0)
    true_wall = time.perf_counter() - t0
    attempt = last_attempt_seconds()

    assert attempt > 1.0, (
        f"the convex-kernel attempt only ran {attempt}s; this test needs an attempt "
        "that actually consumes the budget to have anything to check"
    )
    assert res.wall_time >= 0.8 * true_wall, (
        f"reported wall {res.wall_time}s against a true wall of {true_wall}s with a "
        f"{attempt}s convex-kernel attempt -- the #1422 under-report is back"
    )
