"""Issue #1422: a DECLINED convex-kernel attempt's dual bound must not be thrown away.

The attempt is granted ``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`` -- the
caller's WHOLE budget for any ``time_limit <= 120`` -- so on a model it declines,
the default path is handed ~0 s, takes its #654 deadline short-circuit, and
reports no bound at all. The kernel had nonetheless PROVED one, over a tree whose
``bound`` is a rigorous minimum over its open nodes, and dropped it on the floor.

Measured on ``ball_mk2_30`` (the instance the issue reports) at an 8 s budget::

    before: status=time_limit  bound=None
    after : status=time_limit  bound=-28.878927  bound_valid=True  source=bnb_tree

and over the 39 convex-kernel-eligible instances of the MINLPLib snapshot at the
same budget, 82% of all kernel wall (152.0 s of 185.4 s) goes to models it
declines, 15 of those 20 carrying a finite discarded bound.

This is NOT a fix for the allocation itself -- the budget is still spent. It is
the reason the allocation stops being spent for *nothing*, and it is viable where
the two allocation fixes are not: the fractional cap was built and rejected under
#911, and abandon-on-no-incumbent was falsified for this issue (the kernel finds
its first incumbent essentially at convergence, so "no incumbent yet" predicts
nothing and cutting on it costs certifications at tight budgets).

The soundness story is in ``keep_declined_bound_enabled``. What this file pins is
the part that is testable without an oracle: only the two outcomes whose ``bound``
is a genuine minimum over OPEN nodes are published at all, the sentinel is never
mistaken for a bound, and the two adoption guards hold.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import pytest
from discopt.modeling.core import SolveResult


def _tiny_model(maximize: bool = False):
    """A model the default path solves to proven optimality in milliseconds.

    ``x + y >= 2`` with both columns bounded below by 0 makes the optimum exactly
    2.0 for a minimize and 8.0 for a maximize, so every assertion below can name a
    number instead of a tolerance band.
    """
    m = dm.Model("declined_bound")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    if maximize:
        m.maximize(x + y)
    else:
        m.minimize(x + y)
    m.subject_to(x + y >= 2.0)
    return m


def _kernel_eligible_model(n: int = 4):
    """A model the convex kernel will actually BUILD A SPEC for.

    The publish site lives inside the attempt, so a probe the kernel declines at
    the spec stage exercises none of it -- ``_tiny_model`` is a pure linear MILP
    and builds no spec at all. This is ``ball_mk2``'s shape (a linear objective
    over integers in [-1,1] under one convex ball row), shrunk until the tree is
    instant; the tree is patched anyway, so only eligibility matters here.
    """
    m = dm.Model("kernel_probe")
    xs = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(-sum(xs))
    m.subject_to(sum(x**2 - 0.995825 * x for x in xs) <= 0.0)
    return m


@pytest.fixture
def declining_attempt(monkeypatch):
    """Make the convex kernel decline while publishing a caller-chosen bound.

    Returns a setter. The kernel itself is not the subject here -- ``Model.solve``
    is -- so the published bound is an *input*, exactly as the real pair behaves:
    ``try_convex_solve`` clears the slot on entry and the decline branch fills it.
    """
    import discopt.solvers._convex_kernel as ck

    state = {"bound": None}

    def _fake_try(model, *, time_limit=3600.0, gap_tolerance=1e-4):
        return None  # declined

    monkeypatch.setattr(ck, "try_convex_solve", _fake_try)
    monkeypatch.setattr(ck, "last_attempt_seconds", lambda: 0.0)
    monkeypatch.setattr(ck, "last_attempt_rust_seconds", lambda: 0.0)
    monkeypatch.setattr(ck, "last_declined_bound", lambda: state["bound"])

    def _publish(b):
        state["bound"] = b

    return _publish


@pytest.fixture
def no_default_bound(monkeypatch):
    """Make the default path prove NOTHING, which is what a declined attempt causes.

    Handed ~0 s by #911's deduction, ``solve_model`` takes its #654 deadline
    short-circuit and returns a result with no incumbent and no bound. Patched
    rather than provoked with a real starved solve so the assertions below are
    deterministic instead of budget-sensitive.
    """
    import discopt.solver as solver

    def _starved(model, **kwargs):
        return SolveResult(status="time_limit", objective=None, x={}, bound=None)

    monkeypatch.setattr(solver, "solve_model", _starved)


# --------------------------------------------------------------------------- #
# The defect itself
# --------------------------------------------------------------------------- #


def test_declined_bound_is_adopted_when_the_default_path_proved_nothing(
    declining_attempt, no_default_bound
):
    """The #1422 defect: a proved bound reaching the caller instead of the floor."""
    declining_attempt(-28.878927048801557)
    res = _tiny_model().solve(time_limit=8.0)

    assert res.bound == pytest.approx(-28.878927048801557), (
        f"the declined attempt's dual bound is missing from the result (bound={res.bound}) "
        "-- the #1422 defect"
    )


def test_the_adopted_bound_carries_a_valid_claim(declining_attempt, no_default_bound):
    """#1244: ``bound`` travels with ``bound_valid``/``bound_source`` or buys nothing.

    A bound installed beside a stale ``bound_valid=False`` is discarded by every
    consumer that checks the flag, so adopting it without the triple would look
    like a fix and be a no-op.
    """
    declining_attempt(-28.878927048801557)
    res = _tiny_model().solve(time_limit=8.0)

    assert res.bound_valid is True
    assert res.bound_source == "bnb_tree"


def test_the_flag_off_arm_adopts_nothing(declining_attempt, no_default_bound, monkeypatch):
    """``DISCOPT_CONVEX_KERNEL_KEEP_BOUND=0`` restores the pre-fix behaviour exactly."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_KEEP_BOUND", "0")
    declining_attempt(-28.878927048801557)
    res = _tiny_model().solve(time_limit=8.0)

    assert res.bound is None
    assert res.bound_valid is False


def test_no_published_bound_leaves_the_result_untouched(declining_attempt, no_default_bound):
    """Most declines publish nothing (5 of 20 on the panel); that path must be inert."""
    declining_attempt(None)
    res = _tiny_model().solve(time_limit=8.0)

    assert res.bound is None
    assert res.bound_valid is False


# --------------------------------------------------------------------------- #
# The two adoption guards
# --------------------------------------------------------------------------- #


def test_a_bound_crossing_the_incumbent_is_refused_loudly(declining_attempt, caplog):
    """Guard 1. A dual bound above a found incumbent means one of the two is unsound.

    Nothing here can tell which, so neither is reported: the default path's result
    stands and the contradiction is logged at ERROR (CLAUDE.md §3 -- refuse loudly
    rather than ship a broken certificate). The default path is REAL here: the
    incumbent has to be one the solve actually found for the guard to mean anything.
    """
    declining_attempt(10.0)  # the true optimum is 2.0
    with caplog.at_level(logging.ERROR, logger="discopt.solver"):
        res = _tiny_model().solve(time_limit=8.0)

    assert res.objective == pytest.approx(2.0)
    assert res.bound is None or res.bound <= 2.0 + 1e-6, (
        f"a dual bound of 10.0 was adopted against an incumbent of {res.objective} -- "
        "that is a broken certificate, not a tighter bound"
    )
    assert any("DISCARDING" in r.getMessage() for r in caplog.records), (
        "the contradiction was swallowed; it must be reported at ERROR"
    )


def test_a_looser_bound_never_displaces_a_tighter_one(declining_attempt):
    """Guard 2. Adoption is a merge, not an overwrite.

    The default path proves 2.0 on this model. A declined bound of 1.0 is valid and
    useless; installing it would make the reported certificate strictly worse.
    """
    declining_attempt(1.0)
    res = _tiny_model().solve(time_limit=8.0)

    assert res.bound == pytest.approx(2.0), (
        f"a looser bound (1.0) displaced the default path's 2.0 (bound={res.bound})"
    )


def test_the_merge_is_sense_aware_for_a_maximize_model(declining_attempt, no_default_bound):
    """#860's lesson: on a maximize the dual bound is an UPPER bound.

    An unconditional ``max`` stays SOUND there -- both are valid upper bounds -- but
    keeps the LOOSER one, so sense-blindness is a silent quality regression rather
    than a crash. 12.0 is a valid (weak) upper bound on this model's optimum of 8.0.
    """
    declining_attempt(12.0)
    res = _tiny_model(maximize=True).solve(time_limit=8.0)

    assert res.bound == pytest.approx(12.0)
    assert res.bound_valid is True


def test_a_looser_upper_bound_never_displaces_a_tighter_one_on_a_maximize(declining_attempt):
    """The maximize half of guard 2: larger is LOOSER here, so 12.0 must lose to 8.0."""
    declining_attempt(12.0)
    res = _tiny_model(maximize=True).solve(time_limit=8.0)

    assert res.objective == pytest.approx(8.0)
    assert res.bound == pytest.approx(8.0), (
        f"a looser upper bound (12.0) displaced the default path's 8.0 (bound={res.bound})"
    )


# --------------------------------------------------------------------------- #
# The publish site: which tree outcomes may publish at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "status, publishes",
    [
        ("time_limit", True),
        ("node_limit", True),
        # ``exhausted`` is the tree giving up on numerical non-closure. Its ``bound``
        # is not a clean minimum over open nodes, so it is deliberately excluded.
        ("exhausted", False),
        ("infeasible", False),
        # A certified-but-unverifiable incumbent (#779 declining the cross-check) is
        # excluded for the same reason: the attempt is not trusted, so neither is it.
        ("optimal", False),
    ],
)
def test_only_open_frontier_outcomes_publish_a_bound(monkeypatch, status, publishes):
    """The publish site is a whitelist, not a fallthrough."""
    import discopt.solvers._convex_kernel as ck

    _run_attempt_with(monkeypatch, ck, status=status, bound=-3.5)
    got = ck.last_declined_bound()
    if publishes:
        assert got == pytest.approx(-3.5), f"status={status!r} should publish its bound"
    else:
        assert got is None, f"status={status!r} published {got}; it must not"


def test_the_no_bound_sentinel_is_never_mistaken_for_a_bound(monkeypatch):
    """``BOUND_INF`` (1e19) is the solvers' shared "no bound yet" marker, not ``inf``.

    The kernel reports the sentinel rather than an infinity, so a plain
    ``isfinite`` check would adopt "nothing proved" as if it were a proof -- and
    ``1e19`` as a lower bound certifies essentially any model as optimal.
    """
    import discopt.solvers._convex_kernel as ck
    from discopt.solvers._gap import BOUND_INF

    for sentinel in (-BOUND_INF, -1e20, float("-inf"), float("nan")):
        _run_attempt_with(monkeypatch, ck, status="time_limit", bound=sentinel)
        assert ck.last_declined_bound() is None, f"{sentinel!r} was adopted as a bound"


def test_the_slot_is_cleared_before_the_flag_check(monkeypatch):
    """A later solve on the same thread must never read an earlier attempt's bound.

    The slot is a thread-local, so a stale value would attach a bound proved for
    one model to a different one -- an unsound bound produced by bookkeeping alone.
    """
    import discopt.solvers._convex_kernel as ck

    _run_attempt_with(monkeypatch, ck, status="time_limit", bound=-3.5)
    assert ck.last_declined_bound() == pytest.approx(-3.5)

    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    ck.try_convex_solve(_kernel_eligible_model(), time_limit=1.0)
    assert ck.last_declined_bound() is None, "a flag-off solve read a stale bound"


def _run_attempt_with(monkeypatch, ck, *, status, bound):
    """Drive one convex-kernel attempt whose native tree returns ``status``/``bound``.

    Patches the tree, not the caller: everything above it -- the spec build, the
    whitelist, the sentinel screen, the slot -- is the code under test.
    """
    monkeypatch.delenv("DISCOPT_CONVEX_KERNEL", raising=False)
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_KEEP_BOUND", "1")

    def _fake_tree(spec, **kwargs):
        return {
            "status": status,
            "bound": bound,
            "incumbent": None,
            "incumbent_x": [],
            "node_count": 7,
            "first_incumbent_secs": None,
            "first_incumbent_node": None,
        }

    monkeypatch.setattr(ck, "solve_convex_tree", _fake_tree)
    m = _kernel_eligible_model()
    assert ck.build_convex_spec(m) is not None, (
        "the probe model is no longer convex-kernel eligible, so this test would exercise nothing"
    )
    ck.try_convex_solve(m, time_limit=1.0)


# --------------------------------------------------------------------------- #
# End to end, with the real kernel in the loop
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_a_real_declining_attempt_hands_back_its_bound():
    """The shape #1422 reported, unfaked.

    ``ball_mk2_30``'s MINLPLib form -- 30 integers in [-1,1], a LINEAR objective and
    the single CONVEX ball row -- is convex-kernel eligible and cannot be certified
    in 8 s, so the attempt consumes the budget and declines. Reconstructed rather
    than vendored: the named instance is a gate probe, the shape is the subject
    (CLAUDE.md §2). Its reference optimum is 0.0, so the bound must not exceed it.
    """
    n = 30
    m = dm.Model("ball_mk2_30_shape")
    xs = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(-sum(xs))
    m.subject_to(sum(x**2 - 0.995825 * x for x in xs) <= 0.0)

    from discopt.solvers._convex_kernel import build_convex_spec, last_attempt_seconds

    assert build_convex_spec(m) is not None, (
        "the reconstruction is no longer convex-kernel eligible -- repro drifted"
    )

    res = m.solve(time_limit=8.0)
    attempt = last_attempt_seconds()

    assert attempt > 1.0, (
        f"the convex-kernel attempt only ran {attempt}s; this test needs an attempt "
        "that actually consumes the budget to have anything to check"
    )
    assert res.bound is not None, (
        "the declining attempt spent the whole budget and handed back no bound -- the #1422 defect"
    )
    assert res.bound <= 0.0 + 1e-6, (
        f"bound={res.bound} exceeds the reference optimum 0.0 -- an UNSOUND bound, "
        "which is the one outcome this change may never produce"
    )
    assert res.bound_valid is True
