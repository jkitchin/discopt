"""#1152 — what ``solve(time_limit=T)`` owes, when root setup is the expensive part.

Two slow-tier tests encoded contracts that read as **contradictory**: one required
the solve to return inside ``T`` (#875's 1.25x threshold), the other required the
dual bound produced by a long uninterruptible root operation to survive a short
budget (#654's ``sonet23v4`` guard). Satisfying either as written appeared to break
the other, because the only way to stop overrunning was to decline the operation
that produces the bound.

They are not contradictory. They were two readings of ONE defect: a root-setup
phase polls a deadline between its LP solves but **not inside its relaxation
build**, and it clamps its own budget to *all* of what is left. So the phase both
runs past ``T`` and spends the slice the last-ditch bound producer needs. Measured
on ``casctanks`` at ``time_limit=5`` (in-repo corpus, before the fix):

    root OBBT enters its round build at t=4.39 s with 0.61 s left
    and spends 1.85 s there                       -> wall 6.4 s (1.29x)
    the #654 short-circuit then reports
    "fallback grant 0.000s of the 1.500s reserve" -> bound=None

Both halves are the same missing budget. The fix is issue #1152's option 3 — make
the long operations interruptible so they yield a valid intermediate result — using
the anytime-build mechanism #694/#832 already ships: the constraint-row loop stops
at a ``build_deadline`` and the partial relaxation is still a valid outer
approximation. ``_setup_remaining_budget`` withholds the fallback reserve from the
setup phases so the reserve is a reserve. After the fix the same solve returns in
5.1 s (1.02x) **with** a bound of 1.258.

These tests pin the two halves separately (the OBBT anytime contract as a
deterministic unit test, the end-to-end contract on the instance that exposed it)
so a regression says which half broke.
"""

from __future__ import annotations

import dataclasses
import os
import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver_tuning
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.obbt import obbt_tighten_root

_CORPUS = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _bilinear_model():
    """A wide-box bilinear model whose root OBBT has something to tighten."""
    m = dm.Model("i1152obbt")
    x = m.continuous("x", shape=(6,), lb=-20.0, ub=20.0)
    y = m.continuous("y", shape=(6,), lb=-20.0, ub=20.0)
    for k in range(6):
        m.subject_to(x[k] * y[k] <= 4.0)
        m.subject_to(x[k] + y[k] >= 1.0)
        m.subject_to(x[k] - 0.5 * y[k] <= 3.0)
    m.minimize(sum(x[k] + y[k] for k in range(6)))
    return m


# ---------------------------------------------------------------------------
# Half 1 — the anytime contract of the root-OBBT round build
# ---------------------------------------------------------------------------


def test_root_obbt_build_deadline_only_ever_loosens():
    """A ``build_deadline``-truncated OBBT pass must return a box that CONTAINS the
    untruncated one: dropping constraint rows enlarges the relaxation polytope, so
    every bound read off it moves outward. A truncated pass that tightened *past*
    the full pass would be cutting on rows it never built — the one way this
    mechanism could be unsound.

    The full pass is asserted to tighten something (``n_tightened > 0``): without
    that positive control the comparison below is two identical boxes and the test
    pins nothing (CLAUDE.md §6).
    """
    model = _bilinear_model()
    lb, ub = flat_variable_bounds(model)

    full = obbt_tighten_root(model, lb.copy(), ub.copy(), rounds=1)
    assert full.n_tightened > 0, "the probe did not fire: OBBT tightened nothing to compare against"

    # An already-expired build deadline: the constraint-row loop stops immediately,
    # so the relaxation is the bare column box and OBBT can prove nothing new.
    cut = obbt_tighten_root(
        model, lb.copy(), ub.copy(), rounds=1, build_deadline=time.perf_counter()
    )
    assert cut.n_tightened == 0, (
        f"an expired build deadline still tightened {cut.n_tightened} bounds — the "
        f"deadline is not reaching the round's envelope build"
    )
    assert np.all(cut.lb <= full.lb + 1e-9), "a truncated OBBT pass tightened PAST the full pass"
    assert np.all(cut.ub >= full.ub - 1e-9), "a truncated OBBT pass tightened PAST the full pass"
    # ... and it is still a subset of the input box: a truncated pass never loosens
    # beyond what it was given.
    assert np.all(cut.lb >= lb - 1e-9) and np.all(cut.ub <= ub + 1e-9)


def test_a_future_build_deadline_changes_nothing():
    """The no-regression half: when the build budget is not binding — the normal
    case — the poll must be the only difference."""
    model = _bilinear_model()
    lb, ub = flat_variable_bounds(model)
    base = obbt_tighten_root(model, lb.copy(), ub.copy(), rounds=1)
    dl = obbt_tighten_root(
        model, lb.copy(), ub.copy(), rounds=1, build_deadline=time.perf_counter() + 3600.0
    )
    assert dl.n_tightened == base.n_tightened
    assert np.array_equal(dl.lb, base.lb)
    assert np.array_equal(dl.ub, base.ub)


def test_the_opt_out_is_live():
    """The mechanism is a graduated default with an env opt-out (CLAUDE.md §5)."""
    assert solver_tuning.SolverTuning().root_setup_build_deadline is True
    prev = os.environ.get("DISCOPT_ROOT_SETUP_BUILD_DEADLINE")
    os.environ["DISCOPT_ROOT_SETUP_BUILD_DEADLINE"] = "0"
    try:
        assert solver_tuning.SolverTuning().root_setup_build_deadline is False
    finally:
        if prev is None:
            del os.environ["DISCOPT_ROOT_SETUP_BUILD_DEADLINE"]
        else:
            os.environ["DISCOPT_ROOT_SETUP_BUILD_DEADLINE"] = prev


# ---------------------------------------------------------------------------
# Half 2 — the end-to-end contract on the instance that exposed it
# ---------------------------------------------------------------------------

# ``casctanks`` is one of the four instances issue #1152 names as the overrun class
# (2.7x at a 120 s budget on the owner's machine); it is the only one of the four
# vendored in-repo, and at a 5 s budget on a quiet 4-core Linux box it reproduces
# BOTH halves — 6.43 s (1.29x) with ``bound=None``, against 5.35 s (1.07x) with
# bound 1.2584 once the root-setup budget binds.
#
# The budget matters and is not a knob: at 10 s the same box does not reproduce the
# class at all (both arms return in ~10 s with a bound), so a test written at 10 s
# passes on the legacy path too and pins nothing (CLAUDE.md §6). 5 s it is — and
# because "does this machine reproduce it" is itself a measurement, the test below
# takes it rather than assumes it.
_BUDGET = 5.0
_CASCTANKS_VALID_BOUND = 5.698  # the #654 do-not-regress value; a proven lower bound


def _solve_casctanks(tuning=None):
    from discopt.modeling.core import from_nl

    path = os.path.join(_CORPUS, "casctanks.nl")
    if not os.path.exists(path):  # pragma: no cover - the file is vendored
        pytest.skip("casctanks.nl not in the in-repo corpus")
    kwargs = {"tuning": tuning} if tuning is not None else {}
    t0 = time.perf_counter()
    res = from_nl(path).solve(time_limit=_BUDGET, **kwargs)
    return time.perf_counter() - t0, res


def _assert_sound(res):
    """Whatever it proved must be valid for this MINIMIZE."""
    assert res.status != "infeasible", "FALSE-INFEASIBLE on a feasible instance"
    if res.bound is None:
        return
    ceiling = (
        _CASCTANKS_VALID_BOUND
        if res.objective is None
        else min(_CASCTANKS_VALID_BOUND, res.objective)
    )
    assert res.bound <= ceiling + 1e-4 * max(1.0, abs(ceiling)), (
        f"unsound dual bound {res.bound} > {ceiling} on a minimization"
    )
    if res.objective is not None:
        assert res.bound <= res.objective + 1e-6, "UNSOUND CERT (bound > incumbent)"


@pytest.mark.slow
def test_casctanks_honours_its_time_limit_and_still_proves_a_bound():
    """#1152's two halves at once, on one solve, with both thresholds preserved.

    * #875's threshold, unchanged: the solve returns within 1.25x of ``time_limit``.
    * #654's guard, unchanged in shape: the deadline work must not cost the dual
      bound. Here it is the stronger claim — the bound must EXIST — because the
      reserve now survives root setup, so the last-ditch producer always gets its
      slice and an anytime build always has something to hand back.

    The legacy arm runs FIRST, and its result is the test's premise rather than an
    assumption: unless it shows the defect on *this* machine at *this* budget (an
    overrun past 1.25x, or no bound at all), there is nothing here for the fixed arm
    to fix and the assertions below would pass on the legacy path too. A machine so
    slow that ``casctanks``'s *mandatory* root work alone exceeds 5 s is outside what
    §8 of the plan doc claims, and it is the skip — not a weakened threshold — that
    says so. Both arms are checked for soundness whatever the premise says.
    """
    legacy = dataclasses.replace(solver_tuning.SolverTuning(), root_setup_build_deadline=False)
    wall_off, res_off = _solve_casctanks(legacy)
    _assert_sound(res_off)
    if wall_off <= 1.25 * _BUDGET and res_off.bound is not None:
        pytest.skip(
            f"the #1152 class does not reproduce here: the legacy arm returned in "
            f"{wall_off:.2f}s ({wall_off / _BUDGET:.2f}x) with bound {res_off.bound}"
        )

    wall_on, res_on = _solve_casctanks()
    _assert_sound(res_on)
    assert wall_on < 1.25 * _BUDGET, (
        f"solve took {wall_on:.2f}s against a {_BUDGET:.0f}s time_limit "
        f"({wall_on / _BUDGET:.2f}x); root setup is still spending past the deadline "
        f"(legacy arm: {wall_off:.2f}s)"
    )
    assert res_on.bound is not None, (
        f"the solve reported no dual bound — the root-setup phases spent the fallback "
        f"reserve again (#1152 side B; legacy arm bound: {res_off.bound})"
    )
