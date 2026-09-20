"""#1392 — the distance cap's noise floor must ADD to its allowance, not max with it.

``feasible_distance_cap`` (#1254) answers "how far may this point sit from the
row's surface". ``SMALL_ROW_ABS_FLOOR`` answers a different question — "how small
a residual can this arithmetic even resolve" — and a row's violation carries both
at once. Combining them with ``max`` therefore lets the larger erase the smaller,
and on a row that is nearly flat *because its only improving variable is nearly on
its bound* the distance allowance lands within evaluation noise of the violation.

Measured on ``synthes3`` via ``nlp_bb`` (the numbers below are copied from that
run): the big-M row ``x0 - 10*y9 <= 0`` at ``x0 = 1.0750640431770233e-12``
(lower bound 0), ``y9 = 0`` evaluates to ``1.0751399770470016e-12`` — the body and
its own linear term disagree by 7.6e-17, ordinary double-precision noise on a
unit-scale evaluation. ``x0``'s room to its bound is exactly its value, so the
allowance is 7.6e-17 *below* the violation and the false-primal screen rejected
the point. ``solve`` returned ``status="error"`` with the objective withheld,
where the incumbent was 68.00974056776073 against minlplib's proven ``=opt=``
68.00974052 — correct to 4.8e-8 relative.

The cap's teeth are what #1254 bought, so these tests pin both directions: the
noise-level shortfall is forgiven, and a violation genuinely larger than the
repairable room is still refused.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt._relax.primal_heuristics import passes_false_primal_screen, row_violations
from discopt.validation.feasibility import (
    CANCELLATION_RTOL,
    FEASIBLE_DISTANCE_TOL,
    SMALL_ROW_ABS_FLOOR,
    feasible_distance_cap,
)

# The measured triple from synthes3's row 2. ``GRAD`` is the improving-move
# gradient norm (#1284) at that point: ``|J| * room / FEASIBLE_DISTANCE_TOL``.
SYNTHES3_VIOL = 1.0751399770470016e-12
SYNTHES3_GRAD = 1.075064043178183e-08
SYNTHES3_SCALE = 1.075064043178179e-12


@pytest.mark.unit
def test_a_noise_level_shortfall_is_forgiven():
    """The synthes3 row: the violation exceeds the allowance by 7.6e-17.

    Fails before the fix (``max`` put the cap at 1.075064e-12, just under the
    1.075140e-12 violation); passes after.
    """
    cap = float(feasible_distance_cap(SYNTHES3_GRAD, SYNTHES3_SCALE))
    shortfall = SYNTHES3_VIOL - FEASIBLE_DISTANCE_TOL * SYNTHES3_GRAD
    # The premise of the test: the gap being argued about really is noise-sized.
    assert 0.0 < shortfall < 1e-15, shortfall
    assert SYNTHES3_VIOL <= cap, f"viol {SYNTHES3_VIOL!r} > cap {cap!r}"


@pytest.mark.unit
def test_the_floor_is_added_to_the_allowance():
    """``cap == floor + FEASIBLE_DISTANCE_TOL*grad``, elementwise."""
    grad = np.array([0.0, 1e-9, 1e-8, 1.0, 1e6])
    expected = SMALL_ROW_ABS_FLOOR + FEASIBLE_DISTANCE_TOL * grad
    assert np.allclose(feasible_distance_cap(grad), expected, rtol=0, atol=0)


@pytest.mark.unit
def test_the_cancellation_term_replaces_the_floor_then_adds():
    """The two floors still compete with each other; only their winner adds."""
    grad = np.array([1e-8, 1e-8])
    # First row: 1e-9*1e5 = 1e-4 beats the 1e-12 floor. Second: 1e-9*1e-6 = 1e-15
    # does not, so the absolute floor wins there.
    scale = np.array([1e5, 1e-6])
    floor = np.array([CANCELLATION_RTOL * 1e5, SMALL_ROW_ABS_FLOOR])
    expected = floor + FEASIBLE_DISTANCE_TOL * grad
    assert np.allclose(feasible_distance_cap(grad, scale), expected, rtol=0, atol=0)


@pytest.mark.unit
def test_every_recorded_rejection_survives():
    """A 1e-12 addition cannot rescue a violation orders of magnitude above it."""
    checked = 0
    # #1254: violation 2.0e-07 against every partial ~2.302585e-07.
    assert 2.0e-07 > float(feasible_distance_cap(2.302585e-07))
    checked += 1
    # #770's false primals: violations 0.4 to 17.6 on ordinary rows.
    for viol in (0.4, 17.6):
        assert viol > float(feasible_distance_cap(1.0))
        checked += 1
    # The Scholtes-regularized MPEC point must still be ACCEPTED (dg/dx = 1).
    assert 6.7e-09 <= float(feasible_distance_cap(1.0))
    checked += 1
    assert checked == 4, checked


@pytest.mark.unit
def test_a_non_finite_gradient_is_still_uncapped():
    """An unestimatable gradient must not manufacture a strict test."""
    cap = feasible_distance_cap(np.array([np.inf, np.nan, 1.0]))
    assert cap[0] == np.inf and cap[1] == np.inf
    assert np.isfinite(cap[2])


def _flat_bigm_model(offset: float) -> dm.Model:
    """``x - w - 10*y <= 0`` with ``w`` FIXED at ``-offset``, ``y`` binary.

    The synthes3 structure: ``x`` sits near its lower bound 0, so the only move
    that can reduce the row's violation has room ``x`` and no more — ``w`` is
    fixed and ``y`` would have to move a whole unit. The row therefore evaluates
    to ``x + offset`` while only ``x`` of that is repairable, which puts the
    violation ``offset`` above the first-order distance allowance.
    """
    m = dm.Model("flat_bigm")
    x = m.continuous("x", lb=0.0, ub=2.0)
    w = m.continuous("w", lb=-offset, ub=-offset)
    y = m.binary("y")
    m.subject_to(x - w - 10.0 * y <= 0.0)
    m.minimize(x)
    return m


def _screen(offset: float, x_val: float):
    """``(verdict, violation, threshold)`` for the screen on that constructed row."""
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    ev = make_evaluator(_flat_bigm_model(offset))
    x = np.array([x_val, -offset, 0.0])
    viol = row_violations(ev, x)
    jac = np.asarray(ev.evaluate_jacobian(x), dtype=np.float64)
    cl, cu = _infer_constraint_bounds(ev)
    thr = ph.combined_tolerance(
        ph._scale_from_jacobian(jac, x),
        ph.FALSE_PRIMAL_ATOL,
        ph.FALSE_PRIMAL_RTOL,
        ph._row_gradient_norms(ev, jac, x, cl, cu),
    )
    return passes_false_primal_screen(ev, x), float(viol[0]), float(thr[0])


def test_screen_keeps_a_point_short_of_its_room_by_evaluation_noise():
    """The synthes3 signature, through the screen a real evaluator feeds.

    Fails before the fix: the threshold is 1.075064e-12 and the violation
    1.075140e-12, so the screen declared a FALSE PRIMAL over 7.6e-17.
    """
    ok, viol, thr = _screen(7.6e-17, 1.0750640431770233e-12)
    # The premise, asserted rather than assumed: this reproduces synthes3's row to
    # within a last-digit wobble of the evaluation, and the part of the violation
    # no in-box move can remove really is noise-sized.
    assert viol == pytest.approx(SYNTHES3_VIOL, rel=1e-6)
    unrepairable = viol - FEASIBLE_DISTANCE_TOL * SYNTHES3_GRAD
    assert 0.0 < unrepairable < 1e-15, unrepairable
    assert ok, f"violation {viol!r} rejected against threshold {thr!r}"


def test_screen_still_refuses_a_violation_twice_its_repairable_room():
    """The cap's teeth: the fix must not blind it on the same row shape.

    Same construction, but the unrepairable part of the violation is 1e-6 rather
    than 7.6e-17 — six orders above the noise floor the fix adds.
    """
    ok, viol, thr = _screen(1e-6, 1e-6)
    assert viol == pytest.approx(2e-6, rel=1e-12)
    assert not ok, f"violation {viol!r} accepted against threshold {thr!r}"
