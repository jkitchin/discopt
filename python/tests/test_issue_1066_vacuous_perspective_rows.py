"""#1066: perspective rows at a numerically-zero reference wrecked the master.

``squfl020-150`` at the issue's defaults (``gap_tolerance=1e-4``,
``time_limit=60``) returned ``feasible`` with **no dual bound at all**: the
single-tree master's HiGHS solve reported ``mip_dual_bound = 558.0440365723572``
for a master whose optimum is provably at most ``557.84865`` -- fixing all 6021
columns at the lifted true optimum solves to ``kOptimal`` at 557.84865 with a
max primal infeasibility of 8.35e-15, and fixing more variables cannot lower an
optimum. The #1136 inversion guard caught it and suppressed the bound, which is
correct but leaves the instance uncertified.

The master was 69 330 rows against 6 021 columns, and **43 500 of those rows
(63%) were the same vacuous row**, reaching HiGHS as

    ``0.0174 * x_k  -  70368744177664 * s_k  <=  5.69e-20``

Dividing by ``2**46 = 70368744177664`` recovers what was actually separated:
``2.47e-16 * x_k - s_k <= 8.09e-34``, i.e. ``s_k >= ~0`` -- which ``s_k``'s own
lower bound of 0 already says. Two independent defects compounded:

1. ``_disaggregate_objective_cut`` emitted a perspective row for every reference
   ``z != 0.0``, including ``z ~ 1.2e-16``, whose row carries no information.
2. ``_prepare_cut_row`` chose its power-of-two scale from *every* coefficient,
   including the ``q*z**2 ~ 3e-32`` term. That drove ``headroom`` to 46 -- and
   ``2**46 * 3e-32 = 2.1e-18`` is still under ``small_matrix_value``, so the term
   was dropped anyway. The row was scaled by 7.04e13 to rescue a term it then
   discarded.

Together they took the master's coefficient range to 1.1e-5 .. 7.04e13 (ratio
6.4e18), with a median *per-row* range of 3.4e14 and 43 500 rows above 1e9. On a
matrix like that HiGHS's dual bound is not trustworthy, and measurably was not.

After both fixes ``squfl020-150`` solves to ``optimal``, certified, in 49.3 s:
bound 557.848649960045 <= incumbent 557.848649973387 <= 557.84865. 61 650 of
86 356 references (71%) are refused as vacuous.
"""

import numpy as np
from discopt.solvers import oa
from discopt.solvers.milp_highs import _prepare_cut_row
from discopt.solvers.milp_simplex import _INF

# HiGHS defaults, read here as plain numbers so the test states the window it
# is reasoning about rather than depending on a live solver.
SMALL, LARGE = 1e-9, 1e15


def test_scale_ignores_terms_it_is_going_to_drop_anyway():
    """The measured ``squfl020-150`` row must not be scaled by ``2**46``.

    ``q*z**2`` is too small for ANY admissible scale to lift over
    ``small_matrix_value``, so it is dropped either way; letting it choose the
    scale only destroys the row's conditioning.
    """
    q, z = 1.0, 1.2e-16
    coeffs = np.zeros(4)
    coeffs[0] = 2.0 * q * z  # 2.4e-16 on x_k
    coeffs[1] = -q * z * z  # -1.44e-32 on y_k -- unrescuable
    coeffs[2] = -1.0  # s_k
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([_INF, 1.0, _INF, _INF])

    _idx, vals, _rhs = _prepare_cut_row(coeffs.copy(), 0.0, lb, ub, SMALL, LARGE)

    biggest = float(np.max(np.abs(vals)))
    # Before the fix this was exactly 2**46 = 70368744177664.0.
    assert biggest < 1e8, f"row still scaled into the danger zone: max|a| = {biggest!r}"
    assert biggest != float(2**46)


def test_scale_still_lifts_a_term_that_can_be_rescued():
    """Anti-vacuity: the fix must not disable scaling for rows it should help.

    ``squfl025-040``'s first cut spans 2.05e-19 to 114 -- 21 orders inside a
    24-wide window -- and the docstring of ``_prepare_cut_row`` records that
    failing to lift it killed every ``squfl`` instance under ``lp_nlp_bb``.
    """
    coeffs = np.zeros(3)
    coeffs[0] = 2.05e-19
    coeffs[1] = 114.0
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([10.0, 10.0, 10.0])

    _idx, vals, _rhs = _prepare_cut_row(coeffs.copy(), 1.0, lb, ub, SMALL, LARGE)

    smallest = float(np.min(np.abs(vals)))
    assert len(vals) == 2, "the liftable small term must survive, not be dropped"
    assert smallest > SMALL, f"liftable term was not lifted clear: {smallest!r}"


def _epigraph():
    ep = oa._perspective_epigraph_for([(0, 1, 2.0), (2, 3, 1.0)], n_vars=8)
    assert ep is not None
    return ep


def test_a_numerically_zero_reference_emits_no_row():
    """``z ~ 1e-16`` must be refused exactly as ``z == 0.0`` already was.

    The row would say ``s_k >= 0``, which the column bound already carries.
    """
    ep = _epigraph()
    x_star = np.zeros(8)
    x_star[0] = 1.2e-16  # term 0: 2*q*z = 4.8e-16, far under the floor
    x_star[2] = 3.0  # term 1: an ordinary reference, must still be taken

    before = oa._PERSPECTIVE_DISAGG_NEAR_ZERO[0]
    out = oa._disaggregate_objective_cut(np.zeros(9), 0.0, x_star, ep)

    assert out is not None, "a near-zero reference must not kill the whole cut"
    assert oa._PERSPECTIVE_DISAGG_NEAR_ZERO[0] == before + 1, "the guard did not fire"
    # Only the real reference got a row; the vacuous one did not.
    assert ep.rows == [(1, 3.0)]


def test_the_removal_still_happens_for_a_refused_reference():
    """Skipping the row must not skip the subtraction.

    The term has to leave the aggregate row on every row or none (see
    ``test_split_is_all_or_nothing``); what the guard drops is only the
    *replacement* row, and dropping an underestimator can weaken a bound but
    never invalidate it.
    """
    q0, q1 = 2.0, 1.0
    ep = _epigraph()
    n = 8
    z0, z1 = 1.2e-16, 3.0
    x_star = np.zeros(n)
    x_star[0], x_star[2] = z0, z1

    coeffs = np.zeros(n + 1)
    coeffs[0] = 2.0 * q0 * z0
    coeffs[2] = 2.0 * q1 * z1
    coeffs[n] = -1.0
    rhs = q0 * z0**2 + q1 * z1**2

    out = oa._disaggregate_objective_cut(coeffs.copy(), rhs, x_star, ep)
    assert out is not None
    residual_coeffs, residual_rhs = out

    # BOTH terms left the aggregate row, the refused one included.
    assert residual_coeffs[0] == 0.0
    assert residual_coeffs[2] == 0.0
    assert residual_rhs == 0.0
    assert residual_coeffs[n] == -1.0, "the eta column must survive the split"


def test_an_ordinary_reference_is_still_taken():
    """Anti-vacuity for the guard: normal references must be unaffected."""
    ep = _epigraph()
    x_star = np.zeros(8)
    x_star[0], x_star[2] = 1.5, 3.0
    before = oa._PERSPECTIVE_DISAGG_NEAR_ZERO[0]

    assert oa._disaggregate_objective_cut(np.zeros(9), 0.0, x_star, ep) is not None

    assert oa._PERSPECTIVE_DISAGG_NEAR_ZERO[0] == before, "guard fired on a good reference"
    assert sorted(ep.rows) == [(0, 1.5), (1, 3.0)]


def test_the_threshold_is_on_the_row_coefficient_not_the_reference():
    """A tiny ``z`` with a large ``q`` still produces a usable row.

    The floor is on ``2*q*|z|`` -- what actually reaches the master -- so a
    curvature big enough to lift the coefficient clear keeps its row. Gating on
    ``|z|`` alone would throw that away.
    """
    ep = oa._perspective_epigraph_for([(0, 1, 1e9)], n_vars=8)
    assert ep is not None
    x_star = np.zeros(8)
    x_star[0] = 1e-8  # 2*q*z = 20.0, comfortably representable
    before = oa._PERSPECTIVE_DISAGG_NEAR_ZERO[0]

    assert oa._disaggregate_objective_cut(np.zeros(9), 0.0, x_star, ep) is not None

    assert oa._PERSPECTIVE_DISAGG_NEAR_ZERO[0] == before
    assert ep.rows == [(0, 1e-8)]
