"""#1066: a lazy cut whose coefficients span more orders of magnitude than HiGHS keeps.

HiGHS silently discards any matrix entry with ``|value| <= small_matrix_value``
(1e-9 by default) and reports the discard from ``addRow`` as ``kWarning`` -- not
``kOk``, not ``kError``. ``solve_milp_with_lazy_cuts`` read anything other than
``kOk`` as a rejection and raised, which killed the whole ``squfl`` family under
``mip_nlp_method="lp_nlp_bb", milp_solver="highs"``: ``squfl025-040`` died in 1.0 s
and ``squfl020-150`` in 7.1 s with ``RuntimeError: HiGHS rejected a lazy cut row``.

The row that did it, measured on ``squfl025-040``'s first separated cut: 1026
nonzeros, of which **880** were at or below 1e-9, ``|a|`` spanning 2.05e-19 to
114 -- 21 orders of magnitude of AD dust around a real support of ~146 entries.

Reading the warning as "row added, all good" would have been the other wrong
answer: HiGHS would then have *changed the cut* behind the separator's back. So
the row is fitted to HiGHS's window first -- rescaled by a power of two, which is
exact -- and only what still does not fit is dropped, on the valid side, with a
loud refusal when no valid drop exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from discopt.solvers import SolveStatus

highspy = pytest.importorskip("highspy", reason="the HiGHS master is an opt-in extra")

from discopt.solvers.milp_highs import (  # noqa: E402
    _INF,
    _highs_matrix_window,
    _prepare_cut_row,
    solve_milp_with_lazy_cuts,
)

_SMALL, _LARGE = 1e-9, 1e15


@pytest.mark.smoke
def test_highs_really_does_answer_a_tiny_entry_with_a_warning_not_an_error():
    """The premise of this whole file, asserted rather than assumed.

    If a HiGHS release ever promotes this to ``kError`` or demotes it to ``kOk``,
    every other test here is reasoning about behaviour that no longer exists.
    """
    h = highspy.Highs()
    assert h.setOptionValue("output_flag", False) == highspy.HighsStatus.kOk
    h.addVars(2, np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    st = h.addRow(-highspy.kHighsInf, 5.0, 2, np.array([0, 1], np.int32), np.array([1.0, 2.05e-19]))
    assert st == highspy.HighsStatus.kWarning, st
    assert h.getNumRow() == 1, "HiGHS refused the row outright"
    assert h.getNumNz() == 1, "HiGHS kept the tiny entry after all"


@pytest.mark.smoke
def test_the_window_is_read_from_the_solver_not_hardcoded():
    h = highspy.Highs()
    assert h.setOptionValue("output_flag", False) == highspy.HighsStatus.kOk
    small, large = _highs_matrix_window(h, highspy)
    assert (small, large) == (_SMALL, _LARGE)
    # Moving the option must move what the preparation believes, or the two
    # silently disagree about which entries survive.
    assert h.setOptionValue("small_matrix_value", 1e-6) == highspy.HighsStatus.kOk
    assert _highs_matrix_window(h, highspy)[0] == 1e-6


@pytest.mark.smoke
def test_a_row_that_fits_after_rescaling_keeps_every_entry_exactly():
    """The squfl case: 21 orders of magnitude, a 24-order window. Nothing is dropped."""
    coeffs = np.array([114.0, 2.05e-19, -3.0])
    idx, vals, rhs = _prepare_cut_row(coeffs, -8.74856e-14, *_box(3), _SMALL, _LARGE)

    assert idx.tolist() == [0, 1, 2], "an entry was dropped that rescaling could save"
    assert np.min(np.abs(vals)) > _SMALL, "an entry HiGHS will drop was handed to it"
    assert np.max(np.abs(vals)) < _LARGE, "an entry HiGHS refuses was handed to it"

    # Exactness: one common power-of-two factor, applied bit-for-bit. A non-dyadic
    # factor would round every coefficient and could tighten the cut by an ulp.
    scale = vals[0] / coeffs[0]
    assert math.log2(scale) == int(math.log2(scale)), scale
    np.testing.assert_array_equal(vals, coeffs * scale)
    assert rhs == -8.74856e-14 * scale


@pytest.mark.smoke
def test_a_row_already_inside_the_window_is_passed_through_untouched():
    coeffs = np.array([1.0, -2.0, 0.0, 0.5])
    idx, vals, rhs = _prepare_cut_row(coeffs, 7.0, *_box(4), _SMALL, _LARGE)
    assert idx.tolist() == [0, 1, 3]
    np.testing.assert_array_equal(vals, [1.0, -2.0, 0.5])
    assert rhs == 7.0


def _box(n, lo=0.0, hi=10.0):
    return np.full(n, lo), np.full(n, hi)


@pytest.mark.smoke
def test_a_non_negative_term_drops_for_free_when_rescaling_cannot_save_it():
    """``a_j x_j >= 0`` over the box, so removing it can only make the row easier."""
    # 28 orders of magnitude: wider than the window, so a drop is forced.
    coeffs = np.array([1e10, 1e-18])
    lb, ub = np.array([0.0, 0.0]), np.array([10.0, 100.0])
    idx, vals, rhs = _prepare_cut_row(coeffs, 1e10, lb, ub, _SMALL, _LARGE)

    assert idx.tolist() == [0], "the entry HiGHS would drop was still handed to it"
    scale = vals[0] / coeffs[0]
    assert rhs == 1e10 * scale, "rhs was loosened for a term that needed no slack"
    _assert_implied(coeffs, 1e10, idx, vals, rhs, lb, ub)


@pytest.mark.smoke
def test_a_negative_term_drops_only_after_the_rhs_absorbs_its_worst_case():
    coeffs = np.array([1e10, -1e-18])
    lb, ub = np.array([0.0, 0.0]), np.array([10.0, 100.0])
    idx, vals, rhs = _prepare_cut_row(coeffs, 1e10, lb, ub, _SMALL, _LARGE)

    assert idx.tolist() == [0]
    scale = vals[0] / coeffs[0]
    # Strictly greater, even though the true slack (8.19e-13) is far below the ulp
    # of the rhs (~0.015) and rounds straight off it: the loosening is rounded up,
    # so the row can never come out TIGHTER than the one that is provably implied.
    assert rhs > 1e10 * scale, "the dropped negative term left the row tighter"
    assert rhs == pytest.approx(1e10 * scale + 1e-18 * 100.0 * scale, rel=1e-12)
    _assert_implied(coeffs, 1e10, idx, vals, rhs, lb, ub)


@pytest.mark.smoke
def test_an_unbounded_binding_side_is_refused_loudly_rather_than_quietly_dropped():
    coeffs = np.array([1e10, -1e-18])
    lb, ub = np.array([0.0, 0.0]), np.array([10.0, np.inf])
    with pytest.raises(ValueError, match="unbounded on the binding side"):
        _prepare_cut_row(coeffs, 1e10, lb, ub, _SMALL, _LARGE)


@pytest.mark.smoke
def test_the_1e20_open_bound_sentinel_counts_as_unbounded_not_as_a_finite_1e20():
    """CLAUDE.md's documented trap, on exactly the shape that springs it.

    discopt marks an open bound with ``1e20``, not ``inf``. A finiteness test on
    the *product* passes here -- ``1e-18 * 1e20`` is an unremarkable ``100`` -- so a
    check written that way computes a finite 'compensation' for a term that has no
    worst case and ships a row the cut does not imply.
    """
    coeffs = np.array([1e10, -1e-18])
    lb, ub = np.array([0.0, 0.0]), np.array([10.0, _INF])
    with pytest.raises(ValueError, match="unbounded on the binding side"):
        _prepare_cut_row(coeffs, 1e10, lb, ub, _SMALL, _LARGE)


def _assert_implied(coeffs, rhs, idx, vals, rhs_out, lb, ub, n_samples=2000):
    """Every box point the original row keeps, the prepared row must keep too.

    §6: the sample count is asserted, so a generator that yields nothing cannot
    read as 'no violations found'.
    """
    rng = np.random.default_rng(1066)
    x = rng.uniform(lb, np.minimum(ub, 1e6), size=(n_samples, lb.size))
    kept = x @ coeffs <= rhs
    assert kept.any(), "no sampled point satisfied the original row -- vacuous test"
    new_lhs = x[:, idx] @ vals
    slack = rhs_out - new_lhs
    assert np.all(slack[kept] >= -1e-9 * max(1.0, abs(rhs_out))), (
        f"the prepared row cuts off {int((slack[kept] < 0).sum())} point(s) the cut kept"
    )
    return int(kept.sum())


# --- end to end: the failure #1066 actually hit -------------------------------

_C = np.array([-1.0, -2.0])
_A_UB = np.array([[1.0, 1.0]])
_B_UB = np.array([3.5])
_INTEGRALITY = np.array([1, 1])
_BOUNDS = [(0.0, None), (0.0, None)]


@pytest.mark.smoke
def test_a_wide_dynamic_range_cut_no_longer_kills_the_solve():
    """The regression. Before the fix this raised ``HiGHS rejected a lazy cut row``.

    The cut is ``1e-19 x0 + x1 <= 2``: mathematically ``x1 <= 2`` plus a dust term,
    which is the shape every squfl OA cut has. The answer must still be the vetoed
    optimum's replacement, (1, 2) at -5.
    """
    calls: list[np.ndarray] = []

    def _veto_x1_above_two(x):
        calls.append(np.asarray(x, dtype=float).copy())
        if x[1] > 2.0 + 1e-6:
            return [(np.array([1e-19, 1.0]), 2.0)]
        return None

    res = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=_veto_x1_above_two,
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-5.0, abs=1e-6)
    np.testing.assert_allclose(res.x, [1.0, 2.0], atol=1e-6)
    assert res.callback_stats["lazy_cuts"] >= 1
    assert res.callback_stats["restarts"] >= 1
    assert any(c[1] > 2.0 + 1e-6 for c in calls), "the vetoing branch never fired"


@pytest.mark.smoke
def test_a_cut_with_no_row_in_it_is_refused_instead_of_spinning_the_tree():
    """Adding no row would rebuild an identical tree forever on the same point.

    Rescaling rescues almost anything -- even a 1e-30 coefficient lifts into the
    window -- so the only way to reach this guard is a cut with no nonzero at all.
    That is a broken separator, and it must say so rather than loop.
    """

    def _cut_with_no_coefficients(x):
        if x[1] > 2.0 + 1e-6:
            return [(np.zeros(2), -1.0)]
        return None

    with pytest.raises(ValueError, match="no coefficient HiGHS will accept"):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=_cut_with_no_coefficients,
        )
