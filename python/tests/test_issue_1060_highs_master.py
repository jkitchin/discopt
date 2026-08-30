"""#1060: the opt-in HiGHS master engine for the MIP-NLP family.

Why a HiGHS backend exists at all after #356 removed it: #356 targeted the
*per-node LP* of spatial branch-and-bound, and that removal stands. This is a
different seam -- the *master MILP* of the OA / LP-NLP-BB family -- and it is
where the in-house driver is measurably outclassed. On the ``rsyn0840m`` master
(gap 2295.97) the in-house root loop closes 0.0% with cover+GMI, 10.5% adding
MIR/aggregation c-MIR, 37.7% adding probing implied bounds, and plateaus at
47.6% iterated to tailing-off; HiGHS closes 86.1% at node 0 and finishes the
tree in 92 nodes / 0.42 s. End to end that is the difference between a 60 s
timeout at objective -11.41 and ``optimal`` at 325.5545 in 1.9 s.

``"highs"`` is reachable only by naming it. It is in no fallback order, so
``get_milp_solver()`` and ``milp_solver="auto"`` still resolve to the in-house
simplex and no existing caller's numbers move -- that invariant is asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_backend import get_milp_solver

highspy = pytest.importorskip("highspy", reason="the HiGHS master is an opt-in extra")

from discopt.solvers.milp_highs import (  # noqa: E402
    _INF,
    _to_highs_inf,
    solve_milp,
    solve_milp_with_lazy_cuts,
)

# min -x0 - 2 x1  s.t.  x0 + x1 <= 3.5, integer >= 0 and unbounded above.
# The row keeps it finite: optimum x = (0, 3), objective -6.
_C = np.array([-1.0, -2.0])
_A_UB = np.array([[1.0, 1.0]])
_B_UB = np.array([3.5])
_INTEGRALITY = np.array([1, 1])
_BOUNDS = [(0.0, None), (0.0, None)]


@pytest.mark.smoke
def test_highs_is_opt_in_only_and_auto_is_unchanged():
    """#356's routing must survive: nothing reaches HiGHS without asking for it."""
    assert get_milp_solver().__module__ == "discopt.solvers.milp_simplex"
    assert get_milp_solver(backend="auto").__module__ == "discopt.solvers.milp_simplex"
    assert get_milp_solver(backend="simplex").__module__ == "discopt.solvers.milp_simplex"
    assert get_milp_solver(backend="highs").__module__ == "discopt.solvers.milp_highs"


@pytest.mark.smoke
def test_plain_solve_matches_the_known_optimum_with_a_sound_bound():
    res = solve_milp(_C, _A_UB, _B_UB, bounds=_BOUNDS, integrality=_INTEGRALITY)
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)
    assert res.bound is not None and res.bound <= res.objective + 1e-9
    np.testing.assert_allclose(res.x, [0.0, 3.0], atol=1e-6)


@pytest.mark.smoke
def test_equality_rows_are_marshalled_into_the_two_sided_row_bounds():
    """``A_eq``/``b_eq`` become ``row_lower == row_upper``, not another ``<=``."""
    # min x0 + x1 s.t. x0 + x1 == 2, 0 <= x <= 5 integer.
    res = solve_milp(
        np.array([1.0, 1.0]),
        None,
        None,
        A_eq=np.array([[1.0, 1.0]]),
        b_eq=np.array([2.0]),
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        integrality=np.array([1, 1]),
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(2.0, abs=1e-6)


@pytest.mark.smoke
def test_the_1e20_sentinel_becomes_the_highs_infinity_not_a_finite_bound():
    """discopt's open-bound sentinel is 1e20; HiGHS's is 1e30 (#1060).

    Passing 1e20 through as a literal makes it an ordinary *finite* bound to
    HiGHS, which silently narrows the feasible set. The mapping is the seam
    between the two conventions and has to be exact in both directions.
    """
    out = _to_highs_inf(np.array([-_INF, -1e21, -3.0, 0.0, 7.0, _INF, 1e25]), highspy)
    inf = highspy.kHighsInf
    assert out.tolist() == [-inf, -inf, -3.0, 0.0, 7.0, inf, inf]
    # And a genuinely finite bound must not be inflated.
    assert out[2] == -3.0 and out[4] == 7.0


@pytest.mark.smoke
def test_infeasible_and_unbounded_are_reported_not_guessed():
    infeas = solve_milp(
        np.array([1.0]),
        np.array([[1.0], [-1.0]]),
        np.array([-2.0, -3.0]),  # x <= -2 and x >= 3
        bounds=[(None, None)],
        integrality=np.array([1]),
    )
    assert infeas.status == SolveStatus.INFEASIBLE
    assert infeas.objective is None

    unb = solve_milp(np.array([-1.0]), None, None, bounds=[(0.0, None)], integrality=np.array([1]))
    assert unb.status == SolveStatus.UNBOUNDED
    # No objective and no bound: an unbounded master has neither, and inventing
    # one here is how a false certificate gets manufactured downstream.
    assert unb.objective is None and unb.bound is None


# --- the single-tree (Quesada-Grossmann) entry point ------------------------


@pytest.mark.smoke
def test_lazy_separator_sees_every_integer_feasible_point():
    seen: list[np.ndarray] = []

    res = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=lambda x: seen.append(np.asarray(x, dtype=float).copy()),
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)
    # CLAUDE.md §6: a separator that never fired makes every other assertion here
    # vacuous, and a callback that is silently never registered looks identical
    # to one that accepts everything.
    assert seen, "lazy separator never saw an integer-feasible point"
    assert res.callback_stats["mipsol_calls"] == len(seen)


@pytest.mark.smoke
def test_a_vetoed_point_is_cut_off_and_never_returned_as_the_incumbent():
    """The soundness core: HiGHS's own best solution may be one we rejected.

    HiGHS keeps the interrupted tree's incumbent, which is exactly the point the
    separator asked to remove. Returning it would report an OA-infeasible point
    as the master solution (CLAUDE.md §1), so the wrapper tracks the accepted
    incumbent itself. Here (0, 3) is optimal for the master but forbidden by the
    separator's cut ``x1 <= 2``; the answer must be (1, 2) at -5.
    """
    calls: list[np.ndarray] = []

    def _veto_x1_above_two(x):
        calls.append(np.asarray(x, dtype=float).copy())
        if x[1] > 2.0 + 1e-6:
            return [(np.array([0.0, 1.0]), 2.0)]  # x1 <= 2
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
    assert res.x[1] <= 2.0 + 1e-6, "returned the point the separator cut off"
    # The cut path really ran: at least one cut and one rebuild.
    assert res.callback_stats["lazy_cuts"] >= 1
    assert res.callback_stats["restarts"] >= 1
    assert any(c[1] > 2.0 + 1e-6 for c in calls), "the vetoing branch never fired"
    assert res.bound is not None and res.bound <= res.objective + 1e-9


@pytest.mark.smoke
def test_a_raising_separator_crashes_the_solve_rather_than_accepting_the_point():
    """CLAUDE.md §7 applied to the callback: a broken separator is not an ack."""

    class _Boom(RuntimeError):
        pass

    def _raise(x):
        raise _Boom("separator failed")

    with pytest.raises(_Boom):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=_raise,
        )


@pytest.mark.smoke
def test_mip_start_is_a_structural_seed_and_a_wrong_length_is_refused():
    res = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=lambda x: None,
        mip_start=np.array([1.0, 1.0]),
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)

    with pytest.raises(ValueError, match="mip_start has 3 entries"):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=lambda x: None,
            mip_start=np.array([1.0, 1.0, 0.0]),
        )


@pytest.mark.smoke
def test_a_wrong_length_cut_is_refused_instead_of_being_padded():
    """A short coefficient row would silently become a cut on the wrong columns."""
    with pytest.raises(ValueError, match="lazy cut has 1 coefficients"):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=lambda x: [(np.array([1.0]), 0.0)],
        )


@pytest.mark.smoke
def test_a_hook_this_backend_cannot_honour_is_refused_loudly():
    """HiGHS has no fractional-node hook; silence would lose the caller's cuts.

    Accepting a ``node_callback`` and never calling it would make the SHOT
    profile's MIPNODE cuts vanish with no diagnostic -- the caller would read a
    weaker search as an algorithmic result (CLAUDE.md §3).

    ``terminate_callback`` used to be refused alongside it and no longer is:
    #1066 gave this backend a real check-in point at the master restart, so the
    hook is called rather than ignored. Its behaviour is pinned in
    ``test_issue_1066_lp_nlp_bb_progress_budget.py``.
    """
    with pytest.raises(NotImplementedError, match="no MIPNODE equivalent"):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=lambda x: None,
            node_callback=lambda x: None,
        )


@pytest.mark.smoke
def test_lp_nlp_bb_accepts_highs_and_still_refuses_a_hookless_backend():
    """The public seam #1060's acceptance criterion is written against."""
    from discopt.solvers.oa import _resolve_lp_nlp_bb_backend

    assert _resolve_lp_nlp_bb_backend("highs", shot_profile=False) == "highs"
    assert _resolve_lp_nlp_bb_backend("auto", shot_profile=False) == "simplex"
    with pytest.raises(RuntimeError, match="no separator hook"):
        _resolve_lp_nlp_bb_backend("pounce", shot_profile=False)
    # The SHOT profile needs MIPNODE cuts, which HiGHS cannot provide.
    with pytest.raises(RuntimeError):
        _resolve_lp_nlp_bb_backend("highs", shot_profile=True)
