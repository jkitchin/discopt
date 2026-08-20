"""#1060: an open (``None``) column bound must not become NaN in the simplex marshal.

:func:`discopt.solvers.milp_simplex.solve_milp` mirrors
:func:`discopt.solvers.milp_pounce.solve_milp`, whose documented contract is that
``bounds`` entries may be ``None`` (that side is open) and that ``bounds=None``
means ``(0, +inf)`` per variable. The simplex marshal built its arrays with
``np.array([hi for _, hi in bounds], dtype=np.float64)``, which turns ``None``
into ``nan`` silently — the #1008 guard then rejected the whole solve from inside
the Rust driver, naming a standard-form column index rather than the variable.

This is on the single-tree ``lp_nlp_bb`` path: 169 of the 280 columns of the
``rsyn0840m`` master have no finite upper bound, so *every* free-backend solve of
that master died here before reaching the B&B.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import (
    _INF,
    SimplexBackendUnavailable,
    _marshal_col_bounds,
    solve_milp,
    solve_milp_with_lazy_cuts,
)

# min -x0 - 2 x1  s.t.  x0 + x1 <= 3.5,  x0, x1 integer >= 0 (no upper bound).
# The `<=` row bounds both columns from above, so the open side is genuine but
# the MILP is finite: x = (0, 3), objective -6.
_C = np.array([-1.0, -2.0])
_A_UB = np.array([[1.0, 1.0]])
_B_UB = np.array([3.5])
_INTEGRALITY = np.array([1, 1])


def _skip_without_backend(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except SimplexBackendUnavailable:  # pragma: no cover - build-dependent
        pytest.skip("Rust simplex MILP binding not built")


@pytest.mark.smoke
def test_open_upper_bound_solves_instead_of_raising_on_nan():
    """``(0, None)`` is an open upper bound, not a NaN bound."""
    res = _skip_without_backend(
        solve_milp,
        _C,
        _A_UB,
        _B_UB,
        bounds=[(0.0, None), (0.0, None)],
        integrality=_INTEGRALITY,
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)


@pytest.mark.smoke
def test_open_lower_bound_solves():
    """``(None, hi)`` is an open lower bound; the row keeps the MILP finite."""
    # min x0  s.t.  -x0 <= 2  =>  x0 >= -2, optimum -2.
    res = _skip_without_backend(
        solve_milp,
        np.array([1.0]),
        np.array([[-1.0]]),
        np.array([2.0]),
        bounds=[(None, 10.0)],
        integrality=np.array([1]),
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-2.0, abs=1e-6)


@pytest.mark.smoke
def test_lazy_cut_entry_point_shares_the_open_bound_marshal():
    """The single-tree entry point (#1060's actual caller) takes the same path."""
    seen: list[np.ndarray] = []

    def _accept_everything(x):
        seen.append(np.asarray(x, dtype=float).copy())
        return None

    res = _skip_without_backend(
        solve_milp_with_lazy_cuts,
        _C,
        _A_UB,
        _B_UB,
        bounds=[(0.0, None), (0.0, None)],
        integrality=_INTEGRALITY,
        lazy_callback=_accept_everything,
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)
    # CLAUDE.md §6: prove the separator actually ran, so this arm really did
    # traverse the lazy-cut entry point rather than short-circuiting somewhere.
    assert seen, "lazy separator never saw an integer-feasible point"


@pytest.mark.smoke
def test_open_bound_becomes_the_sentinel_and_explicit_inf_is_passed_through():
    """Only ``None`` is remapped; an explicit ``inf`` must reach the engine as-is.

    Clamping ``±inf`` to the ``1e20`` sentinel *looks* like a tidy-up of the same
    convention, but it is bound-changing and measurably harmful: on the McCormick
    root-LP path ``None`` never occurs and ``±inf`` does, and clamping moves
    ``hda``'s root bound from -64675.2 to -11308304.4 (175x weaker) and
    ``contvar``'s from 185534.9 to 191103.6. Narrowing the sentinel convention is
    its own change under CLAUDE.md §5; this marshal only fixes ``None`` (#1060).
    """
    lb, ub = _marshal_col_bounds([(-np.inf, np.inf), (None, None), (-1.5, 2.5)], 3)
    assert lb.tolist() == [-np.inf, -_INF, -1.5]
    assert ub.tolist() == [np.inf, _INF, 2.5]
    assert not np.isnan(lb).any() and not np.isnan(ub).any()


@pytest.mark.smoke
def test_bounds_none_means_zero_to_infinity():
    """Matches the ``milp_pounce`` contract for an omitted bound list."""
    lb, ub = _marshal_col_bounds(None, 4)
    assert lb.tolist() == [0.0] * 4
    assert ub.tolist() == [_INF] * 4


@pytest.mark.smoke
def test_nan_bound_is_refused_in_the_callers_index_space():
    """A NaN is a caller bug; refuse loudly and name the *caller's* index (§3)."""
    with pytest.raises(ValueError, match=r"bounds\[1\]"):
        _marshal_col_bounds([(0.0, 1.0), (0.0, float("nan"))], 2)


@pytest.mark.smoke
def test_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="2 entries but c has 3 columns"):
        _marshal_col_bounds([(0.0, 1.0), (0.0, 1.0)], 3)


@pytest.mark.smoke
def test_finite_bounds_are_unchanged_bound_neutral():
    """The fix must be bound-neutral for the finite-bound callers that worked."""
    lb, ub = _marshal_col_bounds([(0.0, 1.0), (-3.0, 7.0)], 2)
    assert lb.tolist() == [0.0, -3.0]
    assert ub.tolist() == [1.0, 7.0]
    res = _skip_without_backend(
        solve_milp,
        _C,
        _A_UB,
        _B_UB,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        integrality=_INTEGRALITY,
    )
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)


# --- the second #1060 marshal defect: the mip_start seed --------------------
#
# ``solve_milp_with_lazy_cuts`` padded ``mip_start`` to ``n + m`` with zero
# slacks -- the standard-form layout of every other array it marshals, but the
# wrong length for the seed. ``validate_seed_incumbent`` requires exactly
# ``n_struct`` and rejects anything else *silently* (an unprovable seed must
# never prune the optimum), so every mip_start on this path was discarded with
# no diagnostic. The observable is the driver's pre-search separator call: a
# validated seed is offered to the lazy separator before the search starts.


@pytest.mark.smoke
def test_mip_start_reaches_the_driver_and_is_offered_to_the_separator():
    """A length-``n`` seed must actually seed; before the fix it was dropped."""
    seen: list[np.ndarray] = []

    def _accept_everything(x):
        seen.append(np.asarray(x, dtype=np.float64).copy())
        return None

    res = _skip_without_backend(
        solve_milp_with_lazy_cuts,
        _C,
        _A_UB,
        _B_UB,
        bounds=[(0.0, None), (0.0, None)],
        integrality=_INTEGRALITY,
        lazy_callback=_accept_everything,
        mip_start=np.array([1.0, 1.0]),  # feasible (sum 2 <= 3.5), objective -3
    )
    assert seen, "lazy separator never saw an integer-feasible point"
    # The seed is the *first* point the separator sees: it is offered before the
    # search begins. With the old zero-padded seed this first call was some
    # node's LP solution instead, never (1, 1).
    assert seen[0].shape == (2,)
    np.testing.assert_allclose(seen[0], [1.0, 1.0])
    # Seeding is an optimization, never a certificate change.
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-6.0, abs=1e-6)


@pytest.mark.smoke
def test_mip_start_of_the_wrong_length_is_refused_loudly():
    """A wrong-length seed is a caller bug; silent-drop hid #1060 for good."""
    with pytest.raises(ValueError, match="mip_start has 3 entries"):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=[(0.0, None), (0.0, None)],
            integrality=_INTEGRALITY,
            lazy_callback=lambda x: None,
            mip_start=np.array([1.0, 1.0, 0.0]),
        )
