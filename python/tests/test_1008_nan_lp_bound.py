"""A NaN variable bound must never reach the simplex (#1008 follow-up).

The modeling layer and the LP layer spell "no bound" differently:
``Model.continuous(ub=None)`` stores **NaN**, while the Rust simplex reads the
sentinel ``±1e20`` (``INF``). An untranslated NaN is not merely unhelpful — it is
read *two contradictory ways*, because every comparison against NaN is false:

* the ratio test asks ``ub < INF`` ("is this bound blocking?") → **false** for NaN,
  so the bound is skipped and the step runs to ``t = INF`` → ``unbounded``;
* the unbounded-ray box-recession check asks ``ub >= INF`` ("is this side open?")
  → also **false** for NaN, so the very same bound reads as closed.

Found while the #1008 ray certifier was landing: a Benders recourse LP
(``min -w`` over ``w ∈ [0, NaN]``, ``-w ≤ 0``) reached the simplex with a NaN
upper bound and had been reported ``unbounded`` on the strength of a box no guard
could certify as recessive. The verdict was right by luck, not by derivation —
the ratio test's reading happened to be the one that mattered. This is the same
class as the ``INF``-is-``1e20`` hazard in CLAUDE.md, with the sentinel silently
*absent* rather than silently surviving a multiplication.

Two halves, both pinned here:

1. ``lp_simplex`` translates the box (NaN and ``±inf`` → the sentinel), so the
   simplex sees one unambiguous convention;
2. the PyO3 LP entry points refuse a NaN bound outright, so a caller that skips
   the translation gets a loud ``ValueError`` instead of a verdict derived from
   two incompatible readings of the same number.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_simplex import _LP_INF, _finite_box, solve_lp


def test_finite_box_maps_no_bound_onto_the_lp_sentinel():
    lo, hi = _finite_box(
        np.array([np.nan, -np.inf, -2.5, -1e30]),
        np.array([np.nan, np.inf, 4.0, 1e30]),
    )
    # NaN and -inf are the same statement ("no lower bound") once translated, and a
    # magnitude past the sentinel is already unbounded to the simplex.
    assert lo.tolist() == [-_LP_INF, -_LP_INF, -2.5, -_LP_INF]
    assert hi.tolist() == [_LP_INF, _LP_INF, 4.0, _LP_INF]
    assert not np.isnan(lo).any() and not np.isnan(hi).any()


def test_nan_upper_bound_still_solves_as_unbounded():
    """The Benders recourse LP that exposed this: ``min -w``, ``-w ≤ 0``, ``w ≥ 0``.

    Genuinely unbounded below. Before the translation the NaN reached the simplex
    and the answer depended on which guard was asked; after it the box is the
    sentinel and the ray certifies on its own merits.
    """
    r = solve_lp(
        np.array([-1.0]),
        A_ub=np.array([[-1.0]]),
        b_ub=np.array([0.0]),
        bounds=[(0.0, float("nan"))],
    )
    assert r.status == SolveStatus.UNBOUNDED

    # `None` is the caller-facing spelling of the same thing and must agree.
    r_none = solve_lp(
        np.array([-1.0]),
        A_ub=np.array([[-1.0]]),
        b_ub=np.array([0.0]),
        bounds=[(0.0, None)],
    )
    assert r_none.status == SolveStatus.UNBOUNDED


def test_binding_refuses_a_nan_bound_loudly():
    """A caller that skips the translation gets an error, not a lucky verdict."""
    from discopt._rust import solve_lp_warm_csc_py

    a = np.array([[-1.0, 1.0]])  # one row: structural `w`, plus its slack
    with pytest.raises(ValueError, match="is NaN"):
        solve_lp_warm_csc_py(
            np.ascontiguousarray([-1.0, 0.0]),
            1,
            2,
            np.ascontiguousarray([0, 1, 2], dtype=np.int64),
            np.ascontiguousarray([0, 0], dtype=np.int64),
            np.ascontiguousarray(a.ravel()[a.ravel() != 0.0]),
            np.ascontiguousarray([0.0]),
            np.ascontiguousarray([0.0, 0.0]),
            np.ascontiguousarray([np.nan, _LP_INF]),  # the NaN upper bound
            None,
            None,
            1e-9,
            1000,
            None,
        )
