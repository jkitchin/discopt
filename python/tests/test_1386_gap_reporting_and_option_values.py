"""#1386 -- the reported gap must be the criterion's own number, and an option
value that cannot be satisfied must be refused.

Two defects, one parameter.

1. ``SolveResult.gap`` on the B&B routes was the tree's hybrid gap, whose
   denominator is floored at 1.0, while the convergence test divides by
   ``max(|ub|, |lb|, 1e-10)``. Measured on the six-hump camel, every row of
   which converged exactly as documented::

       gap_tolerance | reported gap | criterion's gap
       0.9           |  8.5706      | 0.8955
       0.99          | 56.0947      | 0.9825
       1.0           | 258.3641     | 1.0000

   so ``assert r.gap <= gap_tolerance`` -- the natural check -- failed on a
   correctly converged solve, by up to 258x.

2. ``gap_tolerance=float("nan")`` and ``gap_tolerance=-1e-3`` were accepted
   silently, though ``solve()`` refuses an unknown option NAME on the explicit
   grounds that "a swallowed option would leave the solver at its default while
   you believe it was set". A NaN tolerance is that same failure: every
   ``rel_gap <= tol`` comparison is False, so the relative arm is switched off.

RETRACTION (CLAUDE.md 11). #1386 as filed also claimed ``status="optimal"`` at a
loose tolerance was itself misleading. Measured, that is wrong and no test here
asserts it: at 0.9/0.99/1.0 the search stops exactly when the relative gap
reaches the requested value (0.8955 / 0.9825 / 1.0) and records
``gap_criterion="relative"``. The criterion does precisely what it documents;
only the reported NUMBER disagreed with it.
"""

from __future__ import annotations

import math

import pytest

from discopt import Model

# Published global minimum of the six-hump camel.
CAMEL_OPT = -1.0316284535


def camel() -> Model:
    m = Model()
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize((4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2)
    return m


# --------------------------------------------------------------------------
# 1. the reported gap agrees with the criterion that set the status
# --------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("gap_tolerance", [1e-4, 1e-1, 0.9, 0.99, 1.0])
def test_a_converged_solve_reports_a_gap_within_its_tolerance(gap_tolerance):
    """The invariant a caller is entitled to: converged => gap <= what I asked for."""
    r = camel().solve(gap_tolerance=gap_tolerance, time_limit=60)

    assert r.status == "optimal"
    assert r.gap is not None
    criterion = (r.solver_stats or {}).get("gap_criterion")
    assert criterion is not None, "a converged exit must name the arm that closed it"
    if criterion == "relative":
        assert r.gap <= gap_tolerance + 1e-12, (
            f"reported gap {r.gap} exceeds the requested tolerance {gap_tolerance} "
            f"on a solve this same criterion just called converged"
        )
    else:
        assert r.gap == pytest.approx(0.0)


@pytest.mark.smoke
@pytest.mark.parametrize("gap_tolerance", [1e-4, 0.9, 1.0])
def test_the_reported_gap_is_the_criterions_arithmetic(gap_tolerance):
    """Recomputed independently from the returned pair, not read back from a stat."""
    from discopt.solver import _relative_gap_from_objective_bound

    r = camel().solve(gap_tolerance=gap_tolerance, time_limit=60)
    criterion = (r.solver_stats or {}).get("gap_criterion")
    if criterion != "relative":
        pytest.skip(f"this tolerance closed on the {criterion!r} arm")

    expected = _relative_gap_from_objective_bound(r.objective, r.bound)
    assert expected is not None
    assert r.gap == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.smoke
def test_the_loose_tolerance_row_that_reported_258x_its_own_gap():
    """The issue's headline row, pinned as an invariant rather than a float."""
    r = camel().solve(gap_tolerance=1.0, time_limit=60)

    assert r.status == "optimal"
    assert (r.solver_stats or {})["gap_criterion"] == "relative"
    assert r.gap is not None and r.gap <= 1.0 + 1e-12, r.gap
    # Soundness is untouched by a reporting fix: the bound is still valid.
    assert r.bound is not None and r.bound <= CAMEL_OPT + 1e-6
    assert r.objective is not None and r.objective >= CAMEL_OPT - 1e-6


@pytest.mark.smoke
def test_a_gap_closed_on_the_absolute_arm_reports_zero():
    """Near a zero optimum the relative number is the degenerate one.

    ``min x`` over ``(x<=3) or (x>=7)``, ``x in [0,10]`` converges at objective
    ~2.5e-09 against a bound of 0.0: absolute gap ~2.5e-09, relative gap exactly
    1.0. Reporting 1.0 for a solve that close to the true optimum would be
    arithmetically honest and practically useless.
    """
    m = Model("loa_near_zero")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 3], [x >= 7]], name="choice")
    m.minimize(x)
    r = m.solve(time_limit=30, gdp_method="loa")

    assert r.status == "optimal"
    assert (r.solver_stats or {})["gap_criterion"] == "absolute"
    assert r.gap == pytest.approx(0.0, abs=1e-9)


@pytest.mark.smoke
def test_an_open_exit_keeps_the_933_floored_gap():
    """Scope control: an exit that met no criterion claims nothing about a
    tolerance, so there is nothing for its gap to be consistent with. #933
    defines and tests the floored ``|obj-bound| / max(1, |obj|)`` there, and
    this fix must not reach it."""
    r = camel().solve(time_limit=0.05)
    if r.objective is None or r.bound is None:
        pytest.skip("the budget was too short to produce a pair to check")

    assert (r.solver_stats or {}).get("gap_criterion") is None
    assert r.gap == pytest.approx(
        abs(r.objective - r.bound) / max(1.0, abs(r.objective)), rel=1e-9
    )


# --------------------------------------------------------------------------
# 2. option values that cannot be satisfied are refused
# --------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [float("nan"), -1e-3, -1.0, float("inf")])
def test_an_unsatisfiable_gap_tolerance_is_refused(bad):
    with pytest.raises(ValueError, match="gap_tolerance"):
        camel().solve(gap_tolerance=bad, time_limit=10)


@pytest.mark.smoke
def test_a_nan_time_limit_is_refused():
    """``Model.solve`` derives ``max(0.0, time_limit - spent)``, which silently
    turns a NaN limit into 0.0 -- so the check has to run before that."""
    with pytest.raises(ValueError, match="time_limit"):
        camel().solve(time_limit=float("nan"))


@pytest.mark.smoke
def test_the_check_also_binds_for_direct_solve_model_callers():
    from discopt.solver import solve_model

    with pytest.raises(ValueError, match="gap_tolerance"):
        solve_model(camel(), gap_tolerance=math.nan, time_limit=10)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "ok_kwargs",
    [
        {"gap_tolerance": 0.0},  # asks for the absolute arm alone
        {"gap_tolerance": 1.0},  # coarse, but measured to behave as documented
        {"gap_tolerance": 1e6},
        {"time_limit": -1.0},  # a computed `deadline - now` can land here
        {"time_limit": 0.0},
    ],
)
def test_values_that_are_coarse_but_satisfiable_are_still_accepted(ok_kwargs):
    """The refusal must not become a policy the evidence does not support."""
    kwargs = {"time_limit": 30, **ok_kwargs}
    r = camel().solve(**kwargs)
    assert r.status in {"optimal", "feasible", "time_limit"}
