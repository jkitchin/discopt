"""A warm start must survive a solve-time reformulation (#1255).

``initial_solution`` is flattened against the variables the USER declared. The
GDP pass then lowers each disjunction into selector binaries at solve time, so by
the time the warm-start site evaluates the point the model has more columns than
the point has entries — and the evaluator raised
``ValueError: objective: x: expected length 5, got 3`` out of a solve that
succeeds without any warm start at all. A hint must never be able to fail a
solve.

Two properties are checked: the solve completes and returns the same optimum it
returns cold, and the completion actually recovers the selector the given point
implies (rather than padding it with a default that the feasibility gate then
throws away). The second is the part that needs a search: the selectors carry
``sum(y) == 1``, so moving off a wrong disjunct onto the right one is a swap —
two flips, the first of them uphill.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp as lower_gdp
from discopt._relax.primal_heuristics import row_violations
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.mpec import complementarity, reformulate_gdp
from discopt.warm_start import complete_initial_point, validate_initial_solution


def _gdp_model():
    """The #1255 model; optimum ``z = -4``."""
    m = dm.Model("warmstart_gdp")
    y = m.continuous("y", lb=-8.0, ub=-1.0)
    n = m.continuous("n", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=-10.0, ub=0.0)
    m.subject_to(10.0**y + n == 1e-3)
    reformulate_gdp(m, [complementarity(n * 1e3, -4.0 - y, name="sat")])
    m.subject_to(10.0 ** (y - z) <= 1.0)
    m.minimize(z)
    return m, y, n, z


@pytest.mark.smoke
def test_warm_started_solve_does_not_raise():
    """The reported crash, end to end."""
    m, y, n, z = _gdp_model()
    r = m.solve(initial_solution={y: -4.0, n: 9e-4, z: -4.0}, time_limit=30)
    assert r.objective is not None, "the warm-started solve must return a result"
    assert abs(float(r.objective) - (-4.0)) < 1e-4, (
        f"warm-started optimum {r.objective!r} differs from the cold optimum -4.0"
    )


@pytest.mark.unit
def test_completion_recovers_the_indicator_the_point_implies():
    """The lowered point must be FEASIBLE, not merely the right length."""
    m, y, n, z = _gdp_model()
    x0 = validate_initial_solution(m, {y: -4.0, n: 9e-4, z: -4.0})
    lowered = lower_gdp(m, method="big-m")
    n_cols = sum(int(v.size) for v in lowered._variables)
    assert n_cols > x0.size, "the GDP lowering must add columns for this to test anything"

    completed = complete_initial_point(lowered, x0)
    assert completed is not None and completed.size == n_cols
    assert np.allclose(completed[: x0.size], x0), "the user's values must be preserved"

    viol = row_violations(make_evaluator(lowered), completed)
    worst = float(np.max(viol)) if viol.size else 0.0
    assert worst <= 1e-9, (
        f"the completed point violates a lowered row by {worst:.3e}; the repair "
        "did not find the disjunct the point satisfies"
    )


@pytest.mark.unit
def test_completion_refuses_a_point_that_is_not_a_prefix():
    """A point that cannot describe this model's columns is declined, not padded."""
    m, _y, _n, _z = _gdp_model()
    n_cols = sum(int(v.size) for v in m._variables)
    assert complete_initial_point(m, np.zeros(n_cols + 1)) is None, "too long: refuse"
    assert complete_initial_point(m, np.array([0.0, np.nan])) is None, "non-finite: refuse"
    same = complete_initial_point(m, np.zeros(n_cols))
    assert same is not None and same.size == n_cols, "an exact-length point passes through"
