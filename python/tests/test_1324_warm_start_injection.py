"""#1324: a warm start is a hint; it must not crash a solve or poison an incumbent.

Two injection sites #1316 did not reach.

1. ``nlp_bb=True`` evaluated the user's point against the *reformulated* model.
   Factorable reformulation appends a column per monomial auxiliary, so the
   point is shorter than the vector the evaluator expects and
   ``evaluate_objective`` raised ``ValueError: objective: x: expected length 3,
   got 2`` -- out of a public ``Model.solve`` call that succeeds without the
   warm start. The spatial path has completed the point across reforms since
   #1255; this one did not.

2. ``_solve_milp_bb`` checked the seed's integrality and its rows but never its
   BOX, so a row-feasible point outside the declared bounds became the tree's
   incumbent and the exit guard had to raise ``MILP-BB returned an infeasible
   point labeled feasible/optimal``. The guard is what kept it from being a
   false certificate; the seed should never have been injected.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import solve_model


def _reformulated_minlp():
    """``n*x`` and ``(x-2.2)**2*n`` both lift, so the solver's model has columns
    the user's point does not."""
    m = dm.Model("i1324_reform")
    n = m.integer("n", lb=0, ub=5)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(n * x <= 7.0)
    m.minimize((x - 2.2) ** 2 * n - 3 * n + 0.5 * x)
    return m, n, x


@pytest.mark.smoke
def test_nlp_bb_warm_start_survives_a_factorable_reformulation():
    """The issue's repro verbatim."""
    m, _n, _x = _reformulated_minlp()
    r0 = m.solve()
    assert r0.status in ("optimal", "feasible")

    r1 = m.solve(nlp_bb=True, warm_start=r0)
    assert r1.status in ("optimal", "feasible", "time_limit", "node_limit")
    assert r1.objective is not None
    assert r1.objective <= r0.objective + 1e-6


@pytest.mark.smoke
def test_nlp_bb_initial_solution_survives_a_factorable_reformulation():
    """Same path, reached through ``initial_solution=`` on an unchanged model."""
    m, n, x = _reformulated_minlp()
    ref = m.solve()
    r = m.solve(nlp_bb=True, initial_solution={n: 4, x: 1.75})
    assert r.status in ("optimal", "feasible", "time_limit", "node_limit")
    assert r.objective is not None
    assert r.objective <= ref.objective + 1e-6


@pytest.mark.smoke
def test_nlp_bb_warm_start_still_works_when_no_reform_happens():
    """No false positive: a model whose columns already line up is untouched."""
    m = dm.Model("i1324_plain")
    y = m.integer("y", lb=0, ub=4)
    z = m.continuous("z", lb=0.0, ub=4.0)
    m.subject_to(y + z <= 5.0)
    m.minimize((z - 1.5) ** 2 - 2.0 * y)
    r0 = m.solve()
    r1 = m.solve(nlp_bb=True, warm_start=r0)
    assert r1.objective == pytest.approx(r0.objective, abs=1e-4)


# ── 2. the MILP-BB seed's box ────────────────────────────────────────────


def _milp_with_a_loose_row():
    """Integer ``y`` in [0,3]^3 whose only row is far from binding, so a point
    OUTSIDE the box is still row-feasible -- and scores better than anything
    inside it, which is what makes an unchecked seed the answer."""
    m = dm.Model("i1324_milpbb")
    y = m.integer("y", shape=(3,), lb=0, ub=3)
    m.subject_to(dm.sum(y[i] for i in range(3)) <= 10.0)
    m.minimize(-(2.0 * y[0] + y[1] + y[2]))
    return m


@pytest.mark.smoke
def test_milp_bb_drops_an_out_of_box_initial_point_and_still_solves():
    """Before the fix this raised ``RuntimeError: MILP-BB returned an infeasible
    point labeled feasible/optimal: bound on x[0]``."""
    m = _milp_with_a_loose_row()
    optimum = solve_model(m, nlp_solver="simplex", abs_gap_tolerance=1e-9).objective
    assert optimum == pytest.approx(-12.0, abs=1e-6)  # y = (3, 3, 3)

    r = solve_model(
        _milp_with_a_loose_row(),
        initial_point=np.array([6.0, 0.0, 1.0]),
        nlp_solver="simplex",
        abs_gap_tolerance=1e-9,
    )
    assert r.status in ("optimal", "feasible")
    assert r.objective == pytest.approx(optimum, abs=1e-6)
    x = np.asarray(r.x["y"], dtype=np.float64).ravel()
    assert np.all(x >= -1e-6) and np.all(x <= 3.0 + 1e-6), f"returned point outside the box: {x}"


@pytest.mark.smoke
def test_milp_bb_still_accepts_an_in_box_initial_point():
    """No false positive: a seed inside the box is injected as before."""
    m = _milp_with_a_loose_row()
    r = solve_model(
        m,
        initial_point=np.array([3.0, 3.0, 3.0]),
        nlp_solver="simplex",
        abs_gap_tolerance=1e-9,
    )
    assert r.objective == pytest.approx(-12.0, abs=1e-6)
