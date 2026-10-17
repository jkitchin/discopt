"""#1721: the ``skip_convex_check`` local NLP runs on the user's model, not its lift.

The factorable lift rewrote Rosenbrock's ``100*(x1 - x0**2)**2`` with an aux
``w = x0**2`` boxed ``w >= 0``. As an NLP that lift has a degenerate KKT point at
``(0, 0, w=0)`` (the equality's multiplier grows like ``1/x0``), and the local
solve converged there: ``status="feasible"``, f = 1 at a point where the source
gradient is (-2, 0). And the user's start, flattened against the two declared
columns, crashed ``np.clip`` against the three lifted ones.
"""

import logging

import discopt.modeling as dm
import numpy as np
import pytest


def _rosenbrock():
    m = dm.Model("rb")
    x = m.continuous("x", shape=(2,), lb=-5, ub=5)
    m.minimize((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
    return m, x


def test_rosenbrock_local_nlp_reaches_the_minimizer():
    m, x = _rosenbrock()
    r = m.solve(skip_convex_check=True)
    assert r.status == "feasible"  # local result: never certified
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(r.value(x), [1.0, 1.0], atol=1e-4)


def test_rosenbrock_initial_solution_accepted():
    m, x = _rosenbrock()
    r = m.solve(skip_convex_check=True, initial_solution={x: np.array([-1.2, 1.0])})
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(r.value(x), [1.0, 1.0], atol=1e-4)


def test_partial_start_on_lifted_ratio_model():
    m = dm.Model("r")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(y <= x**2)
    m.maximize(y / (x + 0.5))
    r = m.solve(skip_convex_check=True, initial_solution={x: 1.0})
    assert r.objective == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_solve_continuous_completes_a_short_start_on_the_lift(caplog):
    # The width guard inside ``_solve_continuous`` itself, for any caller that
    # hands it a reformulated model and a start over the user's columns.
    from discopt._relax.factorable_reform import factorable_reformulate
    from discopt.solver import _solve_continuous

    m, _ = _rosenbrock()
    lifted = factorable_reformulate(m)
    n_lifted = sum(int(v.size) for v in lifted._variables)
    assert n_lifted == 3  # the probe fired: the lift really appended a column
    with caplog.at_level(logging.INFO, logger="discopt"):
        r = _solve_continuous(lifted, 30.0, None, 0.0, "ipm", initial_point=np.array([-1.2, 1.0]))
    assert "continuous NLP warm start extended from 2 to 3 columns" in caplog.text
    assert r.status != "error"
    # No optimality assertion: from this start the LIFTED NLP still falls into
    # the degenerate (0, 0, w=0) point (measured f = 1.0000000350), which is why
    # ``solve_model`` routes the local NLP to the source model instead.
