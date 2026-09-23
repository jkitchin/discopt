"""#1385 -- a model with no columns is optimal at its constant objective.

Before the fix, ``Model(); m.minimize(3.0)`` classified as a pure LP, HiGHS
answered ``kModelEmpty``, that name was absent from ``_solve_lp``'s status
dispatch, and the catch-all returned ``status="error"`` with ``objective=None``
-- the constant discarded and the failure attributed to "a nonlinear term with
no envelope".

The tests assert the invariant (the sole feasible point attains the constant, so
the certificate is exact) rather than pinning one route's floats, and each
control below exists so the fix cannot pass by short-circuiting more than it
should.
"""

from __future__ import annotations

import math

import pytest
from discopt import Model
from discopt.modeling.core import Constant


@pytest.mark.smoke
@pytest.mark.parametrize("constant", [0.0, 3.0, -2.5, 1e6])
def test_variable_free_minimize_is_optimal_at_the_constant(constant):
    m = Model()
    m.minimize(constant)
    r = m.solve()

    assert r.status == "optimal", f"status={r.status!r} (was 'error' before #1385)"
    assert r.objective == pytest.approx(constant)
    assert r.bound == pytest.approx(constant)
    assert r.gap_certified is True
    # The optimum is attained exactly: there is one feasible point and it is it.
    assert r.gap == pytest.approx(0.0)


@pytest.mark.smoke
@pytest.mark.parametrize("constant", [0.0, 7.0, -4.25])
def test_variable_free_maximize_is_optimal_at_the_constant(constant):
    """The sense cannot matter: one feasible point attains the constant either way."""
    m = Model()
    m.maximize(constant)
    r = m.solve()

    assert r.status == "optimal"
    assert r.objective == pytest.approx(constant)
    assert r.bound == pytest.approx(constant)
    assert r.gap_certified is True


@pytest.mark.smoke
def test_the_returned_point_is_the_empty_assignment():
    """``x`` is an empty dict, not ``None``: the model HAS a solution."""
    m = Model()
    m.minimize(2.0)
    r = m.solve()

    assert r.x is not None, "a solved model must carry its point"
    assert dict(r.x) == {}
    assert r.node_count == 0


# --------------------------------------------------------------------------
# Controls: the short circuit must not reach past the case it is written for.
#
# The helper is imported inside each control rather than at module scope so the
# behavioural tests above still COLLECT against a pre-#1385 tree -- that is how
# their fail-before was measured (CLAUDE.md 8).
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_model_with_a_column_is_untouched():
    """Same constant objective, but one declared variable -> the ordinary route."""
    m = Model()
    m.continuous("x", lb=0, ub=1)
    m.minimize(3.0)
    r = m.solve()

    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0)
    assert r.x is not None and "x" in r.x


@pytest.mark.smoke
def test_an_ordinary_nonlinear_solve_is_untouched():
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    m.minimize((x - 3) ** 2)
    r = m.solve()

    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert float(r.x["x"]) == pytest.approx(3.0, abs=1e-3)


@pytest.mark.smoke
def test_helper_declines_a_model_that_declares_a_variable():
    """The gate is "no columns", checked directly so the reason is explicit."""
    from discopt.solver import _constant_objective_result

    m = Model()
    m.continuous("x", lb=0, ub=1)
    m.minimize(3.0)

    assert _constant_objective_result(m, 0.0) is None


@pytest.mark.smoke
def test_helper_declines_a_non_constant_objective_node():
    """Fails closed: an objective that is not a ``Constant`` falls through.

    A zero-column model cannot build one today, so this pins the *policy* --
    fall through to the honest failure rather than guess a value -- against a
    future expression type reaching here.
    """
    from discopt.solver import _constant_objective_result

    m = Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    # Drop the column but keep the non-constant objective: exactly the shape the
    # guard must refuse to answer for.
    m._variables.clear()

    assert _constant_objective_result(m, 0.0) is None


@pytest.mark.smoke
def test_helper_declines_a_vector_constant():
    """A vector objective on a variable-free model has no scalar optimum.

    This pins the *helper's* defensive check, so the invalid objective is
    installed directly rather than through ``minimize`` -- since #1445 the
    boundary refuses it, and internal machinery (deserialization, reformulation
    passes) is the only way such an objective can now reach the solver. The
    assertion is unchanged; only how the state is reached is.
    """
    from discopt.modeling.core import Objective, ObjectiveSense
    from discopt.solver import _constant_objective_result

    m = Model()
    m._objective = Objective(Constant([1.0, 2.0]), ObjectiveSense.MINIMIZE)

    assert _constant_objective_result(m, 0.0) is None


@pytest.mark.smoke
def test_minimize_refuses_a_vector_constant_at_the_boundary():
    """#1445 -- the same objective is now refused where it is written."""
    m = Model()
    with pytest.raises(ValueError, match=r"minimize\(\) needs a scalar objective"):
        m.minimize(Constant([1.0, 2.0]))
    assert m._objective is None, "a refused objective must not be stored"


@pytest.mark.smoke
def test_helper_declines_a_non_finite_constant():
    from discopt.solver import _constant_objective_result

    for bad in (math.inf, -math.inf, math.nan):
        m = Model()
        m.minimize(bad)
        assert _constant_objective_result(m, 0.0) is None, f"accepted {bad!r}"


@pytest.mark.smoke
def test_helper_declines_a_model_with_no_objective():
    from discopt.solver import _constant_objective_result

    m = Model()
    assert _constant_objective_result(m, 0.0) is None
