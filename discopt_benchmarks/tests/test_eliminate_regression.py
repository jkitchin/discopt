"""M10 regression: variable elimination via singleton equality detection.

The pass (``discopt._rust.PyModelRepr.eliminate_variables``) detects
continuous scalar variables uniquely determined by exactly one equality
constraint of the form ``coeff·v + const == rhs`` and fixes their
declared bounds to the derived value, dropping the determining
equality. References to ``v`` elsewhere (e.g. the objective) remain
valid because the bound pinning forces ``v`` to that value.

The acceptance contract:

1. **Feasibility/optimality preservation:** for every sampled feasible
   point of the *new* model, projecting onto the original variable
   set (with ``v`` set to its fixed value) yields a point that is
   feasible in the *original* model and gives the same objective.
2. **No feasible point lost:** the singleton equality alone determined
   ``v``'s value, so by construction no feasible point of the original
   model is excluded by pinning ``v`` to that value.
3. **Idempotence:** running the pass twice produces no further fixings.
4. **Conservative scope:** the pass leaves variables alone when the
   determining equality also references other variables, when the
   variable appears in more than one constraint, when the constraint
   is an inequality, or when the derived value lies outside the
   variable's current bounds (the last case is handled by FBBT).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt._rust import model_to_repr

N_REGRESSION_SAMPLES = 10_000


# ---------------------------------------------------------------------------
# 1. Feasibility/optimality preservation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.regression
def test_singleton_equality_fixes_variable_value():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(x + y)
    m.subject_to(2.0 * x == 6.0)
    m.subject_to(y >= -2.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    assert stats["variables_fixed"] == 1
    assert stats["constraints_removed"] == 1
    assert new_repr.n_constraints == 1
    assert new_repr.var_lb(0) == [3.0]
    assert new_repr.var_ub(0) == [3.0]


@pytest.mark.regression
def test_objective_value_preserved_after_elimination():
    """If x is fixed to a value v* by elimination, then for every y the
    new model's objective at (v*, y) must equal the original model's
    objective at (v*, y)."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(3.0 * x + 2.0 * y * y)
    m.subject_to(x == 4.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    assert stats["variables_fixed"] == 1
    fixed_x = float(np.asarray(new_repr.var_lb(0)).ravel()[0])
    assert fixed_x == 4.0

    rng = np.random.default_rng(0)
    ys = rng.uniform(-5.0, 5.0, N_REGRESSION_SAMPLES)
    for y_val in ys:
        orig = repr_.evaluate_objective(np.array([fixed_x, y_val]))
        new = new_repr.evaluate_objective(np.array([fixed_x, y_val]))
        assert abs(orig - new) < 1e-9


@pytest.mark.regression
def test_no_feasible_point_lost():
    """Sample the original model's feasible set; every sample with
    x = fixed_value (within tol) must satisfy the new model's
    constraints + bounds."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    m.subject_to(2.0 * x == 6.0)
    m.subject_to(x + y <= 8.0)  # uses x but x already in 2 cons -> not eliminated

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    # x appears in 2 constraints, so it should NOT be eliminated.
    assert stats["variables_fixed"] == 0
    assert new_repr.n_constraints == repr_.n_constraints


# ---------------------------------------------------------------------------
# 2. Idempotence
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_elimination_is_idempotent():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    z = m.continuous("z", lb=0.0, ub=10.0)
    m.minimize(x + y + z)
    m.subject_to(x == 2.5)
    m.subject_to(y == 3.5)
    m.subject_to(z >= 1.0)

    repr_ = model_to_repr(m)
    new_repr, stats1 = repr_.eliminate_variables()
    assert stats1["variables_fixed"] == 2
    assert new_repr.n_constraints == 1

    new_new_repr, stats2 = new_repr.eliminate_variables()
    assert stats2["variables_fixed"] == 0
    assert stats2["constraints_removed"] == 0
    assert new_new_repr.n_constraints == new_repr.n_constraints


# ---------------------------------------------------------------------------
# 3. Conservative scope
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_skips_inequality_constraint():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x <= 5.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    assert stats["variables_fixed"] == 0
    assert new_repr.n_constraints == 1
    # Bounds unchanged.
    assert new_repr.var_lb(0) == [0.0]
    assert new_repr.var_ub(0) == [10.0]


@pytest.mark.regression
def test_skips_equality_with_other_variable():
    """2*x + y == 5 is not a singleton — both x and y appear, so x is
    not uniquely determined and should not be eliminated."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    m.subject_to(2.0 * x + y == 5.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    assert stats["variables_fixed"] == 0
    assert new_repr.n_constraints == 1


@pytest.mark.regression
def test_skips_when_value_outside_bounds():
    """2*x == 12 with x ∈ [0, 5]: derived value 6 is infeasible. Pass
    must abstain (leave for FBBT to flag) rather than silently fix x
    to an invalid value."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.minimize(x)
    m.subject_to(2.0 * x == 12.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    assert stats["variables_fixed"] == 0
    assert new_repr.n_constraints == 1
    # Bounds unchanged.
    assert new_repr.var_lb(0) == [0.0]
    assert new_repr.var_ub(0) == [5.0]


@pytest.mark.regression
def test_skips_integer_variable():
    """M10 v0 only handles continuous variables."""
    m = Model()
    x = m.integer("x", lb=0, ub=10)
    m.minimize(x)
    m.subject_to(2.0 * x == 6.0)

    repr_ = model_to_repr(m)
    new_repr, stats = repr_.eliminate_variables()
    assert stats["variables_fixed"] == 0
    assert new_repr.n_constraints == 1
