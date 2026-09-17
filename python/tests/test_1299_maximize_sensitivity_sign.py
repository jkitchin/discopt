"""#1299: every sensitivity quantity must be reported in the sense the user wrote.

Each NLP evaluator and the Rust kernel minimize, so a MAXIMIZE model is solved as
``-f``. Three user-facing sites copied that internal value out without undoing the
flip:

* ``Sensitivity.objective`` (``sensitivity.py``) took POUNCE's raw objective.
* ``DiffSolveResult.objective`` and its ``_sensitivity`` on the continuous path.
* ``_compute_sensitivity_at_solution``, which backs ``SolveResult.gradient`` and
  the fix-and-differentiate integer path.

``Sensitivity.dobj_dp`` was already right -- it differentiates the model's own
objective expression -- which is what makes it the cross-check below.

The model is ``max -(x^2) + p*x`` over ``x in [-10, 10]``, whose value function is
``V(p) = p^2/4`` with ``x*(p) = p/2``, so at ``p = 2``: ``V = 1``, ``dV/dp = 1``.
Every wrong site returned exactly ``-1`` for one of those.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.differentiable import differentiable_solve
from discopt.modeling.core import objective_sense_sign

P0 = 2.0
V_TRUE = P0**2 / 4.0  # 1.0
DV_TRUE = P0 / 2.0  # 1.0


def _quadratic(sense):
    """``max -(x^2) + p x`` and the minimize model with the identical optimum.

    The minimize twin is ``min (x^2) - p x``: same argmin, value ``-V(p)``. It is
    the control -- a fix that negated unconditionally would break it.
    """
    m = dm.Model(f"i1299_{sense}")
    p = m.parameter("p", value=P0)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    if sense == "max":
        m.maximize(-(x * x) + p * x)
    else:
        m.minimize(x * x - p * x)
    return m, p, x


def _expected(sense):
    """(objective, d obj*/dp) in the user's own sense."""
    return (V_TRUE, DV_TRUE) if sense == "max" else (-V_TRUE, -DV_TRUE)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_sensitivity_objective_is_in_the_models_own_sense(sense):
    m, _, _ = _quadratic(sense)
    v, dv = _expected(sense)
    s = m.sensitivity()
    assert s.objective == pytest.approx(v, abs=1e-6)
    # dobj_dp never had the bug; equality of signs is the invariant that broke.
    assert s.dobj_dp[0] == pytest.approx(dv, abs=1e-5)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_differentiable_solve_objective_and_gradient(sense):
    m, p, _ = _quadratic(sense)
    v, dv = _expected(sense)
    r = differentiable_solve(m)
    assert r.objective == pytest.approx(v, abs=1e-6)
    assert float(np.asarray(r.gradient(p))) == pytest.approx(dv, abs=1e-5)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_solve_result_gradient(sense):
    m, p, _ = _quadratic(sense)
    _, dv = _expected(sense)
    r = m.solve()
    assert float(np.asarray(r.gradient(p))) == pytest.approx(dv, abs=1e-5)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_integer_fix_and_differentiate_gradient(sense):
    """The fix-and-differentiate path reported the objective correctly (it comes
    from ``solve_model``) while its gradient came from the unflipped envelope."""
    m = dm.Model(f"i1299_int_{sense}")
    p = m.parameter("p", value=P0)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    z = m.binary("z")
    if sense == "max":
        m.maximize(-(x * x) + p * x - z)
    else:
        m.minimize(x * x - p * x + z)
    v, dv = _expected(sense)
    r = differentiable_solve(m)
    # z* = 0 either way, so the value function is the continuous one.
    assert r.objective == pytest.approx(v, abs=1e-6)
    assert float(np.asarray(r.gradient(p))) == pytest.approx(dv, abs=1e-5)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_l3_objective_and_both_gradients(sense):
    """``differentiable_solve_l3`` reports an objective, an envelope gradient and an
    implicit-differentiation gradient -- all three were in the internal sense."""
    from discopt._relax.differentiable import differentiable_solve_l3

    m, p, _ = _quadratic(sense)
    v, dv = _expected(sense)
    r = differentiable_solve_l3(m, nlp_solver="ipm")
    assert r.objective == pytest.approx(v, abs=1e-6)
    assert float(np.asarray(r.gradient(p))) == pytest.approx(dv, abs=1e-5)
    # The L3 route must not disagree with the L1 envelope value it falls back to.
    assert float(np.asarray(r.implicit_gradient(p))) == pytest.approx(dv, abs=1e-4)


def test_gradient_sign_agrees_across_the_three_entry_points():
    """The defect was that these three disagreed; pin them to each other."""
    m, p, _ = _quadratic("max")
    g_sens = m.sensitivity().dobj_dp[0]
    g_diff = float(np.asarray(differentiable_solve(m).gradient(p)))
    g_solve = float(np.asarray(m.solve().gradient(p)))
    assert g_sens == pytest.approx(g_diff, abs=1e-5)
    assert g_sens == pytest.approx(g_solve, abs=1e-5)
    assert g_sens > 0  # raising p raises the max; all three returned -1 for one


def test_constrained_maximize_gradient_uses_the_multiplier_term():
    """With the bound active the envelope term is the multiplier, not df/dp alone.

    ``max p*x`` s.t. ``x <= 3``, ``p = 2`` gives ``V(p) = 3p`` and ``dV/dp = 3``.
    A sign error in the Lagrangian half shows up here and not in the unconstrained
    case above.
    """
    m = dm.Model("i1299_con")
    p = m.parameter("p", value=P0)
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x <= 3.0)
    m.maximize(p * x)
    r = differentiable_solve(m)
    assert r.objective == pytest.approx(3.0 * P0, abs=1e-5)
    assert float(np.asarray(r.gradient(p))) == pytest.approx(3.0, abs=1e-4)


def test_objective_sense_sign_helper():
    m_max, _, _ = _quadratic("max")
    m_min, _, _ = _quadratic("min")
    assert objective_sense_sign(m_max) == -1.0
    assert objective_sense_sign(m_min) == 1.0
    # No objective yet: nothing has been flipped, so nothing to undo.
    assert objective_sense_sign(dm.Model("empty")) == 1.0
