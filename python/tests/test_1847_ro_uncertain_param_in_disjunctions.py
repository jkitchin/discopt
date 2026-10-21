"""#1847: an uncertain parameter inside an ``if_else`` (or any other non-row entry)
bypassed the robust counterpart.

The formulations robustify plain ``Constraint`` rows and the objective. An
``if_else`` lowers to a ``_DisjunctiveConstraint`` that ``formulate()`` passed
through untouched, as it did an ``if_then`` indicator row, an ``either_or`` /
``Disjunct`` block and a complementarity: a parameter there was solved at its
nominal value and the result reported as robust. Every set kind now refuses, up
front and in the post-check. (The ``dm.custom`` half of the issue is covered by
``test_issue1833_ro_vector_rows.py::test_custom_call_of_uncertain_param_is_refused``.)

``AffineDecisionRule`` had the same blind spot for its recourse variable: a ``y``
inside a disjunct row stayed the retired static variable, decoupled from the
policy. It refuses too.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mpec import complementarity
from discopt.ro import (
    AffineDecisionRule,
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    RobustCounterpart,
)
from discopt.ro.formulations._common import (
    assert_no_uncertain_params_remain,
    uncertain_param_outside_rows,
)

pytestmark = pytest.mark.smoke

SETS = {
    "box": lambda p: BoxUncertaintySet(p, delta=np.array([1.0, 1.0])),
    "polyhedral": lambda p: PolyhedralUncertaintySet(
        p, A=np.vstack([np.eye(2), -np.eye(2)]), b=np.ones(4)
    ),
    "ellipsoidal": lambda p: EllipsoidalUncertaintySet(p, rho=1.0),
}


def _place(where, m, x, y, p):
    if where == "if_else":
        m.subject_to(x + dm.if_else(y >= 1, p[0], 0) <= 13)
    elif where == "if_then":
        z = m.binary("z")
        m.if_then(z, [x + p[0] <= 13])
    elif where == "either_or":
        m.either_or([[x + p[0] <= 13], [x <= 2]])
    elif where == "nested_block":
        # The uncertain row sits two levels down: d0 -> {a, b}.
        a, b = m.make_disjunct("a"), m.make_disjunct("b")
        a.subject_to(x + p[1] <= 13)
        b.subject_to(y <= 1)
        d0, d1 = m.make_disjunct("d0"), m.make_disjunct("d1")
        d0.add_disjunction([a, b])
        d1.subject_to(x <= 2)
        m.add_disjunction([d0, d1])
    elif where == "complementarity":
        m._complementarities.append(complementarity(x - p[0], 20 - x, name="c"))
    else:  # pragma: no cover
        raise AssertionError(where)


WHERE = ["if_else", "if_then", "either_or", "nested_block", "complementarity"]


@pytest.mark.parametrize("kind", sorted(SETS))
@pytest.mark.parametrize("where", WHERE)
def test_uncertain_param_outside_rows_is_refused(kind, where):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=0, ub=5)
    p = m.parameter("p", value=np.array([3.0, 1.0]))
    _place(where, m, x, y, p)
    m.subject_to(x + y <= 11)
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="no robust formulation rewrites"):
        RobustCounterpart(m, SETS[kind](p)).formulate()


@pytest.mark.parametrize("where", WHERE)
def test_post_check_sees_non_row_entries(where):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=0, ub=5)
    p = m.parameter("p", value=np.array([3.0, 1.0]))
    _place(where, m, x, y, p)
    assert uncertain_param_outside_rows(m, {"p"})
    assert not uncertain_param_outside_rows(m, {"other"})
    with pytest.raises(NotImplementedError, match="could not robustify"):
        assert_no_uncertain_params_remain(m, {"p"}, kind="box")


def test_certain_disjunction_still_formulates():
    # A disjunction that holds no uncertain parameter is not robustified and is
    # not refused: only the uncertain row is rewritten.
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=0, ub=5)
    p = m.parameter("p", value=np.array([3.0, 1.0]))
    m.subject_to(x + dm.if_else(y >= 1, 3.0, 0) <= 13)
    m.subject_to(x + p[0] <= 12)
    m.maximize(x)
    RobustCounterpart(m, SETS["box"](p)).formulate()
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(8.0, abs=1e-6)  # x <= 12 - (3 + 1)


def test_affine_rule_refuses_recourse_var_in_a_disjunct():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=0, ub=5)
    p = m.parameter("p", value=np.array([3.0, 1.0]))
    m.subject_to(x + y + p[0] <= 13)
    m.either_or([[y <= 1], [x <= 2]])
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="decoupled from the policy"):
        AffineDecisionRule(y, uncertain_params=[p], prefix="adr").apply()
