"""#1775 G4.5: logic-based discrete-steepest descent (spec §5.3, §5.4).

Before this task ``gdp_method="ldsda"`` was reserved and refused. :class:`LDSDA`
is a local search over external variables -- ``exactly(1)`` groups of Booleans,
from disjunctions or named logical constraints -- with Pyomo's neighbor and line
searches. It proves no bound: ``bound`` is ``None`` and it never reports
``optimal``. Done-when (spec): Pyomo's ``four_stage_dynamic_model`` reaches
-23.3053 as in Pyomo's own test, which needed two fixes found on the way: the
converter folds logical constants (an empty ``lor()`` crashed it), and a local
subproblem fixes its Booleans by bounds (as equality rows the model was left with
fewer degrees of freedom than equalities).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.ldsda import LDSDA, directions

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _landscape():
    """Three disjuncts pin x to 1, 2, 3; f = -1.5x^2 + 5.5x - 2 is 2, 3, 1 there."""
    m = dm.Model("landscape")
    x = m.continuous("x", lb=0, ub=4)
    m.either_or([[x == 1], [x == 2], [x == 3]], name="d")
    m.minimize(-1.5 * x**2 + 5.5 * x - 2)
    return m


def test_directions():
    assert sorted(directions(2, "L2")) == [(-1, 0), (0, -1), (0, 1), (1, 0)]
    assert len(directions(3, "Linf")) == 26 and (0, 0, 0) not in directions(3, "Linf")
    with pytest.raises(ValueError, match="direction_norm"):
        directions(2, "L1")


@pytest.mark.parametrize("subproblem", ["local", "global"])
def test_descends_to_the_better_point(subproblem):
    r = _quiet(LDSDA(_landscape(), starting_point=[2], time_limit=60, subproblem=subproblem).run)
    assert r.objective == pytest.approx(1.0, abs=1e-6)
    assert r.solver_stats["point"] == [3]


def test_stops_at_a_local_optimum_and_proves_nothing():
    """From 1, the only neighbor (2, value 3) is worse: a local optimum at 2,
    although 3 holds 1. The status is ``feasible``, never ``optimal``, and there
    is no bound."""
    r = _quiet(LDSDA(_landscape(), starting_point=[1], time_limit=60).run)
    assert r.objective == pytest.approx(2.0, abs=1e-6) and r.solver_stats["point"] == [1]
    assert r.status == "feasible" and not r.gap_certified and r.bound is None
    assert r.solver_stats["stop_reason"] == "local_optimum"


def test_logic_refuted_points_are_never_solved():
    """``not y[1]`` forbids the start's neighbor 2: the search skips it."""
    m = _landscape()
    d = m._constraints[0]
    m.logical(~d.indicators[1], name="no_two")
    a = LDSDA(m, starting_point=[1], time_limit=60)
    r = _quiet(a.run)
    assert r.solver_stats["n_logic_pruned"] >= 1
    assert r.solver_stats["point"] == [1] and r.objective == pytest.approx(2.0, abs=1e-6)


def _grouped(bad: bool = False):
    """External variables from an exactly(1) logical constraint over free Booleans."""
    m = dm.Model("grouped")
    x = m.continuous("x", lb=0, ub=4)
    b = [m.boolean(f"b{i}") for i in range(3)]
    m.logical(dm.exactly(1, b), name="pick")
    if bad:
        m.logical(dm.atmost(1, b), name="bad")
    for i, bi in enumerate(b):
        m.subject_to(x >= (i + 1) * bi.variable)
    m.minimize((x - 2.2) ** 2)
    return m


def test_logical_constraint_groups():
    r = _quiet(
        LDSDA(_grouped(), starting_point=[1], logical_constraint_list=["pick"], time_limit=60).run
    )
    # b0 -> x >= 1, b1 -> x >= 2, b2 -> x >= 3: the best is b0 or b1 (x = 2.2).
    assert r.objective == pytest.approx(0.0, abs=1e-6)


def test_refusals():
    with pytest.raises(ValueError, match="no logical constraint named"):
        LDSDA(_grouped(), logical_constraint_list=["nope"])
    with pytest.raises(ValueError, match="is not exactly"):
        LDSDA(_grouped(bad=True), logical_constraint_list=["bad"])
    with pytest.raises(ValueError, match="no root disjunction named"):
        LDSDA(_landscape(), disjunction_list=["nope"])
    with pytest.raises(ValueError, match="starting_point"):
        LDSDA(_landscape(), starting_point=[4])
    with pytest.raises(ValueError, match="subproblem"):
        LDSDA(_landscape(), subproblem="nlp")


def test_the_route_and_options():
    r = _quiet(
        _landscape().solve,
        gdp_method="ldsda",
        time_limit=60,
        gdp_options={"starting_point": [2], "direction_norm": "Linf", "subproblem": "global"},
    )
    assert r.algorithm_route.startswith("gdpopt-ldsda") and r.bound is None
    assert r.status == "feasible" and r.objective == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="direction_norm"):
        _landscape().solve(gdp_method="ldsda", gdp_options={"direction_norm": "L1"})
    with pytest.raises(ValueError, match="subproblem"):
        _landscape().solve(gdp_method="ldsda", gdp_options={"subproblem": "nlp"})
    with pytest.raises(ValueError, match="subproblem"):
        _landscape().solve(gdp_method="loa", gdp_options={"subproblem": "local"})


def test_pyomo_four_stage_dynamic_model():
    """The spec's done-when: Pyomo's test reaches -23.305325 with both norms."""
    pytest.importorskip("pyomo.dae")
    pytest.importorskip("cyipopt")  # POUNCE stops at (1, 4); Ipopt converges there
    from discopt.pyomo.gdp import convert_pyomo_gdp
    from pyomo.contrib.gdpopt.tests.four_stage_dynamic_model import build_model
    from pyomo.environ import Constraint, TransformationFactory, Var
    from pyomo.gdp import Disjunct

    for norm in ("L2", "Linf"):
        pm = build_model(mode_transfer=True)
        TransformationFactory("dae.collocation").apply_to(
            pm, nfe=10, ncp=3, scheme="LAGRANGE-RADAU"
        )
        for d in pm.component_data_objects(ctype=Disjunct):
            for c in d.component_objects(ctype=Constraint):
                c._constructed = False  # Pyomo.dae issue 3101, as Pyomo's test does
                c.construct()
        for v in pm.component_data_objects(ctype=Var, descend_into=True):
            if "dxdt" in v.name:
                v.setlb(-300)
                v.setub(300)
        m = _quiet(convert_pyomo_gdp, pm).model
        r = _quiet(
            LDSDA(
                m,
                starting_point=[1, 2],
                logical_constraint_list=["mode_transfer_lc1", "mode_transfer_lc2"],
                direction_norm=norm,
                time_limit=100,
            ).run
        )
        assert r.objective == pytest.approx(-23.305325, abs=1e-4), norm
        assert r.status == "feasible" and r.bound is None
