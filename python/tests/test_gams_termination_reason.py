"""GAMS and Pyomo must preserve a feasible result's termination reason (#1888).

``SolveResult.status`` says what was proven; ``termination`` says why the search
stopped. A feasible incumbent at a time/node/iteration limit was exported as a
normal completion because both bridges read ``status`` alone. No GAMS
installation, license, or optimization run is required.
"""

import pytest
from discopt.gams import link
from discopt.modeling import SolveResult


def _feasible(termination):
    return SolveResult(
        status="feasible",
        termination=termination,
        objective=1.0,
        bound=0.0,
        bound_valid=True,
        gap_certified=False,
        x={"x": 1.0},
    )


@pytest.mark.unit
@pytest.mark.parametrize("has_discrete", [False, True])
@pytest.mark.parametrize(
    "termination, expected_solve_stat",
    [
        ("time_limit", link.SOLVESTAT_RESOURCE),
        ("node_limit", link.SOLVESTAT_RESOURCE),
        ("iteration_limit", link.SOLVESTAT_ITERATION),
        ("interrupted", link.SOLVESTAT_USER_INTERRUPT),
        (None, link.SOLVESTAT_NORMAL),
        ("gap", link.SOLVESTAT_NORMAL),
        ("exhausted", link.SOLVESTAT_NORMAL),
    ],
)
def test_feasible_result_preserves_termination(termination, expected_solve_stat, has_discrete):
    # GAMS: 7 is a feasible point without discrete variables, 8 with them.
    model_stat = link.MODELSTAT_INTEGER if has_discrete else link.MODELSTAT_FEASIBLE
    assert link.status_to_gams(_feasible(termination), has_discrete) == (
        model_stat,
        expected_solve_stat,
    )


@pytest.mark.unit
@pytest.mark.parametrize("termination", ["time_limit", "node_limit", "gap"])
def test_termination_never_changes_a_proven_status(termination):
    """The reason refines a feasible exit only; it never moves a certificate."""
    certified = SolveResult(
        status="optimal",
        termination=termination,
        objective=1.0,
        bound=1.0,
        bound_valid=True,
        gap_certified=True,
        x={"x": 1.0},
    )
    assert certified.gap_certified
    assert link.status_to_gams(certified, False) == (link.MODELSTAT_OPTIMAL, link.SOLVESTAT_NORMAL)
    local = SolveResult(
        status="optimal",
        termination=termination,
        objective=1.0,
        gap_certified=False,
        x={"x": 1.0},
    )
    assert link.status_to_gams(local, False) == (
        link.MODELSTAT_LOCALLY_OPTIMAL,
        link.SOLVESTAT_NORMAL,
    )
    none = SolveResult(status="time_limit", termination=termination)
    assert link.status_to_gams(none, False) == (
        link.MODELSTAT_NO_SOLUTION_RETURNED,
        link.SOLVESTAT_RESOURCE,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "termination, expected",
    [
        ("time_limit", "maxTimeLimit"),
        ("node_limit", "maxIterations"),
        ("iteration_limit", "maxIterations"),
        ("interrupted", "userInterrupt"),
        (None, "feasible"),
        ("gap", "feasible"),
        ("exhausted", "feasible"),
    ],
)
def test_pyomo_feasible_result_preserves_termination(termination, expected):
    pytest.importorskip("pyomo")
    from discopt.pyomo import _mapping

    tc, _ = _mapping.termination_for("feasible", termination)
    assert tc.name == expected
    # A proven status is never refined by the reason.
    tc, _ = _mapping.termination_for("optimal", termination)
    assert tc.name == "optimal"


@pytest.mark.unit
def test_pyomo_build_results_reads_the_termination_field():
    """The bridge must hand ``termination`` to the mapping, not ``status`` alone."""
    pyo = pytest.importorskip("pyomo.environ")
    from discopt.pyomo.solver import DiscoptSolver

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.obj = pyo.Objective(expr=m.x)
    result = _feasible("time_limit")
    result.x = None
    res = DiscoptSolver()._build_results(m, None, result, [], [], [], load_solutions=False)
    assert res.solver.termination_condition.name == "maxTimeLimit"
    assert res.problem.lower_bound == 0.0
