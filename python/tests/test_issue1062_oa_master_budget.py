"""Regression test for #1062: OA must not let the master eat the whole budget.

`solve_oa` handed the master MILP ``time_limit - elapsed`` — every second that
was left. A master that runs to its own time limit still returns a usable
integer assignment, but the fixed-integer NLP subproblem that turns that
assignment into an incumbent lives *after* the candidate loop's
``elapsed >= time_limit`` check, so it never ran. OA then reported
``status=unknown, obj=None`` while holding the answer.

Measured on MINLPLib ``rsyn0840m`` at 60 s with the #1059 convex route on:

    master  60.34 s (100 % of the solve)   fixed-NLP calls 0   obj None

against ``rsyn0805m``, whose master takes 0.14 s and which the same route drives
to a proved optimum in 1.2 s in three iterations. Capping the master's budget
made three fixed-NLP subproblems run (0.02-0.03 s each) and produced a feasible
incumbent on the instance that had returned none.

The integration test below reproduces that shape without depending on a
particular instance's difficulty (§2): it wraps the master so that it consumes
exactly the budget it is given, which is what a hard MILP does. Before the fix
the model comes back with no objective; after it, an incumbent is returned.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import discopt.solvers.oa as oa
import pytest


def test_master_budget_reserves_time_while_there_is_no_incumbent():
    """The unit rule: no incumbent means the master cannot have everything."""
    # With an incumbent in hand OA has something to return, so the master is
    # welcome to the whole remaining budget.
    assert oa._master_time_budget(10.0, has_incumbent=True) == 10.0

    # Without one, it must leave room for the subproblem that creates it.
    budget = oa._master_time_budget(10.0, has_incumbent=False)
    assert budget < 10.0
    assert budget > 0.0

    # The reserve is a fraction, so it survives repeated shrinking rather than
    # collapsing to zero and starving the subproblem late in a solve.
    assert oa._master_time_budget(1e-3, has_incumbent=False) > 0.0

    # Degenerate budgets pass through untouched rather than being scaled into
    # something stranger.
    assert oa._master_time_budget(0.0, has_incumbent=False) == 0.0
    assert oa._master_time_budget(float("inf"), has_incumbent=False) == float("inf")


def _small_convex_minlp():
    """A convex MINLP with a genuine fixed-integer NLP subproblem to solve."""
    m = dm.Model("oa_budget")
    y = m.binary("y", 3)
    x = m.continuous("x", 3, lb=0.0, ub=5.0)
    m.subject_to(y[0] + y[1] + y[2] == 2, name="pick_two")
    for i in range(3):
        m.subject_to(x[i] <= 5.0 * y[i], name=f"link{i}")
    m.subject_to(x[0] + x[1] + x[2] >= 4.0, name="demand")
    m.minimize(sum(x[i] * x[i] for i in range(3)) + 2.0 * (y[0] + y[1] + y[2]))
    return m


@pytest.mark.smoke
def test_oa_returns_an_incumbent_when_the_master_exhausts_its_budget(monkeypatch):
    """A master that uses all the time it is given must not cost OA its answer."""
    real_master = oa._solve_master_milp
    observed: dict[str, int] = {"master": 0, "fixed_nlp": 0}

    def greedy_master(*args, **kwargs):
        """Consume the whole allotted budget, then return the real solution.

        This is what a hard MILP master does; simulating it keeps the test
        independent of any particular instance being slow enough on the day.
        """
        observed["master"] += 1
        budget = kwargs.get("time_limit")
        result = real_master(*args, **kwargs)
        if budget is not None and budget > 0:
            deadline = time.perf_counter() + float(budget)
            while time.perf_counter() < deadline:
                time.sleep(0.01)
        return result

    real_nlp = oa._solve_nlp_subproblem

    def counting_nlp(*args, **kwargs):
        observed["fixed_nlp"] += 1
        return real_nlp(*args, **kwargs)

    monkeypatch.setattr(oa, "_solve_master_milp", greedy_master)
    monkeypatch.setattr(oa, "_solve_nlp_subproblem", counting_nlp)

    result = _small_convex_minlp().solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=3.0)

    # §6: the test is only meaningful if the wrapped master actually ran.
    assert observed["master"] > 0, "the master wrapper never fired — test measured nothing"

    # This is the defect: the subproblem is what manufactures the incumbent, and
    # before the fix it was unreachable once the master had spent the budget.
    assert observed["fixed_nlp"] > 0, (
        "the master consumed the whole budget and OA never ran a fixed-integer "
        "NLP subproblem, so it could not produce an incumbent"
    )
    assert result.objective is not None, (
        f"OA returned no incumbent (status={result.status}) despite having "
        "solved a master to a usable integer assignment"
    )
