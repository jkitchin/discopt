"""#1917: a node NLP with more all-fixed equality rows than free columns.

Ipopt (and POUNCE, its port) counts every equality row against the free columns
before iteration 0, including a row whose columns are all fixed, and refuses with
``Not_Enough_Degrees_Of_Freedom`` (-10) when the rows outnumber them. A GDP
configuration subproblem hits this: it fixes indicators and disjunct-local
parameters by equality rows. Measured on GDPlib ``small_batch`` under
``gdp_method="enumerate"``: every feasible configuration came back ERROR, and the
solve returned ``unknown`` (big-M certifies 167427.66).

``solve_nlp`` now re-solves once without the all-fixed rows when every one of
them holds at the fixed point, and leaves the refusal in place when any of them
fails.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
from discopt.solver import (  # noqa: E402
    _BoundOverrideEvaluator,
    _extract_variable_info,
    _make_evaluator,
)
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.nlp_pounce import solve_nlp  # noqa: E402

#: Ipopt's ``Not_Enough_Degrees_Of_Freedom``.
_NOT_ENOUGH_DEGREES_OF_FREEDOM = -10


def _overdetermined(rhs: float):
    """``x`` fixed at 1 by its box, three equality rows on ``x`` alone, ``y`` free.

    Three equality rows and one free column: refused by the DOF gate. With
    ``rhs == 1`` every fixed row holds; otherwise they do not.
    """
    m = dm.Model("dof")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((y - 2.0) * (y - 2.0) + x)
    m.subject_to(x == rhs)
    m.subject_to(2.0 * x == 2.0 * rhs)
    m.subject_to(3.0 * x == 3.0 * rhs)
    m.subject_to(x + y <= 4.0)
    ev = _make_evaluator(m)
    _, lb, ub, _, _ = _extract_variable_info(m)
    lb, ub = lb.copy(), ub.copy()
    lb[0] = ub[0] = 1.0
    bounds = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-np.inf, 0.0)]
    return _BoundOverrideEvaluator(ev, lb, ub), bounds


def test_the_full_row_set_is_refused_for_degrees_of_freedom(monkeypatch):
    # Guards the premise: without the retry this NLP never starts.
    import discopt.solvers.nlp_pounce as nlp_pounce

    monkeypatch.setattr(nlp_pounce, "_constant_row_retry", lambda *a, **k: None, raising=False)
    ev, bounds = _overdetermined(1.0)
    r = solve_nlp(ev, np.array([1.0, 1.0]), constraint_bounds=bounds)
    assert r.raw_status == _NOT_ENOUGH_DEGREES_OF_FREEDOM


def test_satisfied_constant_rows_are_dropped_and_the_nlp_solves():
    ev, bounds = _overdetermined(1.0)
    r = solve_nlp(ev, np.array([1.0, 1.0]), constraint_bounds=bounds)
    assert r.status == SolveStatus.OPTIMAL
    assert r.x[0] == pytest.approx(1.0)
    assert r.x[1] == pytest.approx(2.0, abs=1e-6)
    assert r.objective == pytest.approx(1.0, abs=1e-6)
    # Multipliers come back in the caller's row space: one per row, the
    # dropped constant rows at zero.
    assert r.multipliers is not None and r.multipliers.shape == (4,)
    assert np.all(r.multipliers[:3] == 0.0)


def test_a_violated_constant_row_keeps_the_refusal():
    # x is fixed at 1 but the rows ask for x == 3: nothing may be dropped, and
    # nothing is declared infeasible either -- the DOF refusal stands.
    ev, bounds = _overdetermined(3.0)
    bounds = [(0.0, 0.0)] * 3 + [(-np.inf, 0.0)]
    r = solve_nlp(ev, np.array([1.0, 1.0]), constraint_bounds=bounds)
    assert r.raw_status == _NOT_ENOUGH_DEGREES_OF_FREEDOM
    assert r.status == SolveStatus.ERROR


@pytest.mark.slow
def test_gdplib_small_batch_enumerate_certifies():
    gr = pytest.importorskip("benchmarks.gdplib_runner")
    pytest.importorskip("gdplib")
    from discopt.pyomo.gdp import convert_pyomo_gdp

    spec = gr.discover_models(include=["small_batch"])
    if not spec:
        pytest.skip("GDPlib small_batch not available")
    m = convert_pyomo_gdp(spec[0].builder()).model
    r = m.solve(gdp_method="enumerate", time_limit=120)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(167427.65, rel=1e-6)
