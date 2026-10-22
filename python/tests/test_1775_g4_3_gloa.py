"""#1775 G4.3: GLOA -- LOA with a relaxation master of every row (spec §5.3).

Before this task ``gdp_method="gloa"`` was reserved and refused. LOA's master
gets OA cuts only on certified-convex rows, so on a nonconvex disjunct its bound
is the linear rows' alone (measured on the ``gdp_parity`` panel: syngas has 78
nonlinear rows, none certified convex; LOA's bound 6.7 against an optimum of
4669.02). GLOA's master is the MILP relaxation of the lowered model, whose rows
relax every term over the box, so its bound needs no certificate: syngas is
certified at the root (bound 4668.81).
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.gdp.algorithms.base import Deadline
from discopt.gdp.algorithms.gloa import GLOA
from discopt.gdp.algorithms.loa import LOA
from discopt.gdp.algorithms.master import RelaxationMaster

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _nonconvex():
    """min x + y over [x^2 + y^2 >= 9] v [x + y >= 5], x, y in [0, 5]; optimum 3."""
    m = dm.Model("nonconvex")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x**2 + y**2 >= 9], [x + y >= 5]], name="d")
    m.minimize(x + y)
    return m


def _bilinear(a: float, b: float, sense: str = "min"):
    """A nonconvex disjunct (x*y >= a) against a linear one, objective nonconvex."""
    m = dm.Model(f"bilinear_{a}_{b}_{sense}")
    x = m.continuous("x", lb=0.5, ub=4)
    y = m.continuous("y", lb=0.5, ub=4)
    m.either_or([[x * y >= a, x <= 2.5], [x + 2 * y >= b, y <= 3]], name="d")
    obj = (x - 2) ** 2 + x * y - y
    if sense == "min":
        m.minimize(obj)
    else:
        m.maximize(-obj)
    return m


def test_the_first_master_bound_needs_no_convexity():
    """The same model, one master solve each: LOA's master has no row for the
    nonconvex disjunct, GLOA's relaxes it."""
    loa = _quiet(LOA(_nonconvex(), time_limit=30, iteration_limit=1).run)
    gloa = _quiet(GLOA(_nonconvex(), time_limit=30, iteration_limit=1).run)
    assert loa.bound == pytest.approx(0.0, abs=1e-6)
    assert 1.0 < gloa.bound <= 3.0 + 1e-6
    assert gloa.solver_stats["relaxation_columns"] > 2


def test_gloa_certifies_the_global_optimum():
    r = _quiet(GLOA(_nonconvex(), time_limit=60).run)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(3.0, abs=1e-5)


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("a, b", [(3.0, 4.0), (5.0, 6.0)])
def test_gloa_agrees_with_the_spatial_bb(a, b, sense):
    """Certified agreement with discopt's own big-M spatial B&B, both senses;
    the bound brackets the optimum from the right side."""
    ref = _quiet(_bilinear(a, b, sense).solve, gdp_method="big-m", time_limit=60)
    r = _quiet(GLOA(_bilinear(a, b, sense), time_limit=60).run)
    assert ref.status == "optimal" and r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(ref.objective, abs=1e-4, rel=1e-4)
    if sense == "min":
        assert r.bound <= ref.objective + 1e-6
    else:
        assert r.bound >= ref.objective - 1e-6


def test_no_master_bound_exceeds_a_feasible_point():
    """Sampled feasibility (CLAUDE.md §5): every first-master bound is at most
    the objective of every sampled feasible point of its model."""
    rng = np.random.default_rng(1775)
    compared = 0
    for a, b in itertools.product((3.0, 5.0), (4.0, 6.0)):
        bound = _quiet(GLOA(_bilinear(a, b), time_limit=30, iteration_limit=1).run).bound
        assert bound is not None
        for x, y in rng.uniform(0.5, 4.0, size=(400, 2)):
            first = x * y >= a and x <= 2.5
            second = x + 2 * y >= b and y <= 3
            if first or second:
                assert bound <= (x - 2) ** 2 + x * y - y + 1e-7
                compared += 1
    assert compared > 0


def test_the_relaxation_master_pads_cuts_and_trims_points():
    class _Relax:
        _c = np.array([1.0, 1.0, 0.0])
        _obj_offset = 0.0
        _objective_bound_valid = True
        _A_ub = np.array([[-1.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        _b_ub = np.array([-1.0, -0.5])
        _bounds = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
        _integrality = np.array([1, 0, 0])
        _build_truncated = False

    master = RelaxationMaster(_Relax(), 2)
    master.add_cut(np.array([0.0, -1.0]), -0.25)  # over the model's two columns
    assert master.cuts[-1].shape == (3,) and master.cuts[-1][2] == 0.0
    r = master.solve(10.0)
    assert r.x is not None and r.x.shape == (2,)
    assert r.bound == pytest.approx(1.0, abs=1e-6)


def test_an_integer_auxiliary_is_refused():
    class _Relax:
        _c = np.zeros(3)
        _obj_offset = 0.0
        _objective_bound_valid = True
        _A_ub = None
        _b_ub = None
        _bounds = [(0.0, 1.0)] * 3
        _integrality = np.array([1, 0, 1])

    with pytest.raises(ValueError, match="integer column"):
        RelaxationMaster(_Relax(), 2)


def test_gloa_refuses_local_subproblems():
    with pytest.raises(ValueError, match="subproblems are global"):
        GLOA(_nonconvex(), subproblem="nlp")


def test_an_init_strategy_runs_through_the_relaxation_master():
    r = _quiet(GLOA(_nonconvex(), time_limit=60, init_algorithm="set_covering").run)
    assert r.status == "optimal" and r.objective == pytest.approx(3.0, abs=1e-5)


def test_model_solve_routes_gloa():
    r = _quiet(_nonconvex().solve, gdp_method="gloa", time_limit=60)
    assert r.status == "optimal" and r.algorithm_route.startswith("gdpopt-gloa")
    r = _quiet(_nonconvex().solve, gdp_method="gloa", time_limit=60, gdp_options={"iterlim": 1})
    assert r.solver_stats["iterations"] == 1
    with pytest.raises(ValueError, match="does not read gdp_options"):
        _nonconvex().solve(gdp_method="gloa", gdp_options={"subproblem": "global"})


def test_the_build_gets_the_callers_deadline(monkeypatch):
    """#912 ``contract``: the relaxation build is handed the caller's whole
    deadline (a truncated build is still a valid, weaker relaxation)."""
    import discopt._relax.milp_relaxation as mr

    seen = []
    real = mr.build_milp_relaxation

    def spy(*a, **k):
        seen.append(k.get("build_deadline"))
        return real(*a, **k)

    monkeypatch.setattr(mr, "build_milp_relaxation", spy)
    a = _quiet(GLOA, _nonconvex(), time_limit=20)
    assert seen == [Deadline.at(a.deadline)]
    _quiet(GLOA, _nonconvex(), time_limit=None)
    assert seen[-1] is None


def _convex_objective():
    """A convex nonlinear objective whose tangents GLOA must keep (LOA's
    objective cuts), on three chained disjunctions; optimum 13.927043508."""
    m = dm.Model("convex_objective")
    xs = [m.continuous(f"x{i}", lb=0.1, ub=5) for i in range(4)]
    for i in range(3):
        m.either_or([[xs[i] + xs[i + 1] >= 3], [xs[i] - xs[i + 1] >= 1]], name=f"d{i}")
    m.minimize(sum(dm.exp(0.7 * v) for v in xs) + sum(1 / v for v in xs))
    return m


def test_a_convex_objective_keeps_its_tangents():
    """Measured on the panel: without the objective's tangents GLOA's master
    bounded a convex objective only by its box envelope and lost positioning
    and small_batch, which LOA certifies; the tangents are written on the
    relaxation's objective column. Here the same: certified, and agreeing with
    the spatial B&B."""
    ref = _quiet(_convex_objective().solve, gdp_method="big-m", time_limit=60)
    r = _quiet(GLOA(_convex_objective(), time_limit=60).run)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(ref.objective, rel=1e-5)
    assert r.bound <= ref.objective + 1e-6


def test_cut_cleaning_moves_a_tiny_coefficient_to_the_rhs_safely():
    class _Relax:
        _c = np.array([1.0, 0.0, 0.0])
        _obj_offset = 0.0
        _objective_bound_valid = True
        _A_ub = None
        _b_ub = None
        _bounds = [(-2.0, 3.0), (0.0, 5.0), (1.0, 4.0)]
        _integrality = np.zeros(3)

    master = RelaxationMaster(_Relax(), 3)
    # Against the largest coefficient 2.0: 1e-12 on column 0 and -5e-13 on
    # column 2 are cleaned at their worst case over the box.
    master.add_cut(np.array([1e-12, 2.0, -5e-13]), 1.0)
    row, rhs = master.cuts[-1], master.cut_rhs[-1]
    assert row[0] == 0.0 and row[2] == 0.0 and row[1] == 2.0
    assert rhs == pytest.approx(1.0 - (1e-12 * -2.0) - (-5e-13 * 4.0))
    rng = np.random.default_rng(4)
    held = 0
    for x in rng.uniform([-2.0, 0.0, 1.0], [3.0, 5.0, 4.0], size=(2000, 3)):
        if 1e-12 * x[0] + 2.0 * x[1] - 5e-13 * x[2] <= 1.0:
            assert row @ x <= rhs + 1e-15
            held += 1
    assert held > 0 and master.cleaned_coefficients == 2


def test_a_milp_stopped_without_an_incumbent_returns_no_point():
    """milp_simplex returned the engine's unset vector (zeros) as ``x`` when a
    node cap stopped the search before any incumbent; a GDP master read it as a
    selection of no disjunct and raised (hda, med_term_purchasing, syngas)."""
    from discopt.solvers import SolveStatus
    from discopt.solvers.milp_simplex import solve_milp

    n = 6
    r = solve_milp(
        c=np.arange(1, n + 1, dtype=float),
        A_ub=np.array([[-3.0, -5.0, -7.0, -9.0, -11.0, -13.0]]),
        b_ub=np.array([-10.5]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n,
        integrality=np.ones(n, dtype=np.int32),
        max_nodes=0,
        time_limit=10,
    )
    assert r.status == SolveStatus.ITERATION_LIMIT and r.objective is None
    assert r.x is None


def test_cut_cleaning_keeps_a_coefficient_without_a_finite_box():
    class _Relax:
        _c = np.array([1.0, 0.0])
        _obj_offset = 0.0
        _objective_bound_valid = True
        _A_ub = None
        _b_ub = None
        _bounds = [(0.0, 1.0), (0.0, 1e20)]
        _integrality = np.zeros(2)

    master = RelaxationMaster(_Relax(), 2)
    master.add_cut(np.array([2.0, 1e-12]), 1.0)
    assert master.cuts[-1][1] == 1e-12 and master.cut_rhs[-1] == 1.0
    assert master.cleaned_coefficients == 0
