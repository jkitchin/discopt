"""Tests for the optional Pyomo ``SolverFactory('discopt')`` plugin.

The bridge round-trips a Pyomo model through a temporary AMPL ``.nl`` file into
discopt and maps the solution back by column order. These tests skip when Pyomo is
not installed (``pip install discopt[pyomo]``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402

pyo = pytest.importorskip("pyomo.environ")

import discopt.pyomo  # noqa: E402,F401  (registers the solver)


@pytest.fixture()
def opt():
    return pyo.SolverFactory("discopt")


def _tiny_minlp(sense=None):
    """min (x-3)^2 + 2y  s.t. x + 5y >= 4 ; x in [0,10], y binary -> x=4, y=0, obj=1."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.y = pyo.Var(domain=pyo.Binary)
    m.obj = pyo.Objective(expr=(m.x - 3) ** 2 + 2 * m.y, sense=sense or pyo.minimize)
    m.c = pyo.Constraint(expr=m.x + 5 * m.y >= 4)
    return m


def test_registration(opt):
    assert "discopt" in pyo.SolverFactory
    assert opt.available() is True


def test_roundtrip_matches_from_nl(opt, tmp_path):
    """The plugin must match a direct from_nl().solve() on the same .nl: same status,
    objective, and variable values aligned by column order (names differ)."""
    import discopt.modeling as dm

    m = _tiny_minlp()
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.obj) == pytest.approx(1.0, abs=1e-3)
    assert m.x.value == pytest.approx(4.0, abs=1e-3)
    assert m.y.value == 0  # exact integer, not 1e-9 drift

    # Independent reference: write the same model and solve via from_nl directly.
    from pyomo.repn.plugins.nl_writer import NLWriter

    nl = str(tmp_path / "ref.nl")
    with open(nl, "w") as f:
        info = NLWriter().write(m, f, linear_presolve=False, scale_model=False)
    ref = dm.from_nl(nl).solve(time_limit=30, gap_tolerance=1e-4)
    assert ref.status == "optimal"
    assert ref.objective == pytest.approx(pyo.value(m.obj), rel=1e-4, abs=1e-6)
    # Column-order alignment: the i-th .nl column value equals the plugin-loaded var.
    flat = []
    import numpy as np

    dref = dm.from_nl(nl)
    for v in dref._variables:
        flat.extend(np.asarray(ref.x[v.name]).ravel())
    pyomo_vals = [info.variables[i].value for i in range(len(info.variables))]
    assert pyomo_vals == pytest.approx(flat, abs=1e-3)


def test_maximize_sense(opt):
    """max -(x-3)^2 over [0,10] -> x=3, obj=0 (no sign flip)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.o = pyo.Objective(expr=-((m.x - 3) ** 2), sense=pyo.maximize)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert m.x.value == pytest.approx(3.0, abs=1e-3)
    assert pyo.value(m.o) == pytest.approx(0.0, abs=1e-3)


def test_integer_rounding(opt):
    """An integral optimum loads as an exact integer."""
    m = _tiny_minlp()
    opt.solve(m)
    assert m.y.value in (0, 1)
    assert float(m.y.value).is_integer()


def test_infeasible(opt):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.o = pyo.Objective(expr=m.x)
    m.c = pyo.Constraint(expr=m.x >= 2)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_options_passthrough(opt, monkeypatch):
    """Pyomo's `timelimit`/options reach Model.solve as the right kwargs."""
    import discopt.modeling as dm

    captured = {}
    orig = dm.Model.solve

    def spy(self, *a, **k):
        captured.update(k)
        return orig(self, *a, **k)

    monkeypatch.setattr(dm.Model, "solve", spy)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5))
    m.o = pyo.Objective(expr=m.x)
    opt.solve(m, options={"timelimit": 7, "mipgap": 1e-3})
    assert captured.get("time_limit") == 7
    assert captured.get("gap_tolerance") == 1e-3


def test_duals_when_exposed(opt):
    """Convex NLP min (x-3)^2 s.t. x>=4 -> KKT multiplier ~2; sign matches the
    AMPL/Pyomo convention (cross-checked against ipopt's value 2.0)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.o = pyo.Objective(expr=(m.x - 3) ** 2, sense=pyo.minimize)
    m.c = pyo.Constraint(expr=m.x >= 4)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert m.x.value == pytest.approx(4.0, abs=1e-3)
    d = m.dual.get(m.c)
    assert d is not None, "discopt exposed duals but the plugin did not load them"
    assert d == pytest.approx(2.0, abs=1e-3)


def test_from_pyomo_matches_solver_plugin(opt):
    """`from_pyomo(m).solve()` must reach the same optimum as `SolverFactory('discopt')`
    on the same model (the issue #381 round-trip acceptance test)."""
    import discopt.modeling as dm

    # Reference: solve via the registered Pyomo solver plugin.
    m_ref = _tiny_minlp()
    res = opt.solve(m_ref)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    ref_obj = pyo.value(m_ref.obj)

    # Import the same model into a discopt Model and solve natively.
    m_imp = _tiny_minlp()
    dmodel = dm.from_pyomo(m_imp)
    assert isinstance(dmodel, dm.Model)
    r = dmodel.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref_obj, rel=1e-4, abs=1e-6)
    assert r.objective == pytest.approx(1.0, abs=1e-3)


def test_from_pyomo_indexed_transportation():
    """A small indexed (Var/Constraint over sets) LP imports and solves correctly."""
    import discopt.modeling as dm

    supply = {0: 20.0, 1: 30.0}
    demand = {0: 10.0, 1: 25.0, 2: 15.0}
    cost = {(0, 0): 2.0, (0, 1): 3.0, (0, 2): 1.0, (1, 0): 5.0, (1, 1): 4.0, (1, 2): 8.0}

    m = pyo.ConcreteModel()
    m.P = pyo.RangeSet(0, 1)
    m.K = pyo.RangeSet(0, 2)
    m.ship = pyo.Var(m.P, m.K, domain=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=sum(cost[i, j] * m.ship[i, j] for i in m.P for j in m.K))
    m.sup = pyo.Constraint(m.P, rule=lambda mm, i: sum(mm.ship[i, j] for j in mm.K) <= supply[i])
    m.dem = pyo.Constraint(m.K, rule=lambda mm, j: sum(mm.ship[i, j] for i in mm.P) >= demand[j])

    dmodel = dm.from_pyomo(m)
    r = dmodel.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal"

    # Independent reference via the registered plugin.
    opt = pyo.SolverFactory("discopt")
    m2 = m.clone()
    opt.solve(m2)
    assert r.objective == pytest.approx(pyo.value(m2.obj), rel=1e-4, abs=1e-6)


def test_from_pyomo_no_variables_raises():
    """A variable-free Pyomo model has nothing to import -> ValueError."""
    import discopt.modeling as dm

    m = pyo.ConcreteModel()
    m.o = pyo.Objective(expr=1.0)
    with pytest.raises(ValueError, match="no variables"):
        dm.from_pyomo(m)


def test_duals_graceful_for_integer_model(opt):
    """Solving an integer model with a `dual` Suffix declared must not error,
    whether or not discopt exposes multipliers (it may surface relaxation duals).
    Any loaded value must be a finite number, never garbage."""
    import math

    m = pyo.ConcreteModel()
    m.y = pyo.Var(domain=pyo.Binary)
    m.o = pyo.Objective(expr=m.y, sense=pyo.minimize)
    m.c = pyo.Constraint(expr=m.y >= 0)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    res = opt.solve(m)
    assert res.solver.termination_condition in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )
    d = m.dual.get(m.c)
    assert d is None or math.isfinite(d)  # absent or finite — never fabricated/NaN


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
