"""#1775 G0.4: Pyomo GDP -> native discopt converter (``discopt.pyomo.gdp``).

Before G0.4 a Pyomo GDP model reached discopt only after a Pyomo transformation
(``gdp.bigm`` / ``gdp.hull``) had lowered every disjunction, through the ``.nl``
bridge: none of discopt's GDP methods could see the disjunctive structure.
``discopt.pyomo.gdp`` did not exist.

NC regime: a new entry point; nothing on the default solve path calls it.
"""

from __future__ import annotations

import warnings

import pytest

pyo = pytest.importorskip("pyomo.environ")
gdp = pytest.importorskip("pyomo.gdp")

from discopt.gdp import gdp_tree  # noqa: E402
from discopt.modeling.core import DisjunctionSemantics  # noqa: E402
from discopt.pyomo.gdp import (  # noqa: E402
    PyomoGDPConversionError,
    check_evaluation_parity,
    convert_pyomo_gdp,
    from_pyomo_gdp,
)

pytestmark = pytest.mark.smoke


def _solve(m, method="big-m"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal", (method, r.status)
    return r.objective


def _expressions():
    """Every expression kind the walker maps, inside rows and the objective."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.5, 10))
    m.y = pyo.Var([1, 2], within=pyo.Integers, bounds=(-3, 3))
    m.w = pyo.Var(bounds=(-2, 2))
    m.f = pyo.Var(initialize=1.5)
    m.f.fix()
    m.p = pyo.Param(initialize=2.5, mutable=True)
    m.e = pyo.Expression(expr=m.x**2 + m.p * m.y[1])
    m.c1 = pyo.Constraint(expr=pyo.inequality(1, m.x + 2 * m.y[1] - m.y[2] / m.x, 5))
    m.c2 = pyo.Constraint(
        expr=pyo.exp(m.x / 10) * m.y[2] + abs(m.w) + pyo.sqrt(m.x + 1) + m.e >= -m.x
    )
    m.c3 = pyo.Constraint(
        expr=pyo.log(m.x)
        + pyo.log10(m.x)
        + pyo.sin(m.w)
        + pyo.cos(m.w)
        + pyo.tan(m.w / 4)
        + pyo.atan(m.w)
        + pyo.sinh(m.w)
        + pyo.cosh(m.w)
        + pyo.tanh(m.w)
        + pyo.asinh(m.w)
        <= 100
    )
    m.c4 = pyo.Constraint(
        expr=pyo.asin(m.w / 4)
        + pyo.acos(m.w / 4)
        + pyo.acosh(m.x + 1)
        + pyo.atanh(m.w / 4)
        + m.f * m.x**m.f
        == m.y[1] + 30
    )
    m.c5 = pyo.Constraint(expr=-((m.x - m.w) ** 2) + 0 * m.y[2] <= 50)
    # A constant base with a variable exponent (GDPlib hda's absorber rows).
    m.c6 = pyo.Constraint(expr=pyo.log((1 - 4.0 ** (0.333 * m.x + 1)) / -3.0) <= 40)
    m.o = pyo.Objective(expr=m.x + m.e - m.w * m.y[2])
    return m


def test_every_expression_kind_evaluates_like_pyomo():
    conv = convert_pyomo_gdp(_expressions())
    rep = check_evaluation_parity(conv, n_points=25, seed=3)
    # c1 is ranged (2 rows) + c2..c6: 7 rows per point, every point evaluable.
    assert rep.row_checks == 25 * 7 and rep.objective_checks == 25
    assert rep.max_abs_error < 1e-9
    # The fixed variable and the mutable Param are constants, not columns.
    assert sorted(v.name for v in conv.model._variables) == ["w", "x", "y[1]", "y[2]"]


def test_domains_and_bounds():
    m = pyo.ConcreteModel()
    m.c = pyo.Var(bounds=(-1, 4))
    m.n = pyo.Var(within=pyo.NonNegativeReals)
    m.b = pyo.Var(within=pyo.Binary)
    m.i = pyo.Var(within=pyo.Integers, bounds=(2, 9))
    m.u = pyo.Var(within=pyo.UnitInterval)
    m.k = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(None, 1))
    m.con = pyo.Constraint(expr=m.c + m.n + m.b + m.i + m.u + m.k >= 0)
    m.o = pyo.Objective(expr=m.c)
    conv = convert_pyomo_gdp(m)
    got = {v.name: (v.var_type.name, float(v.lb), float(v.ub)) for v in conv.model._variables}
    assert got["c"] == ("CONTINUOUS", -1.0, 4.0)
    assert got["n"][:2] == ("CONTINUOUS", 0.0) and got["n"][2] > 1e19
    assert got["b"] == ("BINARY", 0.0, 1.0)
    assert got["i"] == ("INTEGER", 2.0, 9.0)
    assert got["u"] == ("CONTINUOUS", 0.0, 1.0)
    assert got["k"] == ("BINARY", 0.0, 1.0)


def _gdp(xor_inner=False, nonlinear=True):
    """dj: d1 [x <= 3, z >= x/10 with z local to d1] v d2 [x >= 7, inner];
    inner (nested in d2, OR unless xor_inner): [y <= 0] v [y >= 1];
    dj2: [x <= 1] v [x >= 9, y2 == 1]; a logical constraint over a free
    BooleanVar and the indicators; max x + d1 + y2."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.5, 10))
    m.y = pyo.Var([1, 2], within=pyo.Integers, bounds=(-3, 3))
    y2_term = m.y[2] / m.x if nonlinear else m.y[2]
    m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + 2 * m.y[1] - y2_term, 5))
    m.d = gdp.Disjunct([1, 2])
    m.d[1].c = pyo.Constraint(expr=m.x <= 3)
    m.d[1].z = pyo.Var(bounds=(0, 1))
    m.d[1].cz = pyo.Constraint(expr=m.d[1].z >= m.x / 10)
    m.d[2].c = pyo.Constraint(expr=m.x >= 7)
    m.d[2].inner = gdp.Disjunct([1, 2])
    m.d[2].inner[1].c = pyo.Constraint(expr=m.y[1] <= 0)
    m.d[2].inner[2].c = pyo.Constraint(expr=m.y[1] >= 1)
    m.d[2].dj = gdp.Disjunction(expr=[m.d[2].inner[1], m.d[2].inner[2]], xor=xor_inner)
    m.dj = gdp.Disjunction(expr=[m.d[1], m.d[2]])
    m.dj2 = gdp.Disjunction(expr=[[m.x <= 1], [m.x >= 9, m.y[2] == 1]])
    m.b = pyo.BooleanVar()
    m.lc = pyo.LogicalConstraint(
        expr=pyo.atleast(1, m.b, m.d[1].indicator_var).implies(~m.d[2].indicator_var)
        | pyo.xor(m.b, m.d[1].indicator_var)
    )
    m.o = pyo.Objective(expr=m.x + m.d[1].binary_indicator_var + m.y[2], sense=pyo.maximize)
    return m


def test_gdp_structure_maps_onto_the_tree():
    pm = _gdp()
    conv = convert_pyomo_gdp(pm)
    tree = gdp_tree(conv.model)
    nodes = list(tree.walk())
    assert [(len(n.disjuncts), n.semantics) for n in nodes] == [
        (2, DisjunctionSemantics.SELECT_ONE),
        (2, DisjunctionSemantics.OR),  # xor=False
        (2, DisjunctionSemantics.SELECT_ONE),
    ]
    outer, inner, _ = nodes
    assert inner.parent is outer.disjuncts[1]  # nested in d[2]
    assert len(tree.logic) == 1
    # d[1].z is referenced only inside d[1]: a Disjunct.variable local.
    assert [v.name for v in conv.local] == ["d[1].z"]
    assert outer.disjuncts[0].local_vars[0].name == "d[1].z"
    # Pyomo's indicator binary is the discopt indicator column, so the objective
    # prices the same selector.
    block = conv.disjuncts[pm.d[1]]
    assert conv.variables[pm.d[1].binary_indicator_var] is block.indicator.variable
    assert conv.booleans[pm.d[1].indicator_var] is block.indicator
    assert check_evaluation_parity(conv, n_points=10).row_checks > 0


def test_a_variable_used_outside_its_disjunct_stays_global():
    pm = _gdp()
    pm.extra = pyo.Constraint(expr=pm.d[1].z <= 0.9)
    conv = convert_pyomo_gdp(pm)
    assert len(conv.local) == 0
    assert "d[1].z" in [v.name for v in conv.model._variables]


@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm", "auto"])
@pytest.mark.parametrize("xor_inner", [False, True])
def test_optimum_matches_pyomo_bigm_through_nl(method, xor_inner):
    """Spec §8 optimum parity: native -> discopt equals gdp.bigm -> .nl -> discopt."""
    import discopt.pyomo  # noqa: F401  (registers the solver)

    # Linear rows: this checks the GDP semantics; the expression mapping is
    # checked by evaluation parity above.
    native = _solve(from_pyomo_gdp(_gdp(xor_inner, nonlinear=False)), method)
    pm = _gdp(xor_inner, nonlinear=False)
    pyo.TransformationFactory("gdp.bigm").apply_to(pm)
    res = pyo.SolverFactory("discopt").solve(pm, time_limit=60)
    assert str(res.solver.termination_condition) == "optimal"
    # By hand: x = 10 needs dj2's second disjunct (y2 = 1) and dj's d[2]; then
    # c pins y1 in {-3, -2} (the inner [y1 <= 0] holds), and the logic holds with
    # b = 1, d[1] = 0. Optimum 10 + 0 + 1 = 11; d[1] caps x at 1 (dj2), so <= 5.
    assert native == pytest.approx(11.0, abs=1e-6)
    assert pyo.value(pm.o) == pytest.approx(native, abs=1e-6)


def test_an_or_disjunction_lets_both_disjuncts_be_selected():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.d = gdp.Disjunct([1, 2])
    m.d[1].c = pyo.Constraint(expr=m.x <= 4)
    m.d[2].c = pyo.Constraint(expr=pyo.inequality(2, m.x, 6))
    m.dj = gdp.Disjunction(expr=[m.d[1], m.d[2]], xor=False)
    m.o = pyo.Objective(
        expr=m.x + 3 * m.d[1].binary_indicator_var + 3 * m.d[2].binary_indicator_var,
        sense=pyo.maximize,
    )
    # Both selected at x = 4: 10. XOR would give 9 (x = 6, one indicator).
    assert _solve(from_pyomo_gdp(m)) == pytest.approx(10.0, abs=1e-6)


def test_fixed_and_deactivated_disjuncts_are_pinned():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.d = gdp.Disjunct([1, 2, 3])
    m.d[1].c = pyo.Constraint(expr=m.x <= 2)
    m.d[2].c = pyo.Constraint(expr=m.x >= 8)
    m.d[3].c = pyo.Constraint(expr=m.x >= 5)
    m.dj = gdp.Disjunction(expr=[m.d[1], m.d[2], m.d[3]])
    m.o = pyo.Objective(expr=m.x, sense=pyo.maximize)
    assert _solve(from_pyomo_gdp(m)) == pytest.approx(10.0, abs=1e-6)
    m.d[2].deactivate()  # Pyomo: indicator fixed False
    m.d[2].indicator_var.fix(False)
    assert _solve(from_pyomo_gdp(m)) == pytest.approx(10.0, abs=1e-6)  # via d[3]
    m.d[1].indicator_var.fix(True)
    assert _solve(from_pyomo_gdp(m)) == pytest.approx(2.0, abs=1e-6)
    # Deactivated without fixing the indicator: Pyomo refuses it, and so do we.
    m.d[2].indicator_var.unfix()
    with pytest.raises(PyomoGDPConversionError, match="not fixed to False"):
        from_pyomo_gdp(m)


def test_a_fixed_boolean_in_logic_is_pinned():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.d = gdp.Disjunct([1, 2])
    m.d[1].c = pyo.Constraint(expr=m.x <= 2)
    m.d[2].c = pyo.Constraint(expr=m.x >= 8)
    m.dj = gdp.Disjunction(expr=[m.d[1], m.d[2]])
    m.b = pyo.BooleanVar()
    m.lc = pyo.LogicalConstraint(expr=m.b.implies(m.d[1].indicator_var))
    m.o = pyo.Objective(expr=m.x, sense=pyo.maximize)
    assert _solve(from_pyomo_gdp(m)) == pytest.approx(10.0, abs=1e-6)
    m.b.fix(True)  # forces d[1]
    assert _solve(from_pyomo_gdp(m)) == pytest.approx(2.0, abs=1e-6)


def test_a_discretized_dae_model_converts():
    dae = pytest.importorskip("pyomo.dae")
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(0, 1))
    m.x = pyo.Var(m.t, bounds=(0, 5))
    m.u = pyo.Var(bounds=(-1, 1))
    m.dx = dae.DerivativeVar(m.x, wrt=m.t)
    m.ode = pyo.Constraint(m.t, rule=lambda m, t: m.dx[t] == m.u if t > 0 else pyo.Constraint.Skip)
    m.x[0].fix(1.0)
    m.o = pyo.Objective(expr=m.x[1])
    with pytest.raises(PyomoGDPConversionError, match="not discretized"):
        from_pyomo_gdp(m)
    pyo.TransformationFactory("dae.finite_difference").apply_to(m, nfe=4, scheme="BACKWARD")
    conv = convert_pyomo_gdp(m)
    assert check_evaluation_parity(conv, n_points=5).row_checks > 0
    # x(1) = 1 + u, u >= -1: optimum 0.
    assert _solve(conv.model) == pytest.approx(0.0, abs=1e-6)


def test_constant_rows():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.f = pyo.Var(initialize=3.0)
    m.f.fix()
    m.ok = pyo.Constraint(expr=m.f <= 4)
    m.c = pyo.Constraint(expr=m.x >= m.f)
    m.o = pyo.Objective(expr=m.x)
    conv = convert_pyomo_gdp(m)
    assert len(conv.rows) == 1  # the constant row that holds is dropped
    assert _solve(conv.model) == pytest.approx(3.0, abs=1e-6)
    m.bad = pyo.Constraint(expr=m.f >= 5)
    with pytest.raises(PyomoGDPConversionError, match="infeasible as given"):
        from_pyomo_gdp(m)


def _refusal_models():
    def base():
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.y = pyo.Var(bounds=(0, 10))
        m.o = pyo.Objective(expr=m.x)
        return m

    m1 = base()
    m1.c = pyo.Constraint(expr=pyo.Expr_if(IF=m1.x >= 1, THEN=m1.y, ELSE=0) <= 3)

    m2 = base()
    m2.o2 = pyo.Objective(expr=m2.y)

    m3 = base()
    m3.z = pyo.Var(within=pyo.Set(initialize=[1, 3, 5]))
    m3.c = pyo.Constraint(expr=m3.z <= m3.x)

    m4 = base()
    m4.d = gdp.Disjunct()
    m4.d.c = pyo.Constraint(expr=m4.x <= 2)

    m5 = base()
    m5.d = gdp.Disjunct([1, 2])
    m5.d[1].c = pyo.Constraint(expr=m5.x <= 2)
    m5.d[2].c = pyo.Constraint(expr=m5.x >= 4)
    m5.dj = gdp.Disjunction(expr=[m5.d[1], m5.d[2]])
    m5.d[1].lc = pyo.LogicalConstraint(expr=m5.d[2].indicator_var.implies(m5.d[1].indicator_var))

    m6 = base()
    m6.b = pyo.BooleanVar()
    m6.lc = pyo.LogicalConstraint(expr=pyo.land(m6.b, False))

    m7 = base()
    m7.v = pyo.Var([1, 2], bounds=(0, 1))
    m7.sos = pyo.SOSConstraint(var=m7.v, sos=1)

    m8 = base()
    m8.c = pyo.Constraint(expr=pyo.floor(m8.x) <= 3)

    return [
        ("Expr_if", m1, "Expr_ifExpression"),
        ("two objectives", m2, "2 active objectives"),
        ("discrete domain", m3, "not an interval"),
        ("orphan disjunct", m4, "belongs to no active Disjunction"),
        ("logic in a disjunct", m5, "LogicalConstraint inside a Disjunct"),
        ("logic that folds to False", m6, "always false"),
        ("SOS", m7, "SOSConstraint"),
        ("floor/ceil", m8, "the function"),
    ]


@pytest.mark.parametrize(
    "label, pm, match", _refusal_models(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_unsupported_components_refuse_by_name(label, pm, match):
    with pytest.raises(PyomoGDPConversionError, match=match):
        from_pyomo_gdp(pm)


def test_parity_checker_catches_a_wrong_row():
    """The instrument fires: a row whose body is swapped for another fails parity."""
    conv = convert_pyomo_gdp(_gdp())
    recs = conv.rows
    assert len(recs) >= 2
    recs[0].row, recs[1].row = recs[1].row, recs[0].row
    with pytest.raises(AssertionError, match="evaluation parity"):
        check_evaluation_parity(conv, n_points=3)


def test_pyomo_variables_keep_their_values():
    pm = _gdp()
    pm.x.set_value(4.25)
    conv = convert_pyomo_gdp(pm)
    check_evaluation_parity(conv, n_points=3)
    assert pm.x.value == 4.25


def _nested(cost=0.0):
    """d1 holds a nested [e1: x >= 9] v [e2: x <= 10]; d2 is the cheap branch."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.d1 = gdp.Disjunct()
    m.d1.c = pyo.Constraint(expr=m.x >= 8)
    m.d1.e1 = gdp.Disjunct()
    m.d1.e1.c = pyo.Constraint(expr=m.x >= 9)
    m.d1.e2 = gdp.Disjunct()
    m.d1.e2.c = pyo.Constraint(expr=m.x <= 10)
    m.d1.inner = gdp.Disjunction(expr=[m.d1.e1, m.d1.e2])
    m.d2 = gdp.Disjunct()
    m.d2.c = pyo.Constraint(expr=m.x >= 1)
    m.top = gdp.Disjunction(expr=[m.d1, m.d2])
    obj = m.x + cost * m.d1.e1.binary_indicator_var if cost else m.x
    m.o = pyo.Objective(expr=obj)
    return m


@pytest.mark.parametrize("method", ["big-m", "hull"])
def test_a_nested_indicator_referenced_in_logic_keeps_pyomo_semantics(method):
    """Pyomo leaves a nested indicator free when its parent is False; discopt's tree
    selector implies the parent. Requiring the nested indicator True must not force
    d1: Pyomo's bigm and hull both give 1.0 (d2, x = 1). Before the fix: 9.0."""
    pm = _nested()
    pm.L = pyo.LogicalConstraint(expr=pm.d1.e1.indicator_var)
    conv = convert_pyomo_gdp(pm)
    assert _solve(conv.model, method) == pytest.approx(1.0, abs=1e-6)


def test_a_nested_indicator_priced_in_the_objective_keeps_pyomo_semantics():
    """x - 10 * e1: in Pyomo e1 is free under d2, so -9 (x = 1, e1 = 1). Pricing
    the tree selector instead gives -1 (d1, e1, x = 9)."""
    conv = convert_pyomo_gdp(_nested(cost=-10.0))
    assert _solve(conv.model) == pytest.approx(-9.0, abs=1e-6)


def test_a_nested_indicator_fixed_true_does_not_force_its_parent():
    pm = _nested()
    pm.d1.e1.indicator_var.fix(True)
    conv = convert_pyomo_gdp(pm)
    assert _solve(conv.model) == pytest.approx(1.0, abs=1e-6)


def test_an_unreferenced_nested_indicator_adds_no_column():
    """Bound-neutral for models that never reference a nested indicator: no ``W``
    column and no link rows, so the converted model is the one before the fix."""
    conv = convert_pyomo_gdp(_nested())
    names = [c.name for c in conv.model._constraints]
    assert not any("indicator_link" in (n or "") for n in names), names
    # The tree selector is a valid value for an indicator nothing constrains.
    pm_e1 = next(d for d in conv.disjuncts if d.name == "d1.e1")
    assert conv.booleans[pm_e1.indicator_var] is conv.disjuncts[pm_e1].indicator
    assert _solve(conv.model) == pytest.approx(1.0, abs=1e-6)


def test_a_violated_constant_row_in_a_disjunct_deselects_it():
    """A row with only fixed variables that fails makes its disjunct unselectable;
    Pyomo gives 4.0 (d2). Before the fix the converter refused the model."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.p = pyo.Var(initialize=3.0)
    m.p.fix()
    m.d1 = gdp.Disjunct()
    m.d1.bad = pyo.Constraint(expr=m.p >= 5)
    m.d1.c = pyo.Constraint(expr=m.x >= 1)
    m.d2 = gdp.Disjunct()
    m.d2.c = pyo.Constraint(expr=m.x >= 4)
    m.top = gdp.Disjunction(expr=[m.d1, m.d2])
    m.o = pyo.Objective(expr=m.x)
    assert _solve(convert_pyomo_gdp(m).model) == pytest.approx(4.0, abs=1e-6)


def test_parity_draws_a_fixed_indicator_binary_at_its_value():
    """A row over a fixed indicator binary substitutes its value; the parity checker
    used to draw that binary at random and report a false mismatch (GDPlib kaibel)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.d = gdp.Disjunct([1, 2])
    m.d[1].c = pyo.Constraint(expr=m.x >= 2)
    m.d[2].c = pyo.Constraint(expr=m.x >= 1)
    m.top = gdp.Disjunction(expr=[m.d[1], m.d[2]], xor=False)
    m.d[2].binary_indicator_var.fix(1)
    m.r = pyo.Constraint(expr=m.d[1].binary_indicator_var <= m.d[2].binary_indicator_var)
    m.o = pyo.Objective(expr=m.x)
    conv = convert_pyomo_gdp(m)
    rep = check_evaluation_parity(conv, n_points=20)
    assert rep.row_checks > 0


# ── logical constants (#1775 G4.5) ───────────────────────────────────────────
#
# discopt has no logical constant. Pyomo builds them freely -- an empty
# ``lor(... for s in S if s < first)`` in Pyomo's own four_stage_dynamic_model
# crashed the converter with an IndexError -- so the converter folds them: for
# every assignment of the free Booleans, the converted model is feasible exactly
# when Pyomo's formula is true.


def _formulas(m):
    b1, b2, b3 = m.b[1], m.b[2], m.b[3]
    return [
        b1.implies(pyo.lor()),  # x -> False  ==  not x
        pyo.lor().implies(b1),  # always true
        b1.implies(pyo.land()),  # always true
        pyo.land(b1, True, b2),
        pyo.lor(b1, False, b2),
        pyo.lnot(pyo.lor(b1, pyo.lor())),
        b1.equivalent_to(True),
        b1.equivalent_to(False),
        b1.xor(True),
        pyo.exactly(1, b1, True, b2),
        pyo.exactly(2, b1, True, b2),
        pyo.atmost(1, True, b1, b2),
        pyo.atleast(2, False, b1, b2, b3),
        pyo.atleast(1, True, b1),
        pyo.atmost(0, True, b1),
        b1.implies(pyo.lor(b2, pyo.land())),
    ]


def _with(i, values):
    m = pyo.ConcreteModel()
    m.b = pyo.BooleanVar([1, 2, 3])
    m.x = pyo.Var(bounds=(0, 1))
    m.o = pyo.Objective(expr=m.x)
    m.lc = pyo.LogicalConstraint(expr=_formulas(m)[i])
    for k, v in zip((1, 2, 3), values):
        m.b[k].fix(v)
    return m


@pytest.mark.parametrize("i", range(16))
def test_constants_fold_to_the_same_truth_table(i):
    import itertools

    checked = 0
    for values in itertools.product([False, True], repeat=3):
        pm = _with(i, values)
        truth = bool(pyo.value(pm.lc.body))
        try:
            dmodel = convert_pyomo_gdp(pm).model
        except PyomoGDPConversionError as err:
            assert not truth and "always false" in str(err)
            checked += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = dmodel.solve(time_limit=30)
        assert (r.status == "optimal") is truth, (i, values, r.status)
        assert r.status in ("optimal", "infeasible")
        checked += 1
    assert checked == 8
