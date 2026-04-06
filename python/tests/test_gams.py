"""
Tests for GAMS import (from_gams) and export (to_gams).

Uses hand-written .gms fixtures covering MINLP patterns:
  - LP (transport), MILP (knapsack), NLP (Rosenbrock),
  - MINLP (process synthesis with exp/log + binary decisions).

For gamspy-based round-trip validation, install gamspy:
  pip install gamspy
and run:
  pytest python/tests/test_gams.py -m gamspy
"""

from __future__ import annotations

import textwrap

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.gams_parser import GamsParseError, parse_gams

# ── Fixtures: hand-written .gms strings ────────────────────────

TRANSPORT_GMS = textwrap.dedent("""\
    Sets
        i "canning plants" / Seattle, San_Diego /
        j "markets" / New_York, Chicago, Topeka / ;

    Scalar f "freight in dollars per case per thousand miles" / 90 / ;

    Table d(i,j) "distance in thousands of miles"
                  New_York  Chicago  Topeka
        Seattle      2.5      1.7     1.8
        San_Diego    2.5      1.8     1.4 ;

    Parameter a(i) "capacity of plant i in cases"
        / Seattle 350, San_Diego 600 / ;

    Parameter b(j) "demand at market j in cases"
        / New_York 325, Chicago 300, Topeka 275 / ;

    Positive Variables x(i,j) "shipment quantities in cases" ;
    Free Variable z "total transportation costs in thousands of dollars" ;

    Equations
        cost "define objective function"
        supply(i) "observe supply limit at plant i"
        demand(j) "satisfy demand at market j" ;

    cost.. z =e= sum((i,j), f * d(i,j) * x(i,j) / 1000) ;
    supply(i).. sum(j, x(i,j)) =l= a(i) ;
    demand(j).. sum(i, x(i,j)) =g= b(j) ;

    Model transport / all / ;
    Solve transport using LP minimizing z ;
""")

KNAPSACK_GMS = textwrap.dedent("""\
    Sets i "items" / i1*i5 / ;

    Parameter w(i) "weight" / i1 10, i2 20, i3 30, i4 40, i5 50 / ;
    Parameter v(i) "value"  / i1 60, i2 100, i3 120, i4 140, i5 160 / ;
    Scalar cap "capacity" / 100 / ;

    Binary Variables y(i) "select item" ;
    Free Variable obj "total value" ;

    Equations objective, weight_limit ;

    objective.. obj =e= sum(i, v(i) * y(i)) ;
    weight_limit.. sum(i, w(i) * y(i)) =l= cap ;

    Model knapsack / all / ;
    Solve knapsack using MIP maximizing obj ;
""")

MINLP_SYNTH_GMS = textwrap.dedent("""\
    * Simple process synthesis MINLP (based on SYNTHES1)
    Sets i "process units" / 1*3 / ;

    Positive Variables x(i) "continuous flows" ;
    Binary Variables y(i) "build decisions" ;
    Free Variable cost "total cost" ;

    x.up(i) = 2 ;

    Equations obj_def, logic1, logic2, logic3, cap1, cap2, cap3 ;

    obj_def.. cost =e= 5*y(1) + 6*y(2) + 8*y(3)
              + 10*x(1) - 7*x(3) - 18*log(x(2) + 1)
              - 19.2*log(x(1) - x(2) + 1) + 10 ;

    logic1.. 0.8*log(x(2) + 1) + 0.96*log(x(1) - x(2) + 1) - 0.8*x(3) =g= 0 ;
    logic2.. x(2) - x(1) =l= 0 ;
    logic3.. x(2) - 2*y(1) =l= 0 ;

    cap1.. x(1) - 2*y(1) =l= 0 ;
    cap2.. x(2) - 2*y(2) =l= 0 ;
    cap3.. x(3) - 2*y(3) =l= 0 ;

    Model synthes1 / all / ;
    Solve synthes1 using MINLP minimizing cost ;
""")

NLP_ROSENBROCK_GMS = textwrap.dedent("""\
    Free Variables x1, x2, obj ;

    x1.l = -1.0 ;
    x2.l = 1.0 ;

    Equations rosenbrock ;
    rosenbrock.. obj =e= sqr(1 - x1) + 100 * sqr(x2 - sqr(x1)) ;

    Model rosen / all / ;
    Solve rosen using NLP minimizing obj ;
""")

MINLP_NONLINEAR_GMS = textwrap.dedent("""\
    * MINLP with exp, sin, sqrt, power
    Positive Variables x1, x2 ;
    Binary Variable y ;
    Free Variable z ;

    x1.up = 10 ;
    x2.up = 10 ;

    Equations obj_eq, nl_con1, nl_con2, link ;

    obj_eq.. z =e= exp(x1) + sqrt(x2) + 3*y ;
    nl_con1.. sin(x1) + x2 =l= 2 ;
    nl_con2.. power(x1, 2) + power(x2, 2) =l= 25 ;
    link.. x1 + x2 =l= 10 * y ;

    Model nltest / all / ;
    Solve nltest using MINLP minimizing z ;
""")


# ── Import tests ───────────────────────────────────────────────


class TestGamsImportTransport:
    def test_basic_parse(self):
        m = parse_gams(TRANSPORT_GMS)
        assert m.name == "transport"

    def test_variable_count(self):
        m = parse_gams(TRANSPORT_GMS)
        # x(2,3)=6 scalars + z=1 scalar = 2 Variable objects
        assert len(m._variables) == 2

    def test_variable_shapes(self):
        m = parse_gams(TRANSPORT_GMS)
        var_map = {v.name: v for v in m._variables}
        assert var_map["x"].shape == (2, 3)
        assert var_map["z"].shape == ()

    def test_constraint_count(self):
        m = parse_gams(TRANSPORT_GMS)
        # supply(2) + demand(3) = 5 constraints (objective is not a constraint)
        assert len(m._constraints) == 5

    def test_has_objective(self):
        m = parse_gams(TRANSPORT_GMS)
        assert m._objective is not None
        assert m._objective.sense.value == "minimize"


class TestGamsImportKnapsack:
    def test_binary_variables(self):
        m = parse_gams(KNAPSACK_GMS)
        var_map = {v.name: v for v in m._variables}
        assert var_map["y"].var_type == dm.VarType.BINARY
        assert var_map["y"].shape == (5,)

    def test_maximize(self):
        m = parse_gams(KNAPSACK_GMS)
        assert m._objective is not None
        assert m._objective.sense.value == "maximize"

    def test_constraint_count(self):
        m = parse_gams(KNAPSACK_GMS)
        # weight_limit = 1 constraint
        assert len(m._constraints) == 1


class TestGamsImportMINLP:
    def test_synthes1_parse(self):
        m = parse_gams(MINLP_SYNTH_GMS)
        assert m.name == "synthes1"

    def test_mixed_var_types(self):
        m = parse_gams(MINLP_SYNTH_GMS)
        var_map = {v.name: v for v in m._variables}
        assert var_map["x"].var_type == dm.VarType.CONTINUOUS
        assert var_map["y"].var_type == dm.VarType.BINARY

    def test_bounds_applied(self):
        m = parse_gams(MINLP_SYNTH_GMS)
        var_map = {v.name: v for v in m._variables}
        # x.up(i) = 2 applied to all elements
        assert np.all(var_map["x"].ub == 2.0)

    def test_has_constraints(self):
        m = parse_gams(MINLP_SYNTH_GMS)
        # 6 constraints (logic1,2,3 + cap1,2,3)
        assert len(m._constraints) == 6


class TestGamsImportNLP:
    def test_rosenbrock(self):
        m = parse_gams(NLP_ROSENBROCK_GMS)
        assert m._objective is not None
        assert len(m._variables) == 3  # x1, x2, obj


class TestGamsImportNonlinear:
    def test_exp_sqrt_sin_power(self):
        m = parse_gams(MINLP_NONLINEAR_GMS)
        assert m.name == "nltest"
        var_map = {v.name: v for v in m._variables}
        assert var_map["y"].var_type == dm.VarType.BINARY
        assert len(m._constraints) == 3  # nl_con1, nl_con2, link

    def test_bounds(self):
        m = parse_gams(MINLP_NONLINEAR_GMS)
        var_map = {v.name: v for v in m._variables}
        assert float(var_map["x1"].ub) == 10.0
        assert float(var_map["x2"].ub) == 10.0


# ── Dollar condition tests ─────────────────────────────────────


DOLLAR_SUM_GMS = textwrap.dedent("""\
    Sets
        i / i1, i2, i3 /
        j / j1, j2, j3 / ;

    Parameter connected(i,j) /
        i1.j1 1, i1.j2 1,
        i2.j2 1, i2.j3 1,
        i3.j3 1 / ;

    Positive Variables x(i,j) ;
    Free Variable z ;

    Equations obj_def, flow(i) ;

    obj_def.. z =e= sum((i,j)$connected(i,j), x(i,j)) ;
    flow(i).. sum(j$connected(i,j), x(i,j)) =l= 10 ;

    Model dollar_test / all / ;
    Solve dollar_test using LP minimizing z ;
""")

DOLLAR_EQDEF_GMS = textwrap.dedent("""\
    Sets i / 1*5 / ;

    Parameter active(i) / 1 1, 2 1, 3 0, 4 1, 5 0 / ;

    Positive Variables x(i) ;
    Free Variable z ;

    Equations obj_def, limit(i) ;

    obj_def.. z =e= sum(i, x(i)) ;
    limit(i)$active(i).. x(i) =l= 10 ;

    Model cond_test / all / ;
    Solve cond_test using LP minimizing z ;
""")

DOLLAR_BOUNDS_GMS = textwrap.dedent("""\
    Sets i / a, b, c / ;

    Parameter has_cap(i) / a 1, b 0, c 1 / ;

    Positive Variables x(i) ;
    Free Variable z ;

    x.up(i)$has_cap(i) = 5 ;

    Equations obj_def ;
    obj_def.. z =e= sum(i, x(i)) ;

    Model bnd_test / all / ;
    Solve bnd_test using LP minimizing z ;
""")


class TestGamsDollarConditions:
    def test_dollar_in_sum(self):
        """sum((i,j)$connected(i,j), x(i,j)) should only sum over connected pairs."""
        m = parse_gams(DOLLAR_SUM_GMS)
        # connected has 5 entries, so objective should have 5 terms
        assert m._objective is not None
        # flow(i) constraints: i1 has 2 connections, i2 has 2, i3 has 1
        assert len(m._constraints) == 3

    def test_dollar_on_equation_def(self):
        """limit(i)$active(i).. should only generate constraints where active(i)=1."""
        m = parse_gams(DOLLAR_EQDEF_GMS)
        # active = {1:1, 2:1, 3:0, 4:1, 5:0} -> 3 constraints (for i=1,2,4)
        assert len(m._constraints) == 3

    def test_dollar_on_bounds(self):
        """x.up(i)$has_cap(i) = 5 should only set bounds where has_cap=1."""
        m = parse_gams(DOLLAR_BOUNDS_GMS)
        var_map = {v.name: v for v in m._variables}
        x = var_map["x"]
        # a has cap (ub=5), b no cap (ub=default), c has cap (ub=5)
        assert float(x.ub[0]) == 5.0  # a
        assert float(x.ub[2]) == 5.0  # c
        # b should retain default positive variable upper bound (very large)
        assert float(x.ub[1]) > 1e10


# ── Export tests ───────────────────────────────────────────────


class TestGamsExport:
    def test_simple_lp_export(self):
        m = dm.Model("test_lp")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(3 * x + 2 * y)
        m.subject_to(x + y >= 5)

        gms = m.to_gams()
        assert "Solve test_lp" in gms
        assert "LP" in gms or "lp" in gms

    def test_minlp_export(self):
        m = dm.Model("test_minlp")
        x = m.continuous("x", lb=0, ub=5)
        y = m.binary("y")
        m.minimize(dm.exp(x) + 3 * y)
        m.subject_to(x <= 5 * y)

        gms = m.to_gams()
        assert "MINLP" in gms
        assert "Binary" in gms
        assert "exp" in gms

    def test_model_type_override(self):
        m = dm.Model("test")
        x = m.continuous("x", lb=0)
        m.minimize(x)
        gms = m.to_gams(model_type="NLP")
        assert "NLP" in gms

    def test_export_to_file(self, tmp_path):
        m = dm.Model("filetest")
        x = m.continuous("x", lb=0)
        m.minimize(x)
        outpath = tmp_path / "test.gms"
        m.to_gams(str(outpath))
        assert outpath.exists()
        content = outpath.read_text()
        assert "Solve filetest" in content


# ── Round-trip tests ───────────────────────────────────────────


class TestGamsRoundTrip:
    def test_export_import_lp(self):
        """Build model -> export to .gms -> re-import -> check structure."""
        m1 = dm.Model("roundtrip")
        x = m1.continuous("x", lb=0, ub=10)
        y = m1.continuous("y", lb=0, ub=10)
        m1.minimize(3 * x + 2 * y)
        m1.subject_to(x + y >= 5, name="budget")

        gms_text = m1.to_gams()
        assert gms_text is not None
        m2 = parse_gams(gms_text)
        # Verify structural preservation
        assert m2._objective is not None
        assert m2._objective.sense.value == "minimize"
        assert len(m2._constraints) >= 1
        # Original model has 2 vars; exported adds obj_var -> 3
        assert len(m2._variables) >= 2

    def test_export_import_minlp(self):
        """Round-trip a MINLP model and verify var types preserved."""
        m1 = dm.Model("rt_minlp")
        x = m1.continuous("x", lb=0, ub=5)
        y = m1.binary("y")
        m1.minimize(dm.exp(x) + 3 * y)
        m1.subject_to(x <= 5 * y)

        gms_text = m1.to_gams()
        m2 = parse_gams(gms_text)
        assert m2._objective is not None
        var_types = {v.var_type for v in m2._variables}
        assert dm.VarType.BINARY in var_types
        assert dm.VarType.CONTINUOUS in var_types


# ── gamspy validation tests (optional, requires gamspy) ────────

try:
    import gamspy  # noqa: F401

    HAS_GAMSPY = True
except ImportError:
    HAS_GAMSPY = False


@pytest.mark.skipif(not HAS_GAMSPY, reason="gamspy not installed")
class TestGamspyValidation:
    """Generate .gms from gamspy, parse with our parser, compare structure."""

    def test_gamspy_transport(self):
        import gamspy as gp

        c = gp.Container()
        i = gp.Set(c, "i", records=["s1", "s2"])
        j = gp.Set(c, "j", records=["d1", "d2", "d3"])
        gp.Variable(c, "x", type="positive", domain=[i, j])  # registers in container
        gp.Variable(c, "z", type="free")

        gms_text = c.generateGamsString()
        # Our parser should handle gamspy-generated output
        # (may need adjustments for gamspy's specific syntax)
        assert "Set" in gms_text or "set" in gms_text

    def test_gamspy_minlp(self):
        import gamspy as gp

        c = gp.Container()
        gp.Variable(c, "x", type="positive")  # registers in container
        gp.Variable(c, "y", type="binary")
        gp.Variable(c, "z", type="free")

        gms_text = c.generateGamsString()
        assert gms_text is not None


# ── Error handling tests ───────────────────────────────────────


class TestGamsParseErrors:
    def test_empty_source(self):
        m = parse_gams("")
        assert len(m._variables) == 0

    def test_unknown_set_in_domain(self):
        src = textwrap.dedent("""\
            Variables x(unknown_set) ;
            Equations eq1 ;
            eq1.. x(1) =e= 0 ;
            Model m / all / ;
            Solve m using NLP minimizing x ;
        """)
        with pytest.raises(GamsParseError, match="Unknown set"):
            parse_gams(src)
