"""GAMS parser coverage tests: intrinsics, aggregations, tables (#87).

Complements ``test_gams.py``: each fixture exercises parser machinery the
transport/knapsack models do not touch — intrinsic function mapping,
``prod``/dollar-conditioned ``sum`` aggregations, ``alias``, ``loop``
parameter assignment, tables with blanks, and bound statements. The parser
folds a ``obj =e= expr`` defining equation into the objective, so semantics
are verified by evaluating the parsed objective at known points (never just
by "it parsed").
"""

from __future__ import annotations

import os
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.gams_parser import parse_gams

pytestmark = pytest.mark.unit


def _obj(model, x_flat):
    return float(NLPEvaluator(model).evaluate_objective(np.asarray(x_flat, dtype=np.float64)))


def test_intrinsic_functions_evaluate_correctly():
    gms = textwrap.dedent("""\
        Free Variables x, obj ;
        x.lo = 0.5 ; x.up = 2.0 ;

        Equations e ;
        e.. obj =e= exp(x) + log(x) + sqrt(x) + sqr(x) + power(x, 3)
                  + sin(x) + cos(x) + tanh(x) + abs(x) + min(x, 1.5) + max(x, 0.7) ;

        Model m / all / ;
        Solve m using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    xv = 1.3
    expected = (
        np.exp(xv)
        + np.log(xv)
        + np.sqrt(xv)
        + xv**2
        + xv**3
        + np.sin(xv)
        + np.cos(xv)
        + np.tanh(xv)
        + abs(xv)
        + min(xv, 1.5)
        + max(xv, 0.7)
    )
    assert _obj(m, [xv, 0.0]) == pytest.approx(expected, rel=1e-12)


def test_more_intrinsics_log2_log10_trig_inverse():
    gms = textwrap.dedent("""\
        Free Variables x, obj ;
        x.lo = 0.1 ; x.up = 0.9 ;
        Equations e ;
        e.. obj =e= log2(x) + log10(x) + arctan(x) + sinh(x) + cosh(x) ;
        Model m / all / ;
        Solve m using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    xv = 0.6
    expected = np.log2(xv) + np.log10(xv) + np.arctan(xv) + np.sinh(xv) + np.cosh(xv)
    assert _obj(m, [xv, 0.0]) == pytest.approx(expected, rel=1e-12)


def test_signpower_smooth_form():
    gms = textwrap.dedent("""\
        Free Variables x, obj ;
        x.lo = -2 ; x.up = 2 ;
        Equations e ;
        e.. obj =e= signpower(x, 2) ;
        Model m / all / ;
        Solve m using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    for xv in (-1.5, -0.3, 0.0, 0.8):
        assert _obj(m, [xv, 0.0]) == pytest.approx(np.sign(xv) * abs(xv) ** 2, abs=1e-12)


def test_prod_aggregation_and_alias():
    gms = textwrap.dedent("""\
        Sets i / a, b, c / ;
        Alias (i, ip) ;
        Parameter p(i) / a 2, b 3, c 4 / ;
        Positive Variables x(i) ;
        Free Variable obj ;
        x.up(i) = 5 ;
        Equations e, link(i) ;
        e.. obj =e= prod(i, x(i)) ;
        link(i).. x(i) =g= sum(ip, p(ip)) / 9 ;
        Model m / all / ;
        Solve m using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    # Objective: prod over x values.
    assert _obj(m, [1.5, 2.0, 3.0, 0.0]) == pytest.approx(9.0, rel=1e-12)
    # Alias sum: each link row is x(i) - (2+3+4)/9 >= 0, tight at x(i) = 1.
    ev = NLPEvaluator(m)
    g = np.asarray(ev.evaluate_constraints(np.array([1.0, 1.0, 1.0, 0.0])))
    assert g.shape[0] == 3
    ones_residual = g if g.ndim else g.reshape(1)
    # All three rows evaluate to the same residual at the symmetric point.
    assert np.allclose(ones_residual, ones_residual[0])


def test_dollar_condition_in_sum_skips_elements():
    gms = textwrap.dedent("""\
        Sets i / e1*e4 / ;
        Parameter p(i) / e1 1, e2 2, e3 3, e4 4 / ;
        Positive Variables x(i) ;
        Free Variable obj ;
        x.up(i) = 10 ;
        Equations e ;
        e.. obj =e= sum(i$(ord(i) > 2), p(i) * x(i)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # Only e3 and e4 contribute: 3*x3 + 4*x4; x1/x2 coefficients are dropped.
    assert _obj(m, [7.0, 7.0, 1.0, 1.0, 0.0]) == pytest.approx(7.0, rel=1e-12)
    assert _obj(m, [0.0, 0.0, 2.0, 0.5, 0.0]) == pytest.approx(8.0, rel=1e-12)


def test_loop_assignment_builds_parameter():
    # Regression probe for #745 (fixed on main): loop-body parameter
    # assignments must populate the store the equation builder reads.
    gms = textwrap.dedent("""\
        Sets i / k1*k3 / ;
        Parameter p(i) ;
        loop(i,
            p(i) = 2 * ord(i) ;
        ) ;
        Positive Variables x(i) ;
        Free Variable obj ;
        x.up(i) = 10 ;
        Equations e ;
        e.. obj =e= sum(i, p(i) * x(i)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # p = (2, 4, 6) from the loop body.
    assert _obj(m, [1.0, 1.0, 1.0, 0.0]) == pytest.approx(12.0, rel=1e-12)
    assert _obj(m, [1.0, 0.0, 0.5, 0.0]) == pytest.approx(5.0, rel=1e-12)


def test_bound_statements_lo_up_fx():
    gms = textwrap.dedent("""\
        Sets i / a, b / ;
        Positive Variables x(i) ;
        Free Variable obj ;
        x.lo('a') = 0.5 ;
        x.up('a') = 1.5 ;
        x.fx('b') = 2.0 ;
        Equations e ;
        e.. obj =e= sum(i, x(i)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    var_map = {v.name: v for v in m._variables}
    lb = np.asarray(var_map["x"].lb, dtype=np.float64).ravel()
    ub = np.asarray(var_map["x"].ub, dtype=np.float64).ravel()
    assert lb[0] == 0.5 and ub[0] == 1.5
    assert lb[1] == 2.0 and ub[1] == 2.0  # .fx pins both bounds


def test_scalar_constant_function_folding():
    gms = textwrap.dedent("""\
        Scalar s1 / 0 / ;
        Scalar s2 / 0 / ;
        s1 = power(3, 2) + sqrt(16) ;
        s2 = sqrt(s1 + 3) ;
        Free Variables x, obj ;
        x.lo = 0 ; x.up = 20 ;
        Equations e ;
        e.. obj =e= s1 * x + s2 ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # s1 = 13 (power + sqrt fold); s2 = sqrt(16) = 4 (scalar reference).
    assert _obj(m, [1.0, 0.0]) == pytest.approx(17.0, rel=1e-12)
    assert _obj(m, [0.0, 0.0]) == pytest.approx(4.0, rel=1e-12)


def test_two_dimensional_table_with_missing_entries():
    gms = textwrap.dedent("""\
        Sets i / r1, r2 /
             j / c1, c2, c3 / ;
        Table t(i,j)
              c1   c2   c3
        r1    1.0        3.0
        r2         5.0   6.0 ;
        Positive Variables x(i,j) ;
        Free Variable obj ;
        x.up(i,j) = 4 ;
        Equations e ;
        e.. obj =e= sum((i,j), t(i,j) * x(i,j)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # Blanks read as zero: total at all-ones is 1 + 3 + 5 + 6 = 15.
    flat = np.ones(7)
    flat[-1] = 0.0
    assert _obj(m, flat) == pytest.approx(15.0, rel=1e-12)
