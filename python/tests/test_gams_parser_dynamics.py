"""GAMS parser tests for lag/lead indexing and equation-domain conditions (#87).

Second feature slice (see ``test_gams_parser_features.py``): dynamic set
indexing ``x(t+1)``, dollar conditions on equation domains, and conditioned
two-dimensional sums, verified by evaluating the parsed constraints and
objective at hand-computed points.
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


def test_lead_indexing_with_equation_dollar_condition():
    gms = textwrap.dedent("""\
        Sets t / t1*t4 / ;
        Positive Variables x(t) ;
        Free Variable obj ;
        x.up(t) = 10 ;
        Equations e, bal(t) ;
        bal(t)$(ord(t) < 4).. x(t+1) =g= x(t) ;
        e.. obj =e= sum(t, x(t)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    ev = NLPEvaluator(m)
    # The dollar condition drops t4's row (x(t4+1) is off the set): 3 rows.
    assert ev.n_constraints == 3
    # Monotone values: each row's residual has magnitude |x(t+1) - x(t)| = 1.
    g = np.asarray(ev.evaluate_constraints(np.array([1.0, 2.0, 3.0, 4.0, 0.0])))
    assert np.allclose(np.abs(g), 1.0)
    # Constant values: every difference row evaluates to zero.
    g0 = np.asarray(ev.evaluate_constraints(np.array([2.0, 2.0, 2.0, 2.0, 0.0])))
    assert np.allclose(g0, 0.0)


def test_lag_indexing_row_semantics():
    gms = textwrap.dedent("""\
        Sets t / t1*t3 / ;
        Positive Variables x(t) ;
        Free Variable obj ;
        x.up(t) = 10 ;
        Equations e, prev(t) ;
        prev(t)$(ord(t) > 1).. x(t) =e= 2 * x(t-1) ;
        e.. obj =e= sum(t, x(t)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    ev = NLPEvaluator(m)
    assert ev.n_constraints == 2
    # Geometric doubling satisfies every row exactly.
    g = np.asarray(ev.evaluate_constraints(np.array([1.0, 2.0, 4.0, 0.0])))
    assert np.allclose(g, 0.0)
    # Breaking the recursion at the last step shows up in exactly one row.
    g_bad = np.asarray(ev.evaluate_constraints(np.array([1.0, 2.0, 5.0, 0.0])))
    assert np.sum(~np.isclose(g_bad, 0.0)) == 1


def test_conditioned_two_dimensional_sum():
    gms = textwrap.dedent("""\
        Sets i / a, b /
             j / a, b / ;
        Positive Variables x(i,j) ;
        Free Variable obj ;
        x.up(i,j) = 5 ;
        Equations e ;
        e.. obj =e= sum((i,j)$(ord(i) < ord(j)), x(i,j)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    ev = NLPEvaluator(m)
    # Strict upper-triangle only: x(a,b).
    flat = np.array([9.0, 1.0, 2.0, 9.0, 0.0])  # (a,a), (a,b), (b,a), (b,b), obj
    assert float(ev.evaluate_objective(flat)) == pytest.approx(1.0, rel=1e-12)
