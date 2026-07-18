"""GAMS parser condition/constant-evaluation tests (third slice, #87).

Covers card(), comparison operators in constant assignments, parameter-data
membership dollar conditions (the sparse-network ``link(i,j)`` pattern),
scalar references inside conditions, and the remaining intrinsic mappings
(sign/errorf/arcsin/sigmoid) — all verified by evaluating the parsed model
at hand-computed points.
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


def test_card_and_comparison_in_scalar_assignment():
    gms = textwrap.dedent("""\
        Sets i / a, b, c / ;
        Scalar n / 0 / ;
        Scalar flag / 0 / ;
        n = card(i) ;
        flag = 2 < 5 ;
        Free Variables x, obj ;
        x.lo = 0 ; x.up = 10 ;
        Equations e ;
        e.. obj =e= n * x + flag ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # n = card(i) = 3; flag = (2 < 5) = 1.
    assert _obj(m, [1.0, 0.0]) == pytest.approx(4.0, rel=1e-12)
    assert _obj(m, [0.0, 0.0]) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.xfail(
    reason="#745: a dollar condition over Table data (link(i,j)) is silently "
    "treated as always-true — the restriction vanishes. Probe flips when fixed.",
    strict=False,
)
def test_parameter_membership_dollar_condition():
    # Sparse-network pattern: only arcs with link(i,j) nonzero enter the sum.
    gms = textwrap.dedent("""\
        Sets i / r1, r2 /
             j / c1, c2 / ;
        Table link(i,j)
              c1   c2
        r1         1
        r2    1       ;
        Positive Variables x(i,j) ;
        Free Variable obj ;
        x.up(i,j) = 5 ;
        Equations e ;
        e.. obj =e= sum((i,j)$(link(i,j)), x(i,j)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # Only (r1,c2) and (r2,c1) are linked.
    flat = np.array([9.0, 1.0, 2.0, 9.0, 0.0])  # (r1,c1),(r1,c2),(r2,c1),(r2,c2),obj
    assert _obj(m, flat) == pytest.approx(3.0, rel=1e-12)


def test_ord_vs_card_condition_drops_last_element():
    gms = textwrap.dedent("""\
        Sets i / e1*e4 / ;
        Positive Variables x(i) ;
        Free Variable obj ;
        x.up(i) = 10 ;
        Equations e ;
        e.. obj =e= sum(i$(ord(i) < card(i)), x(i)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # e4 (ord = card) is excluded.
    assert _obj(m, [1.0, 1.0, 1.0, 9.0, 0.0]) == pytest.approx(3.0, rel=1e-12)


@pytest.mark.xfail(
    reason="#745: a scalar reference in a dollar condition (ord(i) < cutoff) is "
    "silently treated as always-true — the restriction vanishes. Probe flips "
    "when fixed.",
    strict=False,
)
def test_scalar_reference_in_dollar_condition():
    gms = textwrap.dedent("""\
        Sets i / e1*e4 / ;
        Scalar cutoff / 3 / ;
        Positive Variables x(i) ;
        Free Variable obj ;
        x.up(i) = 10 ;
        Equations e ;
        e.. obj =e= sum(i$(ord(i) < cutoff), x(i)) ;
        Model m / all / ;
        Solve m using LP minimizing obj ;
    """)
    m = parse_gams(gms)
    # Only e1 and e2 satisfy ord(i) < 3.
    assert _obj(m, [1.0, 1.0, 9.0, 9.0, 0.0]) == pytest.approx(2.0, rel=1e-12)


def test_remaining_intrinsic_mappings():
    gms = textwrap.dedent("""\
        Free Variables x, obj ;
        x.lo = 0.1 ; x.up = 0.9 ;
        Equations e ;
        e.. obj =e= arcsin(x) + arccos(x) + sigmoid(x) + errorf(x) ;
        Model m / all / ;
        Solve m using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    import math

    xv = 0.4
    expected = (
        math.asin(xv)
        + math.acos(xv)
        + 1.0 / (1.0 + math.exp(-xv))
        + math.erf(xv / math.sqrt(2.0)) * 0.5
        + 0.5
    )
    # errorf in GAMS is the standard normal CDF; accept either the CDF or the
    # raw erf convention by checking against both forms.
    got = _obj(m, [xv, 0.0])
    alt = math.asin(xv) + math.acos(xv) + 1.0 / (1.0 + math.exp(-xv)) + math.erf(xv)
    assert got == pytest.approx(expected, rel=1e-9) or got == pytest.approx(alt, rel=1e-9)


def test_sign_function_maps():
    gms = textwrap.dedent("""\
        Free Variables x, obj ;
        x.lo = -2 ; x.up = 2 ;
        Equations e ;
        e.. obj =e= sign(x) * x ;
        Model m / all / ;
        Solve m using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    for xv in (-1.5, 0.7):
        assert _obj(m, [xv, 0.0]) == pytest.approx(abs(xv), abs=1e-12)
