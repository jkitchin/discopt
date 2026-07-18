"""
Regression tests for GAMS parser silent-wrong-parse defects (#745).

Covers:
  1. Indexed parameter assignment statements (plain ``p(i) = expr`` and
     inside ``loop(i, ...)``) must actually update the parameter store the
     equation builder reads — previously they silently no-opped, leaving
     all-zero coefficients.
  2. Two-argument ``min``/``max`` in constant assignments must fold to the
     correct value — previously the constant evaluator fell through and the
     assignment was silently dropped (scalar kept its declared value).
  3. Assignments the constant evaluator cannot fold must raise
     ``GamsParseError`` loudly instead of silently building a wrong model.
  4. A per-element level statement ``x.l('a') = 1.0`` must store the initial
     value — previously it raised ``IndexError`` in ``_store_initial_value``.
"""

from __future__ import annotations

import textwrap

import pytest
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    UnaryOp,
)
from discopt.modeling.gams_parser import GamsParseError, parse_gams

pytestmark = pytest.mark.unit


def _constants_in(expr) -> set[float]:
    """Collect all scalar Constant values appearing in an expression DAG."""
    out: set[float] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Constant):
            if node.value.ndim == 0:
                out.add(float(node.value))
        elif isinstance(node, BinaryOp):
            stack.extend([node.left, node.right])
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, FunctionCall):
            stack.extend(node.args)
        elif isinstance(node, IndexExpression):
            stack.append(node.base)
    return out


LOOP_PARAM_GMS = textwrap.dedent("""\
    Sets i / 1*3 / ;
    Parameter p(i) / 1 0, 2 0, 3 0 / ;

    loop(i, p(i) = ord(i) * 10) ;

    Positive Variables x(i) ;
    Free Variable z ;

    Equations obj_def, limit(i) ;
    obj_def.. z =e= sum(i, p(i) * x(i)) ;
    limit(i).. x(i) =l= 100 ;

    Model ltest / all / ;
    Solve ltest using LP minimizing z ;
""")


PLAIN_ASSIGN_GMS = textwrap.dedent("""\
    Sets i / 1*3 / ;
    Parameter p(i) ;
    p(i) = 2 * ord(i) ;

    Positive Variables x(i) ;
    Free Variable z ;

    Equations obj_def ;
    obj_def.. z =e= sum(i, p(i) * x(i)) ;

    Model ptest / all / ;
    Solve ptest using LP minimizing z ;
""")


DOLLAR_ASSIGN_GMS = textwrap.dedent("""\
    Sets i / 1*3 / ;
    Parameter p(i) / 1 7, 2 7, 3 7 / ;
    p(i)$(ord(i) > 1) = 100 ;

    Positive Variables x(i) ;
    Free Variable z ;

    Equations obj_def ;
    obj_def.. z =e= sum(i, p(i) * x(i)) ;

    Model dtest / all / ;
    Solve dtest using LP minimizing z ;
""")


MINMAX_GMS = textwrap.dedent("""\
    Scalar s1 / 0 / ;
    Scalar s2 / 0 / ;
    Scalar s3 / 0 / ;
    Scalar s4 / 0 / ;
    s1 = 13 ;
    s2 = min(13, 10) ;
    s3 = min(s1, 10) ;
    s4 = max(s1, 2) ;

    Positive Variable x ;
    Free Variable z ;

    Equations obj_def ;
    obj_def.. z =e= s2 * x + s3 * x + s4 * x ;

    Model mtest / all / ;
    Solve mtest using LP minimizing z ;
""")


class TestIndexedParamAssignment:
    def test_loop_assignment_builds_parameter(self):
        """loop(i, p(i) = ord(i)*10) must yield coefficients 10, 20, 30."""
        m = parse_gams(LOOP_PARAM_GMS)
        assert m._objective is not None
        consts = _constants_in(m._objective.expression)
        assert {10.0, 20.0, 30.0} <= consts, (
            f"loop-assigned parameter values missing from objective: {consts}"
        )

    def test_plain_indexed_assignment_builds_parameter(self):
        """Direct p(i) = 2*ord(i) must yield coefficients 2, 4, 6."""
        m = parse_gams(PLAIN_ASSIGN_GMS)
        assert m._objective is not None
        consts = _constants_in(m._objective.expression)
        assert {2.0, 4.0, 6.0} <= consts, (
            f"assigned parameter values missing from objective: {consts}"
        )

    def test_dollar_conditioned_indexed_assignment(self):
        """p(i)$(ord(i) > 1) = 100 must only overwrite elements 2 and 3."""
        m = parse_gams(DOLLAR_ASSIGN_GMS)
        assert m._objective is not None
        consts = _constants_in(m._objective.expression)
        assert {7.0, 100.0} <= consts, f"expected mixed 7/100 coefficients: {consts}"

    def test_unevaluable_assignment_raises(self):
        """An RHS the constant evaluator cannot fold must refuse loudly."""
        src = textwrap.dedent("""\
            Scalar s / 0 / ;
            s = uniform(0, 1) ;
            Free Variable z ;
            Equations obj_def ;
            obj_def.. z =e= s ;
            Model utest / all / ;
            Solve utest using LP minimizing z ;
        """)
        with pytest.raises(GamsParseError):
            parse_gams(src)


class TestScalarConstantFolding:
    def test_scalar_constant_min_max_folding(self):
        """min/max in constant assignments must fold, not zero out."""
        m = parse_gams(MINMAX_GMS)
        assert m._objective is not None
        consts = _constants_in(m._objective.expression)
        # s2 = min(13,10) = 10 ; s3 = min(s1,10) = 10 ; s4 = max(s1,2) = 13
        assert 10.0 in consts, f"min() did not fold to 10: {consts}"
        assert 13.0 in consts, f"max() did not fold to 13: {consts}"
        assert 0.0 not in consts, f"min/max silently folded to 0: {consts}"

    def test_scalar_sum_folding(self):
        """sum(i, p(i)) in a constant assignment must fold with env."""
        src = textwrap.dedent("""\
            Sets i / 1*3 / ;
            Parameter p(i) / 1 1, 2 2, 3 3 / ;
            Scalar tot / 0 / ;
            tot = sum(i, p(i)) ;

            Positive Variable x ;
            Free Variable z ;

            Equations obj_def ;
            obj_def.. z =e= tot * x ;

            Model stest / all / ;
            Solve stest using LP minimizing z ;
        """)
        m = parse_gams(src)
        assert m._objective is not None
        consts = _constants_in(m._objective.expression)
        assert 6.0 in consts, f"sum() did not fold to 6: {consts}"


class TestPerElementLevel:
    def test_per_element_level_statement(self):
        """x.l('a') = 1.0 must store an initial value, not raise IndexError."""
        src = textwrap.dedent("""\
            Sets i / a, b / ;
            Positive Variables x(i) ;
            Free Variable z ;

            Equations obj_def ;
            obj_def.. z =e= sum(i, x(i)) ;

            x.l('a') = 1.0 ;

            Model itest / all / ;
            Solve itest using LP minimizing z ;
        """)
        m = parse_gams(src)
        initial = getattr(m, "_gams_initial_values", None)
        assert initial is not None
        assert initial["x"][0] == 1.0
        assert 1 not in initial["x"]
