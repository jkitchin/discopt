"""#1986: a GAMS type statement that repeats only a variable's name dropped its domain.

GAMS lets a variable's type be set in a separate statement::

    Variables x(i,j), z ;
    Positive Variable x ;

GAMS's own ``trnsport.gms`` is written this way. ``from_gams`` read the second
line as a new declaration of a *scalar* ``x``: the model had 2 columns instead
of 7 and solved ``infeasible``. A type statement for an existing symbol now
keeps its domain and only sets the type; a conflicting domain or type is refused.
"""

from __future__ import annotations

import pytest
from discopt.modeling.gams_parser import GamsParseError, _Parser, _tokenize, parse_gams

HEAD = """
Sets i / seattle, san-diego / ;
Sets j / new-york, chicago, topeka / ;
Parameters a(i) / seattle 350, san-diego 600 / ;
Parameters b(j) / new-york 325, chicago 300, topeka 275 / ;
Table d(i,j)
              new-york  chicago  topeka
  seattle        2.5      1.7      1.8
  san-diego      2.5      1.8      1.4 ;
"""

TAIL = """
Equations cost, supply(i), demand(j) ;
cost ..       z =e= sum((i,j), 0.09*d(i,j)*x(i,j)) ;
supply(i) ..  sum(j, x(i,j)) =l= a(i) ;
demand(j) ..  sum(i, x(i,j)) =g= b(j) ;
Model transport / all / ;
Solve transport using lp minimizing z ;
"""

SPLIT = HEAD + "Variables x(i,j), z ;\nPositive Variable x ;\n" + TAIL
ONCE = HEAD + "Positive Variables x(i,j) ;\nVariables z ;\n" + TAIL


def _parser(src: str) -> _Parser:
    p = _Parser(_tokenize(src))
    p.parse()
    return p


def _ncols(m) -> int:
    return sum(v.size for v in m._variables)


def test_type_statement_keeps_domain():
    gv = _parser(SPLIT).variables["x"]
    assert gv.domain == ["i", "j"]
    assert gv.var_type == "positive"


def test_split_declaration_matches_single_declaration():
    split, once = parse_gams(SPLIT), parse_gams(ONCE)
    # 6 shipments; the free objective variable `z` is substituted out (#2002)
    assert _ncols(split) == _ncols(once) == 6
    rs, ro = split.solve(), once.solve()
    assert rs.status == ro.status == "optimal"
    assert rs.objective == pytest.approx(153.675, abs=1e-6)
    assert rs.objective == pytest.approx(ro.objective, abs=1e-9)


def test_untyped_redeclaration_keeps_type():
    p = _parser("Set i / a, b / ;\nPositive Variable x(i) ;\nVariable x ;\n")
    assert p.variables["x"].var_type == "positive"
    assert p.variables["x"].domain == ["i"]


def test_repeated_matching_domain_is_accepted():
    p = _parser("Set i / a, b / ;\nVariable x(i) ;\nBinary Variable x(i) ;\n")
    assert p.variables["x"].var_type == "binary"
    assert p.variables["x"].domain == ["i"]


def test_conflicting_domain_is_refused():
    src = "Sets i / a / ; Sets j / b / ;\nVariable x(i) ;\nPositive Variable x(j) ;\n"
    with pytest.raises(GamsParseError, match="redeclared with domain"):
        _parser(src)


def test_conflicting_type_is_refused():
    src = "Set i / a / ;\nPositive Variable x(i) ;\nBinary Variable x ;\n"
    with pytest.raises(GamsParseError, match="redeclared as binary"):
        _parser(src)
