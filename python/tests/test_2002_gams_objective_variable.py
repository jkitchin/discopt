"""#2002: the GAMS reader's objective-variable extraction.

``Solve m minimizing z`` with ``obj.. z =e= expr`` used to minimize ``expr``,
drop ``obj`` and still create ``z`` as a disconnected column. A bound on ``z``
was then silently lost (a 2.0 optimum reported as 0.0) and a free ``z`` became
a dangling unbounded column. Substituting ``z`` out is exact only when ``obj``
is its sole appearance and ``z`` is free and unbounded; otherwise ``z`` and
every equation are kept and ``z`` itself is minimized.

Also: a post-solve reporting assignment (``MODELSTAT = m.modelstat;``) left
``.modelstat`` in the token stream and raised ``Expected '='``.
"""

import warnings

import pytest
from discopt.modeling.gams_parser import GamsParseError, parse_gams

FREE = """
Variables z, x;
Positive Variables y;
Equations obj, con;
obj.. z =e= x + y;
con.. x + 2*y =g= 3;
x.lo = -5; x.up = 5;
y.up = 10;
Model m /all/;
Solve m using lp minimizing z;
"""


def _names(m):
    return [v.name for v in m._variables]


def test_bound_on_objective_variable_is_kept():
    m = parse_gams(
        """
Variables z, x;
Equations obj;
obj.. z =e= x;
x.lo = 0; x.up = 10;
z.lo = 2; z.up = 100;
Model m /all/;
Solve m using lp minimizing z;
"""
    )
    assert _names(m) == ["z", "x"]
    assert len(m._constraints) == 1
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-7)


def test_free_objective_variable_is_substituted_out():
    m = parse_gams(FREE)
    assert _names(m) == ["x", "y"]
    assert len(m._constraints) == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = m.solve()
    assert not [x for x in w if "very large or infinite declared bounds" in str(x.message)]
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-7)


@pytest.mark.parametrize(
    "extra, n_cons",
    [
        # z appears in a second equation
        ("Equations cap; cap.. z =l= 7;", 3),
        # z on both sides of its definition
        ("", None),
    ],
)
def test_reused_objective_variable_keeps_definition(extra, n_cons):
    src = FREE.replace("Model m", extra + "\nModel m")
    if n_cons is None:
        src = src.replace("obj.. z =e= x + y;", "obj.. z =e= x + y + 0*z;")
        n_cons = 2
    m = parse_gams(src)
    assert _names(m) == ["z", "x", "y"]
    assert len(m._constraints) == n_cons
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-7)


def test_two_defining_equations_both_stay_constraints():
    # Each `z =e= ...` used to call minimize(); the last one silently won and
    # the other was dropped as a constraint.
    m = parse_gams(
        """
Variables z, x, y;
Equations d1, d2;
d1.. z =e= x;
d2.. z =e= y + 1;
x.lo = 0; x.up = 10; y.lo = 0; y.up = 10;
Model m /all/;
Solve m using lp minimizing z;
"""
    )
    assert _names(m) == ["z", "x", "y"]
    assert len(m._constraints) == 2
    r = m.solve()
    assert r.objective == pytest.approx(1.0, abs=1e-7)


def test_reversed_definition_case_insensitive_and_maximize():
    m = parse_gams(
        """
Variables Z, x;
Equations obj;
obj.. 3 - x =e= z;
x.lo = 1; x.up = 2;
Model m /all/;
Solve m using lp maximizing z;
"""
    )
    assert _names(m) == ["x"]
    r = m.solve()
    assert r.objective == pytest.approx(2.0, abs=1e-7)


def test_post_solve_reporting_assignment_is_skipped():
    m = parse_gams(
        FREE
        + """
Scalars MODELSTAT 'model status', SOLVESTAT 'solve status', zl;
MODELSTAT = m.modelstat;
SOLVESTAT = m.solvestat;
zl = z.l;
"""
    )
    assert m.solve().objective == pytest.approx(-1.0, abs=1e-7)


def test_attribute_reference_elsewhere_is_refused():
    with pytest.raises(GamsParseError, match=r"x\.l"):
        parse_gams(FREE.replace("con.. x + 2*y =g= 3;", "con.. x + 2*y =g= x.l;"))
    with pytest.raises(GamsParseError, match=r"x\.lo"):
        parse_gams("Scalar s;\nVariable x;\nx.lo = 1;\ns = x.lo;\n" + FREE)
