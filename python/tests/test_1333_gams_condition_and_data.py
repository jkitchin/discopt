"""#1333: two GAMS-reader defects, each producing a silently wrong model.

1. ``smin``/``smax`` inside an equation's ``$``-condition **failed open**.
   They parsed into an ``ExprFunc`` that discarded the index set, so the
   constant evaluator returned ``None`` -- and ``None`` was read as "generate
   the row". ``c1$(smin(i, d(i)) > 10).. x1 =g= 3`` with ``smin = 1`` produced
   a constraint GAMS does not generate, and the solve returned 3.0 for a model
   whose optimum is -5.

2. Parameter data under **numeric labels** misread signed values. The
   label/value split was a one-token lookahead, which a ``-`` (its own token)
   breaks: in ``/1 2, 2 -1, 3 4/`` the record ``2 -1`` lost its label, so
   ``a('2')`` was never set and ``sum(a)`` came back 6 where GAMS gives 5.
   Alphabetic labels parsed correctly, which is why this went unseen.
"""

import numpy as np
import pytest
from discopt.modeling.gams_parser import GamsParseError, parse_gams

# ── 1. smin / smax in a $-condition ─────────────────────────────────────────

_COND_MODEL = """
Set i /i1*i3/;
Parameter d(i) /i1 2, i2 1, i3 4/;
Variable x1, obj;
Equation c1, odef;
c1$({cond}).. x1 =g= 3;
odef.. obj =e= x1;
x1.lo = -5; x1.up = 5;
Model m /all/;
Solve m using nlp minimizing obj;
"""


@pytest.mark.correctness
@pytest.mark.parametrize(
    "cond,expected",
    [
        # smin(d) = 1, so the condition is false and GAMS omits the row.
        ("smin(i, d(i)) > 10", -5.0),
        # ...and true, so the row IS generated.
        ("smin(i, d(i)) < 10", 3.0),
        # smax(d) = 4.
        ("smax(i, d(i)) > 100", -5.0),
        ("smax(i, d(i)) > 1", 3.0),
    ],
)
def test_smin_smax_in_a_condition_decides_the_row(cond, expected):
    """The condition must be *evaluated*, in both directions."""
    m = parse_gams(_COND_MODEL.format(cond=cond))
    r = m.solve()
    assert r.objective == pytest.approx(expected, abs=1e-6), (
        f"$({cond}): objective {r.objective}, expected {expected}"
    )


@pytest.mark.unit
def test_smin_smax_fold_to_the_right_number():
    """Direct check on the folder, independent of any row decision."""
    src = """
Set i /i1*i4/;
Parameter d(i) /i1 2, i2 -1, i3 4, i4 0.5/;
Variable x1, obj;
Equation odef;
odef.. obj =e= x1 - smin(i, d(i)) - smax(i, d(i));
x1.lo = 0; x1.up = 0;
Model m /all/;
Solve m using nlp minimizing obj;
"""
    m = parse_gams(src)
    r = m.solve()
    # -smin - smax = -(-1) - 4 = -3
    assert r.objective == pytest.approx(-3.0, abs=1e-9)


@pytest.mark.correctness
def test_an_unevaluable_condition_is_refused_not_included():
    """ "I cannot tell" has no safe default for a row's existence."""
    src = """
Set i /i1*i3/;
Variable x1, obj;
Equation c1, odef;
c1$(uniform(0,1) > 0.5).. x1 =g= 3;
odef.. obj =e= x1;
x1.lo = -5; x1.up = 5;
Model m /all/;
Solve m using nlp minimizing obj;
"""
    with pytest.raises(GamsParseError, match="could not be evaluated"):
        parse_gams(src)


@pytest.mark.unit
def test_smin_over_endogenous_values_is_still_refused_by_name():
    """#1325's refusal stands: an endogenous smin has no algebraic form."""
    src = """
Set i /i1*i3/;
Variable x(i), obj;
Equation odef;
odef.. obj =e= smin(i, x(i));
Model m /all/;
Solve m using nlp minimizing obj;
"""
    with pytest.raises(GamsParseError, match="smin"):
        parse_gams(src)


# ── 2. parameter data with numeric labels ───────────────────────────────────


def _sum_of(labels, data):
    src = f"""
Set i /{labels}/;
Parameter a(i) {data};
Variable x1, obj;
Equation odef;
odef.. obj =e= x1 - sum(i, a(i));
x1.lo = 0; x1.up = 0;
Model m /all/;
Solve m using nlp minimizing obj;
"""
    r = parse_gams(src).solve()
    assert r.objective is not None
    return -r.objective


@pytest.mark.correctness
@pytest.mark.parametrize(
    "labels,data,expected",
    [
        ("1*3", "/1 2, 2 -1, 3 4/", 5.0),
        ("1*3", "/1 -1/", -1.0),
        ("1*3", "/1 2, 2 -1.5, 3 4/", 4.5),
        ("1*3", "/1 2, 2 +1, 3 4/", 7.0),
        ("1*3", "/1 2\n 2 -1\n 3 4/", 5.0),
        # The control that always worked and must keep working.
        ("a,b,c", "/a 2, b -1, c 4/", 5.0),
        ("a,b,c", "/a -2.5, b 1, c 4/", 2.5),
    ],
)
def test_signed_parameter_data_reads_its_sign(labels, data, expected):
    assert _sum_of(labels, data) == pytest.approx(expected, abs=1e-9), (
        f"data {data!r} under labels {labels!r}"
    )


@pytest.mark.unit
def test_a_scalar_parameter_is_still_a_value_not_a_label():
    """``Parameter p /3.5/`` is a bare value: the one ambiguous shape."""
    from discopt.modeling.gams_parser import _Parser, _tokenize

    parser = _Parser(_tokenize("3.5/"))
    assert parser._parse_param_data() == {("",): 3.5}

    parser = _Parser(_tokenize("a, b, c/"))
    assert parser._parse_param_data() == {"a": 0.0, "b": 0.0, "c": 0.0}


@pytest.mark.unit
def test_multidimensional_keys_still_parse():
    from discopt.modeling.gams_parser import _Parser, _tokenize

    parser = _Parser(_tokenize("i1.j1 -2.5, i2.j2 3/"))
    assert parser._parse_param_data() == {("i1", "j1"): -2.5, ("i2", "j2"): 3.0}

    parser = _Parser(_tokenize("i1.j1 2, i2.j2 -1, i3.j3 4/"))
    assert parser._parse_param_data() == {
        ("i1", "j1"): 2.0,
        ("i2", "j2"): -1.0,
        ("i3", "j3"): 4.0,
    }


@pytest.mark.correctness
def test_numeric_labels_land_on_the_right_elements():
    """Not just the sum: each element must carry its own value."""
    src = """
Set i /1*3/;
Parameter a(i) /1 2, 2 -1, 3 4/;
Variable x(i), obj;
Equation odef, c(i);
c(i).. x(i) =e= a(i);
odef.. obj =e= sum(i, x(i) * x(i));
Model m /all/;
Solve m using nlp minimizing obj;
"""
    r = parse_gams(src).solve()
    # 2^2 + (-1)^2 + 4^2 = 21; the pre-fix parse gave a('2') = 0 -> 20.
    assert r.objective == pytest.approx(21.0, abs=1e-6)
    xs = np.concatenate([np.ravel(v) for v in r.x.values()])
    assert np.isclose(np.min(xs), -1.0, atol=1e-6), "a('2') = -1 did not reach the model"
