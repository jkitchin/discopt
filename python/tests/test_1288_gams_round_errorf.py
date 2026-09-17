"""#1288: GAMS ``round`` and ``errorf`` must keep their GAMS meaning.

Reference values come from GAMS 53.2.0 (full license) listings saved by the
adversary run: ``r1252_round_fold.lst`` (objective 24.0000), ``r_errorf.lst``
(0.2420 at x = -0.7) and ``w_erf.lst`` (0.6179 for ``errorf(0.3)``).

- ``round`` is half away from zero in GAMS; Python's ``round`` is half to even.
- ``errorf(x)`` is the standard normal CDF Φ(x), not ``erf(x)``.
"""

from __future__ import annotations

import math
import textwrap

import discopt as do
import numpy as np
import pytest
from discopt.export.gams import to_gams
from discopt.modeling.gams_parser import GamsParseError, parse_gams


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_round_fold_matches_gams_listing():
    # The exact text GAMS solved to 24 with x = (3, 13, 7, 1).
    gms = textwrap.dedent("""\
        Scalar p, pm /-2.5/; p = round(0.5);
        Variables z; Positive Variables x1, x2, x3, x4;
        Equations obj, c1, c2, c3, c4;
        obj.. z =e= x1 + x2 + x3 + x4;
        c1.. x1 =l= round(2.5);  c2.. x2 =l= round(0.125, 2)*100;
        c3.. x3 =l= round(pm) + 10;  c4.. x4 =l= p;
        Model m /all/; Solve m using lp maximizing z;
    """)
    r = parse_gams(gms).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(24.0, abs=1e-6)
    xs = {k: float(np.asarray(v)) for k, v in r.x.items()}
    assert [xs["x1"], xs["x2"], xs["x3"], xs["x4"]] == pytest.approx([3, 13, 7, 1], abs=1e-6)


@pytest.mark.parametrize(
    "call, value",
    [
        ("floor(-2.5)", -3.0),
        ("ceil(0-2.5)", -2.0),
        ("floor(2*1.5)", 3.0),
        ("round(-0.5)", -1.0),
        ("round(1.25, 1)", 1.3),
        ("round(-(2.5))", -3.0),
        ("mod(-7, 2+1)", -1.0),
    ],
)
def test_constant_argument_calls_fold(call, value):
    gms = textwrap.dedent(f"""\
        Variables z; Variable x; x.lo = -100; x.up = 100;
        Equations obj, c;
        obj.. z =e= x;
        c.. x =l= {call};
        Model m /all/; Solve m using lp maximizing z;
    """)
    r = parse_gams(gms).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(value, abs=1e-9)


def test_discontinuous_call_over_a_variable_is_still_refused():
    gms = textwrap.dedent("""\
        Variables z; Variable x; x.lo = -3; x.up = 3;
        Equations obj;
        obj.. z =e= floor(-x);
        Model m /all/; Solve m using minlp minimizing z;
    """)
    with pytest.raises(GamsParseError):
        parse_gams(gms)


def test_reader_errorf_is_the_normal_cdf():
    m = parse_gams(
        "Variables z; Variable x; x.fx = -0.7; Equations obj; obj.. z =e= errorf(x);"
        " Model mr /all/; Solve mr using dnlp minimizing z;"
    )
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(_phi(-0.7), abs=1e-7)
    assert round(r.objective, 4) == 0.2420  # r_errorf.lst


def test_data_statement_errorf_is_the_normal_cdf():
    gms = textwrap.dedent("""\
        Scalar q; q = errorf(-0.7);
        Variables z; Variable x; x.lo = -10; x.up = 10;
        Equations obj, c;
        obj.. z =e= x;
        c.. x =l= q;
        Model m /all/; Solve m using lp maximizing z;
    """)
    r = parse_gams(gms).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(_phi(-0.7), abs=1e-9)


def test_writer_erf_round_trips_through_gams_errorf():
    m = do.Model("e")
    a = m.continuous("a", lb=0.3, ub=0.3)
    m.minimize(do.erf(a))
    text = to_gams(m)
    # GAMS evaluates a bare errorf(0.3) to 0.6179 (w_erf.lst) — the writer must
    # not emit that for erf(0.3) = 0.3286.
    assert "errorf(a)" not in text
    r = parse_gams(text).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(math.erf(0.3), abs=1e-7)
