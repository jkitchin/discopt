"""``to_gams`` wraps long statements under GAMS's input line limit (#1739).

GAMS rejects any input line longer than 80000 characters
(``****  98  Non-blank character(s) beyond max input line length``). The writer
emitted every equation name on one ``Equations ...;`` line, every variable of a
type on one ``... Variables ...;`` line and each equation definition on one
line, so a model with ~10,000 rows -- or one dense row -- wrote a ``.gms`` file
that does not compile.

Wrapping must not change the model, so beyond line lengths every test here
re-reads the output with ``parse_gams`` and compares the rows value for value.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

from discopt._relax.dag_compiler import compile_expression
from discopt.export.gams import _MAX_LINE, _wrap_line
from discopt.modeling.gams_parser import parse_gams

pytestmark = pytest.mark.unit

GAMS_LIMIT = 80000


def _row_values(model, x):
    vals = []
    for c in model._constraints:
        body = np.atleast_1d(compile_expression(c.body, model)(x)).ravel()
        vals.append(-body if c.sense == ">=" else body)
    return np.concatenate(vals)


def _assert_lines_short(text):
    longest = max(len(line) for line in text.split("\n"))
    assert longest <= _MAX_LINE < GAMS_LIMIT, longest
    # A continuation in column 1 starting with `*` would be read as a comment
    # and `$` as dollar control; only the header comment may start with `*`.
    body = text.split("\n")[1:]
    assert not [ln for ln in body if ln.startswith(("*", "$"))]


def test_many_scalar_rows_declaration_is_wrapped():
    # The issue's minimal repro: thousands of scalar constraints. 10,000 names
    # of 9 characters put the old one-line `Equations` declaration at ~110k.
    n = 10_000
    m = dm.Model("many_rows")
    x = m.continuous("x", lb=0.0, ub=1.0)
    ys = [m.continuous(f"y_{i:05d}", lb=0.0, ub=2.0) for i in range(n)]
    for i, y in enumerate(ys):
        m.subject_to(y + x <= 2.5, name=f"c_{i:05d}")
    m.minimize(x)

    text = m.to_gams()
    _assert_lines_short(text)
    # One name per line: each declaration statement now spans many lines.
    assert "\nEquations\n    obj_eq,\n    c_00000,\n" in text
    assert "    c_09999;\n" in text
    assert "\nPositive Variables\n" in text

    r = parse_gams(text)
    assert len(r._constraints) == n
    assert {v.name for v in r._variables} >= {v.name for v in m._variables}


def test_dense_row_is_wrapped_and_value_faithful():
    # One equation with thousands of terms, including products, so the wrapper
    # must break before a `*` without starting a line with it.
    n = 3000
    m = dm.Model("dense_row")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=1.0)
    z = m.continuous("z", lb=0.5, ub=2.0)
    coeffs = np.linspace(-2.0, 3.0, n)
    # Bilinear `over=` terms keep the row flat (`a + b + c ...`); a plain
    # linear sum becomes the left-nested fold covered by the next test.
    m.subject_to(dm.sum(lambda i: coeffs[i] * x[i] * z, over=range(n)) <= 50.0)
    m.subject_to(dm.sum(lambda i: x[i] * x[i], over=range(n)) >= 1.0)
    m.minimize(z)

    text = m.to_gams()
    _assert_lines_short(text)

    r = parse_gams(text)
    names = [v.name for v in r._variables]
    rng = np.random.default_rng(0)
    compared = 0
    for _ in range(3):
        xv = 0.1 + 0.9 * rng.random(n)
        zv = 0.5 + 1.5 * rng.random()
        src = np.concatenate([xv, [zv]])
        dst = np.concatenate([xv if nm == "x" else [zv] for nm in names])
        want, got = _row_values(m, src), _row_values(r, dst)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
        compared += want.size
    assert compared == 6


def test_deep_fold_is_wrapped_and_value_faithful():
    # `dm.sum(list)` scalarizes to a left-deep fold whose text opens with one
    # `(` per term and no whitespace, so the wrapper must also break after `(`,
    # and the reader must take that nesting depth without recursing per level.
    n = 3000
    m = dm.Model("deep_fold")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=1.0)
    m.subject_to(dm.sum([x[i] for i in range(n)]) <= 5.0)
    m.subject_to(dm.sum([(i % 7 - 3) * x[i] for i in range(n)]) >= -40.0)
    m.minimize(x[0])
    text = m.to_gams()
    _assert_lines_short(text)
    assert "(" * 200 in text.replace("\n", "").replace(" ", "")

    r = parse_gams(text)
    # The writer's synthetic `obj_var` is substituted back out (#2002).
    assert [v.name for v in r._variables] == ["x"]
    rng = np.random.default_rng(1)
    compared = 0
    for _ in range(3):
        xv = rng.random(n)
        want, got = _row_values(m, xv), _row_values(r, xv)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
        compared += want.size
    assert compared == 6


def test_reader_takes_deep_parenthesis_nesting():
    # The reader's expression parser used one Python frame per grammar level,
    # so ~170 nested parentheses raised RecursionError. Depth is now bounded by
    # memory, not the recursion limit.
    depth = 20_000
    src = (
        "Variables x, obj;\n"
        "Equations e, o;\n"
        f"e.. {'(' * depth}-x{')' * depth} =l= 1;\n"
        "o.. obj =e= x;\n"
        "Model mm / all /;\n"
        "Solve mm using lp minimizing obj;\n"
    )
    r = parse_gams(src)
    (row,) = [c for c in r._constraints if "e" == c.name]
    assert row.sense == "<="
    # body is `-x - 1`; both variables at 2 so the order of `_variables` is moot.
    assert compile_expression(row.body, r)(np.full(2, 2.0)) == pytest.approx(-3.0)


def test_wrap_line_breaks_only_outside_quotes():
    line = "c1.. " + " + ".join(f"x('{i}', 'a b')" for i in range(200)) + " =l= 1;"
    pieces = _wrap_line(line, width=80)
    assert len(pieces) > 1
    assert all(len(p) <= 80 for p in pieces)
    assert all(p.startswith("    ") for p in pieces[1:])
    # Only whitespace changed, and no quoted label was split across lines.
    assert "".join(pieces).replace(" ", "") == line.replace(" ", "")
    assert sum(p.count("'a b'") for p in pieces) == 200
    assert all(p.count("'") % 2 == 0 for p in pieces)


def test_short_lines_are_untouched():
    m = dm.Model("small")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(x <= 0.5, name="c")
    m.minimize(x)
    text = m.to_gams()
    assert "Equations obj_eq, c;" in text
    assert "c.. (x - 0.5) =l= 0.0;" in text
