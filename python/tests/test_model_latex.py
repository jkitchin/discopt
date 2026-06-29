"""Rich LaTeX / HTML representation of Model objects (standard PSE problem form)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling import latex  # noqa: E402


def _minlp():
    m = dm.Model("demo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.binary("y")
    m.minimize((x - 3) ** 2 + 2 * y)
    m.subject_to(x + 5 * y >= 4)
    return m


def test_to_latex_structure():
    s = _minlp().to_latex()
    assert s.startswith("\\begin{aligned}") and s.endswith("\\end{aligned}")
    assert r"\text{minimize}" in s
    assert r"\text{subject to}" in s
    # power -> superscript, product -> \cdot
    assert "^{2}" in s
    assert r"\cdot" in s
    # un-normalised natural constraint form (not "4 - (...) <= 0")
    assert r"4 \le x + 5 \cdot y" in s
    # variable domains
    assert r"0 \le x \le 10" in s
    assert r"y \in \{0, 1\}" in s


def test_operator_rendering():
    m = dm.Model("ops")
    a = m.continuous("a", lb=1, ub=5)
    b = m.continuous("b", lb=1, ub=5)
    m.maximize(dm.exp(a) / (b + 1))
    m.subject_to(a * b <= 6)
    s = m.to_latex()
    assert r"\text{maximize}" in s
    assert r"\frac{" in s  # division
    assert r"\exp" in s  # function name


def test_integer_and_continuous_domains():
    m = dm.Model("d")
    k = m.integer("k", lb=0, ub=7)
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize(k + z)
    s = m.to_latex()
    assert r"\mathbb{Z}" in s
    assert r"-1 \le z \le 1" in s


def test_maximize_sense():
    m = dm.Model("mx")
    x = m.continuous("x", lb=0, ub=10)
    m.maximize(-((x - 3) ** 2))
    assert r"\text{maximize}" in m.to_latex()


def test_constant_vector_rendering():
    m = dm.Model("vec")
    v = m.continuous("v", shape=(3,), lb=0, ub=5)
    m.minimize(np.array([1.0, 2.0, 3.0]) @ v)
    s = m.to_latex()
    assert "bmatrix" in s  # small constant vector rendered inline


def test_truncation_and_full():
    m = dm.Model("big")
    xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(60)]
    m.minimize(sum(xs))
    for i in range(60):
        m.subject_to(xs[i] >= 0.0)
    truncated = m.to_latex(max_rows=5)
    assert r"\vdots" in truncated
    assert "60 constraints" in truncated
    # variable summary instead of 60 rows
    assert r"\text{continuous}" in truncated
    full = m.to_latex(max_rows=None)
    assert r"\vdots" not in full
    assert full.count(r"\\") > 60  # all constraints + vars rendered


def test_repr_latex_and_html():
    m = _minlp()
    lx = m._repr_latex_()
    assert lx.startswith("$$") and lx.endswith("$$")
    html = m._repr_html_()
    assert "discopt-model" in html
    assert "demo" in html  # model name in header
    assert "2 variables" in html and "1 constraint" in html


def test_never_crashes_on_empty_model():
    m = dm.Model("empty")
    # no objective, no constraints, no vars
    assert isinstance(m.to_latex(), str)
    assert isinstance(m.to_html(), str)
    assert r"\text{find}" in m.to_latex()


def test_expr_to_latex_is_total():
    """The visitor must return a string for every node type, never raising."""
    m = dm.Model("nodes")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    exprs = [x[0] ** 2, abs(x[0]), -x[1], dm.sqrt(x[0]), dm.log(x[1] + 1), x[0] * x[1]]
    for e in exprs:
        out = latex.expr_to_latex(e)
        assert isinstance(out, str) and out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
