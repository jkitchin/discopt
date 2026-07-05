"""Regression tests for the display-cluster fixes MO1 and L1/L5/L6/L8.

- **MO1** (`mo/nbi.py`): the NBI quasi-normal was computed with the wrong matrix
  axis (`phi.sum(axis=1)` instead of `axis=0`), deviating from Das-Dennis for
  k >= 3 (invisible at k = 2 by symmetry).
- **L1** (`modeling/latex.py`): `to_latex` / Jupyter `_repr_` crashed with
  `AttributeError` on any model containing a GDP constraint (`if_then`/`either_or`/
  `logical`), because the renderer read `.body`/`.sense` on constraint types that
  carry neither.
- **L5**: an indexed variable whose name ends in a digit rendered an invalid double
  subscript (`y_{1}_{0}` -> MathJax error).
- **L6**: `_fmt_num(inf)` raised `OverflowError` via `int(inf)`.
- **L8**: `Expression._repr_latex_` wrapped the plain repr in `$...$` instead of
  rendering the DAG as math.

Each fails on the pre-fix code.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo.nbi import _quasi_normal
from discopt.modeling.core import Constant, MatMulExpression
from discopt.modeling.latex import (
    _escape_html,
    _fmt_num,
    _latex_text,
    expr_to_latex,
    model_to_html,
    model_to_latex,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------- MO1
def test_mo1_quasi_normal_sums_anchor_vectors_axis0():
    """n̂ = -Φe is the negated sum of the anchor rows (axis=0), not axis=1."""
    # Asymmetric k=3 payoff (phi[i, j] = objective j at anchor i).
    phi = np.array([[0.0, 0.7, 0.9], [0.6, 0.0, 0.3], [0.8, 0.2, 0.0]])
    expected = -(phi[0] + phi[1] + phi[2])  # negated sum of anchor payoff vectors
    np.testing.assert_allclose(_quasi_normal(phi), expected)
    # The buggy axis=1 direction must be different for this asymmetric payoff.
    assert not np.allclose(_quasi_normal(phi), -phi.sum(axis=1))


def test_mo1_symmetric_k2_is_unchanged():
    """At k=2 the ideal/nadir-normalized payoff is symmetric, so both axes agree."""
    phi = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(_quasi_normal(phi), -phi.sum(axis=1))


# ----------------------------------------------------------------------------- L6
def test_l6_fmt_num_handles_non_finite():
    assert _fmt_num(float("inf")) == r"\infty"
    assert _fmt_num(float("-inf")) == r"-\infty"
    assert _fmt_num(float("nan")) == r"\mathrm{nan}"
    # sanity: finite values unchanged
    assert _fmt_num(3.0) == "3"


def test_l6_unbounded_variable_model_renders():
    m = dm.Model("unbounded")
    w = m.continuous("w")  # +/- inf bounds
    m.minimize(w)
    # Must not raise (previously _fmt_num(inf) could crash the domain row).
    assert isinstance(model_to_latex(m), str)


# ----------------------------------------------------------------------------- L5
def test_l5_indexed_digit_name_single_subscript():
    m = dm.Model("l5")
    y1 = m.continuous("y1", shape=(3,), lb=0)
    rendered = expr_to_latex(y1[0])
    assert rendered == "y_{1,0}"
    assert "}_{" not in rendered  # no invalid double subscript


def test_l5_plain_indexed_variable():
    m = dm.Model("l5b")
    x = m.continuous("x", shape=(3,), lb=0)
    assert expr_to_latex(x[0]) == "x_{0}"


# ----------------------------------------------------------------------------- L8
def test_l8_expression_repr_latex_renders_math():
    m = dm.Model("l8")
    x = m.continuous("x", shape=(2,), lb=0)
    out = (x[0] + 2 * x[1])._repr_latex_()
    assert out.startswith("$") and out.endswith("$")
    assert "x_{0}" in out and "x_{1}" in out  # real math, not the plain repr


# ----------------------------------------------------------------------------- L1
def test_l1_gdp_model_renders_without_crashing():
    m = dm.Model("gdp")
    z = m.continuous("z", lb=0, ub=10)
    m.minimize(z)
    m.subject_to(z >= 1, name="c1")
    m.either_or([[z == 0], [z >= 5]], name="switch")
    tex = model_to_latex(m)  # previously raised AttributeError
    assert isinstance(tex, str)
    assert r"\text{[" in tex  # the GDP constraint rendered as a placeholder


def test_l1_indicator_constraint_renders_without_crashing():
    m = dm.Model("ind")
    a = m.binary("a")
    z = m.continuous("z", lb=0, ub=10)
    m.minimize(z)
    m.if_then(a, [z >= 5], name="impl")
    assert isinstance(model_to_latex(m), str)


# ----------------------------------------------------------------------------- L2
def test_l2_fast_api_constraints_render():
    """Fast-path constraints live only in the Rust builder; the renderer must route
    through the X-1 primitive so they show up (previously "0 constraints")."""
    m = dm.Model("fast")
    plants = m.set("plants", ["pitt", "sf"])
    x = m.continuous("x", over=plants, lb=0, ub=10)
    m.minimize(dm.sum(x[p] for p in plants))
    m.constraint(plants, lambda p: x[p] <= 1, name="cap", fast=True)
    # Precondition: these rows are NOT on the Python constraint list.
    assert len(m._constraints) == 0
    tex = model_to_latex(m)
    # Both fast-path rows must appear as real constraints (pre-fix: neither did).
    assert r"\text{subject to}" in tex
    assert r"x_{0} \le 1" in tex
    assert r"x_{1} \le 1" in tex
    # And the HTML header count must include them (pre-fix: "0 constraints").
    html = model_to_html(m)
    assert "2 constraints" in html
    assert isinstance(m._repr_latex_(), str)


def test_l2_fast_api_only_model_repr_does_not_crash():
    m = dm.Model("fastonly")
    s = m.set("s", [0, 1])
    x = m.continuous("x", over=s, lb=0, ub=5)
    m.constraint(s, lambda i: 2 * x[i] <= 3, name="c", fast=True)
    assert isinstance(model_to_latex(m), str)
    assert isinstance(model_to_html(m), str)


# ----------------------------------------------------------------------------- L3
def test_l3_sum_over_expression_renders_as_sum():
    m = dm.Model("l3sum")
    s = m.set("s", [0, 1, 2])
    y = m.continuous("y", over=s, lb=0)
    tex = expr_to_latex(dm.sum(y[i] for i in s))
    assert r"\sum" in tex  # a real summation, not the raw `Σ[3 terms]` repr
    assert "Σ" not in tex
    assert "y_{0}" in tex and "y_{2}" in tex


def test_l3_parameter_renders_as_escaped_symbol():
    m = dm.Model("l3param")
    price = m.parameter("price_A", value=50.0)
    tex = expr_to_latex(price)
    assert tex == r"price\_A"  # escaped underscore, not the `param(price_A)` repr
    assert "param(" not in tex


# ----------------------------------------------------------------------------- L4
def test_l4_negation_parenthesised_under_power():
    m = dm.Model("l4neg")
    x = m.continuous("x", lb=-5, ub=5)
    assert expr_to_latex((-x) ** 2) == r"\left(-x\right)^{2}"


def test_l4_sum_parenthesised_under_power():
    m = dm.Model("l4sum")
    s = m.set("s", [0, 1])
    y = m.continuous("y", over=s, lb=0)
    tex = expr_to_latex(dm.sum(y[i] for i in s) ** 2)
    assert tex.startswith(r"\left(") and tex.endswith(r"\right)^{2}")


def test_l4_matmul_parenthesised_under_power():
    m = dm.Model("l4mm")
    s = m.set("s", [0, 1, 2])
    v = m.continuous("v", over=s, lb=0)
    A = Constant(np.array([[1.0, 2.0, 3.0]]))
    tex = expr_to_latex(MatMulExpression(A, v) ** 2)
    assert tex.startswith(r"\left(") and tex.endswith(r"\right)^{2}")


# ----------------------------------------------------------------------------- L7
def test_l7_latex_text_escapes_math_specials_no_html_entities():
    out = _latex_text("a & b_c%d#e")
    # LaTeX specials escaped, wrapped in \text{}, and NO HTML entity injected.
    assert out == r"\text{a \& b\_c\%d\#e}"
    assert "&amp;" not in out


def test_l7_escape_html_is_html_only():
    assert _escape_html("x<y&z") == "x&lt;y&amp;z"


def test_l7_unknown_node_fallback_stays_math_safe():
    # A stray non-expression object falls through to the escaped \text{} fallback.
    tex = expr_to_latex("raw_string & <tag>")
    assert tex.startswith(r"\text{")
    assert "&amp;" not in tex  # must not inject HTML entities into math mode
    assert r"\_" in tex  # underscore LaTeX-escaped
