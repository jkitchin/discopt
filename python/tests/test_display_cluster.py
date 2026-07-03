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
from discopt.modeling.latex import _fmt_num, expr_to_latex, model_to_latex

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
