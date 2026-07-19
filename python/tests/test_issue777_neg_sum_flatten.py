"""Issue #777 — `_flatten_sum` must distribute `neg` over its operand.

Before the fix, the additive flattening used by the nonlinear bound-tightening
rules treated a ``neg(...)`` node as one opaque leaf, so the sum-of-squares
hidden under a negation was invisible to the quadratic/interval bounding rules
and their (sound) variable bounds were never derived. ``neg(a + b) == -a - b``
is an exact identity, so pushing the sign inward is a pure, sound normalization.

These tests fail on the pre-fix flattening (the neg leaf is not decomposed, so
no rule fires and the box is unchanged) and pass once negation is distributed.
Both directions of soundness are checked: the derived box is a subset of the
input box (nothing looser) and it retains sampled feasible points (nothing
feasible is cut).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.nonlinear_bound_tightening import (
    _flatten_sum,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


def test_flatten_sum_distributes_neg_over_sum():
    """neg(x**2 + y**2) flattens to two squares with negative scale, not one leaf."""
    m = Model("flat")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(x + y)

    expr = -(x**2 + y**2)  # UnaryOp("neg", x**2 + y**2)
    terms: list[tuple[float, object]] = []
    _flatten_sum(expr, 1.0, terms)

    # The opaque neg leaf must have been decomposed into its two square terms;
    # pre-fix this list had length 1 (the whole neg node).
    assert len(terms) == 2
    for scale, _term in terms:
        assert scale == pytest.approx(-1.0)


def test_neg_sum_of_squares_bounds_are_derived():
    """-(9 - x^2 - y^2) <= 0  is  x^2 + y^2 <= 9; derive |x|,|y| <= 3.

    Pre-fix the neg(9 - x^2 - y^2) term is opaque, no rule matches, and the box
    stays at its declared [-100, 100]. With negation distributed the squares are
    exposed and SumOfSquaresUpperBoundRule tightens both variables to [-3, 3].
    """
    m = Model("sos_neg")
    x = m.continuous("x", lb=-100.0, ub=100.0)
    y = m.continuous("y", lb=-100.0, ub=100.0)
    m.subject_to(-(9.0 - x**2 - y**2) <= 0.0)
    m.minimize(x + y)

    lb0, ub0 = flat_variable_bounds(m)
    tlb, tub, stats = tighten_nonlinear_bounds(m, lb0.copy(), ub0.copy())

    assert not stats.infeasible
    assert "sum_of_squares_upper_bound" in stats.applied_rules

    # subset (nothing looser than the input box) — soundness direction (a)
    assert np.all(tlb >= lb0 - 1e-12)
    assert np.all(tub <= ub0 + 1e-12)

    # the derived radius-3 ball bounds
    assert tub[0] == pytest.approx(3.0)
    assert tub[1] == pytest.approx(3.0)
    assert tlb[0] == pytest.approx(-3.0)
    assert tlb[1] == pytest.approx(-3.0)

    # no feasible point cut — soundness direction (b): every point on the
    # constraint boundary x^2 + y^2 == 9 must stay inside the derived box.
    for theta in np.linspace(0.0, 2.0 * np.pi, 37):
        px, py = 3.0 * np.cos(theta), 3.0 * np.sin(theta)
        assert tlb[0] - 1e-9 <= px <= tub[0] + 1e-9
        assert tlb[1] - 1e-9 <= py <= tub[1] + 1e-9
