"""Theorem-style property tests for every relaxation primitive.

Each relaxation rule is exercised as a theorem (see ``relaxation_harness``):
containment over the box, corner exactness for McCormick-exact primitives,
monotone tightening as the box shrinks, and sound convexity classification.

Run just these with::

    pytest python/tests/test_relaxation_theorems.py -m relaxation
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import pytest
from relaxation_harness import (
    assert_containment,
    assert_convexity,
    assert_corner_exactness,
    assert_monotone_tightening,
)

pytestmark = pytest.mark.relaxation


# ───────────────────────── operator matrix ─────────────────────────
# (id, expr_fn, bounds, corner_exact, curvature)
#   corner_exact : True if cv==cc==f at box vertices (McCormick-exact)
#   curvature    : expected DCP verdict over the box, or "UNKNOWN"
CASES = [
    # ---- arithmetic / bilinear ----
    ("affine", lambda x, y: 2 * x + 3 * y - 1, [(-2.0, 2.0), (-2.0, 2.0)], True, "AFFINE"),
    ("bilinear_pos", lambda x, y: x * y, [(0.5, 5.0), (0.5, 5.0)], True, "UNKNOWN"),
    ("bilinear_mixed", lambda x, y: x * y, [(-3.0, 2.0), (-1.0, 4.0)], True, "UNKNOWN"),
    (
        "trilinear",
        lambda x, y, z: x * y * z,
        [(0.5, 2.0), (0.5, 2.0), (0.5, 2.0)],
        False,
        "UNKNOWN",
    ),
    # ---- division / reciprocal ----
    ("reciprocal_pos", lambda x: 1.0 / x, [(0.3, 4.0)], False, "CONVEX"),
    ("division", lambda x, y: x / y, [(1.0, 5.0), (0.5, 3.0)], False, "UNKNOWN"),
    # ---- powers ----
    ("square", lambda x: x**2, [(-3.0, 2.0)], True, "CONVEX"),
    ("cube_pos", lambda x: x**3, [(0.2, 2.5)], False, "CONVEX"),
    ("pow_frac_concave", lambda x: x**0.4, [(0.2, 5.0)], True, "CONCAVE"),
    ("pow_frac_convex", lambda x: x**1.7, [(0.2, 5.0)], True, "CONVEX"),
    # ---- exp / log ----
    ("exp", lambda x: dm.exp(x), [(-2.0, 2.0)], True, "CONVEX"),
    ("log", lambda x: dm.log(x), [(0.2, 6.0)], True, "CONCAVE"),
    ("log2", lambda x: dm.log2(x), [(0.2, 6.0)], True, "CONCAVE"),
    ("log10", lambda x: dm.log10(x), [(0.2, 6.0)], True, "CONCAVE"),
    ("log1p", lambda x: dm.log1p(x), [(0.0, 6.0)], True, "CONCAVE"),
    ("entropy", lambda x: x * dm.log(x), [(0.2, 4.0)], False, "UNKNOWN"),
    # ---- sqrt / abs ----
    ("sqrt", lambda x: dm.sqrt(x), [(0.1, 9.0)], True, "CONCAVE"),
    ("abs", lambda x: abs(x), [(-3.0, 2.0)], True, "CONVEX"),
    # ---- trig (narrow single-curvature regimes) ----
    ("sin_concave", lambda x: dm.sin(x), [(0.2, 2.8)], False, "UNKNOWN"),
    ("cos_regime", lambda x: dm.cos(x), [(0.2, 2.8)], False, "UNKNOWN"),
    ("tan_regime", lambda x: dm.tan(x), [(-1.2, 1.2)], False, "UNKNOWN"),
    # ---- inverse trig ----
    ("atan", lambda x: dm.atan(x), [(-3.0, 3.0)], False, "UNKNOWN"),
    ("asin", lambda x: dm.asin(x), [(-0.9, 0.9)], False, "UNKNOWN"),
    ("acos", lambda x: dm.acos(x), [(-0.9, 0.9)], False, "UNKNOWN"),
    # ---- hyperbolic ----
    ("sinh", lambda x: dm.sinh(x), [(-2.0, 2.0)], False, "UNKNOWN"),
    ("cosh", lambda x: dm.cosh(x), [(-2.0, 2.0)], True, "CONVEX"),
    ("tanh", lambda x: dm.tanh(x), [(-2.0, 2.0)], False, "UNKNOWN"),
    # ---- sigmoidal / special ----
    ("sigmoid", lambda x: dm.sigmoid(x), [(-3.0, 3.0)], False, "UNKNOWN"),
    ("softplus", lambda x: dm.softplus(x), [(-3.0, 3.0)], True, "CONVEX"),
    ("erf", lambda x: dm.erf(x), [(-2.0, 2.0)], False, "UNKNOWN"),
    # ---- min / max ----
    ("maximum", lambda x, y: dm.maximum(x, y), [(-2.0, 3.0), (-1.0, 2.0)], False, "CONVEX"),
    ("minimum", lambda x, y: dm.minimum(x, y), [(-2.0, 3.0), (-1.0, 2.0)], False, "CONCAVE"),
    # ---- signomial / geometric mean (positive domain) ----
    ("geomean", lambda x, y: x**0.5 * y**0.5, [(0.3, 4.0), (0.3, 4.0)], False, "CONCAVE"),
]

_ID = {c[0]: c for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_containment(case):
    """Theorem 1: cv(x) <= f(x) <= cc(x) at interior, boundary and corner points."""
    name, fn, bounds, _, _ = case
    assert_containment(fn, bounds, label=name)


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c[3]],
    ids=[c[0] for c in CASES if c[3]],
)
def test_corner_exactness(case):
    """Theorem 2: McCormick-exact envelopes are tight at the box vertices."""
    name, fn, bounds, _, _ = case
    assert_corner_exactness(fn, bounds, label=name)


# Tightening is checked on the smooth, single-curvature primitives where the
# gap is expected to vanish monotonically; periodic/wide cases are excluded.
_TIGHTENING_IDS = [
    "bilinear_pos",
    "bilinear_mixed",
    "square",
    "cube_pos",
    "exp",
    "log",
    "sqrt",
    "reciprocal_pos",
    "pow_frac_concave",
    "pow_frac_convex",
    "cosh",
    "softplus",
    "entropy",
]


@pytest.mark.parametrize("case", [_ID[i] for i in _TIGHTENING_IDS], ids=_TIGHTENING_IDS)
def test_monotone_tightening(case):
    """Theorem 3: envelope gap is non-increasing and -> 0 as the box shrinks."""
    name, fn, bounds, _, _ = case
    assert_monotone_tightening(fn, bounds, label=name)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_convexity_classification(case):
    """Theorem 4: DCP detector never returns a false convex/concave verdict."""
    name, fn, bounds, _, curv = case
    assert_convexity(fn, bounds, curv, label=name)
