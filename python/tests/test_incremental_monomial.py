"""Property tests for the incremental McCormick monomial patch (cert:T1.2).

The generalized ``_monomial_rows`` (any integer power p>=2, sign-definite box)
must reproduce ``build_milp_relaxation`` row-for-row so the incremental node LP is
identical to the cold build — the soundness guarantee behind the ``ok=False``
fallback. These pin (a) exact per-box agreement on 200 random sign-matched boxes
and (b) the coverage boundary (spanning root box -> fallback).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
from discopt._relax.incremental_mccormick import (  # noqa: E402
    IncrementalMcCormickLP,
    _monomial_aux_bounds,
    _monomial_rows,
)
from discopt._relax.term_classifier import classify_nonlinear_terms  # noqa: E402


def _engine(lb, ub, p):
    m = dm.Model()
    x = m.continuous("x", lb=lb, ub=ub)
    m.minimize(x**p + x)  # +x keeps the objective non-degenerate
    return IncrementalMcCormickLP(m, classify_nonlinear_terms(m))


@pytest.mark.parametrize("p", [2, 3, 4, 5])
@pytest.mark.parametrize("regime", ["positive", "negative"])
def test_monomial_patch_matches_cold_build(p, regime):
    """Patched (A, b, bounds) equals the cold builder to 1e-9 on 200 random
    boxes drawn from the variable's admissible sign regime."""
    if regime == "positive":
        eng = _engine(0.0, 20.0, p)
        rng = np.random.default_rng(p)

        def box():
            a, b = np.sort(rng.uniform(0.0, 20.0, size=2))
            return np.array([a]), np.array([max(b, a + 0.5)])
    else:
        eng = _engine(-20.0, 0.0, p)
        rng = np.random.default_rng(100 + p)

        def box():
            a, b = np.sort(rng.uniform(-20.0, 0.0, size=2))
            return np.array([a]), np.array([max(b, a + 0.5)])

    assert eng.ok, f"engine should validate for x**{p} on a sign-definite box"
    mism = 0
    for _ in range(200):
        lb, ub = box()
        ap, bp, bdp = eng._patch(lb, ub)
        af, bf, bdf, _, _, _ = eng._full_build(lb, ub)
        if ap.shape != af.shape:
            mism += 1
            continue
        if eng._rowset(ap, bp) != eng._rowset(af, bf):
            mism += 1
            continue
        if not np.allclose(bdp, bdf, atol=1e-9, rtol=1e-9):
            mism += 1
    assert mism == 0, f"{mism}/200 boxes mismatched for x**{p} ({regime})"


def test_monomial_rows_unit():
    """Direct check of the row/aux-bound generators vs hand values for x**2.

    Coefficients are exact; each rhs and each aux bound carries the #956 outward
    guard, so they are asserted to reproduce the hand value **from the outward
    side only** and by at most an ulp-scaled amount. The direction is the point:
    an rhs that moved INWARD would cut the graph of the function being relaxed.
    """
    rows = _monomial_rows(1.0, 4.0, 2)
    # tangent at 1: (2, -1, 1); tangent at 4: (8, -1, 16); secant: (-5, 1, -4)
    hand = {(2.0, -1.0): 1.0, (8.0, -1.0): 16.0, (-5.0, 1.0): -4.0}
    seen = 0
    for cx, cs, rhs in rows:
        want = hand.get((cx, cs))
        if want is None:
            continue  # the midpoint tangent, not pinned here
        assert rhs >= want, f"rhs {rhs} moved INWARD of {want}"
        assert rhs <= want + 1e-12 + 1e-13 * abs(want), f"rhs {rhs} beyond the guard on {want}"
        seen += 1
    assert seen == 3, f"probe matched {seen}/3 pinned rows"

    for got, want in (
        (_monomial_aux_bounds(1.0, 4.0, 2), (1.0, 16.0)),
        (_monomial_aux_bounds(-4.0, -1.0, 3), (-64.0, -1.0)),
    ):
        assert got[0] <= want[0] and got[0] >= want[0] - 1e-12 - 1e-13 * abs(want[0])
        assert got[1] >= want[1] and got[1] <= want[1] + 1e-12 + 1e-13 * abs(want[1])


def test_spanning_root_box_falls_back():
    """A monomial whose root box strictly spans zero is unmappable -> ok=False
    (the model falls back to the trusted cold builder)."""
    eng = _engine(-3.0, 4.0, 2)
    assert eng.ok is False


def test_cube_negative_is_concave_and_covered():
    eng = _engine(-5.0, -1.0, 3)
    assert eng.ok is True
