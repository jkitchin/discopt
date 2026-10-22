"""The convexity classifier's linear context is sparse.

``build_linear_context`` stacked one dense ``n_vars``-wide row per linear
constraint. On GDPlib biofuel lowered by big-M (14998 + 2914 rows, 36804 columns)
that was one ``np.vstack`` of 31.6 s and an 8.75 GB peak, all of it before the
solve's deadline applies: ``Model.solve(time_limit=6)`` ran 81 s, and runs of the
parity baseline were killed for memory. CSR built from the coefficient dicts:
0.59 s, 0.31 GB, the same matrices entry for entry (checked on the 15 lowered
GDP panel models) and the same convexity verdicts (87 models).
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.convexity.linear_context import build_linear_context, extract_affine
from discopt._relax.convexity.rules import classify_model

pytestmark = pytest.mark.smoke


def _model():
    m = dm.Model("lc")
    x = m.continuous("x", lb=0, ub=4, shape=(3,))
    y = m.continuous("y", lb=-1, ub=1)
    m.subject_to(x[0] + 2 * x[1] <= 5)
    m.subject_to(3 * x[2] - y >= 1)
    m.subject_to(x[0] - x[2] == 0.5)
    m.subject_to(x[1] ** 2 <= 3)  # nonlinear: no row
    m.minimize(dm.log(1 + x[0] - x[2]) * -1 + y)
    return m


def test_the_matrices_are_sparse_and_exact():
    ctx = build_linear_context(_model())
    assert sp.issparse(ctx.A_ub) and sp.issparse(ctx.A_eq)
    np.testing.assert_array_equal(ctx.A_ub.toarray(), [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, -3.0, 1.0]])
    np.testing.assert_array_equal(ctx.b_ub, [5.0, -1.0])
    np.testing.assert_array_equal(ctx.A_eq.toarray(), [[1.0, 0.0, -1.0, 0.0]])
    np.testing.assert_array_equal(ctx.b_eq, [0.5])


def test_each_row_matches_extract_affine():
    m = _model()
    ctx = build_linear_context(m)
    rows = [c for c in m._constraints if extract_affine(c.body, m, ctx.n_vars) is not None]
    assert len(rows) == ctx.A_ub.shape[0] + ctx.A_eq.shape[0] == 3


def test_the_linear_rows_still_refine_a_range():
    """``x0 - x2 == 0.5`` makes ``1 + x0 - x2`` exactly 1.5: the box alone says [-3, 5]."""
    m = _model()
    ctx = build_linear_context(m)
    coeffs, const = extract_affine(1 + m._variables[0][0] - m._variables[0][2], m, ctx.n_vars)
    lo, hi = ctx.affine_range(coeffs, const)
    assert math.isclose(lo, 1.5, abs_tol=1e-5) and math.isclose(hi, 1.5, abs_tol=1e-5)


def test_no_linear_rows_is_box_only():
    m = dm.Model("nl")
    x = m.continuous("x", lb=1, ub=2)
    m.subject_to(x**2 <= 3)
    m.minimize(x)
    ctx = build_linear_context(m)
    assert ctx.A_ub.shape == (0, 1) and ctx.A_eq.shape == (0, 1)
    assert ctx.affine_range(np.array([1.0]), 0.0) == (1.0, 2.0)


def test_the_classifier_still_uses_the_rows():
    """``-log(1 + x0 - x2)`` is convex only because the linear rows keep its
    argument positive; the box alone does not."""
    convex, _ = classify_model(_model())
    assert convex
