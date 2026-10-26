"""The convex kernel's linear/nonlinear row split must rest on a proof, not a sample.

``_build`` called a row linear when its Jacobian at two random points agreed to
``np.isclose(..., atol=1e-9)`` -- numpy's default ``rtol=1e-5`` included. On
``1e6*x - x**2 + z <= 5`` (x binary) the two samples differ by ~2 in 1e6, so the row
was replaced by one tangent, skipped the convexity check, and the kernel certified
5.2874 on a model whose optimum is 5 + ln 2 (found by the #1952-#1958 sweep).
"""

from __future__ import annotations

import math

import discopt
import discopt.modeling as dm
import pytest
from discopt.solvers import _convex_kernel


def _model():
    m = discopt.Model("near_linear_row")
    x = m.integer("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=10)
    w = m.continuous("w", lb=-5, ub=5)
    m.subject_to(1e6 * x - x**2 + z <= 5)
    m.subject_to(dm.exp(w) <= 2)
    m.maximize(z + w)
    return m


@pytest.mark.smoke
def test_near_linear_nonconvex_row_is_not_certified_wrong():
    r = _model().solve(time_limit=30, threads=1)
    opt = 5 + math.log(2)
    assert r.bound >= opt - 1e-7, (r.status, r.objective, r.bound, r.algorithm_route)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=1e-6)


@pytest.mark.smoke
def test_kernel_refuses_the_unproven_row():
    # The row is concave in x after moving everything left: not a convex <= row.
    m = _model()
    assert _convex_kernel.build_convex_spec(m) is None


@pytest.mark.smoke
def test_proven_linear_rows_still_take_the_kernel():
    m = discopt.Model("convex_minlp")
    x = m.continuous("x", lb=0, ub=4)
    y = m.binary("y")
    m.subject_to(dm.exp(x) <= 20 * y + 1)
    m.subject_to(x + y <= 4)
    m.minimize(-3 * x + 4 * y)
    obj_lin, rows = _convex_kernel._proven_linear(m)
    assert obj_lin and list(rows) == [False, True]
    assert _convex_kernel.build_convex_spec(m) is not None
