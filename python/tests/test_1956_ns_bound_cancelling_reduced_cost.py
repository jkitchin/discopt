"""#1956: the Neumaier-Shcherbina safe bound must be proved, not margin-guarded.

``ns_safe_bound_csc`` rounded ``(Aᵀy)_j`` to ``f64`` before subtracting it from ``c_j``.
A true reduced cost below that rounding error came out exactly ``0``, the column was
skipped and its ``rc_j·bound`` term lost; the ``1e-12·S`` margin does not cover a lost
term times a large bound. On the default pure-LP route this certified ``optimal`` with
a bound above the exact optimum. The bound now encloses every rounding error, so these
tests use zero tolerance against the exact (rational) optimum.
"""

from __future__ import annotations

from fractions import Fraction

import discopt
import numpy as np
import pytest
from discopt import _rust

C1 = 1.1 * 0.7  # binary64 product
DELTA = Fraction(C1) - Fraction(1.1) * Fraction(0.7)  # 4.44e-18 > 0, exact


@pytest.mark.smoke
@pytest.mark.parametrize("R", [1e8, 1e12, 1e14])
def test_default_lp_route_bound_never_exceeds_exact_optimum(R):
    m = discopt.Model("ns_cancel")
    x1 = m.continuous("x1", lb=0, ub=R)
    x2 = m.continuous("x2", lb=-2 * R, ub=2 * R)
    m.subject_to(1.1 * x1 + x2 == 0)
    m.minimize(-C1 * x1 - 0.7 * x2)
    r = m.solve(time_limit=30, threads=1)
    p_star = -DELTA * Fraction(R)
    if r.bound_valid:
        assert Fraction(r.bound) <= p_star, (r.status, r.objective, r.bound, float(p_star))
    if r.gap_certified:
        # A certificate is only acceptable when the incumbent really is within tolerance.
        assert abs(Fraction(r.objective) - p_star) <= Fraction(1e-6), (r.objective, float(p_star))


@pytest.mark.smoke
def test_rust_ns_bound_on_fixed_columns_is_not_above_zero():
    # Both columns fixed at 2^60: g(y) = cᵀx = 0 for every y. Before: 63.999999999935.
    X = 2.0**60
    f = lambda v: np.array(v, dtype=np.float64)  # noqa: E731
    i = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
    g = _rust.ns_safe_bound_csc_py(
        f([1, 1]), f([1, -1]), 2, 2, i([0, 2, 3]), i([0, 1, 0]),
        f([1.0, 2.0**-54, -1.0]), f([0, 64]), f([X, X]), f([X, X]),
    )  # fmt: skip
    assert g is not None and g <= 0.0
    assert g >= -1e-6
