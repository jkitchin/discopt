"""#1955: the native convex kernel's linear-row FBBT must round outward.

``ConvexKernelSpec::propagate_fbbt`` summed the other terms of a row in nearest
rounding. On ``x0 + 1e16*x1 - x2 - 1e16*x3 <= 0`` with ``x1 = x2 = x3 = 1`` the sum
``1e16 - 1`` ties to ``1e16``, so FBBT set ``hi[x0] = 0`` although ``x0 = 1`` gives
an exactly zero residual. The kernel then certified the optimum with ``x0`` fixed
at 0, a bound 1 above the true optimum. The Rust unit tests in
``convex_kernel.rs`` pin the arithmetic; these pin the kernel tree's own bound and
the end-to-end certificate on the option set that used to route the model to the
kernel. With outward rounding the kernel's node LP cannot certify this 1e16 row
(it ends ``exhausted`` with no incumbent), so ``try_convex_solve`` declines and the
default path certifies the true optimum.
"""

from __future__ import annotations

import math
import warnings

import discopt.modeling as dm
import pytest

EXACT_OPT = -1.0 - math.log(2.5) + 0.1


def _build() -> dm.Model:
    m = dm.Model("rowprop_1955")
    x0 = m.continuous("x0", lb=0, ub=1)
    x1 = m.continuous("x1", lb=1, ub=2)
    x2 = m.continuous("x2", lb=0, ub=1)
    x3 = m.continuous("x3", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    b = m.binary("b")
    m.subject_to(x0 + 1e16 * x1 - x2 - 1e16 * x3 <= 0)
    m.subject_to(dm.exp(z) <= 1.5 + b)
    m.minimize(-x0 - z + 0.1 * b)
    return m


@pytest.mark.smoke
def test_convex_kernel_tree_bound_stays_below_exact_optimum():
    from discopt.solvers._convex_kernel import build_convex_spec, solve_convex_tree

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # deliberate 1e16 row scaling
        spec = build_convex_spec(_build())
    assert spec is not None, "model must stay kernel-eligible for this test to bite"
    out = solve_convex_tree(spec, time_limit_s=20)
    # Before #1955 the tree returned bound -0.8163 with x0 fixed at 0.
    assert out["bound"] <= EXACT_OPT + 1e-6, out


@pytest.mark.smoke
def test_solve_certifies_exactly_feasible_optimum():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # deliberate 1e16 row scaling
        r = _build().solve(time_limit=30, threads=1, deterministic=True)
    # The bound must never sit above the exact optimum (certificate invariant).
    assert r.bound <= EXACT_OPT + 1e-6, (r.status, r.objective, r.bound)
    if r.status == "optimal":
        assert r.objective == pytest.approx(EXACT_OPT, abs=1e-6)
