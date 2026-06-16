"""Adversarial soundness tests for the square-of-affine-in-lifted-vars envelope.

Issue #155 lifts ``(affine-in-lifted-vars)**2`` residuals to a univariate square
envelope instead of distributing them into catastrophic high-degree monomials.
The soundness mandate is that the envelope must *underestimate* ``r**2`` over the
*true* range of the residual ``r``; a wrong lifted-var range would make the dual
bound unsound. These tests construct squares whose residual ranges straddle zero
and sit fully on either side, and check the root McCormick LP bound never exceeds
the true minimum of the objective sampled over the (integer) box.
"""

import itertools
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.milp_relaxation import _extract_affine_square
from discopt._jax.model_utils import flat_variable_bounds


def _true_min_over_box(fn, lo, hi):
    """Brute-force minimum of ``fn(x0, x1)`` over the integer box [lo, hi]^2."""
    best = float("inf")
    for x0, x1 in itertools.product(range(lo, hi + 1), range(lo, hi + 1)):
        best = min(best, fn(x0, x1))
    return best


def _root_bound(model):
    relaxer = MccormickLPRelaxer(model)
    lb, ub = flat_variable_bounds(model)
    res = relaxer.solve_at_node(lb, ub)
    assert res.status == "optimal", f"root LP status {res.status}"
    return res.lower_bound


# (label, build_objective(x0, x1) -> Expression, true_fn(x0, x1) -> float, box)
# Each residual is affine in lifted products (x0*x1 etc.) and its range crosses
# zero, is strictly positive, or is strictly negative depending on the constant.
_CASES = [
    # Mixed-sign residual: r = x0*x1 - 9 ranges over [-9, 16] on [0,4]^2.
    ("mixed_sign", lambda x0, x1: (x0 * x1 - 9) ** 2, lambda a, b: (a * b - 9) ** 2, (0, 4)),
    # Strictly positive residual: r = x0*x1 + 5 ranges over [5, 21] on [0,4]^2.
    ("positive", lambda x0, x1: (x0 * x1 + 5) ** 2, lambda a, b: (a * b + 5) ** 2, (0, 4)),
    # Strictly negative residual: r = -x0*x1 - 2 ranges over [-18, -2] on [0,4]^2.
    (
        "negative",
        lambda x0, x1: (-x0 * x1 - 2) ** 2,
        lambda a, b: (-a * b - 2) ** 2,
        (0, 4),
    ),
    # Affine sum residual: r = 1.5 - x0 + x0*x1 (the nvs16 r1 shape).
    (
        "nvs16_r1_shape",
        lambda x0, x1: (1.5 - x0 * (1 - x1)) ** 2,
        lambda a, b: (1.5 - a * (1 - b)) ** 2,
        (0, 5),
    ),
]


@pytest.mark.correctness
@pytest.mark.parametrize("label, build_obj, true_fn, box", _CASES)
def test_affine_square_bound_is_sound(label, build_obj, true_fn, box):
    lo, hi = box
    m = dm.Model()
    x = m.integer("x", shape=2, lb=lo, ub=hi)
    m.minimize(build_obj(x[0], x[1]))

    # The objective square must be recognized as an affine-square residual.
    assert _extract_affine_square(m._objective.expression, m) is not None, (
        f"[{label}] objective square not recognized as affine-in-lifted"
    )

    bound = _root_bound(m)
    assert bound is not None, f"[{label}] objective dropped — no root bound"

    true_min = _true_min_over_box(true_fn, lo, hi)
    # The soundness invariant: a valid lower bound never exceeds the true optimum.
    assert bound <= true_min + 1e-6, f"[{label}] UNSOUND bound {bound} > true min {true_min}"


@pytest.mark.correctness
def test_pure_quadratic_form_is_not_hijacked():
    """A square of a *linear* residual (no lifted product) is left to the existing
    bilinear/monomial pipeline — the affine-square detector must not fire on it."""
    m = dm.Model()
    x = m.continuous("x", shape=2, lb=0.0, ub=4.0)
    m.minimize((2 * x[0] + 3 * x[1] - 1) ** 2)
    assert _extract_affine_square(m._objective.expression, m) is None


@pytest.mark.correctness
def test_single_product_square_is_not_hijacked():
    """``(x0*x1)**2`` is a single-term square; the detector defers to the
    existing monomial/product handling rather than lifting a one-term residual."""
    m = dm.Model()
    x = m.continuous("x", shape=2, lb=0.0, ub=4.0)
    m.minimize((x[0] * x[1]) ** 2)
    assert _extract_affine_square(m._objective.expression, m) is None
