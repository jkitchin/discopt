"""#2244: the in-tree row check must not accept a non-finite constraint activity.

``_check_constraint_feasibility`` returned True for a NaN activity when the
gradient screen was unavailable: ``NaN > tol`` is False, and the no-gradient
fallback then accepts. +/-inf against an infinite bound produced a NaN violation
and took the same route.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from discopt.solver import _check_constraint_feasibility

pytestmark = pytest.mark.smoke


def _evaluator(value):
    # No ``evaluate_jacobian``: the gradient screen raises and the fallback runs.
    return SimpleNamespace(
        n_constraints=1,
        evaluate_constraints=lambda x: np.array([value], dtype=np.float64),
    )


BOUNDS = [
    pytest.param([-1e20], [0.0], id="sentinel-upper-0"),
    pytest.param([-np.inf], [np.inf], id="free-inf"),
    pytest.param([-1e20], [1e20], id="free-sentinel"),
]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "+inf", "-inf"])
@pytest.mark.parametrize("cl,cu", BOUNDS)
def test_nonfinite_activity_rejected(value, cl, cu):
    assert _check_constraint_feasibility(_evaluator(value), np.array([1.0]), cl, cu) is False


def test_nonfinite_row_among_finite_rows_rejected():
    ev = SimpleNamespace(n_constraints=2, evaluate_constraints=lambda x: np.array([0.0, np.nan]))
    assert _check_constraint_feasibility(ev, np.array([1.0]), [-1e20, -1e20], [0.0, 0.0]) is False


def test_nonfinite_augmented_row_beyond_bounds_is_not_checked():
    # Rows past len(cl) (e.g. appended cuts) are outside this check by design.
    ev = SimpleNamespace(n_constraints=2, evaluate_constraints=lambda x: np.array([0.0, np.nan]))
    assert _check_constraint_feasibility(ev, np.array([1.0]), [-1e20], [0.0]) is True


def test_finite_residual_within_tol_without_gradient_still_accepted():
    # Above the small-row floor, so the gradient screen is attempted and its
    # absolute-test-only fallback is what accepts.
    assert _check_constraint_feasibility(_evaluator(5e-5), np.array([1.0]), [-1e20], [0.0]) is True


def test_finite_residual_beyond_tol_rejected():
    assert _check_constraint_feasibility(_evaluator(1e-2), np.array([1.0]), [-1e20], [0.0]) is False
