"""#1660: quadratic curvature verdicts need a PSD *proof*, not ``lambda_min >= -1e-10``.

``-1e-11 * x**2 + y**2`` has ``lambda_min = -1e-11``. The absolute licence called it
CONVEX, the convex-QP route took it, and the solve certified ``-1e-11`` (bound
``-2e-7``) where the optimum on ``x in [-1e4, 1e4]`` is ``-1e-3``. A curvature
verdict is box-free, so a negative eigenvalue of any size refutes it.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import classify_model
from discopt._relax.convexity.eigenvalue import exact_psd, psd_proved
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.convexity.patterns import (
    is_homogeneous_psd_quadratic,
    quadratic_curvature,
)


def _model():
    m = dm.Model("q")
    x = m.continuous("x", lb=-1e4, ub=1e4)
    y = m.continuous("y", lb=-1, ub=1)
    return m, x, y


def test_witness_is_not_convex_and_solve_certifies_the_true_optimum():
    m, x, y = _model()
    m.minimize(-1e-11 * x**2 + y**2)
    assert classify_model(m)[0] is False
    r = m.solve(time_limit=60)
    assert r.objective == pytest.approx(-1e-3, abs=1e-6)
    assert r.bound <= -1e-3 + 1e-6


@pytest.mark.parametrize(
    "build, expected",
    [
        (lambda x, y: -1e-11 * x**2 + y**2, Curvature.UNKNOWN),
        (lambda x, y: -1e-11 * x * x + y * y, Curvature.UNKNOWN),
        (lambda x, y: -1e-13 * x**2 + y**2, Curvature.UNKNOWN),  # below the old support cut
        (lambda x, y: -1e-11 * x**2, Curvature.CONCAVE),  # was AFFINE
        (lambda x, y: 1e-11 * x**2 - y**2, Curvature.UNKNOWN),
        (lambda x, y: x * x - 2 * x * y + y * y, Curvature.CONVEX),  # singular PSD: exact band
        (lambda x, y: -(x * x) + 2 * x * y - y * y, Curvature.CONCAVE),
        (lambda x, y: 1e-11 * x**2 + y**2, Curvature.CONVEX),
        (lambda x, y: 3 * x + y, None),  # not a quadratic at all, or affine
    ],
)
def test_quadratic_curvature(build, expected):
    m, x, y = _model()
    got = quadratic_curvature(build(x, y), m)
    if expected is None:
        assert got in (None, Curvature.AFFINE)
    else:
        assert got == expected


def test_homogeneous_psd_needs_exact_homogeneity_and_proof():
    m, x, y = _model()
    assert is_homogeneous_psd_quadratic(x**2 + y**2, m)
    assert is_homogeneous_psd_quadratic(x * x - 2 * x * y + y * y, m)
    assert not is_homogeneous_psd_quadratic(-1e-11 * x**2 + y**2, m)
    assert not is_homogeneous_psd_quadratic(x**2 + y**2 + 1e-11 * x, m)
    m2 = dm.Model("v")
    v = m2.continuous("v", shape=(3,), lb=-1, ub=1)
    A = np.array([[1.0, 2.0, 0.0], [0.0, 1e-13, 1.0]])
    assert is_homogeneous_psd_quadratic(dm.sum((A @ v) * (A @ v)), m2)


def test_sqrt_of_nearly_psd_form_is_not_convex():
    m, x, y = _model()
    m.minimize(dm.sqrt(-1e-11 * x**2 + y**2 + 1.0))
    assert classify_model(m)[0] is False


def test_psd_proved_bands():
    n = 0
    for Q, ok in [
        (np.diag([-1e-11, 1.0]), False),
        (np.diag([-1e-300, 1.0]), False),
        (np.array([[1.0, -1.0], [-1.0, 1.0]]), True),
        (np.diag([1e12, 0.0]), True),
        (np.diag([1e-20, 1.0]), True),
        (np.zeros((0, 0)), True),
        (np.array([[np.nan]]), False),
    ]:
        assert psd_proved(Q) is ok, Q
        n += 1
    assert exact_psd(np.diag([-1e-300, 1.0])) is False
    assert n == 7
