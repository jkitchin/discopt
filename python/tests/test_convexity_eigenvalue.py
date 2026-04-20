"""Soundness tests for interval Gershgorin eigenvalue bounds.

The primary invariant: for every concrete symmetric matrix A inside
the interval matrix H, every eigenvalue of A lies in
``[gershgorin_lambda_min(H), gershgorin_lambda_max(H)]``. Violating
this breaks the convexity certificate.

References
----------
Gershgorin (1931), "Über die Abgrenzung der Eigenwerte einer Matrix."
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax.convexity.eigenvalue import (
    gershgorin_lambda_max,
    gershgorin_lambda_min,
)
from discopt._jax.convexity.interval import Interval


def _sample_symmetric(H: Interval, n_samples: int, seed: int) -> np.ndarray:
    """Sample ``n_samples`` concrete symmetric matrices inside ``H``."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(H.lo)
    hi = np.asarray(H.hi)
    n = lo.shape[0]
    out = np.zeros((n_samples, n, n), dtype=np.float64)
    for s in range(n_samples):
        A = rng.uniform(lo, hi)
        # Symmetrize by mirroring upper triangle.
        A = np.triu(A) + np.triu(A, 1).T
        # Ensure diagonals are inside the interval (after symmetrization
        # they already are, but be explicit).
        out[s] = A
    return out


def _assert_bounds_enclose_spectrum(H: Interval, n_samples: int = 16, seed: int = 0):
    lower = gershgorin_lambda_min(H)
    upper = gershgorin_lambda_max(H)
    for A in _sample_symmetric(H, n_samples, seed=seed):
        eigs = np.linalg.eigvalsh(A)
        assert lower <= eigs.min() + 1e-9, (
            f"Gershgorin lower {lower} exceeds sample λ_min {eigs.min()}"
        )
        assert eigs.max() <= upper + 1e-9, (
            f"sample λ_max {eigs.max()} exceeds Gershgorin upper {upper}"
        )


class TestDiagonalMatrices:
    def test_positive_diagonal(self):
        # diag([2, 3]) with no off-diagonal — eigenvalues equal diag.
        H = Interval(
            lo=np.array([[2.0, 0.0], [0.0, 3.0]]),
            hi=np.array([[2.0, 0.0], [0.0, 3.0]]),
        )
        lo = gershgorin_lambda_min(H)
        hi = gershgorin_lambda_max(H)
        assert lo <= 2.0 and hi >= 3.0

    def test_interval_positive_diagonal(self):
        H = Interval(
            lo=np.array([[1.0, 0.0], [0.0, 2.0]]),
            hi=np.array([[3.0, 0.0], [0.0, 4.0]]),
        )
        assert gershgorin_lambda_min(H) <= 1.0
        assert gershgorin_lambda_max(H) >= 4.0

    def test_negative_diagonal(self):
        H = Interval(
            lo=np.array([[-4.0, 0.0], [0.0, -3.0]]),
            hi=np.array([[-2.0, 0.0], [0.0, -1.0]]),
        )
        lo = gershgorin_lambda_min(H)
        assert lo <= -4.0


class TestDenseIntervalMatrices:
    def test_small_2x2(self):
        H = Interval(
            lo=np.array([[3.0, -0.5], [-0.5, 2.0]]),
            hi=np.array([[4.0, 0.5], [0.5, 3.0]]),
        )
        _assert_bounds_enclose_spectrum(H)

    def test_3x3_positive_definite(self):
        H = Interval(
            lo=np.array([[5.0, -0.3, -0.2], [-0.3, 4.0, -0.1], [-0.2, -0.1, 3.0]]),
            hi=np.array([[6.0, 0.3, 0.2], [0.3, 5.0, 0.1], [0.2, 0.1, 4.0]]),
        )
        _assert_bounds_enclose_spectrum(H)
        # Diagonals are at least 3 and off-diag radii ≤ 0.5; Gershgorin
        # gives a positive lower bound.
        assert gershgorin_lambda_min(H) > 0.0

    def test_indefinite_2x2(self):
        """Saddle-like: diagonals opposite sign → Gershgorin proves indefinite."""
        H = Interval(
            lo=np.array([[1.0, -0.1], [-0.1, -2.0]]),
            hi=np.array([[2.0, 0.1], [0.1, -1.0]]),
        )
        _assert_bounds_enclose_spectrum(H)
        # Lower bound must be ≤ -1 (concave eigenvalue exists).
        assert gershgorin_lambda_min(H) <= -1.0

    def test_wide_off_diagonal_kills_psd(self):
        """Large off-diagonal radii flip the verdict to indefinite."""
        H = Interval(
            lo=np.array([[2.0, -5.0], [-5.0, 2.0]]),
            hi=np.array([[3.0, 5.0], [5.0, 3.0]]),
        )
        _assert_bounds_enclose_spectrum(H)
        assert gershgorin_lambda_min(H) < 0.0


class TestUnboundedEntries:
    def test_infinite_entry_gives_infinite_bound(self):
        """One unbounded entry forces the eigenvalue bound to be non-finite."""
        lo = np.array([[1.0, -np.inf], [-np.inf, 1.0]])
        hi = np.array([[2.0, np.inf], [np.inf, 2.0]])
        H = Interval(lo=lo, hi=hi)
        assert gershgorin_lambda_min(H) == -np.inf
        assert gershgorin_lambda_max(H) == np.inf


class TestErrorPaths:
    def test_non_square_rejected(self):
        H = Interval(lo=np.zeros((2, 3)), hi=np.zeros((2, 3)))
        with pytest.raises(ValueError):
            gershgorin_lambda_min(H)


class TestRandomSoundness:
    """Fuzz the bounds against a corpus of random symmetric intervals."""

    @pytest.mark.parametrize("seed", range(6))
    def test_random_interval_matrices(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 6)
        centre = rng.uniform(-3.0, 3.0, size=(n, n))
        centre = 0.5 * (centre + centre.T)
        radius = rng.uniform(0.0, 0.5, size=(n, n))
        radius = 0.5 * (radius + radius.T)
        H = Interval(lo=centre - radius, hi=centre + radius)
        _assert_bounds_enclose_spectrum(H, n_samples=24, seed=seed + 100)
