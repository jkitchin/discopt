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
    psd_2x2_sufficient,
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


# ──────────────────────────────────────────────────────────────────────
# 2×2 sufficient PSD test
# ──────────────────────────────────────────────────────────────────────


class TestPsd2x2Sufficient:
    """``psd_2x2_sufficient`` returns ``True`` only when every concrete
    symmetric matrix in ``H`` is provably PSD by Sylvester's criterion."""

    def test_pure_diagonal_psd(self):
        H = Interval(
            lo=np.array([[1.0, 0.0], [0.0, 2.0]]),
            hi=np.array([[3.0, 0.0], [0.0, 5.0]]),
        )
        assert psd_2x2_sufficient(H) is True

    def test_zero_matrix_is_psd(self):
        H = Interval(lo=np.zeros((2, 2)), hi=np.zeros((2, 2)))
        assert psd_2x2_sufficient(H) is True

    def test_negative_diagonal_lower_bound_rejected(self):
        H = Interval(
            lo=np.array([[-0.1, 0.0], [0.0, 1.0]]),
            hi=np.array([[1.0, 0.0], [0.0, 2.0]]),
        )
        assert psd_2x2_sufficient(H) is False

    def test_indefinite_via_off_diagonal(self):
        # Diagonals are 1, off-diagonal radius is 2 — clearly indefinite.
        H = Interval(
            lo=np.array([[1.0, -2.0], [-2.0, 1.0]]),
            hi=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )
        assert psd_2x2_sufficient(H) is False

    def test_borderline_psd_passes(self):
        # Det = 1 * 1 - 0.5^2 = 0.75 > 0, both diagonals nonneg.
        H = Interval(
            lo=np.array([[1.0, -0.5], [-0.5, 1.0]]),
            hi=np.array([[1.5, 0.5], [0.5, 1.5]]),
        )
        assert psd_2x2_sufficient(H) is True

    def test_unbounded_rejected(self):
        H = Interval(
            lo=np.array([[1.0, -np.inf], [-np.inf, 1.0]]),
            hi=np.array([[2.0, np.inf], [np.inf, 2.0]]),
        )
        assert psd_2x2_sufficient(H) is False

    def test_non_2x2_rejected(self):
        H = Interval(lo=np.eye(3), hi=np.eye(3))
        assert psd_2x2_sufficient(H) is False

    def test_strictly_positive_definite_passes(self):
        # H = v vᵀ + ε I with ε > 0 — strictly PD; the outward-rounded
        # determinant test certifies it. (The exactly rank-1 case
        # ``v vᵀ`` has det = 0 and is certified through the structural
        # ``Rank1Factor`` path in the certificate, not through this
        # sufficient 2×2 numerical test.)
        v = np.array([1.0, 2.0])
        eps = 1.0
        H_mat = np.outer(v, v) + eps * np.eye(2)
        H = Interval(lo=H_mat, hi=H_mat)
        assert psd_2x2_sufficient(H) is True
