"""Sound eigenvalue bounds for interval Hessians.

Given an interval-valued symmetric matrix ``H`` produced by
:mod:`interval_ad`, derive a rigorous lower bound on
``min_{A ∈ H} λ_min(A)`` and a rigorous upper bound on
``max_{A ∈ H} λ_max(A)``. When the lower bound is ≥ 0, every
concrete Hessian in the interval box is positive semidefinite and the
expression is convex on the argument box — the full chain that makes
the convexity certificate sound.

The primary routine uses the interval extension of Gershgorin's
theorem {cite:p}`Gershgorin1931`: every eigenvalue of a matrix lies
in the union of disks centred at the diagonal entries, with radii
equal to the row off-diagonal absolute sums. For an interval matrix,
widening the disks to cover every concrete realisation gives a sound
bound. This is cheap and scales linearly in the Hessian's non-zero
footprint; soundness holds even with loose magnitudes.

Hertz-Rohn vertex enumeration {cite:p}`Hertz1992,Rohn1994` provides a
tighter bound for symmetric interval matrices but costs ``O(2^n)`` and
is not included in this module — room for a follow-up when Gershgorin
proves too loose in practice.

References
----------
Gershgorin (1931), "Über die Abgrenzung der Eigenwerte einer Matrix."
Hertz (1992), "The extreme eigenvalues and stability of real symmetric
  interval matrices."
Rohn (1994), "Positive definiteness and stability of interval
  matrices."
"""

from __future__ import annotations

import numpy as np

from .interval import Interval, _round_down, _round_up

# Unit roundoff for IEEE-754 binary64 round-to-nearest (2**-53).
_UNIT_ROUNDOFF = 2.0**-53


def _row_offdiag_abs_sum_upper(abs_sup: np.ndarray) -> np.ndarray:
    """Sound per-row upper bound on the off-diagonal absolute sums.

    ``abs_sup`` is the entry-wise ``max(|H_lo|, |H_hi|)`` matrix with its
    diagonal already zeroed, so row ``i`` summed gives ``Σ_{j≠i} |H_ij|``.

    The previous implementation accumulated each row in a Python double loop,
    pushing every partial one ULP toward ``+∞`` — O(n²) scalar ``np.nextafter``
    calls and the dominant cost of the whole certificate. This computes the
    row sums with a single vectorised ``np.sum`` and then inflates each by the
    standard Higham recursive-summation error factor

        ``S ≤ Ŝ / (1 − γ_m)``,   ``γ_m = m·u / (1 − m·u)``,   ``m = n − 1``,

    valid because all summands are nonnegative (so ``Σ|x_i| = S``). numpy's
    pairwise summation has an even smaller error than the recursive bound, so
    ``γ_{n−1}`` is a safe over-estimate. A final outward round absorbs the
    division's own roundoff. The result is therefore a rigorous upper bound,
    matching the loop's guarantee at O(n²) vectorised cost instead of O(n²)
    interpreted scalar ops.
    """
    raw = np.asarray(np.sum(abs_sup, axis=1), dtype=np.float64)
    m = abs_sup.shape[1] - 1  # off-diagonal terms per row (diagonal is zeroed)
    if m <= 0:
        return raw
    mu = m * _UNIT_ROUNDOFF
    # mu < 1 for any n < 2**53; guard defensively so the bound stays sound
    # (and finite) even in the absurd-size limit.
    if mu >= 0.5:
        return _round_up(np.full_like(raw, np.inf))
    gamma = mu / (1.0 - mu)
    return _round_up(raw / (1.0 - gamma))


def gershgorin_lambda_min(H: Interval) -> float:
    """Sound lower bound on ``λ_min`` over the interval Hessian ``H``.

    For a symmetric matrix ``A`` each eigenvalue satisfies
    ``λ_k(A) ≥ A_ii − Σ_{j ≠ i} |A_ij|`` for some row ``i``.
    Widening to cover every concrete ``A`` in the interval matrix
    gives

        λ_min ≥ min_i ( inf(H_ii) − Σ_{j ≠ i} max(|H_ij|_lo, |H_ij|_hi) ).

    The subtraction and summation are performed with outward rounding
    (lower endpoint rounded toward ``−∞``) so floating-point roundoff
    never breaks the inequality.

    Returns
    -------
    float
        Lower bound on ``λ_min``. ``-inf`` when any Hessian entry is
        unbounded.
    """
    lo = np.asarray(H.lo, dtype=np.float64)
    hi = np.asarray(H.hi, dtype=np.float64)
    if lo.ndim != 2 or lo.shape[0] != lo.shape[1]:
        raise ValueError(f"Expected square Hessian; got shape {lo.shape}")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return float(-np.inf)

    # |A_ij| supremum over the interval: max(|lo|, |hi|).
    abs_sup = np.maximum(np.abs(lo), np.abs(hi))
    # Remove diagonal contribution so row sums hold Σ_{j ≠ i} |A_ij|.
    np.fill_diagonal(abs_sup, 0.0)

    # Sound per-row upper bound on the off-diagonal sums (vectorised).
    row_sum = _row_offdiag_abs_sum_upper(abs_sup)

    # Diagonal lower bound minus sum — round down.
    diag_lo = np.diag(lo)
    bounds = _round_down(diag_lo - row_sum)
    return float(bounds.min())


def gershgorin_lambda_max(H: Interval) -> float:
    """Sound upper bound on ``λ_max`` over the interval Hessian ``H``."""
    lo = np.asarray(H.lo, dtype=np.float64)
    hi = np.asarray(H.hi, dtype=np.float64)
    if lo.ndim != 2 or lo.shape[0] != lo.shape[1]:
        raise ValueError(f"Expected square Hessian; got shape {lo.shape}")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return float(np.inf)

    abs_sup = np.maximum(np.abs(lo), np.abs(hi))
    np.fill_diagonal(abs_sup, 0.0)

    row_sum = _row_offdiag_abs_sum_upper(abs_sup)

    diag_hi = np.diag(hi)
    bounds = _round_up(diag_hi + row_sum)
    return float(bounds.max())


def psd_2x2_sufficient(H: Interval) -> bool:
    """Sufficient PSD test for a 2×2 interval Hessian.

    Returns ``True`` only when every concrete symmetric matrix in
    ``H`` is provably PSD via Sylvester's criterion: both diagonal
    entries are nonneg, and the worst-case 2×2 determinant is nonneg
    (``H[0,0].lo · H[1,1].lo ≥ max(|H[0,1].lo|, |H[0,1].hi|)²``). This
    is *sufficient* — when it returns ``False`` the matrix may still
    be PSD; the caller should fall through to Gershgorin.

    The off-diagonal magnitude squared is computed with an upward
    round, and the diagonal product with a downward round, so
    floating-point roundoff cannot push a borderline determinant
    incorrectly into the nonneg bucket.
    """
    lo = np.asarray(H.lo, dtype=np.float64)
    hi = np.asarray(H.hi, dtype=np.float64)
    if lo.shape != (2, 2):
        return False
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return False
    if lo[0, 0] < 0.0 or lo[1, 1] < 0.0:
        return False
    off = max(abs(lo[0, 1]), abs(hi[0, 1]), abs(lo[1, 0]), abs(hi[1, 0]))
    prod = float(lo[0, 0]) * float(lo[1, 1])
    sq = float(off) * float(off)
    # Outward rounding is a no-op when the float product is exactly
    # zero (no roundoff to absorb); skipping it there prevents the
    # nextafter shift from manufacturing a spurious negative
    # determinant on the all-zero / rank-deficient corner.
    prod_lo = prod if prod == 0.0 else float(_round_down(np.float64(prod)))
    sq_hi = sq if sq == 0.0 else float(_round_up(np.float64(sq)))
    return bool(prod_lo >= sq_hi)


__all__ = ["gershgorin_lambda_min", "gershgorin_lambda_max", "psd_2x2_sufficient"]
