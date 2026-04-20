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

    n = lo.shape[0]
    # |A_ij| supremum over the interval: max(|lo|, |hi|).
    abs_sup = np.maximum(np.abs(lo), np.abs(hi))
    # Remove diagonal contribution so row_sums holds Σ_{j ≠ i} |A_ij|.
    abs_sup[np.arange(n), np.arange(n)] = 0.0

    # Sum with outward rounding. ``np.sum`` is not directed-rounded,
    # so we accumulate pair-wise and push each partial toward +∞.
    row_sum = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s = _round_up(np.float64(s + abs_sup[i, j]))
        row_sum[i] = s

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

    n = lo.shape[0]
    abs_sup = np.maximum(np.abs(lo), np.abs(hi))
    abs_sup[np.arange(n), np.arange(n)] = 0.0

    row_sum = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s = _round_up(np.float64(s + abs_sup[i, j]))
        row_sum[i] = s

    diag_hi = np.diag(hi)
    bounds = _round_up(diag_hi + row_sum)
    return float(bounds.max())


__all__ = ["gershgorin_lambda_min", "gershgorin_lambda_max"]
