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

from .interval import (
    Interval,
    _round_down,
    _round_down_exact0,
    _round_up,
    _round_up_exact0,
)

# Unit roundoff for IEEE-754 binary64 round-to-nearest (2**-53).
_UNIT_ROUNDOFF = 2.0**-53

#: Multiplier on ``u·‖H‖`` for a semidefiniteness *decision slack* (#1397).
#:
#: Every semidefiniteness verdict in this package compares a computed eigenvalue
#: (or a rigorous bound on one) against zero, and needs some slack: a genuinely
#: singular PSD Hessian — a quadratic in a subset of the model's variables, which
#: is the ordinary case — has an exact zero eigenvalue that arithmetic renders as
#: a small negative. The slack cannot be an absolute constant, because the error
#: it absorbs is O(u·‖H‖): 32 puts it just above the measured worst case for
#: ``eigvalsh`` (5.42·u·‖H‖ over 1800 trials, see ``solver._hessian_is_psd_with_
#: margin``) while staying far below any tolerance in the solver.
_PSD_DECISION_K = 32.0


def psd_decision_slack(magnitude: float) -> float:
    """Slack for a ``λ ≥ 0`` test on a matrix of magnitude ``magnitude`` (#1397).

    Returns ``_PSD_DECISION_K · u · magnitude``: the arithmetic's own error at
    that magnitude, and nothing else. An absolute constant in this position is
    dimensionally incoherent — an eigenvalue carries the units of the matrix — and
    fails in both directions at once. Measured over a sweep in ``‖Q‖_F`` (2000
    verdicts, ``n ≤ 32``) against the previous absolute ``1e-10``:

    * 149 genuinely PSD matrices with an *exact* zero eigenvalue were refused
      once ``‖Q‖_F ≳ 1e7`` (at ``‖Q‖_F = 9.3e11`` the zero computes as
      ``-2.7e-05``), silently costing the convexity certificate;
    * 240 *indefinite* matrices were certified convex, the worst admitting a
      **relative** nonconvexity of ``1.97e-12`` — 8900·u — because an absolute
      ``1e-10`` is enormous next to a small ``‖Q‖``. This slack caps that
      admission at ``2·K·u ≈ 1.4e-14``.

    ``magnitude`` is a Frobenius norm at every call site, which bounds the
    spectral norm above: erring high only widens the slack for a matrix whose
    entries are genuinely large, and the bound on admitted nonconvexity stays
    relative.

    A non-finite or negative ``magnitude`` yields ``0.0`` — no slack, so the
    caller's test reduces to the exact ``λ ≥ 0`` comparison rather than being
    handed an infinite licence.
    """
    mag = float(magnitude)
    if not np.isfinite(mag) or mag <= 0.0:
        return 0.0
    return _PSD_DECISION_K * _UNIT_ROUNDOFF * mag


def interval_magnitude(H: Interval) -> float:
    """Frobenius norm of the entry-wise absolute supremum of an interval matrix.

    The magnitude to hand :func:`psd_decision_slack` for a verdict taken on an
    interval Hessian: ``max(|H_lo|, |H_hi|)`` dominates ``|A|`` entry-wise for
    every concrete ``A ∈ H``, so this bounds ``‖A‖_2`` above for all of them.
    Returns ``inf`` when any entry is unbounded, which yields no usable slack and
    leaves the caller's own non-finite guard to abstain.
    """
    lo = np.asarray(H.lo, dtype=np.float64)
    hi = np.asarray(H.hi, dtype=np.float64)
    return float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi)), "fro"))


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
    # ``_exact0`` (#957): ``raw`` is a sum of absolute values, so it is >= 0 and
    # the divisor is just under 1 — the quotient is >= ``raw`` and cannot
    # underflow to a false zero. A row with no off-diagonal entries therefore
    # keeps a row sum of exactly 0 instead of ``5e-324``, which would otherwise
    # drag the Gershgorin bounds below (above) their true values by one
    # subnormal and make a diagonal Hessian miss a boundary ``λ_min >= 0`` test.
    return _round_up_exact0(raw / (1.0 - gamma))


def gershgorin_row_lower_bounds(H: Interval) -> np.ndarray:
    """Sound **per-row** Gershgorin lower bounds on ``λ_min`` over ``H``.

    For a symmetric matrix ``A`` each eigenvalue satisfies
    ``λ_k(A) ≥ A_ii − Σ_{j ≠ i} |A_ij|`` for some row ``i``. Widening to cover
    every concrete ``A`` in the interval matrix gives, row by row,

        b_i = inf(H_ii) − Σ_{j ≠ i} max(|H_ij|_lo, |H_ij|_hi),

    and ``λ_min ≥ min_i b_i``. The summation is inflated by the Higham
    recursive-summation factor and the subtraction is rounded toward ``−∞``, so
    floating-point roundoff never breaks either inequality.

    The per-row bounds are what αBB needs: ``α_i = max(0, −b_i/2)`` perturbs only
    as much as each variable's own row requires, which is tighter than applying
    the global minimum to every variable. Exposed (#1397) because
    ``_alphabb_rigorous.rigorous_alpha`` had reimplemented this formula with a
    plain round-to-nearest ``np.sum`` and no outward rounding — so its "rigorous"
    bound could sit *above* the true one, leaving ``α`` below ``−λ_min/2``, the
    αBB body nonconvex, and the resulting node bound above the true minimum: a
    false dual bound. Measured by calling ``rigorous_alpha`` itself and grading
    its output against this formula in exact rational arithmetic over the same
    float entries: 513 of 1120 rows had ``α`` provably below ``−λ_min/2``, worst
    shortfall 5.76e-4, at every scale from ``‖A‖_F = 1e0`` to 1e12 (the error is
    O(u·‖A‖), so it also grows with the problem's scale without limit).

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` lower bounds. A row with an unbounded entry yields
        ``-inf``, which is the sound bound for it and makes the corresponding
        ``α`` infinite.
    """
    lo = np.asarray(H.lo, dtype=np.float64)
    hi = np.asarray(H.hi, dtype=np.float64)
    if lo.ndim != 2 or lo.shape[0] != lo.shape[1]:
        raise ValueError(f"Expected square Hessian; got shape {lo.shape}")

    # |A_ij| supremum over the interval: max(|lo|, |hi|).
    abs_sup = np.maximum(np.abs(lo), np.abs(hi))
    # Remove diagonal contribution so row sums hold Σ_{j ≠ i} |A_ij|.
    np.fill_diagonal(abs_sup, 0.0)

    # Sound per-row upper bound on the off-diagonal sums (vectorised).
    row_sum = _row_offdiag_abs_sum_upper(abs_sup)

    diag_lo = np.diag(lo)
    with np.errstate(invalid="ignore"):
        raw = diag_lo - row_sum
    # ``(+inf) − (+inf)`` is the only NaN this subtraction can produce; ``-inf``
    # is the sound lower bound for such a row. Mapped before rounding so the
    # outward round never sees a NaN.
    raw = np.where(np.isnan(raw), -np.inf, raw)
    # ``_exact0`` (#957): IEEE-754 subtraction of two doubles is zero only when
    # the exact difference is zero, so a Gershgorin disc that lands exactly on
    # the origin stays there instead of becoming ``-5e-324`` — the difference
    # between certifying and declining a boundary-PSD Hessian.
    return np.asarray(_round_down_exact0(raw), dtype=np.float64)


def gershgorin_lambda_min(H: Interval) -> float:
    """Sound lower bound on ``λ_min`` over the interval Hessian ``H``.

    The minimum of :func:`gershgorin_row_lower_bounds`; see there for the
    derivation and the rounding discipline.

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

    return float(gershgorin_row_lower_bounds(H).min())


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
    bounds = _round_up_exact0(diag_hi + row_sum)  # ``_exact0``: see λ_min above
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
    # Outward rounding is skipped when the exact result is zero, so the
    # nextafter shift cannot manufacture a spurious negative determinant on the
    # all-zero / rank-deficient corner. The test is on the *factors*, not on the
    # float result (#957): a product of two nonzero factors can flush to zero by
    # underflow while the exact value is nonzero, and ``sq``'s round is the
    # upward one, where treating an underflowed ``off**2 ~ 1e-400`` as an exact
    # zero would understate the off-diagonal term and let a matrix with a
    # (barely) negative determinant pass. ``prod`` needs no such care in either
    # direction — both diagonal entries are nonneg by the guard above, so ``0``
    # is a valid lower bound whether the product is exactly or only nearly zero
    # — but it is written the same way so the two lines cannot drift apart.
    prod_lo = 0.0 if (lo[0, 0] == 0.0 or lo[1, 1] == 0.0) else float(_round_down(np.float64(prod)))
    sq_hi = 0.0 if off == 0.0 else float(_round_up(np.float64(sq)))
    return bool(prod_lo >= sq_hi)


__all__ = [
    "gershgorin_lambda_min",
    "gershgorin_lambda_max",
    "gershgorin_row_lower_bounds",
    "psd_2x2_sufficient",
    "psd_decision_slack",
    "interval_magnitude",
]
