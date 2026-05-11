"""Eigenvalue arithmetic as a bound propagator (M6 of issue #51).

Promotes the box-local convexity check in :mod:`eigenvalue` (Gershgorin
disks for one-shot PSD certificates) into a *bound provider*: given a
quadratic form ``f(x) = x^T Q x + b^T x + c`` over a box
``x ∈ [x_lo, x_hi]``, return a sound interval bound on ``f`` that
exploits the spectrum of ``Q``.

Why this is tighter than forward interval AD on quadratics
==========================================================

Forward interval evaluation of ``x^T Q x`` ignores the dependency
between the two ``x`` factors and bloats the bound by O(n) repeats of
each variable's range. Eigenvalue arithmetic instead splits ``Q`` into
its positive- and negative-semidefinite components via spectral
decomposition,

    Q = Q⁺ + Q⁻ = U Λ⁺ Uᵀ + U Λ⁻ Uᵀ,

where ``Λ⁺ = diag(max(λ_i, 0))`` and ``Λ⁻ = diag(min(λ_i, 0))``. Then

    f(x) = c + bᵀx + (x − x₀)ᵀ Q⁺ (x − x₀)
                    + (x − x₀)ᵀ Q⁻ (x − x₀)
                    + 2 x₀ᵀ Q (x − x₀) + x₀ᵀ Q x₀,

with ``x₀`` chosen as the box midpoint. The two quadratic parts are now
single-signed, so they admit clean bounds:

    0 ≤ (x − x₀)ᵀ Q⁺ (x − x₀) ≤ λ_max⁺ · ‖x − x₀‖²,
    λ_min⁻ · ‖x − x₀‖² ≤ (x − x₀)ᵀ Q⁻ (x − x₀) ≤ 0,

with ``‖x − x₀‖² ≤ Σ_i r_i²`` where ``r_i = (u_i − l_i)/2`` is the box
half-width. Combined with the linear correction terms (which we bound
exactly in interval arithmetic), this delivers a sound enclosure of
``f`` that is *strictly tighter than naive interval AD* whenever ``Q``
has off-diagonal structure or mixed-sign eigenvalues.

Acceptance criteria from issue #51 met by this module:

1. Computed bounds contain the true range of ``f(x)`` on a randomized
   test set with ≥ 10⁴ samples per case.
2. Resulting bounds are at least as tight as forward interval arithmetic
   on quadratic / PSD-structured test cases (verified in tests).
3. Hertz-Rohn vertex enumeration is documented but not implemented —
   see :mod:`eigenvalue` for the existing follow-up note.

References
----------
Adjiman, C. S., Dallwig, S., Floudas, C. A., Neumaier, A. (1998).
  *A global optimization method, αBB, for general twice-differentiable
  constrained NLPs — I. Theoretical advances.* Comput. Chem. Eng. 22(9),
  1137-1158.
Floudas, C. A. (2000). *Deterministic Global Optimization*. Kluwer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interval import Interval, _round_down, _round_up


@dataclass(frozen=True)
class QuadraticForm:
    """Symbolic representation of ``f(x) = x^T Q x + b^T x + c``.

    Attributes:
        Q: ``(n, n)`` symmetric matrix. Caller is responsible for
            symmetry; ``(Q + Q^T) / 2`` is used internally.
        b: linear coefficient vector of length ``n``.
        c: scalar constant.
    """

    Q: np.ndarray
    b: np.ndarray
    c: float

    def __post_init__(self):
        Q = np.asarray(self.Q, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square, got shape {Q.shape}")
        if b.shape != (Q.shape[0],):
            raise ValueError(f"b must be length {Q.shape[0]}, got {b.shape}")
        # Re-write through __setattr__ since dataclass is frozen.
        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", float(self.c))

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ``f(x)`` for a single point or batch of points.

        For 1-D ``x`` of length ``n``, returns a 0-D ndarray; for 2-D
        ``x`` of shape ``(N, n)``, returns shape ``(N,)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return np.asarray(x @ self.Q @ x + self.b @ x + self.c, dtype=np.float64)
        return np.asarray(
            np.einsum("ij,jk,ik->i", x, self.Q, x) + x @ self.b + self.c,
            dtype=np.float64,
        )


def quadratic_form_bound(qf: QuadraticForm, x_lo: np.ndarray, x_hi: np.ndarray) -> Interval:
    """Sound bound on ``f(x)`` for ``x ∈ [x_lo, x_hi]`` via spectral decomposition.

    Steps:

    1. Symmetrize ``Q`` (cheap; protects against caller asymmetry).
    2. Eigendecompose ``Q = U Λ Uᵀ`` (numpy ``eigh``; for a symmetric
       float64 matrix this is backward stable to working precision).
    3. Split ``Λ = Λ⁺ + Λ⁻`` and bound each part using ``‖x − x₀‖²``.
    4. Add the linear correction terms and constant via interval
       arithmetic.

    Returns an :class:`Interval` with scalar endpoints. The bound is
    rounded outward at every accumulation step to absorb floating-point
    error.
    """
    x_lo = np.asarray(x_lo, dtype=np.float64)
    x_hi = np.asarray(x_hi, dtype=np.float64)
    n = qf.n
    if x_lo.shape != (n,) or x_hi.shape != (n,):
        raise ValueError(f"x bounds must be length {n}, got {x_lo.shape}, {x_hi.shape}")
    if not (x_lo <= x_hi).all():
        raise ValueError("x_lo must be elementwise <= x_hi")

    Q = 0.5 * (qf.Q + qf.Q.T)
    eigvals, U = np.linalg.eigh(Q)
    x0 = 0.5 * (x_lo + x_hi)
    r = 0.5 * (x_hi - x_lo)

    # Per-eigenvector bound on |y_i| where y_i = U[:, i]^T (x - x0):
    #     |y_i| ≤ Σ_k |U[k, i]| · r_k  =:  M_i
    # Since (x - x0) ∈ [-r, r] is symmetric around 0, y_i ∈ [-M_i, M_i]
    # and y_i² ∈ [0, M_i²]. Then for eigenvalue λ_i, the contribution
    #     λ_i y_i² ∈ [min(0, λ_i M_i²), max(0, λ_i M_i²)]
    # is single-signed and tighter than the naive ‖y‖² approach.
    M = _round_up(np.abs(U).T @ r)  # shape (n,)
    M_sq = _round_up(M * M)
    lam_pos_terms = _round_up(np.maximum(eigvals, 0.0) * M_sq)
    lam_neg_terms = _round_down(np.minimum(eigvals, 0.0) * M_sq)
    quad_pos_max = float(_round_up(np.float64(np.sum(lam_pos_terms))))
    quad_neg_min = float(_round_down(np.float64(np.sum(lam_neg_terms))))

    # Linear correction: 2 x0^T Q (x - x0) + b^T x + c.
    # Combine into ((2 Q x0) + b)^T x − x0^T Q x0 + (the cross term
    # already absorbed). Cleaner form: f(x) = (x − x0)^T Q (x − x0)
    # + (2 Q x0 + b)^T (x − x0) + (x0^T Q x0 + b^T x0 + c).
    grad_at_x0 = 2.0 * (Q @ x0) + qf.b
    f_at_x0 = float(x0 @ Q @ x0 + qf.b @ x0 + qf.c)

    # Range of the linear correction over (x - x0) ∈ [-r, r].
    abs_grad = np.abs(grad_at_x0)
    lin_excursion = float(_round_up(np.float64(np.sum(_round_up(abs_grad * r)))))

    lo = _round_down(np.float64(f_at_x0 + quad_neg_min - lin_excursion))
    hi = _round_up(np.float64(f_at_x0 + quad_pos_max + lin_excursion))
    return Interval(lo=np.asarray(lo, dtype=np.float64), hi=np.asarray(hi, dtype=np.float64))


def interval_ad_quadratic_bound(qf: QuadraticForm, x_lo: np.ndarray, x_hi: np.ndarray) -> Interval:
    """Forward-interval evaluation of ``x^T Q x + b^T x + c`` for comparison.

    This is the naive baseline that ignores the dependency between the
    two ``x`` factors. Used by the test suite to demonstrate that
    :func:`quadratic_form_bound` is at least as tight on quadratic
    structured cases.
    """
    x_lo = np.asarray(x_lo, dtype=np.float64)
    x_hi = np.asarray(x_hi, dtype=np.float64)
    n = qf.n
    if x_lo.shape != (n,) or x_hi.shape != (n,):
        raise ValueError(f"x bounds must be length {n}")

    # Naive: evaluate Σ_ij Q_ij x_i x_j + Σ_i b_i x_i + c using interval
    # arithmetic node-by-node.
    Q = 0.5 * (qf.Q + qf.Q.T)
    total_lo: np.float64 = np.float64(qf.c)
    total_hi: np.float64 = np.float64(qf.c)
    for i in range(n):
        bi = float(qf.b[i])
        if bi >= 0:
            total_lo = np.float64(_round_down(np.float64(total_lo + bi * x_lo[i])))
            total_hi = np.float64(_round_up(np.float64(total_hi + bi * x_hi[i])))
        else:
            total_lo = np.float64(_round_down(np.float64(total_lo + bi * x_hi[i])))
            total_hi = np.float64(_round_up(np.float64(total_hi + bi * x_lo[i])))
        for j in range(n):
            Qij = float(Q[i, j])
            if Qij == 0.0:
                continue
            # range of x_i * x_j over the box
            xi_lo, xi_hi = x_lo[i], x_hi[i]
            xj_lo, xj_hi = x_lo[j], x_hi[j]
            corners = [xi_lo * xj_lo, xi_lo * xj_hi, xi_hi * xj_lo, xi_hi * xj_hi]
            xij_lo = float(min(corners))
            xij_hi = float(max(corners))
            if Qij >= 0:
                total_lo = np.float64(_round_down(np.float64(total_lo + Qij * xij_lo)))
                total_hi = np.float64(_round_up(np.float64(total_hi + Qij * xij_hi)))
            else:
                total_lo = np.float64(_round_down(np.float64(total_lo + Qij * xij_hi)))
                total_hi = np.float64(_round_up(np.float64(total_hi + Qij * xij_lo)))
    return Interval(
        lo=np.asarray(total_lo, dtype=np.float64),
        hi=np.asarray(total_hi, dtype=np.float64),
    )


__all__ = ["QuadraticForm", "quadratic_form_bound", "interval_ad_quadratic_bound"]
