"""Exact-rational positive-semidefiniteness test (Tier-2 convexity witness).

A quadratic body is convex iff its (constant) Hessian is positive semidefinite.
Over exact rationals this is decidable with **no floating point**: run a
symmetric LDLᵀ-style elimination and check every pivot is ``>= 0`` (and that a
zero pivot leaves its row/column consistent). We use complete diagonal pivoting
so genuinely semidefinite (not just definite) matrices are handled.

This mirrors what the Lean checker will do (an exact `Rat` factorization), so a
Tier-2 certificate's convexity claim is machine-checkable, not asserted.
"""

from __future__ import annotations

from fractions import Fraction

Matrix = list  # list[list[Fraction]], symmetric


def is_psd(mat: list[list[Fraction]], tol: Fraction = Fraction(0)) -> tuple[bool, str]:
    """Return ``(True, reason)`` iff the symmetric rational ``mat`` is PSD.

    Uses symmetric Gaussian elimination with diagonal pivoting (exact rationals):

    * pick the largest remaining diagonal entry as pivot;
    * if that maximum is ``< -tol`` -> a negative direction exists -> not PSD;
    * if it is ``<= tol`` (numerically zero), the whole remaining trailing block
      must be zero (a nonzero off-diagonal with a zero diagonal exposes a
      negative eigenvalue) -> otherwise not PSD; then stop (rank found);
    * otherwise eliminate the pivot row/column (Schur complement) and continue.

    ``tol`` defaults to exact ``0``; pass a small rational to absorb the rounding
    in an emitted (float-derived) Hessian.
    """
    n = len(mat)
    # Work on a mutable copy; verify symmetry as we go.
    a = [[Fraction(mat[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if a[i][j] != a[j][i]:
                return False, f"Hessian not symmetric at ({i},{j})"

    active = list(range(n))
    while active:
        # Diagonal pivot: largest remaining diagonal entry.
        p = max(active, key=lambda idx: a[idx][idx])
        piv = a[p][p]
        if piv < -tol:
            return False, f"negative pivot {float(piv):.3g} at index {p} (not PSD)"
        if piv <= tol:
            # Remaining diagonal all <= tol; require the whole trailing block ~0.
            for i in active:
                for j in active:
                    if abs(a[i][j]) > tol:
                        return False, (
                            f"zero pivot but nonzero entry {float(a[i][j]):.3g} at "
                            f"({i},{j}) (indefinite direction, not PSD)"
                        )
            break
        active.remove(p)
        # Schur complement: a[i][j] -= a[i][p]*a[p][j]/piv for remaining i,j.
        for i in active:
            aip = a[i][p]
            if aip == 0:
                continue
            for j in active:
                a[i][j] -= aip * a[p][j] / piv
    return True, "PSD (all pivots >= 0)"


def is_zero_matrix(mat: list[list[Fraction]], tol: Fraction = Fraction(0)) -> bool:
    """True iff every entry is within ``tol`` of zero (affine-body Hessian check)."""
    return all(abs(Fraction(x)) <= tol for row in mat for x in row)
