"""
Sparse KKT system assembly and direct solve for IPM.

Assembles the augmented KKT system in sparse format and solves via
scipy.sparse.linalg.spsolve (SuperLU). This avoids forming dense n x n
matrices, reducing memory from O(n^2) to O(nnz) and solve time from
O(n^3) to O(nnz * log(nnz)) for typical sparse problems.

The KKT system has the structure:

    [ H + D_x + delta_w*I    J^T  ] [ dx ]   [ -r_d ]
    [        J            -delta_c*I ] [ dy ] = [ -r_p ]

where:
  - H is the Hessian of the Lagrangian (sparse)
  - J is the constraint Jacobian (sparse)
  - D_x = diag(sigma) from barrier terms (diagonal)
  - delta_w, delta_c are inertia correction parameters
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def assemble_kkt_sparse(
    H: sp.spmatrix,
    J: sp.spmatrix,
    sigma: np.ndarray,
    delta_w: float = 0.0,
    delta_c: float = 0.0,
) -> sp.csc_matrix:
    """Assemble the augmented KKT matrix in sparse CSC format.

    Args:
        H: (n, n) sparse Hessian of the Lagrangian.
        J: (m, n) sparse constraint Jacobian.
        sigma: (n,) diagonal barrier scaling from bound slacks.
        delta_w: Inertia correction for the (1,1) block.
        delta_c: Inertia correction for the (2,2) block (negative).

    Returns:
        (n+m, n+m) sparse CSC KKT matrix.
    """
    n = H.shape[0]
    m = J.shape[0]

    # (1,1) block: H + diag(sigma) + delta_w * I
    D_x = sp.diags(sigma, format="csc")
    block_11 = sp.csc_matrix(H) + D_x
    if delta_w > 0:
        block_11 = block_11 + delta_w * sp.eye(n, format="csc")

    # (2,2) block: -delta_c * I
    if delta_c > 0:
        block_22 = -delta_c * sp.eye(m, format="csc")
    else:
        block_22 = sp.csc_matrix((m, m))

    # Assemble: [[block_11, J^T], [J, block_22]]
    J_csc = sp.csc_matrix(J)
    top = sp.hstack([block_11, J_csc.T], format="csc")
    bottom = sp.hstack([J_csc, block_22], format="csc")
    kkt = sp.vstack([top, bottom], format="csc")

    return kkt


def solve_kkt_direct(
    kkt: sp.csc_matrix,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve the KKT system directly via sparse LU (SuperLU).

    Args:
        kkt: (n+m, n+m) sparse CSC KKT matrix.
        rhs: (n+m,) right-hand side vector.

    Returns:
        (n+m,) solution vector.
    """
    return np.asarray(spla.spsolve(kkt, rhs))


def solve_kkt_factored(
    kkt: sp.csc_matrix,
    rhs: np.ndarray,
    lu: Optional[spla.SuperLU] = None,
) -> tuple[np.ndarray, spla.SuperLU]:
    """Solve KKT system with LU factorization reuse.

    On first call, factorizes KKT and returns the factorization.
    On subsequent calls with the same pattern, reuses the factorization.

    Args:
        kkt: (n+m, n+m) sparse CSC KKT matrix.
        rhs: (n+m,) right-hand side vector.
        lu: Optional pre-computed LU factorization.

    Returns:
        Tuple of (solution, lu_factorization).
    """
    if lu is None:
        lu = spla.splu(kkt)
    solution = lu.solve(rhs)
    return solution, lu


def detect_inertia_sparse(
    H_block: sp.spmatrix,
    n: int,
    threshold: float = -1e-8,
) -> tuple[bool, float]:
    """Check if the (1,1) block has correct inertia for the IPM.

    For the IPM to converge, the (1,1) block should be positive definite
    (or at least positive semidefinite). We check the smallest eigenvalue.

    Args:
        H_block: (n, n) sparse matrix (H + sigma + delta_w*I).
        n: Dimension.
        threshold: Minimum acceptable eigenvalue.

    Returns:
        Tuple of (is_ok, min_eigenvalue).
    """
    if n <= 0:
        return True, 0.0

    if n <= 10:
        # Small enough for dense eigenvalue computation
        H_dense = H_block.toarray()
        eigvals = np.linalg.eigvalsh(H_dense)
        min_eig = float(eigvals[0])
    else:
        try:
            from scipy.sparse.linalg import eigsh

            eigvals = eigsh(sp.csc_matrix(H_block), k=1, which="SA", return_eigenvectors=False)
            min_eig = float(eigvals[0])
        except Exception:
            # Fallback: assume inertia is bad if we can't compute it
            return False, -1.0

    return min_eig >= threshold, min_eig
