"""PSD / eigenvalue cuts for the lifted (moment) relaxation of QCQP.

Term-wise McCormick / RLT relaxations lift each product ``x_i x_j`` to an
independent variable ``X_ij`` but never enforce that the moment matrix

    M(x, X) = [[1, xᵀ], [x, X]]

is positive semidefinite — yet ``M ⪰ 0`` holds for *every* feasible point
(where ``X = x xᵀ``, so ``M = [1; x][1; x]ᵀ ⪰ 0``). That missing condition is the
dominant source of the relaxation gap on nonconvex QCQP.

This module separates it dynamically: at a relaxation point ``(x*, X*)`` where the
moment matrix has a negative eigenvalue ``λ_min`` with eigenvector ``v``, the
inequality

    vᵀ M(x, X) v ≥ 0

is **linear** in ``(x, X)`` and valid for the whole feasible region (it equals
``(v₀ + v_restᵀ x)² ≥ 0`` at any true point), while it is *violated* at ``(x*, X*)``
— so it is a sound cut that tightens the relaxation toward the SDP bound, with no
SDP solver: a single dense ``eigh`` on the (small) moment submatrix.

The separator is purely numeric and indexing-driven so it plugs into whatever
column layout the relaxation uses (original variables + lifted product columns).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from discopt._jax.cutting_planes import LinearCut

__all__ = [
    "psd_cut_from_submatrix",
    "moment_matrix",
    "separate_psd_cuts_on_relaxation",
    "psd_strengthen_relaxation_bound",
]


def moment_matrix(x_vals: np.ndarray, X_vals: np.ndarray) -> np.ndarray:
    """Assemble the (k+1)×(k+1) moment matrix ``[[1, xᵀ], [x, X]]`` (symmetrized)."""
    x = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    X = np.asarray(X_vals, dtype=np.float64)
    k = x.size
    if X.shape != (k, k):
        raise ValueError(f"X_vals must be ({k}, {k}); got {X.shape}")
    Xs = 0.5 * (X + X.T)
    M = np.empty((k + 1, k + 1), dtype=np.float64)
    M[0, 0] = 1.0
    M[0, 1:] = x
    M[1:, 0] = x
    M[1:, 1:] = Xs
    return M


def psd_cut_from_submatrix(
    x_vals: np.ndarray,
    X_vals: np.ndarray,
    orig_cols: Sequence[int],
    prod_cols: np.ndarray,
    n_total: int,
    *,
    tol: float = 1e-7,
) -> Optional[LinearCut]:
    """Separate one PSD cut from a moment submatrix, or ``None`` if PSD enough.

    Parameters
    ----------
    x_vals : (k,) array
        Values of the ``k`` original variables in this submatrix at the point.
    X_vals : (k, k) array
        Values of the lifted products ``X_ij ≈ x_i x_j`` at the point.
    orig_cols : (k,) sequence of int
        Column index in the relaxation LP of each original variable.
    prod_cols : (k, k) int array
        Column index of the lifted variable representing ``X_ij`` (symmetric; the
        same column may back both ``(i, j)`` and ``(j, i)``).
    n_total : int
        Total number of columns in the relaxation LP (length of the cut vector).
    tol : float
        Only emit a cut when ``λ_min(M) < -tol`` (the point violates PSD).

    Returns
    -------
    LinearCut or None
        A valid inequality ``coeffs · z ≥ rhs`` violated at ``(x*, X*)``. The cut
        is valid for every feasible point because ``vᵀ M v = (v₀ + v_restᵀ x)² ≥ 0``
        whenever ``X = x xᵀ`` — it never removes a feasible point.
    """
    x = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    k = x.size
    prod_cols = np.asarray(prod_cols)
    if prod_cols.shape != (k, k):
        raise ValueError(f"prod_cols must be ({k}, {k}); got {prod_cols.shape}")

    M = moment_matrix(x, X_vals)
    eigvals, eigvecs = np.linalg.eigh(M)  # ascending; symmetric
    lam = float(eigvals[0])
    if lam >= -tol:
        return None  # already PSD to tolerance: nothing to separate

    v = eigvecs[:, 0]
    v0 = float(v[0])
    vr = v[1:]

    # vᵀ M v = v0² + 2 v0 Σ_a vr_a x_a + Σ_{a,b} vr_a vr_b X_ab ≥ 0
    # → coeffs·z ≥ -v0², with coeffs on the (x, X) columns.
    coeffs = np.zeros(n_total, dtype=np.float64)
    for a in range(k):
        coeffs[int(orig_cols[a])] += 2.0 * v0 * vr[a]
    for a in range(k):
        for b in range(k):
            coeffs[int(prod_cols[a, b])] += vr[a] * vr[b]

    rhs = -(v0 * v0)
    return LinearCut(coeffs=coeffs, rhs=float(rhs), sense=">=")


def _diag_col(info: dict, i: int) -> Optional[int]:
    """Relaxation column holding the lifted square ``X_ii = x_i**2`` (or None)."""
    mono = info.get("monomial", {})
    usq = info.get("univariate_square", {})
    if (i, 2) in mono:
        return int(mono[(i, 2)])
    if (i, 2) in usq:
        return int(usq[(i, 2)])
    return None


def _moment_blocks_for_set(info: dict, S: Sequence[int]):
    """Column indices needed for the moment submatrix over variables ``S``.

    Returns ``(orig_cols, diag_cols, off_cols)`` or ``None`` if any required
    lifted column (a square or a cross-product within ``S``) is missing.
    """
    orig = info.get("original", {})
    bil = info.get("bilinear", {})
    S = list(S)
    if any(i not in orig for i in S):
        return None
    diag_cols = []
    for i in S:
        di = _diag_col(info, i)
        if di is None:
            return None
        diag_cols.append(di)
    k = len(S)
    off = np.full((k, k), -1, dtype=int)
    for a in range(k):
        for b in range(a + 1, k):
            key = (min(S[a], S[b]), max(S[a], S[b]))
            if key not in bil:
                return None
            off[a, b] = off[b, a] = int(bil[key])
    return [int(orig[i]) for i in S], diag_cols, off


def _moment_clique_cut(
    info: dict, x_full: np.ndarray, S: Sequence[int], n_total: int, *, tol: float
) -> Optional[LinearCut]:
    """Separate a single moment cut over the variable clique ``S`` (size >= 2)."""
    blocks = _moment_blocks_for_set(info, S)
    if blocks is None:
        return None
    orig_cols, diag_cols, off = blocks
    k = len(S)
    x_vals = np.array([x_full[c] for c in orig_cols], dtype=np.float64)
    X_vals = np.empty((k, k), dtype=np.float64)
    prod_cols = np.empty((k, k), dtype=int)
    for a in range(k):
        prod_cols[a, a] = diag_cols[a]
        X_vals[a, a] = x_full[diag_cols[a]]
        for b in range(a + 1, k):
            prod_cols[a, b] = prod_cols[b, a] = off[a, b]
            X_vals[a, b] = X_vals[b, a] = x_full[off[a, b]]
    return psd_cut_from_submatrix(x_vals, X_vals, orig_cols, prod_cols, n_total, tol=tol)


def _lifted_cliques(info: dict, max_dim: int) -> list[tuple[int, ...]]:
    """Greedy cliques of variables whose pairwise products + squares are all lifted.

    A clique ``S`` (all pairs in ``bilinear``, all squares lifted) is exactly the
    set over which a dense moment submatrix can be formed. Dense ``k>=3`` cliques
    capture multi-variable moment coupling that pairwise 2x2 minors miss.
    """
    bil = info.get("bilinear", {})
    verts = [i for i in info.get("original", {}) if _diag_col(info, i) is not None]
    adj: dict[int, set] = {i: set() for i in verts}
    for i, j in bil:
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    cliques: set[tuple[int, ...]] = set()
    # Seed a greedy clique from each vertex, extending by most-connected neighbours.
    for seed in sorted(verts, key=lambda v: -len(adj[v])):
        clique = [seed]
        cand = sorted(adj[seed], key=lambda v: -len(adj[v]))
        for v in cand:
            if len(clique) >= max_dim:
                break
            if all(v in adj[u] for u in clique):
                clique.append(v)
        if len(clique) >= 2:
            cliques.add(tuple(sorted(clique)))
    return sorted(cliques, key=lambda c: (-len(c), c))


def separate_psd_cuts_on_relaxation(
    info: dict,
    x_full: np.ndarray,
    n_total: int,
    *,
    tol: float = 1e-7,
    max_cuts: int = 64,
    max_dim: int = 6,
) -> list[LinearCut]:
    """Separate moment (PSD) cuts from a McCormick relaxation point.

    ``info`` is the column-map dict returned by ``build_milp_relaxation``: it maps
    ``original`` variables, ``bilinear`` pairs ``(i, j) -> col(X_ij)``, and
    ``monomial``/``univariate_square`` ``(i, 2) -> col(X_ii)`` to relaxation
    columns. Cuts are separated over **cliques** of variables whose pairwise
    products and squares are all lifted (up to ``max_dim`` variables): the moment
    submatrix

        [[1, x_S^T], [x_S, X_SS]]

    is checked for PSD-ness at the current point; a negative eigenvalue yields a
    valid cut. Dense ``k>=3`` cliques capture multi-variable moment coupling that
    pairwise 2x2 minors cannot; pairwise cuts remain the ``k=2`` special case.
    """
    cuts: list[LinearCut] = []
    seen_pairs: set[tuple[int, int]] = set()
    # Dense cliques first (strongest), then any remaining pairwise products.
    for S in _lifted_cliques(info, max_dim):
        if len(cuts) >= max_cuts:
            break
        cut = _moment_clique_cut(info, x_full, S, n_total, tol=tol)
        if cut is not None:
            cuts.append(cut)
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                seen_pairs.add((S[a], S[b]))

    orig = info.get("original", {})
    bil = info.get("bilinear", {})
    for i, j in bil:
        if len(cuts) >= max_cuts:
            break
        if (min(i, j), max(i, j)) in seen_pairs:
            continue
        di = _diag_col(info, i)
        dj = _diag_col(info, j)
        if di is None or dj is None or i not in orig or j not in orig:
            continue
        cut = _moment_clique_cut(info, x_full, (i, j), n_total, tol=tol)
        if cut is not None:
            cuts.append(cut)
    return cuts


def psd_strengthen_relaxation_bound(
    milp,
    info: dict,
    *,
    max_rounds: int = 5,
    tol: float = 1e-7,
    time_limit_per_lp: Optional[float] = 1.0,
):
    """Iteratively add PSD cuts to a McCormick relaxation LP and re-solve.

    Returns ``(z_before, z_after, n_cuts)`` where the ``z`` values are valid dual
    bounds (``min`` of the relaxation objective, in the user's objective scale).
    Because every PSD cut is valid for the whole feasible region, ``z_after`` is a
    *valid* bound that is ``>= z_before`` — a genuine tightening that never
    excludes the optimum. A sound no-op (``z_after == z_before``, ``0`` cuts) on
    any failure (no exact LP oracle, non-optimal solve, no separable cut).
    """
    import scipy.sparse as sp

    from discopt._jax.obbt import get_exact_lp_solver
    from discopt.solvers import SolveStatus

    _lp = get_exact_lp_solver()
    if _lp is None:
        return (None, None, 0)

    c = np.asarray(milp._c, dtype=np.float64)
    bounds = list(milp._bounds)
    n_total = len(bounds)
    offset = float(milp._obj_offset)

    A_ub = milp._A_ub
    b_ub = np.asarray(milp._b_ub, dtype=np.float64) if milp._b_ub is not None else np.zeros(0)
    if A_ub is None:
        A_ub = sp.csr_matrix((0, n_total))

    def _solve(A, b):
        return _lp(c=c, A_ub=A, b_ub=b, bounds=bounds, time_limit=time_limit_per_lp)

    res = _solve(A_ub, b_ub)
    if res.status != SolveStatus.OPTIMAL or res.objective is None or res.x is None:
        return (None, None, 0)
    z_before = float(res.objective) + offset
    z_cur = z_before
    total_cuts = 0

    A_cur = sp.csr_matrix(A_ub)
    b_cur = np.asarray(b_ub, dtype=np.float64)
    for _ in range(max(1, max_rounds)):
        cuts = separate_psd_cuts_on_relaxation(info, np.asarray(res.x), n_total, tol=tol)
        if not cuts:
            break
        # Cut is coeffs.z >= rhs  ==>  (-coeffs).z <= -rhs for the A_ub<=b_ub form.
        rows = sp.csr_matrix(np.vstack([-c0.coeffs for c0 in cuts]))
        rhs = np.array([-c0.rhs for c0 in cuts], dtype=np.float64)
        A_cur = sp.vstack([A_cur, rows], format="csr")
        b_cur = np.concatenate([b_cur, rhs])
        total_cuts += len(cuts)

        res = _solve(A_cur, b_cur)
        if res.status != SolveStatus.OPTIMAL or res.objective is None or res.x is None:
            break
        z_new = float(res.objective) + offset
        if z_new <= z_cur + 1e-9:
            z_cur = max(z_cur, z_new)
            break
        z_cur = z_new
    return (z_before, z_cur, total_cuts)
