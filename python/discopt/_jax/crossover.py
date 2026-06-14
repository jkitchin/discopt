"""Interior-point → vertex crossover (roadmap Phase 2 keystone).

An interior-point LP solve returns the *analytic center* of the optimal face,
which smears fractional structure: cover/clique cuts and branching separate a
*vertex* sharply but a symmetric interior point weakly (see the Phase-3 cut
findings). This module pushes an interior optimum to a vertex of the optimal
face, exposing that structure.

Given ``min c^T x  s.t.  A x = b, l <= x <= u`` with an optimal point ``x``
(value ``z* = c^T x``), the optimal face is ``{x : A x = b, c^T x = z*,
l <= x <= u}``. We repeatedly move along a direction ``d`` supported on the
*free* variables (those strictly inside their bounds) with ``A d = 0`` and
``c^T d = 0`` — so the move stays feasible and keeps the objective — until a
free variable hits a bound (ratio test). Fixing that variable removes a degree
of freedom; when no free direction remains, ``x`` is a vertex of the optimal
face. Terminates in at most ``n`` steps.

Note: this is used to *find* violated cuts; the cuts themselves are validated
by their own (cover/clique) structure, so crossover numerics never affect
soundness.
"""

from __future__ import annotations

import numpy as np

# Skip the (cubic-ish) push on very wide problems; interior-point separation is
# the fallback. Cut soundness does not depend on this.
_MAX_CROSSOVER_VARS = 400


def _null_direction(M: np.ndarray, tol: float):
    """A unit null-space direction of ``M`` (column-rank-deficient), else None."""
    if M.shape[1] == 0:
        return None
    # SVD: right-singular vectors with ~zero singular value span the null space.
    _, s, vt = np.linalg.svd(M, full_matrices=True)
    k = M.shape[1]
    smax = s[0] if s.size else 1.0
    rank = int(np.sum(s > tol * max(1.0, smax)))
    if rank >= k:
        return None
    return vt[rank]


def _max_step(x, d, xl, xu, free, tol):
    """Largest signed ``t`` keeping ``x + t d`` within bounds on ``free`` vars.

    Tries ``+d`` then ``-d`` (both preserve feasibility/objective); returns the
    signed step that drives some free variable to a bound, or ``None`` if the
    face is unbounded along ``d``."""

    def step(direction):
        t = np.inf
        for j in free:
            dj = direction[j]
            if dj > tol:
                t = min(t, (xu[j] - x[j]) / dj)
            elif dj < -tol:
                t = min(t, (xl[j] - x[j]) / dj)
        return t

    tp = step(d)
    if np.isfinite(tp) and tp > tol:
        return tp
    tn = step(-d)
    if np.isfinite(tn) and tn > tol:
        return -tn
    return None


def crossover_to_vertex(
    x,
    A,
    b,
    c,
    xl,
    xu,
    *,
    tol: float = 1e-7,
    max_iter: int | None = None,
):
    """Push interior optimum ``x`` to a vertex of the LP optimal face.

    Returns a point with the same objective and (to tolerance) the same
    feasibility as ``x``, but at a vertex of the optimal face. Returns ``x``
    unchanged if the problem is too wide (``> _MAX_CROSSOVER_VARS``) or the face
    is unbounded along a push direction.
    """
    x = np.array(x, dtype=np.float64).copy()
    n = x.shape[0]
    if n == 0 or n > _MAX_CROSSOVER_VARS:
        return x
    A = np.asarray(A, dtype=np.float64) if A is not None and np.size(A) else np.zeros((0, n))
    c = np.asarray(c, dtype=np.float64).ravel()
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)
    m = A.shape[0]
    if max_iter is None:
        max_iter = n + 1

    for _ in range(max_iter):
        free = np.where((x > xl + tol) & (x < xu - tol))[0]
        if free.size == 0:
            break
        c_row = c[free][None, :]
        M = np.vstack([A[:, free], c_row]) if m > 0 else c_row
        d_free = _null_direction(M, tol)
        if d_free is None:
            break  # free columns independent -> vertex of the optimal face
        d = np.zeros(n)
        d[free] = d_free
        t = _max_step(x, d, xl, xu, free, tol)
        if t is None or abs(t) < tol:
            break
        x = np.clip(x + t * d, xl, xu)
    return x
