"""On-demand separation of the exact multilinear convex/concave hull.

The convex hull of a single multilinear monomial ``w = prod_i x_i`` over a box
is a polytope in ``(x, w)`` space (Rikun 1997): its facets are inequalities in
the original variables and the product variable *only* — no intermediate
product variables are required. That makes the hull separable on demand:

Given a relaxation point ``(x*, w*)``, the convex envelope value at ``x*`` is

    env(x*) = min_lambda  sum_v lambda_v * f(v)
              s.t.  sum_v lambda_v * v = x*,  sum_v lambda_v = 1,  lambda >= 0

over the ``2^n`` box vertices ``v`` (``f(v) = prod v``). By LP duality the
optimal dual ``(a, b)`` of the equality rows is a supporting hyperplane:
``a . v + b <= f(v)`` for every box point, with ``a . x* + b = env(x*)``. So

    w >= a . x + b                      (convex underestimator cut)

is a *valid* inequality (it never cuts the true manifold ``w = f(x)``) that
separates ``(x*, w*)`` whenever ``w* < env(x*)``. The concave overestimator is
the symmetric construction on ``-f``.

This scales the exact hull past the dense ``2^n``-cut RLT cap: the ``2^n`` only
appears in the small per-term separation LP, not in the main relaxation's
columns or rows. Every separated cut is sound regardless of how many are added,
so the loop can stop at any round and the bound stays valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

try:  # SciPy's HiGHS LP gives exact vertex optima and equality marginals.
    from scipy.optimize import linprog

    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False


@dataclass(frozen=True)
class EnvelopeCut:
    """A valid multilinear hull cut: ``w {>=,<=} a . x + b``.

    ``sense="under"`` is the convex underestimator (``w >= a.x + b``),
    ``sense="over"`` the concave overestimator (``w <= a.x + b``). ``a`` is
    indexed over the term's factors (same order as the caller's columns).
    """

    a: np.ndarray
    b: float
    sense: str


def _solve_envelope(verts: np.ndarray, fv: np.ndarray, x_star: np.ndarray, maximize: bool):
    """Return ``(env_value, a, b)`` of the (concave if maximize) vertex envelope."""
    n = verts.shape[1]
    m = verts.shape[0]
    a_eq = np.vstack([verts.T, np.ones(m)])
    b_eq = np.append(x_star, 1.0)
    c = -fv if maximize else fv
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=[(0.0, None)] * m, method="highs")
    if not res.success:
        return None
    duals = np.asarray(res.eqlin.marginals, dtype=np.float64)
    if duals.shape[0] != n + 1 or not np.all(np.isfinite(duals)):
        return None
    if maximize:
        env = -float(res.fun)
        a = -duals[:n]
        b = -float(duals[n])
    else:
        env = float(res.fun)
        a = duals[:n]
        b = float(duals[n])
    return env, a, b


def separate_multilinear_envelope(
    lb: np.ndarray,
    ub: np.ndarray,
    x_star: np.ndarray,
    w_star: float,
    *,
    tol: float = 1e-6,
    max_factors: int = 12,
) -> list[EnvelopeCut]:
    """Separate violated convex/concave hull cuts for ``w = prod_i x_i``.

    Returns the (at most two) supporting-hyperplane cuts that ``(x*, w*)``
    violates — convex underestimator and/or concave overestimator. Each is a
    valid relaxation cut (it never excludes a true ``(x, prod x)`` point), so the
    returned list is always sound to add. Returns ``[]`` when the point is
    already inside the hull, the bounds are not finite, the factor count exceeds
    ``max_factors`` (``2^n`` vertices), or SciPy is unavailable.
    """
    if not _SCIPY:
        return []
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    n = lb.shape[0]
    if n < 2 or n > max_factors:
        return []
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        return []
    if not np.isfinite(w_star):
        return []
    # Clamp the query point into the box (LP feasibility); a point outside is a
    # numerical artifact and clamping keeps the supporting hyperplane valid.
    x_star = np.clip(x_star, lb, ub)

    verts = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in range(n)])))
    fv = np.prod(verts, axis=1)

    cuts: list[EnvelopeCut] = []
    under = _solve_envelope(verts, fv, x_star, maximize=False)
    if under is not None:
        env, a, b = under
        if w_star < env - tol:
            cuts.append(EnvelopeCut(a=a, b=b, sense="under"))
    over = _solve_envelope(verts, fv, x_star, maximize=True)
    if over is not None:
        env, a, b = over
        if w_star > env + tol:
            cuts.append(EnvelopeCut(a=a, b=b, sense="over"))
    return cuts
