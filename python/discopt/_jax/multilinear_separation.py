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

import os
from dataclasses import dataclass
from itertools import product

import numpy as np

# Cold-simplex pivot cap for the multilinear vertex-hull LP. The hull LP has
# ``2^n`` λ columns and ``n+1`` rows; on the widest ones (``n``≈10 → 1024
# columns) the *cold* Rust simplex either converges in a few hundred pivots
# (≈1 ms — ~100× faster than the POUNCE IPM's ~150 ms) or, on a degenerate/
# ill-conditioned instance, spins to its 100 000-pivot default (≈1.6 s) — the
# same stall class F2 guarded on the *warm* dual path, but the standalone hull
# LP is solved cold. The two regimes are cleanly separated (hundreds vs 10⁵
# pivots), so a small absolute cap detects a stall in ≈tens of ms; on a cap trip
# (or any non-optimal exit) we fall back to POUNCE for that single LP, so no LP
# is ever slower than the pre-F3 POUNCE baseline while the converging majority
# gets the simplex win. The cap grows slowly with rows (``n+1``) for headroom.
_HULL_SIMPLEX_STALL_K = 100
_HULL_SIMPLEX_STALL_C = 500
_HULL_SIMPLEX_STALL_MAX = 3000


def _separation_lp_simplex_enabled() -> bool:
    """Whether F3 routes the hull LP to the Rust simplex (env, default ON).

    Honors the same ``DISCOPT_SEPARATION_LP_SIMPLEX`` flag as the edge-concave
    separator (:func:`discopt._jax.edge_concave._separation_lp_solver`) and
    strong branching; ``"0"`` restores the POUNCE-IPM-only path.
    """
    return os.environ.get("DISCOPT_SEPARATION_LP_SIMPLEX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


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
    """Return ``(env_value, a, b)`` of the (concave if maximize) vertex envelope.

    The supporting-hyperplane LP ``min/max f(v)·λ s.t. Vᵀλ = x*, 1ᵀλ = 1, λ ≥ 0``
    is solved with the in-house pure-Rust warm simplex by default (F3 lever;
    ``DISCOPT_SEPARATION_LP_SIMPLEX=0`` restores the POUNCE IPM — see
    :func:`_separation_lp_simplex_enabled`). The simplex is tried first with a
    size-derived pivot cap; on a cap trip or any non-optimal exit it falls back
    to POUNCE for that LP, so no single LP is slower than the pre-F3 POUNCE
    baseline. Only the dual *slope* ``a`` (the marginals on the ``Vᵀλ = x*``
    rows) is taken from whichever engine converged; the intercept ``b`` is then
    recomputed to the exact validity boundary over the box vertices —
    ``b = minᵥ(f(v) − a·v)`` (under) / ``maxᵥ`` (over). Because a multilinear
    function attains its box extrema at vertices, the resulting hyperplane
    ``a·x + b`` under/over-estimates ``f`` *everywhere*, so the cut is rigorously
    valid for ANY slope — robust to the backend's analytic-center dual on a
    degenerate LP (a different facet, still valid) or any dual sign/scale
    convention. ``None`` if no backend is available or neither engine converged.
    """
    from discopt.solvers import SolveStatus

    n = verts.shape[1]
    m = verts.shape[0]
    a_eq = np.vstack([verts.T, np.ones(m)])
    b_eq = np.append(x_star, 1.0)
    c = -fv if maximize else fv
    bounds = [(0.0, np.inf)] * m

    duals: np.ndarray | None = None
    if _separation_lp_simplex_enabled():
        try:
            from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE
            from discopt.solvers.lp_simplex import solve_lp as _simplex_solve_lp

            if SIMPLEX_AVAILABLE:
                cap = min(
                    _HULL_SIMPLEX_STALL_K * (n + 1) + _HULL_SIMPLEX_STALL_C,
                    _HULL_SIMPLEX_STALL_MAX,
                )
                res = _simplex_solve_lp(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, max_iter=int(cap))
                if res.status == SolveStatus.OPTIMAL and res.dual_values is not None:
                    _d = np.asarray(res.dual_values, dtype=np.float64)
                    if _d.shape[0] == n + 1 and np.all(np.isfinite(_d)):
                        duals = _d
        except ImportError:
            duals = None
    if duals is None:
        # POUNCE path: the off-switch, or the per-LP fallback when the capped
        # simplex stalled / returned a non-usable dual (validity is unaffected —
        # the intercept is recomputed from the vertices below for ANY slope).
        from discopt.solvers.lp_pounce import solve_lp as _pounce_solve_lp

        try:
            res = _pounce_solve_lp(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
        except ImportError:  # pragma: no cover - POUNCE is a core dependency
            return None
        if res.status != SolveStatus.OPTIMAL or res.dual_values is None:
            return None
        _d = np.asarray(res.dual_values, dtype=np.float64)
        if _d.shape[0] != n + 1 or not np.all(np.isfinite(_d)):
            return None
        duals = _d

    a = -duals[:n] if maximize else duals[:n]
    if not np.all(np.isfinite(a)):
        return None
    # Recompute the intercept to the exact validity boundary over the vertices,
    # so ``a·v + b`` bounds ``f(v)`` at every vertex (hence everywhere) — this is
    # what makes the cut sound without trusting the engine's reported intercept/scale.
    resid = fv - verts @ a  # f(v) − a·v
    if maximize:  # concave overestimator: a·v + b >= f(v)  ->  b = maxᵥ(f(v)−a·v)
        b = float(np.max(resid))
    else:  # convex underestimator: a·v + b <= f(v)  ->  b = minᵥ(f(v)−a·v)
        b = float(np.min(resid))
    env = float(a @ x_star + b)  # value of this valid hyperplane at x*
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
    ``max_factors`` (``2^n`` vertices), or the LP solve did not converge.
    """
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
