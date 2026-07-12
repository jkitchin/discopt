"""Rigorous 1-D convex/concave hull envelopes for single-variable subtrees (H-UNI).

Task ``cert:LR-2`` (``docs/dev/lever-a-root-tightness-plan.md`` §4). The existing
composite-univariate path (:func:`milp_relaxation._collect_composite_univariate_relaxations`)
lifts a single-variable node ``f(x)`` only when its curvature is *certified*
convex or concave, in which case tangents+secant are a rigorous outer band. Many
tight-relaxation targets are neither: nvs09's per-variable composite
``g(x) = (ln(x-2))**2 + (ln(10-x))**2`` on ``x∈[3,9]`` has ``min f'' ≈ -0.097``
(not convex) but its *composed* relaxation (lift ``ln`` then square) lets each
square independently reach 0, losing ~0.8 of bound per variable (~8 total) — the
LR-0 probe (``docs/dev/lr0-logspace-entry-2026-07-11.md``) measured this exactly.

This module builds the **exact 1-D convex underestimator hull** and **concave
overestimator hull** of an arbitrary continuous ``f`` on ``[lo, hi]``, as a small
set of secant lines ``(slope, intercept)`` in the single variable.

**Soundness by construction (#632 review finding 2).** An earlier version shifted
each facet only by the deviation *measured on a finite grid* — proven at the grid
nodes but not between them. A dense-sampling probe found real (tiny, ~1e-6)
violations between nodes on sharply-curved ``f``; that is exactly the
"sampled/assumed convexity" CLAUDE.md §3 forbids on the certificate path. The
correction is now **rigorous**: on each fine cell ``[u,v]`` (nodes ``u,v`` of the
fine grid), ``facet(x) − f(x)`` decomposes into an affine part (max at the nodes —
covered by the measured grid slack) plus the interpolation remainder
``chord_f(x) − f(x)``, which is bounded by ``⅛·max(0, f''_hi)·h²`` for a convex
cell (and symmetrically ``⅛·max(0, −f''_lo)·h²`` for the concave/overestimator
side), where ``[f''_lo, f''_hi]`` is a **sound interval enclosure** of ``f''`` over
the cell and ``h`` the fine spacing. Adding that proven ``O(h²)`` term to the
measured slack makes every facet a bound *everywhere*, not just at the nodes. The
caller supplies the enclosure (``curvature_enclosure``) — in the build via a
sound interval Hessian (``convexity.interval_ad``); the envelope refuses loudly
(returns ``None``) if the enclosure is unbounded or the resulting shift cannot be
driven under the usefulness tolerance, rather than ship an assumed-tight facet.

The construction is porting the validated LR-0 probe builder
(``docs/dev/lr0_probe/lr0_envelopes.py::exact_convex_underenvelope_rows``) into a
shippable, unit-tested form. It is used only behind the default-OFF
``DISCOPT_UNIVARIATE_ENVELOPE`` flag.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

# A ``curvature_enclosure`` maps a cell ``[a, b]`` to a SOUND enclosure
# ``(f''_lo, f''_hi)`` of the second derivative of ``f`` over that cell. Non-finite
# entries signal an unbounded Hessian (the envelope then abstains).
CurvatureEnclosure = Callable[[float, float], "tuple[float, float]"]

# Number of cells over which the rigorous second-order remainder correction is
# evaluated. More cells → a tighter (smaller) proven shift, since the interval
# Hessian enclosure per cell is tighter; the correction is SOUND for any count
# (enclosure monotonicity under subset), so this only trades cost for tightness.
_CURV_CELLS = 512

# Sampling / rigor budget. The coarse grid seeds the hull; the fine grid measures
# the worst-case hull-vs-f deviation used to shift each facet into a proven bound.
_COARSE_N = 4_001
_FINE_MULT = 10
# Cap on secant pieces per side. A raw sampled hull can have thousands of
# near-collinear facets; that many rows per lifted node is both slow and
# ill-conditioned. We keep at most this many vertices (endpoints always kept,
# interior vertices thinned to the largest-curvature ones) and re-prove the
# thinned hull against the fine grid, so soundness is preserved by construction.
_MAX_LINES = 64
# A hull whose proven correction shift exceeds this (relative to the function's
# range on the box) is too coarse to be a *useful* envelope; we refuse rather than
# emit a valid-but-vacuous band. This is a usefulness guard, never a soundness one:
# the shift always makes the facet sound; this only rejects a hull that would add
# no tightening. The shift lowers every facet uniformly, so it is a direct upper
# bound on the bound loss vs the true 1-D hull; 1e-3·range keeps that loss small
# relative to typical certificate tolerances (nvs09: ~1.8e-4 shift on a range ~4).
_MAX_REL_SLACK = 1e-3


def _lower_convex_hull(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    """Vertices of the lower convex hull of points ``(xs, ys)`` (xs ascending).

    Monotone-chain: a point is kept only while the last turn is a left turn
    (positive cross product), which traces the lower boundary of the convex hull.
    """
    hull: list[tuple[float, float]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        while len(hull) >= 2:
            (x0, y0), (x1, y1) = hull[-2], hull[-1]
            # cross((x1-x0,y1-y0),(x-x0,y-y0)) <= 0 → (x1,y1) not a lower vertex
            cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
            if cross <= 0.0:
                hull.pop()
            else:
                break
        hull.append((float(x), float(y)))
    return hull


def _rdp_indices(pts: list[tuple[float, float]], eps: float) -> list[int]:
    """Ramer–Douglas–Peucker vertex selection: indices whose polyline stays within
    vertical distance ``eps`` of the full ``pts``. Endpoints always kept.

    Vertical (not perpendicular) distance is used because the hull is a graph
    ``y(x)`` and the downstream slack correction measures vertical deviation.
    """
    n = len(pts)
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        m = (y1 - y0) / (x1 - x0) if x1 > x0 else 0.0
        worst_d = -1.0
        worst_i = -1
        for i in range(i0 + 1, i1):
            xi, yi = pts[i]
            d = abs(yi - (y0 + m * (xi - x0)))
            if d > worst_d:
                worst_d = d
                worst_i = i
        if worst_i != -1 and worst_d > eps:
            keep[worst_i] = True
            stack.append((i0, worst_i))
            stack.append((worst_i, i1))
    return [i for i in range(n) if keep[i]]


def _thin_hull(
    hull: list[tuple[float, float]], max_lines: int, f_range: float
) -> list[tuple[float, float]]:
    """Simplify ``hull`` to at most ``max_lines`` facets with bounded added slack.

    Uses Ramer–Douglas–Peucker with the smallest ``eps`` (bisected) that yields
    ``≤ max_lines+1`` vertices, so the retained polyline stays within ``eps`` of
    the full hull. The added vertical deviation is therefore ≤ ``eps``, keeping
    the caller's slack correction small (and bound loss small). Soundness is
    unaffected — the caller re-measures and shifts regardless — this only controls
    *how much* bound the thinning costs.
    """
    if len(hull) <= max_lines + 1:
        return hull
    # Bisect eps to land at/under the facet cap. Upper bound: the full hull's
    # vertical span (any polyline within that eps collapses to the endpoints).
    span = max(f_range, 1e-12)
    lo_eps, hi_eps = 0.0, span
    best = _rdp_indices(hull, hi_eps)
    for _ in range(40):
        mid = 0.5 * (lo_eps + hi_eps)
        idxs = _rdp_indices(hull, mid)
        if len(idxs) <= max_lines + 1:
            best = idxs
            hi_eps = mid
        else:
            lo_eps = mid
    return [hull[i] for i in best]


def _pwl_eval(hull: list[tuple[float, float]], xf: np.ndarray) -> np.ndarray:
    hx = np.array([p[0] for p in hull], dtype=np.float64)
    hy = np.array([p[1] for p in hull], dtype=np.float64)
    return np.asarray(np.interp(xf, hx, hy), dtype=np.float64)


def _hull_lines(
    xs: np.ndarray,
    ys: np.ndarray,
    xf: np.ndarray,
    ff: np.ndarray,
    f_range: float,
    *,
    upper: bool,
    remainder: float,
) -> Optional[tuple[tuple[float, float], ...]]:
    """Proven under- (``upper=False``) or over- (``upper=True``) estimator lines.

    ``xs``/``ys`` are the coarse hull-seed samples; ``xf``/``ff`` the fine grid
    used to measure the hull's deviation from ``f`` at the fine nodes. ``remainder``
    is the proven ``O(h²)`` second-order bound on ``f``'s deviation from its chord
    *between* fine nodes for this side (under vs over) — added to the measured node
    slack so the shift is a bound everywhere, not just at nodes. Returns a tuple of
    ``(slope, intercept)`` lines such that, on the box, ``slope*x + intercept ≤
    f(x)`` (under) resp. ``≥ f(x)`` (over). ``None`` if the hull cannot be made a
    *useful* rigorous bound (caller falls back to the existing path).
    """
    if not np.isfinite(remainder):
        return None  # unbounded interval Hessian → no proven bound → abstain
    # For an OVER estimator (concave hull) we build the LOWER hull of -f and negate.
    sign = -1.0 if upper else 1.0
    hull = _lower_convex_hull(xs, sign * ys)
    if len(hull) < 2:
        return None

    hull = _thin_hull(hull, _MAX_LINES, f_range)

    # Rigor: the piecewise-linear hull of the *samples* can sit slightly on the
    # wrong side of f between samples — and thinning vertices can raise it further.
    # Measure the worst deviation at the fine nodes and shift the whole (thinned)
    # hull to the safe side by that (nonnegative) slack. The affine (facet − chord)
    # part is maximised at a fine node, so this covers it; the remaining
    # chord-vs-f interpolation error between nodes is bounded by the proven
    # ``remainder`` (⅛·max(0,±f'')·h²). Their sum shifts every facet to a bound
    # everywhere. Deviation is measured against sign*f, matching the hull space.
    hull_vals = _pwl_eval(hull, xf)
    slack = float(np.max(hull_vals - sign * ff))  # how far hull rose above sign*f
    slack = max(slack, 0.0) + remainder

    denom = max(abs(f_range), 1.0)
    if slack / denom > _MAX_REL_SLACK:
        return None  # too coarse to be a useful (tight) envelope — abstain, don't ship

    lines: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(hull[:-1], hull[1:]):
        if x1 <= x0:
            continue
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0 - slack  # lower the (sign-space) facet by the proven slack
        if not (np.isfinite(m) and np.isfinite(b)):
            return None
        # Map back out of sign space: sign*f ≥ m*x + b  ⇒  f ≥ sign*(m*x+b) [under]
        # (for under sign=+1 this is identity; for over sign=-1 it flips to ≤).
        lines.append((float(sign * m), float(sign * b)))
    if not lines:
        return None
    return tuple(lines)


def _second_order_remainder(
    lo: float,
    hi: float,
    curvature_enclosure: CurvatureEnclosure,
    h: float,
    n_cells: int,
) -> tuple[float, float]:
    """Proven ``O(h²)`` interpolation-remainder bound per side over ``[lo, hi]``.

    Partitions the box into ``n_cells`` cells, takes a sound interval enclosure
    ``[d_lo, d_hi]`` of ``f''`` on each, and returns
    ``(under_remainder, over_remainder)``:

    * ``under_remainder = ⅛·max_cell(max(0, d_hi))·h²`` — the most a chord of ``f``
      can rise above ``f`` on a convex cell (bounds ``chord_f − f`` for the
      underestimator side);
    * ``over_remainder  = ⅛·max_cell(max(0, −d_lo))·h²`` — symmetric for the
      concave/overestimator side.

    A cell's ``f''`` enclosure encloses that of every sub-cell (monotone under
    subset), so bounding over ``n_cells`` cells and applying the fine spacing ``h``
    is a valid bound for every fine cell. Returns ``(inf, inf)`` if any enclosure is
    unbounded — the caller then abstains rather than ship an uncertified facet.
    """
    edges = np.linspace(lo, hi, n_cells + 1)
    d_hi_max = 0.0  # max(0, f''_hi) over cells → convex-side remainder
    d_lo_neg_max = 0.0  # max(0, -f''_lo) over cells → concave-side remainder
    for i in range(n_cells):
        d_lo, d_hi = curvature_enclosure(float(edges[i]), float(edges[i + 1]))
        if not (np.isfinite(d_lo) and np.isfinite(d_hi)):
            return float("inf"), float("inf")
        d_hi_max = max(d_hi_max, d_hi)
        d_lo_neg_max = max(d_lo_neg_max, -d_lo)
    c = 0.125 * h * h
    return c * d_hi_max, c * d_lo_neg_max


def univariate_hull_envelope(
    lo: float,
    hi: float,
    value_batch: Callable[[np.ndarray], np.ndarray],
    curvature_enclosure: CurvatureEnclosure,
    *,
    n_curv_cells: int = _CURV_CELLS,
) -> Optional[
    tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], tuple[float, float]]
]:
    """Exact 1-D outer band of ``f`` on ``[lo, hi]`` for a general (curvature-free) f.

    ``value_batch`` maps a 1-D numpy array of ``x`` values to ``f(x)`` values
    (vectorized; called exactly twice — once for the coarse hull seed, once for
    the fine correction grid — so per-node cost is two batched evaluations, not
    hundreds of thousands of scalar dispatches).

    ``curvature_enclosure`` maps a cell ``[a, b]`` to a SOUND interval enclosure
    ``(f''_lo, f''_hi)`` of ``f``'s second derivative on that cell (in the build,
    a sound interval Hessian). It is what makes the returned facets rigorous
    *between* the fine-grid nodes, not merely at them (#632 review finding 2); it
    is called ``n_curv_cells`` times.

    Returns ``(lower_lines, upper_lines, col_bounds)`` where ``lower_lines`` are
    proven underestimator secants (``≤ f`` everywhere on the box), ``upper_lines``
    proven overestimator secants (``≥ f`` everywhere), and ``col_bounds`` a sound
    box on the lifted value (widened outward by the proven hull slacks so it never
    excludes a true value). ``None`` to abstain (caller keeps the existing
    relaxation for this node).

    Soundness relies on: (a) the monotone-chain lower hull is a valid PL
    underestimator of its sample points; (b) the fine-node deviation shift covers
    the affine ``facet − chord`` part; and (c) the proven ``⅛·max(0,±f'')·h²``
    second-order remainder covers the chord-vs-``f`` interpolation error between
    nodes. No convexity of ``f`` is assumed anywhere.
    """
    if not (hi > lo):
        return None

    xs = np.linspace(lo, hi, _COARSE_N)
    xf = np.linspace(lo, hi, _FINE_MULT * _COARSE_N)
    try:
        ys = np.asarray(value_batch(xs), dtype=np.float64).ravel()
        ff = np.asarray(value_batch(xf), dtype=np.float64).ravel()
    except Exception:
        return None
    if ys.shape != xs.shape or ff.shape != xf.shape:
        return None
    if not (np.all(np.isfinite(ys)) and np.all(np.isfinite(ff))):
        return None
    f_range = float(np.max(ff) - np.min(ff))

    # Proven second-order remainder that certifies the facets between fine nodes.
    h_fine = (hi - lo) / (_FINE_MULT * _COARSE_N - 1)
    try:
        under_rem, over_rem = _second_order_remainder(
            lo, hi, curvature_enclosure, h_fine, n_curv_cells
        )
    except Exception:
        return None

    lower = _hull_lines(xs, ys, xf, ff, f_range, upper=False, remainder=under_rem)
    upper = _hull_lines(xs, ys, xf, ff, f_range, upper=True, remainder=over_rem)
    if lower is None or upper is None:
        return None

    # Column bounds: the aux value lies within [min underestimator floor over box,
    # max overestimator ceiling over box]. Both are proven bounds, so the box is
    # sound. Evaluate each line at the two endpoints (lines are affine → extremes
    # are at endpoints).
    def _line_span(slope: float, intercept: float) -> tuple[float, float]:
        a = slope * lo + intercept
        b = slope * hi + intercept
        return (min(a, b), max(a, b))

    col_lo = min(_line_span(s, b)[0] for s, b in lower)
    col_hi = max(_line_span(s, b)[1] for s, b in upper)
    if not (np.isfinite(col_lo) and np.isfinite(col_hi)) or col_hi < col_lo:
        return None
    return lower, upper, (float(col_lo), float(col_hi))
