"""Analytical 1-D convex/concave envelopes for single-variable subtrees (H-UNI).

Task ``cert:LR-2`` (``docs/dev/lever-a-root-tightness-plan.md`` §4/§0.1). The
composite-univariate path (:func:`milp_relaxation._collect_composite_univariate_relaxations`)
lifts a single-variable node ``f(x)`` only when its curvature is *certified*
convex/concave. Many tight-relaxation targets are neither: nvs09's per-variable
composite ``g(x) = (ln(x-2))**2 + (ln(10-x))**2`` on ``x∈[3,9]`` is
convex→concave→convex (two inflections) — its *composed* relaxation loses ~0.8 of
bound per variable (~8 total).

This module builds a proven polyhedral outer band of an arbitrary continuous
``f`` on ``[lo, hi]`` as a small set of secant/tangent lines ``(slope, intercept)``.

**No sampling (#632, lever-a §0.1).** An earlier revision approximated the hull by
sampling ``f`` on a 40 010-point grid and shifting facets by the *measured*
deviation — which is neither how SOTA factorable solvers (BARON/Couenne/SCIP) build
univariate envelopes, nor what the plan specified, and it only certified the facets
*at* the sample points. This construction is analytical instead:

1. **Verified curvature.** Partition ``[lo, hi]`` into pieces of definite ``f''``
   sign via a sound interval Hessian (``convexity.interval_ad.interval_hessian``),
   bisecting until each piece is sign-definite; tiny indefinite cells at inflections
   are tolerated; refuse (``None``) past a subdivision budget.
2. **Candidate lines.** Tangents at knots + chords (global + adjacent knot pairs).
3. **Rigorous global validity.** A line is kept only if ``line ≤ f`` (under) resp.
   ``≥ f`` (over) is *proven* on every piece: on a concave/convex piece the extreme
   of ``f − line`` is at an endpoint or the single interior tangency, bounded by a
   sound interval enclosure there; on indefinite cells by a direct interval bound.
   Each kept line is then shifted outward by the check tolerance, so it is a bound
   everywhere, not merely at tested points.

It is used only behind the default-OFF ``DISCOPT_UNIVARIATE_ENVELOPE`` flag.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

# Soundness margin. The validity check proves ``line ≤ f + _TOL`` (under) on each
# piece; shifting the accepted line outward by ``_TOL`` makes it ``≤ f`` everywhere,
# absorbing floating-point noise in the point/interval evaluations. Far below the
# solver's abs tolerance (1e-6), so tightness loss is immaterial.
_TOL = 1e-9
# Tangent knots across the box (more → tighter piecewise-linear band, more LP rows).
# Only the globally-valid ones survive (typically far fewer than this), and _thin
# caps the emitted count, so a generous knot set buys tightness cheaply.
_N_TANGENTS = 25
# Curvature-bisection budget: number of interval-Hessian sign probes before refusing.
# Sized to resolve legitimate multi-inflection targets (nvs09's convex→concave→convex
# needs ~191) while abstaining on pathological needles (a 1e4-sharp Gaussian needs
# ~2.2e4 probes) — those fall back to the composed relaxation. Cost is bounded.
_PIECE_BUDGET = 1000
# Cap on emitted lines per side (keeps the per-node LP small / well-conditioned).
_MAX_LINES = 32


def _flat_size(model) -> int:
    return int(sum(v.size for v in model._variables))


def _make_scalar_fns(expr, model, flat_idx: int):
    """Return ``(fv, fp)``: jitted value and derivative of ``f`` as a scalar
    function of coordinate ``flat_idx`` (other coordinates are irrelevant to a
    single-variable composite, so a zero base row is used)."""
    import jax
    import jax.numpy as jnp

    from discopt._jax.dag_compiler import compile_expression

    f = compile_expression(expr, model)
    n = _flat_size(model)

    def _scal(t):
        x = jnp.zeros(n).at[flat_idx].set(t)
        return jnp.reshape(f(x), ())

    _fj = jax.jit(_scal)
    _gj = jax.jit(jax.grad(_scal))
    fv = lambda t: float(_fj(jnp.asarray(float(t))))  # noqa: E731
    fp = lambda t: float(_gj(jnp.asarray(float(t))))  # noqa: E731
    return fv, fp


def _fpp_sign(expr, model, var, flat_idx: int, a: float, b: float) -> Optional[int]:
    """Sound curvature sign of ``f`` over ``[a, b]``: ``+1`` (convex, ``f'' ≥ 0``),
    ``-1`` (concave, ``f'' ≤ 0``), ``0`` (interval straddles 0), or ``None`` if the
    interval Hessian is unbounded (abstain)."""
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_ad import interval_hessian

    iad = interval_hessian(expr, model, {var: Interval(np.array([a]), np.array([b]))})
    h_lo = float(np.asarray(iad.hess.lo, dtype=float)[flat_idx, flat_idx])
    h_hi = float(np.asarray(iad.hess.hi, dtype=float)[flat_idx, flat_idx])
    if not (np.isfinite(h_lo) and np.isfinite(h_hi)):
        return None
    if h_lo >= 0.0:
        return 1
    if h_hi <= 0.0:
        return -1
    return 0


def _curvature_pieces(
    expr, model, var, flat_idx: int, lo: float, hi: float
) -> Optional[list[tuple[float, float, int]]]:
    """Partition ``[lo, hi]`` into ``(a, b, sign)`` pieces of definite ``f''`` sign,
    tolerating tiny indefinite cells (sign ``0``) at inflections. ``None`` if the
    curvature cannot be resolved within ``_PIECE_BUDGET`` probes (pathological f)."""
    min_w = (hi - lo) * 1e-4
    out: list[tuple[float, float, int]] = []
    budget = [_PIECE_BUDGET]

    def rec(a: float, b: float) -> bool:
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        s = _fpp_sign(expr, model, var, flat_idx, a, b)
        if s in (1, -1):
            out.append((a, b, s))
            return True
        if s is None:
            return False
        if (b - a) <= min_w:
            out.append((a, b, 0))  # inflection neighbourhood — measure-tiny
            return True
        m = 0.5 * (a + b)
        return rec(a, m) and rec(m, b)

    if not rec(lo, hi):
        return None
    # merge adjacent same-sign pieces
    merged = [out[0]]
    for a, b, s in out[1:]:
        pa, pb, ps = merged[-1]
        if s == ps:
            merged[-1] = (pa, b, s)
        else:
            merged.append((a, b, s))
    return merged


def _root_slope(fp: Callable[[float], float], m: float, a: float, c: float, it: int = 60):
    """Bisect for ``x*`` in ``[a, c]`` with ``f'(x*) = m`` (the interior extremum of
    ``f − line``). Returns ``None`` if ``m`` is outside ``[f'(a), f'(c)]`` (extremum
    at an endpoint)."""
    ga, gc = fp(a) - m, fp(c) - m
    if ga == 0.0:
        return a
    if gc == 0.0:
        return c
    if ga * gc > 0.0:
        return None
    for _ in range(it):
        mid = 0.5 * (a + c)
        gm = fp(mid) - m
        if ga * gm <= 0.0:
            c = mid
        else:
            a, ga = mid, gm
    return 0.5 * (a + c)


def _make_interval_f(expr, model, var):
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_eval import evaluate_interval

    def f_enclosure(a: float, b: float) -> tuple[float, float]:
        iv = evaluate_interval(expr, model, {var: Interval(np.array([a]), np.array([b]))})
        return float(np.asarray(iv.lo).reshape(-1)[0]), float(np.asarray(iv.hi).reshape(-1)[0])

    return f_enclosure


def _line_valid(
    m: float,
    b: float,
    pieces: list[tuple[float, float, int]],
    fv: Callable[[float], float],
    fp: Callable[[float], float],
    f_encl: Callable[[float, float], tuple[float, float]],
    *,
    under: bool,
) -> bool:
    """Prove ``m*x + b ≤ f`` (``under``) resp. ``≥ f`` on every piece of ``[lo, hi]``.

    The extreme of ``d(x) = f(x) − (m*x + b)`` is located from the piece curvature
    (endpoints for a piece where ``d`` is monotone/curving away, the single interior
    tangency ``f'=m`` otherwise) and bounded there by a **sound interval enclosure**
    of ``f`` over a tiny neighbourhood — so the bound is rigorous, not a point
    sample. Indefinite cells are bounded by a direct interval enclosure.
    """

    def line(x: float) -> float:
        return m * x + b

    for a, c, s in pieces:
        la, lc = line(a), line(c)
        if s == 0:
            f_lo, f_hi = f_encl(a, c)
            if under:
                if f_lo - max(la, lc) < -_TOL:
                    return False
            else:
                if min(la, lc) - f_hi < -_TOL:
                    return False
            continue
        # d = f - line. d'' has sign s. For under we need min d >= -tol; for over we
        # need max d <= tol i.e. min(-d) >= -tol.
        # Where is the relevant extreme of d on [a,c]?
        #   under & convex(s=1):  d convex  -> min at interior tangency (f'=m) or ends
        #   under & concave(s=-1):d concave -> min at an endpoint
        #   over  & concave(s=-1):d concave -> max at interior tangency or ends
        #   over  & convex(s=1):  d convex  -> max at an endpoint
        interior = (under and s == 1) or ((not under) and s == -1)
        # endpoint residuals (point evals; tol absorbs fp noise)
        if under:
            worst = min(fv(a) - la, fv(c) - lc)
        else:
            worst = min(la - fv(a), lc - fv(c))
        if interior:
            # d = f - line is convex (under) / concave (over) on this piece, so its
            # extreme is the single interior stationary point x* where f'(x*) = m.
            # The bisection converges x* to ~machine precision, and d is smooth there,
            # so the point residual equals the true extreme up to ~½·d''·(Δx*)² —
            # far below _TOL — making the point eval a sound bound.
            xs = _root_slope(fp, m, a, c)
            if xs is not None:
                d = (fv(xs) - line(xs)) if under else (line(xs) - fv(xs))
                worst = min(worst, d)
        if worst < -_TOL:
            return False
    return True


def _col_bounds(
    lowers: list[tuple[float, float]],
    uppers: list[tuple[float, float]],
    lo: float,
    hi: float,
) -> Optional[tuple[float, float]]:
    def span(m: float, b: float) -> tuple[float, float]:
        u, v = m * lo + b, m * hi + b
        return (min(u, v), max(u, v))

    col_lo = min(span(m, b)[0] for m, b in lowers)
    col_hi = max(span(m, b)[1] for m, b in uppers)
    if not (np.isfinite(col_lo) and np.isfinite(col_hi)) or col_hi < col_lo:
        return None
    return float(col_lo), float(col_hi)


def univariate_hull_envelope(
    expr,
    model,
    var,
    flat_idx: int,
    lo: float,
    hi: float,
) -> Optional[
    tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], tuple[float, float]]
]:
    """Proven 1-D outer band of the single-variable composite ``expr`` on ``[lo, hi]``.

    ``expr`` must depend on exactly one scalar variable ``var`` (flat index
    ``flat_idx``). Returns ``(lower_lines, upper_lines, col_bounds)`` with
    ``lower_lines`` proven underestimator secants (``≤ f`` everywhere on the box),
    ``upper_lines`` proven overestimators (``≥ f``), and a sound box on the lifted
    value. ``None`` to abstain (degenerate box, unbounded/unresolvable curvature, or
    no useful line survives) — the caller keeps the existing composed relaxation.
    """
    if not (hi > lo):
        return None
    try:
        pieces = _curvature_pieces(expr, model, var, flat_idx, lo, hi)
        if pieces is None:
            return None
        fv, fp = _make_scalar_fns(expr, model, flat_idx)
        f_encl = _make_interval_f(expr, model, var)

        knots = np.linspace(lo, hi, _N_TANGENTS)
        cand: list[tuple[float, float]] = []
        for t in knots:  # tangent lines
            tf = float(t)
            m = fp(tf)
            cand.append((m, fv(tf) - m * tf))
        # chords: global endpoints + each adjacent knot pair (captures concave secants)
        pair_ids = [(0, _N_TANGENTS - 1)] + [(i, i + 1) for i in range(_N_TANGENTS - 1)]
        for i, j in pair_ids:
            x0, x1 = float(knots[i]), float(knots[j])
            if x1 <= x0:
                continue
            m = (fv(x1) - fv(x0)) / (x1 - x0)
            cand.append((m, fv(x0) - m * x0))

        lowers: list[tuple[float, float]] = []
        uppers: list[tuple[float, float]] = []
        for m, b in cand:
            if not (np.isfinite(m) and np.isfinite(b)):
                continue
            if _line_valid(m, b, pieces, fv, fp, f_encl, under=True):
                lowers.append((m, b - _TOL))  # shift down: proven ≤ f everywhere
            if _line_valid(m, b, pieces, fv, fp, f_encl, under=False):
                uppers.append((m, b + _TOL))  # shift up: proven ≥ f everywhere
    except Exception:
        return None

    if not lowers or not uppers:
        return None
    lowers = _thin(lowers, lo, hi, upper=False)
    uppers = _thin(uppers, lo, hi, upper=True)
    cb = _col_bounds(lowers, uppers, lo, hi)
    if cb is None:
        return None
    return tuple(lowers), tuple(uppers), cb


def _thin(
    lines: list[tuple[float, float]], lo: float, hi: float, *, upper: bool
) -> list[tuple[float, float]]:
    """Drop dominated lines and cap the count, keeping the tightest band.

    A line is kept only if it is the tightest (max for lower, min for upper) at ``lo``
    or ``hi`` or the midpoint, then the strongest ``_MAX_LINES`` by area under/over the
    band are retained. Dropping a line only *loosens* the (still sound) band."""
    if len(lines) <= 1:
        return lines
    g = np.linspace(lo, hi, 513)
    vals = np.array([m * g + b for m, b in lines])  # (n_lines, n_grid)
    # A line is kept only if it is active (tightest) at some grid point — i.e. it
    # lies on the upper envelope (lower) / lower envelope (upper). Dominated lines
    # never bind and are dropped; the band is unchanged.
    active = np.argmax(vals, axis=0) if not upper else np.argmin(vals, axis=0)
    counts: dict[int, int] = {}
    for i in active.tolist():
        counts[i] = counts.get(i, 0) + 1
    keep = sorted(counts, key=lambda i: -counts[i])
    if len(keep) > _MAX_LINES:
        keep = keep[:_MAX_LINES]  # retain the widest-binding facets
    return [lines[i] for i in sorted(keep)]
