"""Incremental McCormick LP for the LP-node spatial branch-and-bound engine.

``build_milp_relaxation`` re-walks the expression DAG on every call (~14 ms on
nvs17) even though, across a spatial-B&B tree, the LP *structure* (columns, row
sparsity, the fixed linear/model rows, the objective) is identical for every box —
only the McCormick envelope rows and the auxiliary-variable bounds depend on the
node box. This class builds the structure **once** and per node recomputes only the
box-dependent rows/bounds in closed form (numpy, ~0.1 ms), then warm-starts the LP
from the parent basis. That removes both the per-node DAG re-walk and the cold LP
solve — the throughput needed to close the nvs17/19/24 family by branching (the
no-cut-SCIP regime: thousands of fast nodes).

**Soundness gate.** The closed-form envelopes are validated to reproduce
``build_milp_relaxation`` exactly (row-set and bounds, to tolerance) on random
boxes at construction. If validation fails — an unhandled term type, a different
discretization, anything — :attr:`ok` is ``False`` and the caller falls back to the
trusted per-node builder. The incremental path can therefore never change a result,
only its speed.

A term's envelope rows are identified NUMERICALLY (each closed-form row must match
exactly one candidate on the probe box), not by support containment alone: a lifted
*model constraint* over the same variables has the same support — ``x0*x1 >= 60``
becomes ``{aux: -1} <= -60`` — and counting it as an envelope row declined the whole
structure (#861). Unmatched candidates are box-independent and stay unpatched.

Scope: the box-dependent terms it regenerates are bilinear products ``w=x_i*x_j``
(4 McCormick rows), integer powers ``s=x_i**p`` (secant + 3 tangents, matching an
empty ``DiscretizationState``) and affine squares ``w=(c*x_j+d)**2``. A monomial is
mapped on any box when ``p`` is even (convex on all of R) and on a sign-definite box
when ``p`` is odd; an odd power whose ROOT box straddles zero is declined, because
there the envelope's facet *count* changes with the node's sign regime. Any other
lifted term (trilinear, univariate, fractional power, piecewise) makes validation
fail -> fallback.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import scipy.sparse as sp

from discopt._jax.outward_rounding import (
    envelope_1d_slack,
    envelope_product_slack,
    widen,
)

logger = logging.getLogger(__name__)

_TOL = 1e-7

# Nonzero-footprint budget for the incremental structure. The fast path now holds
# ``base_A`` SPARSE (CSR) and ``_patch`` copies only its ``.data`` array (~nnz
# floats) per node, so the footprint is ``O(nnz)``, not ``O(rows*cols)`` — the
# earlier dense ``base_A`` (~14.85 GB on qap, ~30 GB peak) is gone. This cap only
# guards a pathologically dense lift whose per-node ``.data`` copy would itself be
# large; above it the incremental path is declined (``ok=False``) and
# ``solve_at_node`` uses the per-node SPARSE cold build. Sound either way: the
# structure is only an accelerator whose rows are validated bit-identical to the
# cold build (``_validate``), so declining changes speed, never the bound. 5e7
# nonzeros ~ 0.4 GB of f64 data — far above qap's ~172k nnz and any well-posed
# lift, far below a memory-thrashing structure.
_MAX_INCREMENTAL_NNZ = 50_000_000

# Warn-once keys for oversize declines, so a large model logs a single line
# rather than one per relaxer construction.
_incremental_oversize_warned: set = set()


class _IncrementalStructureTooLarge(Exception):
    """The lifted relaxation is too large to hold densely in the incremental
    fast path; the caller falls back to the sparse per-node cold build."""


def _bilinear_rows(i, j, a, li, ui, lj, uj):
    """The 4 McCormick inequalities for w=x_i*x_j over [li,ui]x[lj,uj], each as
    ``(coeff_on_i, coeff_on_j, coeff_on_w, rhs)`` of an ``... <= rhs`` row.

    Each rhs carries the #956 outward guard, computed by the shared helper from
    the same enclosures ``_emit_mccormick`` uses, so the cold build and this patch
    stay bit-identical (see :mod:`discopt._jax.outward_rounding`).
    """
    # (coeff_on_i, coeff_on_j, coeff_on_w, rhs, coef_a, coef_b) — `coef_a`/`coef_b`
    # are the row's multipliers on the two factors, which is what sizes the guard.
    spec = [
        (lj, li, -1.0, li * lj, lj, li),  # w >= lj*xi + li*xj - li*lj
        (uj, ui, -1.0, ui * uj, uj, ui),  # w >= uj*xi + ui*xj - ui*uj
        (-uj, -li, 1.0, -li * uj, uj, li),  # w <= uj*xi + li*xj - li*uj
        (-lj, -ui, 1.0, -ui * lj, lj, ui),  # w <= lj*xi + ui*xj - ui*lj
    ]
    return [
        (ci, cj, cw, rhs + envelope_product_slack(ca, cb, li, ui, lj, uj, 0.0, 0.0, rhs))
        for ci, cj, cw, rhs, ca, cb in spec
    ]


def _square_rows(i, a, li, ui):
    """The 3 rows for s=x_i**2 over [li,ui]: 2 endpoint tangents + 1 secant, each
    ``(coeff_on_i, coeff_on_s, rhs)`` of an ``... <= rhs`` row."""
    fl, fu = li * li, ui * ui
    spec = [
        (2.0 * li, -1.0, fl, 2.0 * li),  # s >= 2*li*xi - li^2  (tangent at li)
        (2.0 * ui, -1.0, fu, 2.0 * ui),  # s >= 2*ui*xi - ui^2  (tangent at ui)
        (-(li + ui), 1.0, -li * ui, li + ui),  # s <= (li+ui)*xi - li*ui (secant)
    ]
    return [
        (cx, cs, rhs + envelope_1d_slack(d, li, ui, fl, fu, 0.0, rhs)) for cx, cs, rhs, d in spec
    ]


def _bilinear_aux_bounds(li, ui, lj, uj):
    corners = (li * lj, li * uj, ui * lj, ui * uj)
    lo, hi = min(corners), max(corners)
    # #956: widened outward so a corner product can never fall outside the aux
    # column's own bounds by a rounding.
    return widen(lo, hi)


def _square_aux_bounds(li, ui):
    if li >= 0:
        lo, hi = li * li, ui * ui
    elif ui <= 0:
        lo, hi = ui * ui, li * li
    else:
        lo, hi = 0.0, max(li * li, ui * ui)
    # #956: widened outward (see `_bilinear_aux_bounds`).
    return widen(lo, hi)


def _monomial_rows(li, ui, p):
    """The 4 envelope rows for ``s = x**p`` over ``[li,ui]`` (secant + tangents at
    ``li``, the midpoint, and ``ui``), each ``(coeff_on_x, coeff_on_s, rhs)`` of an
    ``... <= rhs`` row. Generalizes :func:`_square_rows` (p=2) to any integer power
    ``p >= 2``.

    Validity domain. ``x**p`` has *single* curvature — hence a secant/tangent
    envelope — on:

    * any finite box when ``p`` is EVEN (``f'' = p(p-1)x**(p-2) >= 0`` everywhere,
      so the atom is convex on all of R including boxes that straddle zero); or
    * a sign-definite box when ``p`` is odd (convex for ``li >= 0``, concave for
      ``ui <= 0``).

    In the convex case the three tangents underestimate and the secant
    overestimates; in the concave case the roles flip. On a sign-STRADDLING box with
    ODD ``p`` the atom is S-shaped and neither branch is valid — the caller must not
    reach here (``_build_structure`` declines that monomial; the cold build emits a
    different, 2-facet hull there instead).

    Matches the uniform engine's ``_emit_1d`` envelope exactly — which underestimates
    a convex atom with tangents at both endpoints AND the box midpoint (a tighter,
    3-tangent hull), so the closed-form patch must emit the same midpoint tangent to
    reproduce the cold build row-for-row (validated).
    """
    # Unbounded box: there is no secant and no finite tangent point, and evaluating
    # the formulae anyway yields NaN coefficients (``inf - inf`` in the tangent
    # intercept, ``0 * inf`` in the slope) that flow straight into ``A``/``b``. NaN in
    # an LP row is strictly worse than a missing row — comparisons against it are all
    # False, so it can silently disable fathoming rather than merely loosen a bound.
    # Emit four VACUOUS rows (``0 <= 0``) instead, which is exactly what the cold
    # build's polytope is here: ``_emit_1d`` bails on ``not _finite(lo, hi)`` and
    # emits no envelope rows at all, leaving the aux interval bound as the whole
    # relaxation. So this makes the patch AGREE with the cold build on a box where it
    # previously disagreed, and ``_rowset``'s vacuity filter drops these rows on the
    # patched side exactly as the cold side has none. (Latent on main too, and not
    # reachable from the newly-admitted class — every validation box is finite — but
    # it is the same non-finite-endpoint defect as the aux enclosure below, found by
    # the regression test written for it.)
    if not (math.isfinite(li) and math.isfinite(ui)):
        return [(0.0, 0.0, 0.0)] * 4
    mid = 0.5 * (li + ui)
    fl, fm, fu = li**p, mid**p, ui**p
    dfl, dfm, dfu = p * li ** (p - 1), p * mid ** (p - 1), p * ui ** (p - 1)
    # Degenerate box (variable fixed by integer branching, li==ui): the exact value
    # is fl and the secant slope is 0/0. Use the endpoint derivative so the "secant"
    # collapses to the tangent at the pinned point (s <= fl at x=li, matching the
    # tangent's s >= fl) — sound and NaN-free. Guarded on EXACT zero width only: for
    # ANY positive width the true secant is the sound convex overestimator, whereas
    # ``dfl`` would under-cut it. Mirrors the cold build, which emits no envelope
    # rows at a fixed variable and pins the aux via its [fl,fl] bound.
    slope = dfl if ui <= li else (fu - fl) / (ui - li)
    convex = (p % 2 == 0) or (li >= 0.0)
    if convex:
        spec = [
            (dfl, -1.0, dfl * li - fl, dfl),  # tangent at li: s >= f'(li)*(x-li)+f(li)
            (dfm, -1.0, dfm * mid - fm, dfm),  # tangent at midpoint
            (dfu, -1.0, dfu * ui - fu, dfu),  # tangent at ui
            (-slope, 1.0, fl - slope * li, slope),  # secant (overestimator): s <= ...
        ]
    else:
        spec = [
            (-dfl, 1.0, fl - dfl * li, dfl),  # tangent at li (overestimator): s <= ...
            (-dfm, 1.0, fm - dfm * mid, dfm),  # tangent at midpoint
            (-dfu, 1.0, fu - dfu * ui, dfu),  # tangent at ui
            (slope, -1.0, slope * li - fl, slope),  # secant (underestimator): s >= ...
        ]
    # #956: each rhs is a cancelling combination of box-endpoint quantities
    # (`f'(x0)*x0 - f(x0)`), so it is relaxed outward by the shared ulp-scaled
    # guard — computed from the same `t`-space quantities `_emit_1d` has, which is
    # what keeps the cold build and this patch bit-identical under `_validate`.
    # `t == x` here (the base is the bare variable), so `cst` is 0.
    return [
        (cx, cs, rhs + envelope_1d_slack(d, li, ui, fl, fu, 0.0, rhs)) for cx, cs, rhs, d in spec
    ]


def _monomial_aux_bounds(li, ui, p):
    """Enclosure of ``x**p`` over ``[li,ui]``, matching the cold build's aux column.

    The cold build takes this bound from ``evaluate_interval``, i.e. from
    ``Interval.__pow__``, so this closed form must reproduce *that* — not merely a
    sound enclosure — or the incremental rows describe a different polytope than the
    path they must be bound-neutral against (``_validate`` compares aux bounds too).
    ``Interval.__pow__`` is:

    * ``p == 2``: the exact square image ``[0, max(li^2,ui^2)]`` when the box
      straddles zero, else ``[min, max]`` of the endpoint squares;
    * ``p >= 3``: repeated interval MULTIPLICATION by ``[li,ui]`` (``p-1`` times),
      whose corner min/max is reproduced below. That is deliberately *looser* than
      the exact image on a straddling box (e.g. ``x**4`` over ``[-2,3]`` encloses
      ``[-54, 81]``, not ``[0, 81]``) — matching it exactly is what keeps the fast
      path bound-NEUTRAL. Do not "tighten" this: a tighter aux bound here is a
      different relaxation, which is precisely what the incremental path may not be.

    A NON-FINITE endpoint delegates to ``Interval.__pow__`` itself rather than being
    reproduced here, because the closed form below cannot match it there. Two ways it
    diverges, both load-bearing:

    * ``0 * ±inf`` is ``NaN`` in IEEE and a bare ``min``/``max`` propagates it,
      collapsing the enclosure to ``[NaN, NaN]``. That is worse than a wide bound: it
      reaches the LP, and ``NaN <= incumbent`` is always ``False``, so it can
      silently disable fathoming. ``Interval.__mul__`` maps that corner to 0 (C-36 /
      #723).
    * ``Interval`` outward-rounds after *every* step, and on an unbounded box that
      changes the answer rather than the last ulp: ``[0,inf]**2`` rounds its lower
      end to ``-5e-324``, which at the next multiply gives ``-inf`` instead of a
      NaN-to-zero corner — so ``[0,inf]**3`` is ``[-inf, inf]``, not ``[0, inf]``.
      Reporting the tighter one would make the patched aux TIGHTER than the cold
      build, which this function's whole contract forbids.

    Delegating costs ~65 us versus ~2 us for the loop, so it is confined to the
    non-finite case; a finite box (every real node box, and every box the corpus
    exercises) keeps the fast path, where the loop reproduces ``Interval.__pow__``
    to well within the comparison tolerance (its per-step rounding is a 1-ulp
    effect there, not a change of value).

    The old form (``min/max`` of the two endpoint powers) was correct only on a
    sign-definite box, where ``x**p`` is monotone; it was UNSOUND on a straddling box
    for even ``p`` (it would floor ``x**2`` at ``min(li^2,ui^2) > 0`` and cut off the
    true point ``x=0``). That was unreachable while every monomial was gated to a
    sign-definite root box, and is the reason the bound had to be generalized before
    the gate could be relaxed (#861). ``test_monomial_aux_bounds_match_interval_pow``
    pins the parity with ``Interval.__pow__`` across powers and sign regimes,
    half-infinite and doubly-infinite boxes included.
    """
    if not (math.isfinite(li) and math.isfinite(ui)):
        from discopt._jax.convexity.interval import Interval

        enc = Interval.from_bounds(np.float64(li), np.float64(ui)) ** int(p)
        return float(enc.lo), float(enc.hi)
    if p == 2:
        return _square_aux_bounds(li, ui)  # already widened there
    lo, hi = li, ui
    for _ in range(p - 1):
        corners = (lo * li, lo * ui, hi * li, hi * ui)
        lo, hi = min(corners), max(corners)
    # #956: widened outward. This stays within the `rel=1e-9` parity the cold
    # build's `Interval.__pow__` is pinned to (the guard is ~3.6e-15 relative),
    # and `Interval` already outward-rounds every step of its own product — this
    # is the same treatment applied to the closed form that reproduces it.
    return widen(lo, hi)


def _affine_square_rows(coeff, const, li, ui):
    """The 4 envelope rows for ``w = (coeff*x + const)**2`` over ``x in [li,ui]``,
    each ``(coeff_on_x, coeff_on_w, rhs)`` of an ``... <= rhs`` row.

    Let ``t = coeff*x + const`` range over ``[t_lo, t_hi]``. ``t**2`` is convex for
    every ``t`` (no sign gating), so the hull is the secant overestimator plus
    tangents at ``t_lo``, the midpoint, and ``t_hi`` — matching the uniform engine's
    ``_emit_1d`` on the affine base LinForm exactly (validated)."""
    tl, tu = coeff * li + const, coeff * ui + const
    t_lo, t_hi = (tl, tu) if tl <= tu else (tu, tl)
    mid = 0.5 * (t_lo + t_hi)
    # t_hi + t_lo == (t_hi**2 - t_lo**2)/(t_hi - t_lo); at a degenerate base box
    # (t_lo == t_hi) it already equals 2*t_lo == f'(t_lo), so no divide-by-zero
    # guard is needed here (unlike the general power secant).
    slope = t_hi + t_lo
    a = t_lo * t_lo - slope * t_lo
    f_lo, f_hi = t_lo * t_lo, t_hi * t_hi
    spec = [
        (-slope * coeff, 1.0, a + slope * const, slope),  # secant (overestimator)
        (2.0 * t_lo * coeff, -1.0, f_lo - 2.0 * t_lo * const, 2.0 * t_lo),  # tangent @ t_lo
        (2.0 * mid * coeff, -1.0, mid * mid - 2.0 * mid * const, 2.0 * mid),  # tangent @ mid
        (2.0 * t_hi * coeff, -1.0, f_hi - 2.0 * t_hi * const, 2.0 * t_hi),  # tangent @ t_hi
    ]
    # #956: outward guard in `t`-space, matching `_emit_1d` term for term.
    return [
        (cx, cw, rhs + envelope_1d_slack(d, t_lo, t_hi, f_lo, f_hi, const, rhs))
        for cx, cw, rhs, d in spec
    ]


def _affine_square_aux_bounds(coeff, const, li, ui):
    """min/max of ``(coeff*x+const)**2`` over ``x in [li,ui]`` (0 if the base
    straddles zero)."""
    tl, tu = coeff * li + const, coeff * ui + const
    t_lo, t_hi = (tl, tu) if tl <= tu else (tu, tl)
    return _square_aux_bounds(t_lo, t_hi)


class IncrementalMcCormickLP:
    """Build the McCormick LP structure once; patch box-dependent rows per node."""

    def __init__(self, model, terms, deadline=None, box=None):
        """``deadline`` — absolute ``time.perf_counter()`` budget for construction.

        Overrides the ambient ``model._solve_deadline`` stash. Pass it whenever the
        caller has a budget of its own that is not the enclosing ``solve_model``'s:
        ``_solve_deadline`` is written once per ``solve_model`` call and never
        cleared, so a *later* in-process consumer reads an already-expired deadline
        and silently declines to build (#844).

        ``box`` — the caller's ROOT box ``(lb, ub)`` over the original columns,
        normally the *presolved* one (post-FBBT/OBBT). Both the probe box and the
        validation boxes are generated inside it (#861), so they describe boxes the
        B&B tree can actually reach. Defaults to the model's declared bounds.

        Passing the presolved box matters for two reasons beyond realism. A model
        whose declared box is unbounded (``ex1233``: 28 infinite bounds) or whose
        raw-box relaxation has no valid objective bound (``st_e04``) cannot be
        judged at all from the declared bounds — the structure build fails before
        anything can be compared. And a tightened box is the one the tree actually
        branches in, so anchoring to it makes ``_validate``'s comparison boxes
        reachable rather than hypothetical.
        """
        self.ok = False
        self.model = model
        self.terms = terms
        self._box = None
        if box is not None:
            _blb = np.asarray(box[0], dtype=np.float64).ravel()
            _bub = np.asarray(box[1], dtype=np.float64).ravel()
            if _blb.size == _bub.size == len(model._variables):
                self._box = (_blb.copy(), _bub.copy())
        # Why this structure declined, or ``None`` when it was admitted (#861). The
        # reason used to exist ONLY as the ``logger.debug`` line below, so a caller
        # measuring coverage had to scrape a log to learn anything — and
        # ``getattr(inc, "reason", None)`` returning ``None`` for a missing attribute
        # reads as "declined for no reason", which cost a wrong triage pass. Storing
        # it makes the decline reason a first-class, measurable property; nothing in
        # the solve path branches on it (it is diagnostic only).
        self.decline_reason: str | None = None
        self._validated_regimes = frozenset()  # sign regimes _validate exercised
        self._deadline = (
            deadline if deadline is not None else getattr(model, "_solve_deadline", None)
        )
        # #654: if the solve budget is already spent when we get here, don't build
        # the incremental structure at all (its own cold builds cost seconds on
        # large factorable models) — leave ``ok=False`` and let ``solve_at_node``
        # use the trusted per-node cold build. No-op when budget remains (the common
        # case) or when no deadline was stashed (construction outside a solve).
        if self._deadline is not None and time.perf_counter() > self._deadline:
            self.decline_reason = "deadline already spent before construction"
            return
        try:
            self._build_structure()
            self._validate()
        except Exception as exc:
            # Declining is always SOUND (the caller falls back to the per-node cold
            # build) but it is never free: it costs the fast path, the cuts and the
            # feasibility pump. A silently-declined structure is exactly how #844's
            # overshoot hid for two rounds of measurement, so record why.
            self.ok = False
            self.decline_reason = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "IncrementalMcCormickLP declined to build: %s: %s", type(exc).__name__, exc
            )

    # -- construction ------------------------------------------------------ #

    def _full_build(self, lb, ub):
        from discopt._jax.discretization import DiscretizationState
        from discopt._jax.milp_relaxation import build_milp_relaxation

        # Skip the separable objective-floor cut (#640 Bucket 1): it is a
        # box-dependent OBJECTIVE row (support = the objective columns) that the
        # closed-form patch does not regenerate, so including it would break the
        # row-for-row match. Omitting it only *loosens* the incremental relaxation
        # (a valid lower bound is dropped, never invented), so the fast-path bound
        # stays sound — and never tighter than the cold path, which keeps it.
        relax, info = build_milp_relaxation(
            self.model,
            self.terms,
            DiscretizationState(),
            bound_override=(lb, ub),
            skip_separable_floor=True,
            skip_convex_lift=True,
        )
        if not relax._objective_bound_valid or relax._A_ub is None:
            raise ValueError("relaxation has no valid bound / no rows")
        # SPARSE: keep the lifted constraint matrix as CSR — never densify. The
        # structure holds it sparse (``_build_structure``) and ``_patch`` rewrites
        # only the box-dependent product-row *values* in place (fixed sparsity
        # pattern), so the footprint is O(nnz), not O(rows*cols). (Before this the
        # matrix was ``.todense()``'d, ~14.85 GB per copy on qap's 85756x21649 lift,
        # which forced the whole structure to be declined for large lifts — T6/T10.)
        A = sp.csr_matrix(relax._A_ub, dtype=np.float64)
        A.sort_indices()
        b = np.asarray(relax._b_ub, dtype=np.float64).ravel()
        bnds = np.asarray(relax._bounds, dtype=np.float64)
        c = np.asarray(relax._c, dtype=np.float64).ravel()
        return A, b, bnds, c, info, relax

    def _root_box(self):
        """The root box every probe/validation box is generated inside: the caller's
        (presolved) ``box`` when given, else the model's declared bounds."""
        if self._box is not None:
            return self._box[0].copy(), self._box[1].copy()
        return (
            np.array([float(np.min(v.lb)) for v in self.model._variables]),
            np.array([float(np.max(v.ub)) for v in self.model._variables]),
        )

    @staticmethod
    def _finite_root_interval(rl, ru, i):
        """``(lo, hi)`` finite stand-in for a root interval with an infinite end.

        The probe box needs finite, distinct, sign-matched endpoints to expose each
        product's row support; an infinite endpoint cannot serve. Substitute a finite
        window ANCHORED at the finite end (so the probe still sits inside the reachable
        region on that side) and fall back to the historical synthetic magnitudes only
        when BOTH ends are infinite. This is a structure-identification device, not a
        bound: the layout it reveals is box-independent, and ``_validate`` re-checks
        every row against a cold build on the real boxes."""
        lo_inf, hi_inf = not math.isfinite(rl), not math.isfinite(ru)
        if lo_inf and hi_inf:
            return 1.0, 7.0 + i
        if lo_inf:
            return ru - (8.0 + i), ru
        if hi_inf:
            return rl, rl + (8.0 + i)
        return rl, ru

    def _probe_box(self):
        """The box the STRUCTURE is identified on — now generated inside the root box.

        Requirements, unchanged: strictly *sign-matched* to each variable's root sign
        regime (so the cached convex/concave power rows match the cold build's regime)
        and with distinct endpoints so every McCormick coefficient is nonzero and a
        product's row support reveals ``{factors, aux}`` cleanly.

        What changed (#861): the endpoints are now interior points of the model's real
        root interval instead of the synthetic ``[1, 7+k]`` / ``[-(7+k), -1]``. The
        synthetic box is frequently UNREACHABLE — on ``gear`` the root box is
        ``[12,60]^4`` and the probe ran on ``[1,7+k]``, where the engine takes a
        different decomposition route for the ratio ``x0*x1/(x2*x3)`` (log/reciprocal
        columns), so the structure was built over a 15-column layout while every real
        node builds 10 and ``_validate`` reported a ``column-count mismatch``. That was
        the single largest decline bucket after the row-classifier fix.

        The inset keeps the probe strictly interior (no endpoint sits exactly on a root
        bound, and no coordinate is 0) so coefficients stay nonzero; a degenerate root
        interval (``lb == ub``, an already-fixed variable) is passed through as the
        point it is."""
        root_lb, root_ub = self._root_box()
        n = root_lb.size
        lb_p = np.empty(n)
        ub_p = np.empty(n)
        for k in range(n):
            rl, ru = self._finite_root_interval(float(root_lb[k]), float(root_ub[k]), k)
            if not (ru > rl):  # pinned at the root — nothing to probe with
                lb_p[k] = ub_p[k] = rl
                continue
            w = ru - rl
            # Distinct per-variable insets keep the endpoints from lining up across
            # variables (which would make two products' rows numerically identical and
            # ambiguous to the row matcher).
            lo = rl + (0.11 + 0.017 * (k % 7)) * w
            hi = ru - (0.13 + 0.013 * (k % 5)) * w
            if not (hi > lo):  # very narrow interval: fall back to the raw endpoints
                lo, hi = rl, ru
            # Keep the probe sign-matched to the ROOT regime. A spanning root gets the
            # positive side (matching the historical probe, which used a positive box
            # for every non-negative-definite variable) so an even power's cached rows
            # are built in the same regime the cold build uses there.
            if self._root_sign[k] < 0:
                hi = min(hi, -_TOL * (1.0 + abs(hi)))
                lo = min(lo, hi - 1e-6 * (1.0 + abs(hi)))
            else:
                lo = max(lo, _TOL * (1.0 + abs(lo)))
                hi = max(hi, lo + 1e-6 * (1.0 + abs(lo)))
            lb_p[k], ub_p[k] = lo, hi
        return lb_p, ub_p

    def _build_structure(self):
        n = len(self.model._variables)
        # Per-variable ROOT sign regime (cert:T1.2). ``+1`` = ``lb>=0``, ``-1`` =
        # ``ub<=0``, ``0`` = spans zero. Branching only shrinks boxes, so a
        # sign-definite root sign holds at every node of the subtree; a spanning root
        # reaches BOTH regimes below it. An ODD-power monomial's envelope changes
        # *shape* across that split — 4 secant/tangent rows on a sign-definite box vs
        # the 2-facet S-hull on a straddling one — which a fixed sparsity pattern
        # cannot express, so those stay unmappable (#861). An EVEN power is convex on
        # all of R, so its envelope is the same 4 rows in every regime and it is
        # mapped regardless of root sign.
        root_lb, root_ub = self._root_box()
        self._root_sign = np.where(root_lb >= 0.0, 1, np.where(root_ub <= 0.0, -1, 0))
        lb_p, ub_p = self._probe_box()
        self._probe_lb, self._probe_ub = lb_p, ub_p
        A, b, bnds, c, info, _relax_probe = self._full_build(lb_p, ub_p)  # A is CSR (sparse)
        # Decline only genuinely huge structures — now measured by NONZEROS (the
        # sparse footprint), not dense cells. qap's lift is ~172k nnz (trivial);
        # this guards a pathological lift whose sparse `.data` copy per node would
        # itself be large. Above it, fall back to the per-node cold build.
        if A.nnz > _MAX_INCREMENTAL_NNZ:
            if A.nnz not in _incremental_oversize_warned:
                _incremental_oversize_warned.add(A.nnz)
                logger.info(
                    "Incremental McCormick structure declined: lift has %d nonzeros "
                    "> cap %d; using the per-node sparse cold build instead.",
                    A.nnz,
                    _MAX_INCREMENTAL_NNZ,
                )
            raise _IncrementalStructureTooLarge(f"lift nnz {A.nnz} exceeds budget")
        self.n = n
        self.ncol = A.shape[1]
        self.c = c
        # Constant term of the (minimize-equivalent) relaxation objective. The LP
        # solved here is ``min c·x``, but the relaxation's objective value is
        # ``c·x + obj_offset`` — the cold path adds it (``MilpRelaxationModel.solve``),
        # so a bound returned WITHOUT it is not on the same scale as the cold build's,
        # and is not a valid bound at all when the offset is negative (measured: a
        # node whose true McCormick optimum is -92 came back as +8 — a dual bound
        # ABOVE the true optimum, the false-fathom class). Carried here and added
        # back at every bound-returning exit; ``c_override`` solves (feasibility
        # pump) are surrogates and deliberately excluded.
        self.obj_offset = float(getattr(_relax_probe, "_obj_offset", 0.0) or 0.0)
        self.base_A = A  # CSR, sorted indices; product-row VALUES rewritten per node
        self.base_b = b.copy()
        self.base_bounds = bnds.copy()
        self.bilinear = dict(info.get("bilinear", {}))
        self.monomial = dict(info.get("monomial", {}))  # any integer power p >= 2
        self.affine_square = dict(info.get("affine_square", {}))  # (var,aux)->(coeff,const)

        # C-44: per-column identity vector for this fixed lifted layout. The
        # incremental structure never rebuilds columns (``_patch`` only rewrites
        # coefficients of existing rows), so this identity vector is stable across
        # every node — a root cut pool (captured on the cold path over its own
        # varmap) is remapped onto these positions by matching identity. Built
        # from the same ``info`` (full varmap) the cold path returns.
        try:
            from discopt._jax.mccormick_lp import column_identities

            self.col_identities = column_identities(info, self.ncol, self.n)
        except Exception:
            self.col_identities = None

        # Map each product to its row indices. SPARSE + efficient: each product row
        # contains its aux column, so a product's rows are found among the rows that
        # touch that aux (a CSC lookup) — not by scanning all rows for every product
        # (the old dense ``supp <= {...}`` loop was O(rows*products) ~1.8e9 on qap).
        indptr, indices, data = A.indptr, A.indices, A.data
        csc = A.tocsc()
        col_ptr, col_rows = csc.indptr, csc.indices

        def _rows_with_col(c):
            return col_rows[col_ptr[c] : col_ptr[c + 1]]

        def _support(k):
            lo, hi = indptr[k], indptr[k + 1]
            return {int(indices[t]) for t in range(lo, hi) if abs(data[t]) > _TOL}

        def _entries(k):
            lo, hi = indptr[k], indptr[k + 1]
            return {int(indices[t]): float(data[t]) for t in range(lo, hi)}

        def _select(cand, expected, label):
            """The subset of ``cand`` that IS this term's box-dependent envelope.

            Support containment alone does not identify an envelope row (#861). A
            *model constraint* over the same variables is lifted to a row whose
            support is also contained in ``{operands, aux}`` — ``x0*x1 >= 60``
            becomes the single-column row ``{aux: -1} <= -60`` — so the old
            ``len(rows) != 4`` check counted lifted model rows as envelope rows and
            declined the whole structure (6 corpus instances: prob02, prob03,
            st_e01, st_e08, st_e09, st_e11).

            Identify them NUMERICALLY instead: on the probe box the closed-form
            envelope this class regenerates is exactly what the cold build emitted,
            so each expected row matches exactly one candidate. Any candidate left
            over is box-INDEPENDENT (a lifted model row) and is simply not patched —
            it stays in ``base_A`` untouched. That classification is not taken on
            faith: were such a row actually box-dependent, the cold build would emit
            different values on a different validation box and :meth:`_validate`
            would fail with a row-set mismatch, declining as before.

            Requiring EXACTLY ONE match per expected row is what keeps this safe: an
            ambiguous match (two candidates equal within tolerance) means we cannot
            say which row the patch owns, so we decline rather than guess.

            Returns the matched rows in ASCENDING index order — deliberately NOT in
            match order. ``_patch`` zips this list against the closed-form list
            positionally, and ascending order is what it has always used; measured
            over the 31 admitted instances (630 terms), the match order is a
            *rotation* of ascending order on 30 of them, so assigning in match order
            would permute the rows of every node LP and put the exactly-unchanged
            node-count gate at risk for zero benefit. Either assignment describes the
            same polytope (:meth:`_rowset` is order-free); this one is byte-identical
            to the pre-#861 behaviour wherever the term already had exactly 4 rows.
            """
            chosen: list[int] = []
            used: set[int] = set()
            for coeffs, rhs in expected:
                hits = []
                for k in cand:
                    if k in used:
                        continue
                    ent = _entries(k)
                    if not math.isclose(float(b[k]), rhs, rel_tol=1e-9, abs_tol=1e-9):
                        continue
                    if all(
                        math.isclose(ent.get(c, 0.0), v, rel_tol=1e-9, abs_tol=1e-9)
                        for c, v in coeffs.items()
                    ):
                        hits.append(k)
                if len(hits) != 1:
                    raise ValueError(
                        f"{label}: {len(hits)} of {len(cand)} candidate rows match the "
                        f"closed-form envelope row {coeffs} <= {rhs} (need exactly 1)"
                    )
                used.add(hits[0])
                chosen.append(hits[0])
            return sorted(chosen)

        def _coeffs(pairs):
            """Column -> coefficient, SUMMING duplicates (a term whose operands
            coincide, e.g. ``x_i*x_i``, contributes twice to the same column)."""
            d: dict[int, float] = {}
            for c, v in pairs:
                d[int(c)] = d.get(int(c), 0.0) + float(v)
            return d

        self.bilin_rows = {}
        for (i, j), a in self.bilinear.items():
            cand = [int(k) for k in _rows_with_col(a) if _support(k) <= {i, j, a}]
            expected = [
                (_coeffs([(i, ci), (j, cj), (a, cw)]), rhs)
                for ci, cj, cw, rhs in _bilinear_rows(i, j, a, lb_p[i], ub_p[i], lb_p[j], ub_p[j])
            ]
            self.bilin_rows[(i, j, a)] = _select(cand, expected, f"bilinear ({i},{j})")
        # monomial x_i**p, any p >= 2. Only an ODD power needs a sign-definite root
        # box (see the ``_root_sign`` note above); an even power is convex on all of
        # R and keeps its 4-row envelope across a sign change, so it is admitted on a
        # straddling root too — the case that used to decline the whole structure on
        # sign-mixed integer QCQPs (#861).
        self.mono_rows = {}
        for (i, p), a in self.monomial.items():
            if self._root_sign[i] == 0 and p % 2 == 1:
                raise ValueError(
                    f"monomial x_{i}^{p}: odd power on a root box spanning zero "
                    "(the envelope switches between the 4-row secant/tangent hull and "
                    "the 2-facet S-hull, which the fixed row pattern cannot express)"
                )
            cand = [int(k) for k in _rows_with_col(a) if _support(k) <= {i, a}]
            expected = [
                (_coeffs([(i, ci), (a, cs)]), rhs)
                for ci, cs, rhs in _monomial_rows(lb_p[i], ub_p[i], p)
            ]
            self.mono_rows[(i, a, p)] = _select(cand, expected, f"monomial x_{i}^{p}")
        # affine square (c*x_j + d)**2 -> aux: 4 secant/tangent rows over {j, aux}.
        self.affsq_rows = {}
        for (j, a), (coeff, const) in self.affine_square.items():
            cand = [int(k) for k in _rows_with_col(a) if _support(k) <= {j, a}]
            expected = [
                (_coeffs([(j, cx), (a, cw)]), rhs)
                for cx, cw, rhs in _affine_square_rows(coeff, const, lb_p[j], ub_p[j])
            ]
            self.affsq_rows[(j, a, coeff, const)] = _select(
                cand, expected, f"affine square ({j},{a})"
            )
        # the union of all product rows must be exactly the box-dependent rows
        self._prod_rows = set()
        for rs in self.bilin_rows.values():
            self._prod_rows |= set(rs)
        for rs in self.mono_rows.values():
            self._prod_rows |= set(rs)
        for rs in self.affsq_rows.values():
            self._prod_rows |= set(rs)

        # Fixed-pattern rewrite maps for the sparse ``_patch`` hot path: the base
        # CSR ``.data`` template, each product row's data-index span (to zero it,
        # matching the dense ``A[k]=0.0``), and the data index of each target column
        # ``(row, col)``. The sparsity pattern never changes across nodes — only the
        # product-row coefficient *values* do — so a node solve copies ``.data`` and
        # overwrites a few hundred entries instead of rebuilding the whole matrix.
        self._base_data = np.asarray(data, dtype=np.float64).copy()
        self._base_indices = np.asarray(indices)
        self._base_indptr = np.asarray(indptr)
        self._base_shape = A.shape
        self._row_span = {k: (int(indptr[k]), int(indptr[k + 1])) for k in self._prod_rows}
        self._pos: dict[tuple[int, int], int] = {}

        def _index_of(k, col):
            lo, hi = indptr[k], indptr[k + 1]
            for t in range(lo, hi):
                if int(indices[t]) == col:
                    return int(t)
            # Probe box is built so every McCormick coefficient is nonzero, so each
            # target column is present. A miss means the pattern can't represent a
            # box where this coefficient is nonzero -> refuse (ok=False -> cold path).
            raise ValueError(f"incremental pattern missing entry ({k},{col})")

        for (i, j, a), rows in self.bilin_rows.items():
            for k in rows:
                for col in (i, j, a):
                    self._pos[(k, col)] = _index_of(k, col)
        for (i, a, p), rows in self.mono_rows.items():
            for k in rows:
                for col in (i, a):
                    self._pos[(k, col)] = _index_of(k, col)
        for (j, a, coeff, const), rows in self.affsq_rows.items():
            for k in rows:
                for col in (j, a):
                    self._pos[(k, col)] = _index_of(k, col)

    @property
    def box_dependent_cols(self) -> frozenset[int]:
        """Structural columns whose BOUNDS drive a patched envelope row.

        :meth:`_patch` writes closed-form McCormick / secant-tangent coefficients
        built directly from ``lb[k]``/``ub[k]`` of these columns. An infinite endpoint
        on one of them produces ``inf``/``nan`` coefficients — silently, since the
        fixed sparsity pattern has nowhere to drop a row (unlike the cold builder,
        whose ``_Builder.add_row`` discards any non-finite payload and merely loosens).
        A caller that may hand this structure a partially infinite box must therefore
        check these columns first; :meth:`_validate` guarantees the set is complete,
        because any box-dependent row it did not map makes ``ok`` False.
        """
        cols: set[int] = set()
        for i, j in self.bilinear:
            cols.update((int(i), int(j)))
        for i, _p in self.monomial:
            cols.add(int(i))
        for j, _a in self.affine_square:
            cols.add(int(j))
        return frozenset(cols)

    def box_is_patchable(self, lb, ub) -> bool:
        """Whether :meth:`_patch` can produce a finite system over ``[lb, ub]``."""
        cols = self.box_dependent_cols
        if not cols:
            return True
        idx = np.fromiter(cols, dtype=int, count=len(cols))
        return bool(
            np.all(np.isfinite(np.asarray(lb, dtype=float)[idx]))
            and np.all(np.isfinite(np.asarray(ub, dtype=float)[idx]))
        )

    # -- per-node patch ---------------------------------------------------- #

    def _patch(self, lb, ub):
        """Return (A, b, bounds) for the McCormick LP over [lb,ub].

        SPARSE hot path: copy the base CSR ``.data`` template and overwrite only the
        box-dependent product-row entries at their precomputed positions (fixed
        pattern), then wrap it back into a CSR sharing the immutable indptr/indices.
        Equivalent, row for row, to the dense ``A[k]=0; A[k,col]=coef`` it replaces
        (each product row's stored support is exactly its target columns, so zeroing
        the row's data span then setting the targets reproduces it) — bit-identity is
        gated by :meth:`_validate`.
        """
        data = self._base_data.copy()
        b = self.base_b.copy()
        bounds = self.base_bounds.copy()
        bounds[: self.n, 0] = lb
        bounds[: self.n, 1] = ub
        pos = self._pos
        span = self._row_span
        for (i, j, a), rows in self.bilin_rows.items():
            li, ui, lj, uj = lb[i], ub[i], lb[j], ub[j]
            for k, (ci, cj, cw, rhs) in zip(rows, _bilinear_rows(i, j, a, li, ui, lj, uj)):
                lo, hi = span[k]
                data[lo:hi] = 0.0  # zero the whole row (matches dense A[k]=0.0)
                data[pos[(k, i)]] = ci
                data[pos[(k, j)]] = cj
                data[pos[(k, a)]] = cw
                b[k] = rhs
            bounds[a, 0], bounds[a, 1] = _bilinear_aux_bounds(li, ui, lj, uj)
        for (i, a, p), rows in self.mono_rows.items():
            li, ui = lb[i], ub[i]
            for k, (ci, cs, rhs) in zip(rows, _monomial_rows(li, ui, p)):
                lo, hi = span[k]
                data[lo:hi] = 0.0
                data[pos[(k, i)]] = ci
                data[pos[(k, a)]] = cs
                b[k] = rhs
            bounds[a, 0], bounds[a, 1] = _monomial_aux_bounds(li, ui, p)
        for (j, a, coeff, const), rows in self.affsq_rows.items():
            li, ui = lb[j], ub[j]
            for k, (cx, cw, rhs) in zip(rows, _affine_square_rows(coeff, const, li, ui)):
                lo, hi = span[k]
                data[lo:hi] = 0.0
                data[pos[(k, j)]] = cx
                data[pos[(k, a)]] = cw
                b[k] = rhs
            bounds[a, 0], bounds[a, 1] = _affine_square_aux_bounds(coeff, const, li, ui)
        A = sp.csr_matrix(
            (data, self._base_indices, self._base_indptr), shape=self._base_shape, copy=False
        )
        return A, b, bounds

    # -- soundness gate ---------------------------------------------------- #

    @staticmethod
    def _rowset(A, b, bounds=None):
        """Canonical hashable representation of the polytope's rows (order-free).

        Sparse-native (O(nnz), never densified): each row is the sorted tuple of its
        nonzero ``(col, round(val,6))`` pairs plus ``round(rhs,6)``. Entries rounding
        to 0 are dropped, so an explicit structural zero (the fixed-pattern ``_patch``
        can leave a zeroed target entry) compares equal to its absence in the cold
        build — the two matrices match iff they encode the same polytope.

        ``bounds`` (the ``(ncol,2)`` column box) additionally drops rows that are
        **vacuous over that box**: ``max_{x in box} a·x <= rhs``, i.e. the row cuts
        off nothing the bounds do not already exclude. Removing such a row provably
        leaves the feasible set unchanged, so the comparison stays an exact
        polytope-identity test — it just stops demanding that two identical polytopes
        be spelled with the same redundant rows. This is what lets a *pinned* box
        (``lb==ub``, reached whenever integer branching fixes a variable) validate:
        the cold build emits no 1-D envelope rows at zero width (``_emit_1d`` bails
        under ``_MIN_WIDTH``) and pins the aux via its ``[f(v),f(v)]`` bound, while
        the fixed-pattern ``_patch`` must write *something* into its four reserved
        rows and writes the tangents/secant collapsed at ``v`` — which are exactly
        tight there, hence vacuous, hence dropped here. Any row that genuinely cuts
        the box survives on both sides and a real mismatch is still caught.
        """
        M = sp.csr_matrix(A)
        M.sort_indices()
        indptr, indices, data = M.indptr, M.indices, M.data
        b = np.asarray(b, dtype=np.float64).ravel()
        vacuous = None
        if bounds is not None:
            # Row maximum over the box, VECTORIZED over all nonzeros at once (this
            # runs on the whole lift — up to ~172k nnz on qap — twice per validation
            # box, so a per-row numpy call would re-introduce the #654 pre-B&B
            # overrun). Each term contributes its larger endpoint product; an exact
            # zero coefficient contributes nothing (guarding ``0 * inf -> nan`` on an
            # unbounded column). A non-finite maximum leaves the row in place, which
            # is the conservative direction: an undroppable row can only cause a
            # (sound) mismatch, never a false match.
            bnd = np.asarray(bounds, dtype=np.float64)
            col_lo, col_hi = bnd[:, 0], bnd[:, 1]
            contrib = np.where(
                data == 0.0, 0.0, np.maximum(data * col_lo[indices], data * col_hi[indices])
            )
            row_max = np.asarray(
                sp.csr_matrix((contrib, indices, indptr), shape=M.shape).sum(axis=1)
            ).ravel()
            # The slack exists only to absorb the last-ulp difference between a row
            # built by the patch and the same row built by the cold build; it is NOT
            # meant to forgive a row that genuinely cuts. Measured over 22 admitted
            # models x 132 validation boxes, the worst amount by which a DROPPED row
            # actually cut its box was 7.1e-15 — i.e. the tolerance runs ~6 orders of
            # magnitude above anything the corpus exercises. If that margin ever
            # closes, this is the number to re-measure: the relative form would
            # tolerate a ~1-unit cut on a row whose maximum is ~1e9, so a future
            # large-coefficient lift is where it would first matter.
            slack = 1e-9 * (1.0 + np.abs(b) + np.abs(row_max))
            vacuous = np.isfinite(row_max) & (row_max <= b + slack)
        out = []
        for k in range(M.shape[0]):
            if vacuous is not None and vacuous[k]:
                continue
            entries = tuple(
                (int(indices[t]), rv)
                for t in range(indptr[k], indptr[k + 1])
                if (rv := round(float(data[t]), 6)) != 0.0
            )
            out.append((entries, round(float(b[k]), 6)))
        return sorted(out)

    @staticmethod
    def _box_sign_regime(lo, hi):
        """Classify a single variable's box ``[lo,hi]`` into a sign regime label so
        the validation set can prove it spans several. ``"pos"`` (``lo>0``),
        ``"neg"`` (``hi<0``), ``"span"`` (``lo<0<hi``, strictly crosses zero),
        ``"degen"`` (``lo==hi``), ``"zero_lb"`` (``lo==0<hi``, the boundary)."""
        if lo == hi:
            return "degen"
        if lo == 0.0:
            return "zero_lb"
        if lo > 0.0:
            return "pos"
        if hi <= 0.0:
            return "neg"
        return "span"

    def _validation_boxes(self):
        """The validation boxes fed to :meth:`_validate`, as ``(lo, hi)`` pairs.

        Every box is a *reachable* B&B sub-box of the root: branching only shrinks a
        box, so a var that is sign-definite at the root (``_root_sign != 0``) keeps
        that sign — a positive var never gets ``lb<0``, a negative var never gets
        ``ub>0``. A **spanning** var (``_root_sign==0``) carries only even-power
        monomials (odd ones are gated out in :meth:`_build_structure`) and its real
        nodes DO carry negative / zero-spanning bounds, so the boxes below
        deliberately drive those vars through negative-lb, zero-spanning
        (``lb<0<ub``), mixed-sign and degenerate (``lb==ub``) regimes — exactly the
        sign regimes that dominate real nodes and that the earlier ``lb>=0``-only set
        never exercised (C-21). Since #861 those same boxes are what proves an
        even-power envelope on a straddling box reproduces the cold build: the
        ``span``/``span_wide`` trials put ``lb<0<ub`` on every spanning var.

        Since #861 every interval is generated INSIDE the model's real root box
        rather than at synthetic absolute magnitudes. The old set used fixed
        magnitudes (``[0.5+0.3i, …]``, and ``lb=0`` on even trials) which for most
        models are *not reachable*: on ``gear``, whose root box is ``[12,60]^4``, it
        compared against boxes with ``lb=0``. Anchoring makes ``_validate`` compare
        the patch against the cold build on boxes the tree can actually branch into
        — which is the whole point of the check — and it is what lets a model whose
        decomposition route depends on its bounds (a ratio's log-space route needs
        strictly positive operands) be validated at all.

        Measured when this changed: **36 admitted before, 36 after, 0 flips** — no
        currently-admitted model relied on an unreachable comparison box, so no
        latent patch/cold divergence was hiding behind one.
        """
        # Per trial, ``kind`` says how each spanning var sits relative to zero;
        # sign-definite vars follow their root sign with a varying width/offset.
        kinds = ["shift_pos", "zero_lb", "span", "neg", "span_wide", "degen"]
        root_lb, root_ub = self._root_box()
        boxes = []
        for t, kind in enumerate(kinds):
            lo = np.empty(self.n)
            hi = np.empty(self.n)
            for i in range(self.n):
                rl, ru = self._finite_root_interval(float(root_lb[i]), float(root_ub[i]), i)
                w = ru - rl
                if w <= 0.0:  # pinned at the root: the only reachable box is the point
                    lo[i] = hi[i] = rl
                    continue
                if self._root_sign[i] != 0:
                    # Sign-definite root: every sub-box keeps that sign automatically,
                    # so vary WHERE in the root interval the box sits — full width,
                    # lb-touching, interior, ub-touching — rather than its sign. No
                    # degenerate trial: pinning a sign-definite var is outside the
                    # scope these envelopes are validated for (see :meth:`_rowset` on
                    # the pinned-box vacuity rule, which covers integer branching).
                    frac = [
                        (0.0, 1.0),
                        (0.0, 0.5),
                        (0.25, 0.75),
                        (0.5, 1.0),
                        (0.05 * (1 + i % 3), 0.95),
                        (0.4, 1.0),
                    ][t]
                    lo[i], hi[i] = rl + frac[0] * w, rl + frac[1] * w
                    continue
                # Spanning root (rl < 0 < ru): drive the negative / zero-spanning /
                # pinned regimes real nodes reach, all INSIDE [rl, ru].
                neg, pos = -rl, ru  # both > 0 here
                if kind == "shift_pos":
                    lo[i], hi[i] = 0.1 * pos, pos
                elif kind == "zero_lb":
                    lo[i], hi[i] = 0.0, 0.7 * pos
                elif kind == "span":
                    lo[i], hi[i] = -0.5 * neg, 0.5 * pos
                elif kind == "neg":
                    lo[i], hi[i] = rl, -0.1 * neg
                elif kind == "span_wide":
                    lo[i], hi[i] = rl, ru
                else:  # degen — a spanning var pinned by branching
                    lo[i] = hi[i] = -0.25 * neg
                if hi[i] < lo[i]:  # degenerate slice of a very narrow root interval
                    lo[i] = hi[i] = 0.5 * (rl + ru)
            boxes.append((lo, hi))
        return boxes

    def _validate(self):
        # Reachable, sign-diverse validation boxes (C-21 / cert:T1.2): each box is a
        # sub-box of the root (so the patched convex/concave power rows are compared
        # against a cold build in the *same* regime), but spanning vars are driven
        # through negative-lb, zero-spanning, mixed-sign and degenerate boxes — the
        # sign regimes real nodes reach. The patched row-set + aux bounds must
        # reproduce the cold ``build_milp_relaxation`` exactly on every one.
        # #654: this row-for-row self-check cold-builds ``build_milp_relaxation``
        # once per validation box — tens of seconds on large factorable models
        # (sonet*, qap), the dominant uninterruptible pre-B&B overrun. Bound it by
        # the solve deadline (stashed on the model in solver.py, anchored before all
        # preprocessing): once the budget is spent, stop *before starting* a new box
        # and leave ``ok=False`` — the engine then falls back to the trusted per-node
        # cold build (a valid, if unaccelerated, relaxation). Skipping validation only
        # forgoes the incremental speedup, never soundness; and checking before each
        # box bounds the overrun to at most one in-flight build (baron-gap-plan §8:
        # never truncate an in-flight bound-producing op).
        _deadline = self._deadline
        rng_boxes = self._validation_boxes()
        regimes = set()
        for lb, ub in rng_boxes:
            if _deadline is not None and time.perf_counter() > _deadline:
                self.ok = False
                self.decline_reason = "deadline spent during validation"
                return
            for i in range(self.n):
                regimes.add(self._box_sign_regime(float(lb[i]), float(ub[i])))
            Ap, bp, bdp = self._patch(lb, ub)
            # Conflict resolution (#860 x #861): main's validation STRUCTURE — the
            # vacuity-filtered row sets and the bounds-first ordering, which is what
            # lets a pinned box validate — plus #860's objective-offset assertion.
            # Both are needed: dropping main's version would re-break the pinned-box
            # case, dropping #860's would let a box-dependent offset through.
            Af, bf, bdf, _, _, relax_f = self._full_build(lb, ub)
            if Ap.shape[1] != Af.shape[1]:
                raise ValueError("column-count mismatch")
            # Bounds first: the row comparison drops box-vacuous rows, so the two
            # boxes must be known equal before that filter can be trusted to mean
            # the same thing on both sides.
            if not np.allclose(bdp, bdf, atol=1e-6, rtol=1e-6):
                raise ValueError("bounds mismatch")
            # Row COUNTS may legitimately differ (a pinned variable's envelope is 4
            # vacuous rows on the patched side and no rows at all on the cold side),
            # so identity is decided by the box-filtered row sets, which is the exact
            # polytope test — see :meth:`_rowset`.
            if self._rowset(Ap, bp, bdp) != self._rowset(Af, bf, bdf):
                raise ValueError("row-set mismatch")
            # The objective constant is captured once (probe box) and added back to
            # every returned bound, so it must be box-INDEPENDENT for that to be
            # exact. It is, for every shape the engine lifts (the aux substitution
            # moves the nonlinear part into columns and leaves the model's own
            # constant), but assert it here rather than assume: a box-dependent
            # offset makes ``ok=False`` and the caller falls back to the cold build.
            off_f = float(getattr(relax_f, "_obj_offset", 0.0) or 0.0)
            if abs(off_f - self.obj_offset) > 1e-9 * (1.0 + abs(off_f)):
                raise ValueError(
                    f"objective offset is box-dependent ({off_f} vs {self.obj_offset})"
                )
        self._validated_regimes = frozenset(regimes)
        self.ok = True

    # -- solve ------------------------------------------------------------- #

    def assemble(self, lb, ub, cut_rows=None):
        """Patched McCormick LP rows over [lb,ub] with optional appended cut rows.

        ``cut_rows`` is a list of ``(coeffs, rhs)`` inequalities ``coeffs·x <= rhs``
        over the structural+aux columns (length ``ncol``). Returns ``(A, b, bounds)``.
        """
        A, b, bounds = self._patch(lb, ub)  # A is CSR (sparse)
        if cut_rows:
            extra_A = sp.csr_matrix(
                np.array([np.asarray(co, dtype=np.float64)[: self.ncol] for co, _ in cut_rows])
            )
            extra_b = np.array([float(r) for _, r in cut_rows])
            A = sp.vstack([A, extra_A], format="csr")
            b = np.concatenate([b, extra_b])
        return A, b, bounds

    def solve_assembled(self, A, b, bounds, in_basis=None, c_override=None):
        """Solve a pre-assembled LP ``min c·x s.t. A x <= b, bounds``.

        The returned value is the RELAXATION's objective, ``c·x + obj_offset`` — the
        same scale ``MilpRelaxationModel.solve`` reports on the cold path. A
        ``c_override`` solve is a surrogate (feasibility pump), not a bound, so the
        offset is not applied to it."""
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_simplex import solve_lp_warm_std

        cobj = self.c if c_override is None else np.asarray(c_override, dtype=np.float64)
        off = 0.0 if c_override is not None else self.obj_offset
        try:
            result, out_basis = solve_lp_warm_std(
                cobj, sp.csr_matrix(A), b, bounds, in_basis=in_basis
            )
        except Exception:
            return None, None, None
        if result is None or result.status != SolveStatus.OPTIMAL or result.objective is None:
            return None, None, None
        return float(result.objective) + off, np.asarray(result.x, dtype=float), out_basis

    def solve_assembled_full(
        self, A, b, bounds, in_basis=None, c_override=None, *, return_cert=False
    ):
        """Like :meth:`solve_assembled`, but return the terminal *status* too so a
        caller can tell a (certified) ``infeasible`` apart from any other
        non-optimal verdict (time limit / numerical error).

        Returns ``(status, bound, x, out_basis, farkas_certified)`` where
        ``status`` is one of ``"optimal"``, ``"infeasible"`` (the LP feasible set
        is empty over this box — a rigorous fathoming proof, since the McCormick
        polytope is a valid outer approximation), or ``"other"`` (no certified
        verdict). ``bound``/``x`` are populated only for ``"optimal"``.

        ``bound`` is the **Neumaier–Shcherbina safe lower bound** built from the
        simplex's own row duals (issue #356) — sound at any conditioning, so it is
        never above the true LP optimum even when an ill-conditioned lifted basis
        makes the raw vertex objective drift high. ``farkas_certified`` is ``True``
        only when an ``"infeasible"`` verdict was independently proven by a
        verified Farkas dual ray; a caller can then fathom rigorously without any
        second (HiGHS/equilibration) solve.

        When ``return_cert`` is set the tuple is extended to
        ``(..., farkas_certified, cert)`` with the :class:`LpWarmCert` carrying the
        node LP's row duals / column status / safe bound (cert:T2.4a) -- a pure
        side-channel; ``bound``/``x`` are computed identically whether or not it is
        requested."""
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_simplex import LpWarmCert, solve_lp_warm_std

        cobj = self.c if c_override is None else np.asarray(c_override, dtype=np.float64)
        off = 0.0 if c_override is not None else self.obj_offset
        _empty = LpWarmCert(safe_bound=None, farkas_certified=False)

        def _ret(status, bound, x, out_basis, farkas, cert=_empty):
            if return_cert:
                return status, bound, x, out_basis, farkas, cert
            return status, bound, x, out_basis, farkas

        try:
            result, out_basis, cert = solve_lp_warm_std(
                cobj, sp.csr_matrix(A), b, bounds, in_basis=in_basis, return_cert=True
            )
        except Exception:
            return _ret("other", None, None, None, False)
        if result is None:
            return _ret("other", None, None, None, False)
        if result.status == SolveStatus.INFEASIBLE:
            return _ret("infeasible", None, None, None, bool(cert.farkas_certified), cert)
        if result.status != SolveStatus.OPTIMAL or result.bound is None:
            return _ret("other", None, None, None, False)
        # Shift BOTH the reported bound and the certificate's safe bound onto the
        # relaxation's own objective scale (``c·x + obj_offset``), so no consumer has
        # to remember to add the constant back — the cert flows on to DBBT / reduced
        # costs in ``mccormick_lp`` and would otherwise carry a different origin from
        # the bound sitting beside it.
        if off and cert.safe_bound is not None:
            cert = cert._replace(safe_bound=float(cert.safe_bound) + off)
        return _ret(
            "optimal",
            float(result.bound) + off,
            np.asarray(result.x, dtype=float),
            out_basis,
            False,
            cert,
        )

    def solve(self, lb, ub, in_basis=None, c_override=None, cut_rows=None):
        """Solve the McCormick LP over [lb,ub] (plus optional cut rows); return
        (bound, x, out_basis) or (None, None, None). Warm-starts from ``in_basis``.
        The bound is the relaxation objective ``c·x + obj_offset`` (cold-path scale).
        ``c_override`` replaces the objective (feasibility pump) — the returned bound
        is then the surrogate, not a dual bound, and carries no offset."""
        A, b, bounds = self.assemble(lb, ub, cut_rows)
        return self.solve_assembled(A, b, bounds, in_basis=in_basis, c_override=c_override)
