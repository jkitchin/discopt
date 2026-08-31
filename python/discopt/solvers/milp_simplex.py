"""Warm-started-simplex MILP backend (Rust ``solve_milp_py``).

A ``solve_milp(c, A_ub, b_ub, ..., integrality, ...)`` adapter, signature- and
``MILPResult``-compatible with :mod:`discopt.solvers.milp_pounce` and
:mod:`discopt.solvers.milp_pounce`, so it can be selected through
:func:`discopt.solvers.lp_backend.get_milp_solver`. It marshals the ``A_ub x <= b_ub``
form into the engine's standard form ``A_eq z = b`` (one explicit slack per row) and runs
the pure-Rust warm-started-simplex branch-and-bound.

Soundness: ``MILPResult.objective`` is the incumbent (an upper bound on a non-optimal
exit) and ``MILPResult.bound`` is the engine's **dual lower bound** — equal to the
incumbent once the solve is proven optimal, and a valid lower bound otherwise. Callers
that need a lower bound (AMP/OA/GDP-LOA) must read ``bound``, never ``objective``. If the
Rust binding is unavailable it raises :class:`SimplexBackendUnavailable` so the selector
can fall back.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Callable, NamedTuple, Optional, Union, cast

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus

logger = logging.getLogger(__name__)


class SimplexBackendUnavailable(RuntimeError):
    """Raised when the Rust ``solve_milp_py`` binding cannot be imported."""


_NS_MARGIN_REL = 1e-9
"""Magnitude-scaled relative margin for the safe-bound / Farkas-ray evaluations.

The two dot products below run in plain float64 (not directed-rounding interval
arithmetic), so a margin proportional to the operands' magnitude is subtracted to
dominate their rounding error and keep the returned bound a *rigorous*
under-estimate (and the Farkas test a rigorous proof). Mirrors the constant in
:func:`discopt._relax.obbt._ns_safe_lp_lower_bound`."""

_INF = 1e20  # discopt's effective-infinity sentinel for free variable bounds.

# A column bound list in the scipy/``milp_pounce`` shape: ``None`` on a side means
# that side is open (issue #1060). ``Sequence`` rather than ``list`` so the common
# ``list[tuple[float, float]]`` of a fully-bounded caller still type-checks --
# ``list`` is invariant, ``Sequence`` is not.
BoundList = Sequence[tuple[Optional[float], Optional[float]]]

#: The root cut budget every MILP that reaches the Rust driver *through Python*
#: has run at since #334: at most 16 cuts, in a single round, first-come (no
#: efficacy/orthogonality selection). Those numbers were not chosen against cut
#: quality — they were chosen against cut *cost*. Until #1102 every round of the
#: root loop re-derived the augmented LP from a cold slack basis, so a second
#: round cost a full root solve. Measured on the ``rsyn0840m`` OA master at
#: ``root_cuts=500, cut_rounds=15``: 14 cold root solves = 23.1 s of a 24.2 s cut
#: loop. #1102 made the round warm (re-optimize from the previous round's optimal
#: basis), and the budget was never revisited.
_LEGACY_CUT_PROFILE: dict[str, object] = {
    "root_cuts": 16,
    "cut_rounds": 1,
    "cut_select": False,
}

#: The budget the loop is worth now that a round is warm (#1066) — reached ONLY
#: through :func:`solve_milp`'s escalation, never as a blanket default. Applying
#: it unconditionally was measured and rejected: it costs the ``tls2`` masters
#: their proof (see the escalation comment in :func:`solve_milp`). ``cut_select``
#: is the half that bounds the *cost*: it keeps only the strongest, most diverse
#: cuts up to the cap, so a 200-cut budget does not densify every node LP the way
#: 200 first-come Gomory rows would. Measured on the four OA masters #1066 still
#: loses (``master0``, 60 s cap, nodes to proven optimality):
#:
#: ==================  ===================  ====================
#: master              legacy 16/1          200/10/select
#: ==================  ===================  ====================
#: ``rsyn0820m``       139 267 n, 12.9 s    4 785 n, 0.8 s
#: ``rsyn0830m``       529 573 n, 57.1 s    1 197 n, 0.3 s
#: ``rsyn0840m``       no proof in 60 s     120 241 n, 33.5 s
#: ``rsyn0820m02m``    bound −5108.7        bound −4151.2
#: ==================  ===================  ====================
#:
#: The cap matters as much as the rounds: ``16/10/select`` leaves ``rsyn0840m``
#: unproven, and dropping GMI (``gmi_cuts=False``) is far worse than the legacy
#: profile on three of the four — the tableau cuts are what close this class.
_STRONG_CUT_PROFILE: dict[str, object] = {
    "root_cuts": 200,
    "cut_rounds": 10,
    "cut_select": True,
}

#: Whether an unset ``DISCOPT_MILP_CUT_BUDGET`` enables the escalation. Escalating
#: changes which cuts the root loop adds on a master the probe cannot close, hence
#: that master's dual bound — a bound-changing knob (CLAUDE.md §5 regime 2). So it
#: ships default-off and is flipped only by a graduation panel that clears both
#: bars — cert-clean and net-positive — over the corpus. That panel passed on
#: 2026-08-29 (79 instances, 292 soundness checks, 0 violations; certificates
#: 56 -> 57, five dual bounds tighter and none looser, total wall -4.7 %), so the
#: default is ON. ``DISCOPT_MILP_CUT_BUDGET=0`` remains the opt-out and keeps the
#: single-legacy-solve path exactly as it was. The panel is recorded in
#: ``docs/dev/performance-plan.md`` §23.
_MILP_CUT_BUDGET_DEFAULT = True

#: Default separation rounds one node may run against a supplied
#: ``node_callback`` (#1141). Each round costs one callback plus one warm re-solve
#: of that node's LP and buys a tighter bound there; because the rows are global,
#: what one node does not separate the next one will, so a small budget is the
#: point rather than a compromise. Overridable per call and by
#: ``DISCOPT_MILP_NODE_HOOK_ROUNDS``.
_NODE_HOOK_ROUNDS_DEFAULT = int(os.environ.get("DISCOPT_MILP_NODE_HOOK_ROUNDS", "2"))

#: Default cap on the rows a supplied ``node_callback`` may fold into the shared
#: relaxation over one solve. A fractional cut is optional (it only tightens), so
#: unlike a lazy cut it is budgeted: an unbounded stream of gradient cuts would
#: densify every node LP and trade the node win back for wall time. Overridable
#: per call and by ``DISCOPT_MILP_NODE_HOOK_CUT_CAP``.
_NODE_HOOK_CUT_CAP_DEFAULT = int(os.environ.get("DISCOPT_MILP_NODE_HOOK_CUT_CAP", "500"))

_U64 = 2.0**-53  # float64 unit roundoff


def _gamma(p):
    """Higham's ``γ_p = p·u/(1−p·u)``: rigorous relative bound on the forward
    error of a length-``p`` float64 dot product / recursive sum (Higham 2002,
    Lemma 3.1 — valid for ANY summation order, so numpy's pairwise reduction is
    covered). ``p`` may be an ndarray."""
    pu = np.asarray(p, dtype=np.float64) * _U64
    return pu / (1.0 - pu)


def _safe_lp_lower_bound_sharp(
    y: np.ndarray,
    c: np.ndarray,
    a_std,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[float]:
    """Sharp-margin Neumaier–Shcherbina safe bound (#309, flag-gated).

    Same weak-duality bound as :func:`_safe_lp_lower_bound_std` — ``g(y) = bᵀy +
    Σ_k min_{z_k∈box} rc_k z_k`` is valid for ANY ``y`` — but instead of the flat
    ``1e-9``-relative margin it subtracts a *provably sufficient* forward-error
    bound assembled from the data:

    * per-column reduced-cost error ``E_k ≤ γ_{nnz_k+1}·(|c_k| + (|A|ᵀ|y|)_k)``
      (dot product + one subtraction, Higham 2002 §3.1);
    * a column whose computed ``rc_k`` is within ``E_k`` of zero has an
      *uncertain sign*: its box term is enclosed by the four interval corners
      ``{rc_k∓E_k}×{lb_k, ub_k}`` (rigorous, no side selection needed). If such
      a column still has an infinite side after FBBT the enclosure is ``−∞`` and
      the function abstains — the legacy path silently contributes 0 there,
      which a flat margin cannot cover (latent soundness gap, recorded in
      ``docs/dev/ns-sharp-margin-2026-07-16.md``);
    * summation errors ``γ_n·Σ|term|`` and ``γ_{m+1}·Σ|b_j y_j|``.

    The assembled margin is inflated by 1.0625 (exact power-of-two headroom) to
    dominate the O(u²) terms Higham's first-order gammas drop and the float64
    evaluation of the margin expression itself. On the gear4 piece LPs this
    replaces a 2.9e-4 loss with ~1e-6 (measured; see the dev doc §2).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or not np.all(np.isfinite(y)):
        return None
    lb = np.where(np.asarray(lb, dtype=np.float64) <= -_INF, -np.inf, lb)
    ub = np.where(np.asarray(ub, dtype=np.float64) >= _INF, np.inf, ub)
    c = np.asarray(c, dtype=np.float64)
    a_csc = a_std if sp.issparse(a_std) else sp.csc_matrix(np.asarray(a_std, dtype=np.float64))
    a_csc = a_csc.tocsc()
    n = a_csc.shape[1]
    rc = c - np.asarray(a_csc.T @ y).ravel()
    if not np.all(np.isfinite(rc)):
        return None

    abs_a = a_csc.copy()
    abs_a.data = np.abs(abs_a.data)
    aty_abs = np.asarray(abs_a.T @ np.abs(y)).ravel()
    col_nnz = np.diff(a_csc.indptr)
    err_rc = _gamma(col_nnz + 1) * (np.abs(c) + aty_abs)

    cert_pos = rc > err_rc
    cert_neg = rc < -err_rc
    uncertain = ~(cert_pos | cert_neg)
    # FBBT-recover a finite valid box wherever the selected/enclosed side is
    # infinite (superset of the legacy trigger: uncertain columns need BOTH
    # sides). FBBT bounds still contain the feasible set, so g stays ≤ p*.
    need = (
        (cert_pos & ~np.isfinite(lb))
        | (cert_neg & ~np.isfinite(ub))
        | (uncertain & ~(np.isfinite(lb) & np.isfinite(ub)))
    )
    if need.any():
        lb, ub = _fbbt_eq_bounds(a_csc, np.asarray(b, dtype=np.float64), lb, ub)

    term = np.zeros_like(rc)
    err_term = np.zeros_like(rc)
    term[cert_pos] = rc[cert_pos] * lb[cert_pos]
    term[cert_neg] = rc[cert_neg] * ub[cert_neg]
    # Certain-sign columns: |true term − computed| ≤ E_k·|side| + u·|computed|.
    side = np.zeros_like(rc)
    side[cert_pos] = lb[cert_pos]
    side[cert_neg] = ub[cert_neg]
    cert = cert_pos | cert_neg
    err_term[cert] = err_rc[cert] * np.abs(side[cert]) + _U64 * np.abs(term[cert])
    if uncertain.any():
        # Interval-corner enclosure of min_{z∈box} rc_true·z over rc_true ∈
        # [rc−E, rc+E]. np.minimum propagates the −inf of an unbounded side.
        rl, rh = rc[uncertain] - err_rc[uncertain], rc[uncertain] + err_rc[uncertain]
        lo_u, hi_u = lb[uncertain], ub[uncertain]
        with np.errstate(invalid="ignore"):  # 0·±inf → nan, handled below
            corners = np.minimum(np.minimum(rl * lo_u, rl * hi_u), np.minimum(rh * lo_u, rh * hi_u))
        # 0·±inf = nan in IEEE: an exact-zero interval edge on an infinite side
        # bounds a term that is exactly 0 from that corner; treat nan as that
        # corner not binding only when the WHOLE interval is the point 0.
        point_zero = (rl == 0.0) & (rh == 0.0)
        corners = np.where(point_zero, 0.0, corners)
        if not np.all(np.isfinite(corners)):
            return None
        term[uncertain] = corners
        err_term[uncertain] = 2.0 * _U64 * np.abs(corners)
    if not np.all(np.isfinite(term)):
        return None

    b64 = np.asarray(b, dtype=np.float64)
    by = float(b64 @ y)
    s = float(term.sum())
    g = by + s
    if not np.isfinite(g):
        return None
    margin = (
        float(err_term.sum())
        + float(_gamma(max(n, 1)) * np.abs(term).sum())
        + float(_gamma(y.size + 1) * (np.abs(b64) * np.abs(y)).sum())
        + 4.0 * _U64 * (abs(by) + abs(s) + abs(g))
    ) * 1.0625
    return g - margin


class LpWarmCert(NamedTuple):
    """Verified-certificate side-channel from :func:`solve_lp_warm_std`.

    * ``safe_bound`` — on an ``optimal`` solve, a Neumaier–Shcherbina safe lower
      bound computed from the simplex's own row duals: ``<=`` the true LP optimum
      at *any* conditioning, so a caller can use it as a rigorous bound without an
      independent second solve. ``None`` when unavailable.
    * ``farkas_certified`` — on an ``infeasible`` solve, ``True`` iff the
      simplex's Farkas dual-ray candidate was independently verified to prove the
      feasible set empty (a rigorous fathoming proof). ``False`` otherwise — the
      caller must then fall back rather than trust the bare verdict.
    * ``dual`` — on an ``optimal`` solve, the simplex's row-dual vector ``y`` (one
      entry per constraint row of the *standard-form* ``[A_ub | I] z = b`` system,
      so ``len(dual) == m``). ``None`` when unavailable. Additive side-channel for
      duality-based bound tightening (cert:T2.4a); it never changes the reported
      objective/bound (those are computed identically whether or not it is read).
    * ``col_status`` — on an ``optimal`` solve, the final column-status vector for
      the standard-form columns (structural first, then slacks) — the warm-start
      basis to thread into a downstream re-solve. ``None`` when unavailable.
    """

    safe_bound: Optional[float]
    farkas_certified: bool
    dual: Optional[np.ndarray] = None
    col_status: Optional[np.ndarray] = None


def _fbbt_eq_bounds(
    a_std: "object",
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    rounds: int = 3,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Feasibility-based bound tightening on the equality system ``A_std z = b``.

    Returns ``(lb, ub)`` tightened so that every derived bound is *implied* by the
    equalities plus the incoming box — i.e. the result still contains the whole
    feasible set ``{z : A_std z = b, lb <= z <= ub}``. Used to give the unbounded
    lifted/slack columns a finite, **valid** box for the Neumaier–Shcherbina safe
    bound: a superset-preserving tightening keeps ``g(y) <= p*`` sound while making
    the box-min term finite (an open side whose reduced cost selects it would
    otherwise collapse the whole bound to ``-inf``).

    Each equality ``Σ_j a_ij z_j = b_i`` bounds column ``k`` two-sidedly from the
    min/max activity of the *other* columns, with an explicit per-row infinity
    tally so a single open bound still propagates (vectorised over the sparse
    matrix; the per-element Python loop is too slow for this per-solve path).
    """
    coo = sp.csr_matrix(a_std).tocoo()
    rows = coo.row
    cols = coo.col
    vals = coo.data
    n_rows = coo.shape[0]
    lb = np.array(lb, dtype=np.float64)
    ub = np.array(ub, dtype=np.float64)
    if vals.size == 0:
        return lb, ub
    pos = vals > 0.0
    b_r = b[rows]

    for _ in range(rounds):
        # Min activity uses lb where coeff>0, ub where coeff<0; max activity swaps.
        min_used = np.where(pos, lb[cols], ub[cols])
        max_used = np.where(pos, ub[cols], lb[cols])
        min_term = vals * min_used  # may be -inf
        max_term = vals * max_used  # may be +inf
        min_inf = ~np.isfinite(min_term)
        max_inf = ~np.isfinite(max_term)
        n_min_inf = np.zeros(n_rows)
        np.add.at(n_min_inf, rows, min_inf.astype(np.float64))
        n_max_inf = np.zeros(n_rows)
        np.add.at(n_max_inf, rows, max_inf.astype(np.float64))
        sum_min = np.zeros(n_rows)
        np.add.at(sum_min, rows, np.where(min_inf, 0.0, min_term))
        sum_max = np.zeros(n_rows)
        np.add.at(sum_max, rows, np.where(max_inf, 0.0, max_term))
        # Activity of the row excluding column j (finite iff every *other* term is).
        minrest_finite = (n_min_inf[rows] - min_inf.astype(np.float64)) == 0
        maxrest_finite = (n_max_inf[rows] - max_inf.astype(np.float64)) == 0
        minrest = sum_min[rows] - np.where(min_inf, 0.0, min_term)
        maxrest = sum_max[rows] - np.where(max_inf, 0.0, max_term)
        # z_j = (b_i - rest)/a_ij; the rest-interval endpoints give z_j's bounds.
        # The quotient is computed densely over every stored coefficient, so on the
        # ill-conditioned narrow/RLT boxes this runs on (coefficient spreads of ~1e26
        # over ~1e-300 denormals — the same class as milp_relaxation.py's Stage-1B
        # guard, #732) a handful of entries overflow to inf / divide by a denormal /
        # go nan. Those results are rigorously discarded below by the ``*_valid`` mask
        # AND the ``np.isfinite`` filter, so the transient inf/nan never reaches a
        # bound — suppress the spurious RuntimeWarnings without changing any value
        # (bound-neutral by construction; the arithmetic is byte-identical).
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            lo_cand = np.where(pos, (b_r - maxrest) / vals, (b_r - minrest) / vals)
            hi_cand = np.where(pos, (b_r - minrest) / vals, (b_r - maxrest) / vals)
        lo_valid = np.where(pos, maxrest_finite, minrest_finite)
        hi_valid = np.where(pos, minrest_finite, maxrest_finite)

        new_lo = np.full(lb.shape[0], -np.inf)
        sel = lo_valid & np.isfinite(lo_cand)
        if sel.any():
            np.maximum.at(new_lo, cols[sel], lo_cand[sel])
        upd_lo = new_lo > lb + tol
        if upd_lo.any():
            lb = np.where(upd_lo, np.maximum(lb, new_lo), lb)

        new_hi = np.full(ub.shape[0], np.inf)
        sel = hi_valid & np.isfinite(hi_cand)
        if sel.any():
            np.minimum.at(new_hi, cols[sel], hi_cand[sel])
        upd_hi = new_hi < ub - tol
        if upd_hi.any():
            ub = np.where(upd_hi, np.minimum(ub, new_hi), ub)

        if not (upd_lo.any() or upd_hi.any()):
            break
    return lb, ub


def _safe_lp_lower_bound_std(
    y: np.ndarray,
    c: np.ndarray,
    a_std: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[float]:
    """Neumaier–Shcherbina safe lower bound on ``min cᵀz s.t. A z = b, lb<=z<=ub``
    from *free-sign* equality multipliers ``y`` (length m).

    Weak duality gives, for ANY ``y``,

        g(y) = bᵀy + Σ_k min_{z_k∈[lb_k,ub_k]} (c − Aᵀy)_k z_k  ≤  min cᵀz,

    so ``g(y)`` is a valid lower bound regardless of how ``y`` was obtained — it
    stays sound even when an ill-conditioned basis makes the reported vertex
    objective drift *above* the true optimum (the nvs22 false-certificate class).
    A magnitude-scaled margin is subtracted so the float64 evaluation error cannot
    push ``g`` above the true optimum. Returns ``None`` when no usable (finite)
    bound exists (e.g. an unbounded box term)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or not np.all(np.isfinite(y)):
        return None
    # Map the ±1e20 sentinels to true infinities so an infinite box side with a
    # nonzero reduced cost yields −inf (an unusable bound → None), not a spurious
    # large-finite contribution.
    lb = np.where(np.asarray(lb, dtype=np.float64) <= -_INF, -np.inf, lb)
    ub = np.where(np.asarray(ub, dtype=np.float64) >= _INF, np.inf, ub)
    c = np.asarray(c, dtype=np.float64)
    at_y = a_std.T @ y if not sp.issparse(a_std) else (a_std.T @ y)
    rc = c - np.asarray(at_y).ravel()
    pos = rc > 0.0
    neg = rc < 0.0
    # A box-min term is -inf when the reduced cost selects an open side. The
    # lifted relaxation's objective-epigraph / sqrt-/division-lift aux columns and
    # the row slacks carry +/-inf bounds, and a roundoff-flipped tiny reduced cost
    # on such a column would otherwise collapse the whole safe bound to -inf (the
    # nvs05/nvs22/st_e36/chance root-bound drop). Recover a finite, *valid* box for
    # exactly those columns by feasibility-based bound tightening: FBBT bounds still
    # contain the feasible set, so g(y) stays <= p* (sound) while becoming finite.
    # Gated on actually needing it, so well-bounded LPs keep the cheap path.
    if (pos & ~np.isfinite(lb)).any() or (neg & ~np.isfinite(ub)).any():
        # FBBT's float64 division roundoff (~ulp·|bound|) is dominated by the
        # magnitude-scaled ``margin`` subtracted from g below, so the derived box
        # needs no extra outward loosening — and adding one perturbs the (large)
        # slack/aux bounds enough to break the safe bound's rescaling invariance
        # without improving soundness. Use the FBBT bounds directly.
        lb, ub = _fbbt_eq_bounds(a_std, np.asarray(b, dtype=np.float64), lb, ub)
    # min_{z_k∈[lb,ub]} rc_k z_k = lb_k if rc_k>0, ub_k if rc_k<0, else 0 (the
    # rc_k==0 case contributes 0 even when that bound is infinite).
    contrib = np.zeros_like(rc)
    contrib[pos] = rc[pos] * lb[pos]
    contrib[neg] = rc[neg] * ub[neg]
    # Any term still open after FBBT (a genuinely unbounded selected side) leaves
    # no usable bound — abstain rather than return a spurious value.
    if not np.all(np.isfinite(contrib)):
        return None
    by = float(np.asarray(b, dtype=np.float64) @ y)
    g = by + float(contrib.sum())
    if not np.isfinite(g):
        return None
    margin = _NS_MARGIN_REL * (1.0 + abs(by) + float(np.abs(contrib).sum()))
    return g - margin


def _safe_lp_lower_bound(
    y: np.ndarray,
    c: np.ndarray,
    a_std,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[float]:
    """Dispatch to the sharp-margin NS evaluation when ``ns_sharp_margin`` is on
    (#309), else the legacy flat-margin one. Both are rigorous lower bounds; the
    sharp one is tighter and additionally abstains on sign-uncertain columns
    with an unbounded side (see :func:`_safe_lp_lower_bound_sharp`)."""
    from discopt.solver_tuning import current as _tuning_current

    if _tuning_current().ns_sharp_margin:
        return _safe_lp_lower_bound_sharp(y, c, a_std, b, lb, ub)
    return _safe_lp_lower_bound_std(y, c, a_std, b, lb, ub)


# #671: geometric RHS-regularization schedule for the numerical-failure refinement.
# A small ``tau`` moves the ill-conditioned relaxation off its near-singular
# optimal configuration so the in-house simplex can certify the neighbour and
# return a good dual; too-small ``tau`` stays singular (drifted dual), too-large
# loosens the relaxation. Sweeping several orders and taking the *max* NS bound is
# rigorously safe (max of valid lower bounds) and robust to where the usable window
# sits for a given instance. Absolute perturbation, matching the measured hda sweet
# spot (~3e-3; docs/dev/issue-671-gsw-iterative-refinement-2026-07-18.md).
_REFINE_TAUS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def _refined_safe_bound_regularized(
    solve_lp_warm_csc_py,
    c_std: np.ndarray,
    a_std,
    b_vec: np.ndarray,
    lb_std: np.ndarray,
    ub_std: np.ndarray,
    m: int,
    n: int,
    *,
    time_limit: Optional[float] = None,
) -> Optional[float]:
    """#671 tight dual bound for a numerically-failed node LP, in-house only.

    Re-solve ``[A|I] z = b + tau`` for each ``tau`` in :data:`_REFINE_TAUS` with the
    in-house simplex, and for every recovered dual evaluate the Neumaier–Shcherbina
    safe bound against the **original** ``b_vec`` (never ``b+tau``). ``g(y)`` is a
    valid lower bound for *any* ``y`` (weak duality), so the regularization affects
    only the *tightness* of the recovered dual, never the *soundness* of the bound.
    Returns the **max** over the sweep (the tightest sound bound), or ``None`` if no
    tau yielded a usable dual — in which case the caller keeps candidate A's
    drifted-dual bound. Never returns a value above the true optimum.
    """
    indptr = np.ascontiguousarray(a_std.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(a_std.indices, dtype=np.int64)
    data = np.ascontiguousarray(a_std.data, dtype=np.float64)
    c_c = np.ascontiguousarray(c_std)
    lb_c = np.ascontiguousarray(lb_std)
    ub_c = np.ascontiguousarray(ub_std)
    best: Optional[float] = None
    # ``time_limit`` is the budget of the ONE node LP this sweep is recovering a bound
    # for, so spend it across the whole sweep rather than handing each of the seven
    # perturbations a fresh copy (which would multiply the caller's budget by 7).
    _deadline = None if time_limit is None else time.perf_counter() + max(0.0, time_limit)
    for tau in _REFINE_TAUS:
        _remaining = None if _deadline is None else max(0.0, _deadline - time.perf_counter())
        if _remaining == 0.0:
            break
        b_reg = np.ascontiguousarray(b_vec + tau)
        try:
            _status, _x, _obj, _iters, _cs, _bv, dual, _ray = solve_lp_warm_csc_py(
                c_c,
                m,
                n + m,
                indptr,
                indices,
                data,
                b_reg,
                lb_c,
                ub_c,
                None,
                None,
                time_limit_s=_remaining,
            )
        except Exception as exc:  # noqa: BLE001 - the next perturbation is tried instead
            logger.debug(
                "dual-bound refinement LP failed at tau=%g: %s: %s", tau, type(exc).__name__, exc
            )
            continue
        if dual is None or not np.size(dual):
            continue
        g = _safe_lp_lower_bound(
            np.asarray(dual, dtype=np.float64), c_std, a_std, b_vec, lb_std, ub_std
        )
        if g is not None and np.isfinite(g) and (best is None or g > best):
            best = float(g)
    return best


def _cold_dual_start_enabled() -> bool:
    """``DISCOPT_LP_COLD_DUAL_START`` — use the dual slack start on a cold,
    deadline-free pure LP too. Resolved per call so a test can set the env var."""
    from discopt.solver_tuning import current as _tuning_current

    return bool(_tuning_current().lp_cold_dual_start)


def _dual_start_slack_basis(
    c_arr: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    m: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Sign-matched slack start basis for a deadline-carrying COLD pure-LP solve
    (#928), or ``None`` when it is not dual-feasible.

    Slacks basic (``B = I``, so ``y = B⁻ᵀc_B = 0`` and every reduced cost is
    ``c_j``), each structural column nonbasic at the bound its objective sign
    selects (``c_j > 0`` → lower, ``c_j < 0`` → upper, ``c_j = 0`` → lower). That
    basis is dual-feasible exactly when every selected side is finite, which this
    checks; the Rust ``PreparedDual::prepare`` re-verifies the same precondition
    exactly, so an accepted basis here can never make the engine converge wrong.

    Why: the primal simplex proves NO usable lower bound mid-run — on the hda
    separated-relaxation LP the floor recovered from a deadline-cut cold primal
    was -141697 at *any* deadline fraction (15/40/75% of the full solve) against
    an optimum of -64473. The dual simplex maintains dual feasibility, so its
    dual objective is a monotone anytime lower bound and a deadline exit banks
    the best bound proved so far. That is why a finite ``time_limit`` engages it.

    A cold solve with **no** deadline engages it only under
    ``DISCOPT_LP_COLD_DUAL_START`` (default OFF, so the historical cold-primal
    path stays bit-identical by default). The reason there is speed rather than
    bankability: on equality-rich lifted relaxations — every equality reaching the
    LP layer as two opposing, always-tight ``<=`` rows — the cold primal is
    massively degenerate and stalls, exhausting ``max_iter`` on LPs this start
    solves in seconds to the same optimum. See ``SolverTuning.lp_cold_dual_start``.
    """
    if m <= 0:
        return None
    need_lb = c_arr > 0.0
    need_ub = c_arr < 0.0
    if ((need_lb & (lb <= -_INF)) | (need_ub & (ub >= _INF))).any():
        return None  # a selected side is open → dual-infeasible → keep the primal
    n = c_arr.shape[0]
    # Rust basis encoding: AT_LOWER=0, BASIC=1, AT_UPPER=2 (crate::lp::basis).
    col_status = np.concatenate(
        [np.where(need_ub, 2, 0).astype(np.int8), np.ones(m, dtype=np.int8)]
    )
    basic_vars = np.arange(n, n + m, dtype=np.int64)
    return col_status, basic_vars


def _farkas_certified_std(
    ray: np.ndarray,
    a_std: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> bool:
    """Verify a Farkas dual-ray candidate proves ``A z = b, lb<=z<=ub`` is empty.

    The system is infeasible iff some free-sign ``y`` has ``bᵀy`` exceeding the
    box-maximum of ``(Aᵀy)ᵀz`` — i.e. the ``c=0`` safe bound ``g₀(y) > 0`` (the
    margin inside :func:`_safe_lp_lower_bound_std` already makes the strict
    inequality rigorous). The simplex hands us a ray up to an overall sign, so we
    try ``±ray``; a candidate that fails to verify simply returns ``False`` and
    the caller falls back — it can never produce an unsound fathom."""
    ray = np.asarray(ray, dtype=np.float64)
    if ray.size == 0 or not np.all(np.isfinite(ray)):
        return False
    zeros_c = np.zeros(a_std.shape[1], dtype=np.float64)
    for sign in (1.0, -1.0):
        g0 = _safe_lp_lower_bound(sign * ray, zeros_c, a_std, b, lb, ub)
        if g0 is not None and g0 > 0.0:
            return True
    return False


def milp_cut_budget_enabled() -> bool:
    """Is the #1066 cut-budget escalation switched on?

    Escalating changes which cuts the root loop adds on a master the probe cannot
    close, hence that master's dual bound and node count — CLAUDE.md §5 regime 2.
    ``DISCOPT_MILP_CUT_BUDGET=0`` restores the single :data:`_LEGACY_CUT_PROFILE`
    solve exactly, and that path stays tested.

    Read per call, not cached at import, so a test can flip it without reloading
    the module, and so an A/B panel can drive both arms from one build.
    """
    raw = os.environ.get("DISCOPT_MILP_CUT_BUDGET")
    if raw is None:
        return _MILP_CUT_BUDGET_DEFAULT
    return raw.strip() != "0"


#: Node cap for the escalation probe (#1066). Measured, not chosen by taste: on
#: the captured OA masters the two classes separate cleanly by node count. Every
#: master the legacy budget closes *fast* closes well inside the cap (tls2
#: masters 0/3/6 at 241, 515 and 595 nodes, each in 0.0 s), while every master
#: that needs the strong budget was still open at 20 000 nodes (all four rsyn
#: masters). 5 000 sits ~8x above the first class and far below the second, so
#: the probe decides correctly on both while costing ~0.2 s where it escalates.
_PROBE_MAX_NODES = 5_000

#: Statuses that make a probe result final — a *proof*, not a budget cut-off.
#: Escalating past one of these would spend the strong budget re-deriving an
#: answer already certified.
_PROBE_FINAL_STATUSES = frozenset(
    {SolveStatus.OPTIMAL, SolveStatus.INFEASIBLE, SolveStatus.UNBOUNDED}
)


class _StdForm(NamedTuple):
    """The engine's standard form ``[A_ub | I] z = b``, built column-major.

    ``n`` structural columns come first, then one slack per row; ``lb``/``ub``/``c``
    are over all ``n + m`` columns. ``lp_kwargs`` carries either the pure-LP
    short-circuit or the root cut budget, both described in
    :func:`_marshal_std_form`, and is forwarded by every entry point in this
    module — so the two never diverge between the plain and lazy-cut drivers.
    """

    m: int
    n: int
    col_ptr: np.ndarray
    row_idx: np.ndarray
    vals: np.ndarray
    b: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    c: np.ndarray
    int_cols: np.ndarray
    lp_kwargs: dict


def _marshal_col_bounds(
    bounds: Optional[BoundList],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Marshal a scipy-style ``[(lo, hi), ...]`` list into the engine's bound arrays.

    Mirrors :func:`discopt.solvers.milp_pounce.solve_milp`'s documented contract:
    ``bounds=None`` means ``(0, +inf)`` per variable, and ``None`` on one side of a
    pair means that side is open.

    An open side becomes the ``±_INF`` sentinel (``1e20``) — what the Rust LP layer
    reads as unbounded. An **explicit** ``±inf`` is passed through untouched: that
    is what every pre-#1060 caller already sent (measured: on the McCormick root-LP
    path, ``None`` never occurs and ``±inf`` does), so clamping it here would be a
    silent bound-*changing* edit riding along with a marshaling fix. It is not
    hypothetical — clamping ``hda``'s infinite bounds to ``1e20`` moves its root LP
    bound from ``-64675.2`` to ``-11308304.4``, 175x weaker, and moves ``contvar``'s
    too. Narrowing the sentinel convention is its own change under CLAUDE.md §5,
    with its own differential gate; this function only fixes ``None``.

    The previous code built these arrays with
    ``np.array([hi for _, hi in bounds], dtype=np.float64)``, which turns a ``None``
    into ``nan`` **silently**: every caller holding an open-above column (169 of the
    280 columns of the ``rsyn0840m`` LP/NLP-BB master, issue #1060) had its solve
    rejected by the #1008 NaN guard from deep inside the driver, reported against a
    standard-form column index that does not name the offending variable. NaN is
    refused here instead — loudly, and in the caller's own index space (§3).
    """
    if bounds is None:
        return np.zeros(n, dtype=np.float64), np.full(n, _INF, dtype=np.float64)
    if len(bounds) != n:
        raise ValueError(
            f"bounds has {len(bounds)} entries but c has {n} columns; "
            "one (lo, hi) pair per structural variable is required"
        )
    lb = np.empty(n, dtype=np.float64)
    ub = np.empty(n, dtype=np.float64)
    for j, (lo, hi) in enumerate(bounds):
        lo_f = -_INF if lo is None else float(lo)
        hi_f = _INF if hi is None else float(hi)
        if np.isnan(lo_f) or np.isnan(hi_f):
            raise ValueError(
                f"bounds[{j}] = ({lo!r}, {hi!r}) contains NaN; an LP bound must be "
                "finite, None, or +/-inf (NaN is read as both open and closed by "
                "different guards -- see issue #1008)"
            )
        lb[j] = lo_f
        ub[j] = hi_f
    return lb, ub


def _marshal_std_form(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    bounds: Optional[BoundList],
    integrality: Optional[np.ndarray],
) -> _StdForm:
    """Marshal ``A_ub x <= b_ub, A_eq x == b_eq`` into the driver's CSC standard form.

    Shared verbatim by :func:`solve_milp` and :func:`solve_milp_with_lazy_cuts` so
    the two entry points cannot drift: a lazy-cut row is written in the *caller's*
    ``x`` space, which only means anything if both paths lay the columns out the
    same way (structural first, then one slack per row).
    """
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    # Assemble all rows as `<=` (A_eq becomes a pair of `<=` rows) then slack.
    # SPARSE throughout: a dense `[A_ub | I]` would materialize an m×(n+m) matrix
    # (~73 GB for qap's 85k×21k McCormick relaxation) that the Rust driver never
    # needs — it consumes CSC. We keep every block sparse and hand the driver the
    # CSC triplets of the standard-form matrix; nothing is ever densified here.
    blocks: list[sp.spmatrix] = []
    rhs: list[float] = []
    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        au = (
            sp.csr_matrix(cast("sp.spmatrix", A_ub))
            if sp.issparse(A_ub)
            else sp.csr_matrix(np.asarray(A_ub, dtype=np.float64).reshape(-1, n))
        )
        blocks.append(au)
        rhs.extend(np.asarray(b_ub, dtype=np.float64).ravel().tolist())
    if A_eq is not None and b_eq is not None and np.size(b_eq) > 0:
        ae = (
            sp.csr_matrix(cast("sp.spmatrix", A_eq))
            if sp.issparse(A_eq)
            else sp.csr_matrix(np.asarray(A_eq, dtype=np.float64).reshape(-1, n))
        )
        be = np.asarray(b_eq, dtype=np.float64).ravel()
        blocks.append(ae)
        rhs.extend(be.tolist())
        blocks.append(-ae)
        rhs.extend((-be).tolist())

    if blocks:
        a_ub_sp = sp.vstack(blocks, format="csr")
    else:
        a_ub_sp = sp.csr_matrix((0, n), dtype=np.float64)
    b_vec = np.asarray(rhs, dtype=np.float64)
    m = a_ub_sp.shape[0]

    # Standard form A_eq z = b with one slack per row: [A_ub | I] z = b_ub, built
    # directly as CSC (never densified). ``sort_indices`` gives ascending row
    # order within each column, which ``SparseCols::from_csc`` requires.
    a_std_sp = sp.hstack([a_ub_sp, sp.identity(m, dtype=np.float64, format="csr")], format="csc")
    a_std_sp.sum_duplicates()
    a_std_sp.sort_indices()
    csc_col_ptr = np.ascontiguousarray(a_std_sp.indptr, dtype=np.int64)
    csc_row_idx = np.ascontiguousarray(a_std_sp.indices, dtype=np.int64)
    csc_vals = np.ascontiguousarray(a_std_sp.data, dtype=np.float64)

    lb, ub = _marshal_col_bounds(bounds, n)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, _INF)])
    c_std = np.concatenate([c_arr, np.zeros(m)])

    if integrality is not None:
        int_mask = np.asarray(integrality, dtype=np.int64).ravel()
        int_cols = np.flatnonzero(int_mask != 0).astype(np.int64)
    else:
        int_cols = np.zeros(0, dtype=np.int64)

    # Pure-LP short-circuit (THRU-2b): when there are no integer columns this is a
    # plain LP, yet the MILP driver still runs its integer-search machinery — root
    # cut rounds, GMI, primal heuristics, strong branching. With no integer
    # variables none of that can fire (GMI needs a fractional integer; the
    # heuristics round nothing; there is no candidate to branch on), so it is pure
    # overhead on the root LP whose optimum/infeasibility is the whole answer. This
    # path is the fallback the McCormick node relaxer reaches when the warm sparse
    # simplex breaks down numerically on a hard, ill-conditioned lifted LP
    # (``solve_lp_warm_csc_py`` -> ``numerical`` at iters=0); the driver's LP
    # presolve then decides it, but the wasted cut/heuristic passes inflate the
    # solve (nvs24 node LP 10.9 s -> 5.5 s with the machinery off). Turning the
    # machinery off on a genuine LP is bound-neutral by construction: the root LP
    # optimum and the infeasibility verdict are unchanged — only inert integer-side
    # work is skipped. It never triggers when ``int_cols`` is non-empty.
    _pure_lp = int(int_cols.size) == 0
    _lp_kwargs: dict = (
        dict(
            root_cuts=0,
            cut_rounds=0,
            gmi_cuts=False,
            heuristics=False,
            strong_branch=False,
        )
        if _pure_lp
        # A genuine MILP marshals at the LEGACY budget unconditionally. The #1066
        # strong budget is not a blanket default: it is reached only through
        # :func:`solve_milp`'s escalation, which spends it on the masters a cheap
        # probe fails to close. Marshaling it here would also hand it to
        # :func:`solve_milp_with_lazy_cuts`, a path no #1066 measurement covers.
        else dict(_LEGACY_CUT_PROFILE)
    )
    return _StdForm(
        m=m,
        n=n,
        col_ptr=csc_col_ptr,
        row_idx=csc_row_idx,
        vals=csc_vals,
        b=np.ascontiguousarray(b_vec),
        lb=np.ascontiguousarray(lb_std),
        ub=np.ascontiguousarray(ub_std),
        c=np.ascontiguousarray(c_std),
        int_cols=np.ascontiguousarray(int_cols),
        lp_kwargs=_lp_kwargs,
    )


def _certified_bound(obj: float, bound: float) -> float:
    """The dual bound to publish for a driver exit of ``"optimal"``.

    ``"optimal"`` from the Rust driver means *optimal within* ``gap_tol``
    (:func:`TreeManager::gap` <= ``opts.gap_tol``), NOT gap zero. This module used
    to publish ``bound = objective`` there, on the reasoning that a proven optimum
    has incumbent == dual bound. That reasoning holds only at gap zero; at any
    positive tolerance it over-states the dual bound by up to that tolerance, and
    the engine's gap is normalised by ``max(|incumbent|, 1.0)`` — so on an
    objective of magnitude 0.1 a 1e-4 "relative" tolerance is 1e-4 ABSOLUTE,
    i.e. 1e-3 relative. That is not a rounding artefact: measured on the #1141
    convex-MINLP panel, an OA master whose true optimum was −0.10091959 exited
    ``optimal`` with incumbent −0.10088167 and published −0.10088167 as a lower
    bound — 3.8e-5 ABOVE the true optimum, a false certificate that
    ``solve_lp_nlp_bb`` then republished as the MINLP's dual bound.

    The engine's own ``global_lower_bound`` is the honest answer: the frontier
    minimum, already floored by the unresolved-fathom floor and capped at the
    incumbent (``TreeManager::update_global_lower_bound``). It equals the
    incumbent exactly when the tree really did drain, so a genuinely proven
    optimum loses nothing. Take the incumbent only when the engine's bound is not
    finite (an empty frontier reports ``+inf``), and never publish a bound above
    the incumbent.
    """
    if not np.isfinite(bound):
        return float(obj)
    return min(float(bound), float(obj))


def solve_milp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[BoundList] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 1_000_000,
) -> MILPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq, bounds, integrality``
    with the Rust warm-started-simplex B&B.

    Mirrors :func:`discopt.solvers.milp_pounce.solve_milp`. The returned
    ``objective`` is the engine's dual lower bound (see module docstring).
    """
    try:
        from discopt._rust import solve_milp_csc_py
    except ImportError as err:  # pragma: no cover - exercised via the selector
        raise SimplexBackendUnavailable(
            "discopt._rust.solve_milp_csc_py is unavailable; build the Rust extension"
        ) from err

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    std = _marshal_std_form(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)
    m = std.m

    # Interactive debugger: install the Rust checkpoint hook only when a debugger
    # is attached now, so the pure-Rust search stays bound-neutral otherwise.
    from discopt import debug as _debug

    def _drive(
        *,
        cut_kwargs: dict,
        node_cap: int,
        budget: Optional[float],
        seed: Optional[np.ndarray],
    ) -> MILPResult:
        """One call into the Rust driver, decoded into a :class:`MILPResult`."""
        status, x_full, obj, bound, nodes, _iters = solve_milp_csc_py(
            std.c,
            m,
            n + m,  # total columns: structural + one slack per row
            std.col_ptr,
            std.row_idx,
            std.vals,
            std.b,
            std.lb,
            std.ub,
            std.int_cols,
            n,  # n_struct: structural columns precede the slacks
            0.0,  # obj_const: caller (MilpRelaxationModel) applies its own offset
            int(node_cap),
            float(gap_tolerance),
            # #928: pass ``None`` for "no limit" and the number — INCLUDING an
            # exact 0.0 — for a real budget. The previous
            # ``0.0 if time_limit is None`` spelling collapsed the two: the
            # binding mapped 0.0 back to "no deadline", so a caller whose shared
            # budget was already spent by earlier attempts
            # (``MilpRelaxationModel.solve`` under ``DISCOPT_LP_WARM_DEADLINE``)
            # launched an *unbounded* B&B at the one moment it must not start at
            # all. The escalation below relies on this too: its second attempt
            # gets the budget the probe left, and an exhausted budget must return
            # at once rather than run unbounded.
            time_limit_s=None if budget is None else max(0.0, float(budget)),
            initial_incumbent=seed,
            debug_hook=_debug.rust_hook(),
            **cut_kwargs,
        )

        if status == "infeasible":
            return MILPResult(status=SolveStatus.INFEASIBLE, node_count=int(nodes))
        if status == "unbounded":
            return MILPResult(status=SolveStatus.UNBOUNDED, node_count=int(nodes))

        x_struct = np.asarray(x_full, dtype=np.float64)[:n]
        if status == "optimal":
            # Optimal WITHIN ``gap_tolerance`` — publish the engine's dual bound,
            # not the incumbent (see :func:`_certified_bound`).
            return MILPResult(
                status=SolveStatus.OPTIMAL,
                x=x_struct,
                objective=float(obj),
                bound=_certified_bound(obj, bound),
                node_count=int(nodes),
            )

        # node_limit / feasible: ``objective`` is the incumbent (upper bound) and
        # ``bound`` is the engine's dual lower bound (sound) if finite. Callers that
        # need a lower bound must read ``bound``, never ``objective``.
        return MILPResult(
            status=SolveStatus.ITERATION_LIMIT,
            x=x_struct,
            objective=float(obj) if np.isfinite(obj) else None,
            bound=float(bound) if np.isfinite(bound) else None,
            node_count=int(nodes),
        )

    legacy_kwargs = dict(_LEGACY_CUT_PROFILE)
    if not milp_cut_budget_enabled() or std.int_cols.size == 0:
        # Flag off, or a pure LP whose short-circuit already replaced the cut
        # budget with "no integer machinery at all": one solve, legacy budget,
        # byte-for-byte the pre-#1066 path.
        return _drive(cut_kwargs=std.lp_kwargs, node_cap=max_nodes, budget=time_limit, seed=None)

    # --- #1066 escalation --------------------------------------------------
    # Measurement (docs/dev/performance-plan.md §23) killed the obvious fix of
    # simply raising the root cut budget for every MILP. Raising it is a large
    # win where the master is hard (rsyn0830m master: 49.2 s -> 0.3 s) and a
    # large *loss* where it is easy (tls2 masters close in 241-595 nodes at the
    # legacy budget and in 0.0 s; at the strong budget the extra rows derail the
    # search and it is still open at 60 s). Neither budget dominates, and nothing
    # in the master's shape predicts which class it is in — so the policy
    # measures instead of guessing: run the cheap budget under a node cap, and
    # spend the strong budget only on a master that cap fails to close.
    #
    # Soundness: both attempts bound the same MILP, so either one's dual bound is
    # valid and either one's incumbent is feasible. The merge takes the better of
    # each, which cannot invent a bound neither attempt proved.
    t_probe = time.perf_counter()
    probe = _drive(
        cut_kwargs=legacy_kwargs,
        node_cap=min(int(max_nodes), _PROBE_MAX_NODES),
        budget=time_limit,
        seed=None,
    )
    if probe.status in _PROBE_FINAL_STATUSES:
        # The cheap budget proved it. This is the tls2 class, and it is the
        # common case — escalating here would be pure overhead.
        return probe

    remaining = None
    if time_limit is not None:
        remaining = max(0.0, float(time_limit) - (time.perf_counter() - t_probe))
        if remaining <= 0.0:
            # The probe spent the whole budget. A second attempt with no time
            # cannot improve on it, and #928 makes 0.0 a real (spent) budget
            # rather than "unlimited" — return what we proved.
            return probe

    second = _drive(
        cut_kwargs=dict(_STRONG_CUT_PROFILE),
        node_cap=max(1, int(max_nodes) - int(probe.node_count)),
        budget=remaining,
        # Hand over the probe's incumbent so the strong attempt starts with the
        # pruning power the probe already paid for. The driver re-validates the
        # seed and silently drops one it cannot prove feasible, so this can never
        # manufacture a certificate.
        seed=probe.x if probe.x is not None else None,
    )
    return _merge_escalation(probe, second)


def _merge_escalation(probe: MILPResult, second: MILPResult) -> MILPResult:
    """Combine the #1066 probe and strong attempts into one sound result.

    Both attempts bound the *same* MILP, so the tightest valid dual bound is the
    larger of the two (the driver minimizes) and the best incumbent is the
    smaller objective. Node counts add: the caller is told what the search cost.

    A ``second`` that reached a proof is returned as the proof it is — with one
    exception. If it contradicts a feasible point the probe actually found (it
    claims infeasibility, or an optimum worse than that point), the two arms
    disagree about the same problem and at least one is wrong. That is a
    soundness question, not a performance one, so the escalation declines to be
    the arm that produces a false certificate: it keeps the probe's result, which
    is backed by a point, and says so loudly. CLAUDE.md §1/§3.
    """
    probe_obj = probe.objective
    if probe_obj is not None and probe.x is not None:
        contradiction = (
            second.status == SolveStatus.INFEASIBLE
            or second.status == SolveStatus.UNBOUNDED
            or (
                second.status == SolveStatus.OPTIMAL
                and second.objective is not None
                and second.objective > probe_obj + 1e-6 * (1.0 + abs(probe_obj))
            )
        )
        if contradiction:
            logger.error(
                "#1066 escalation: the strong-cut attempt returned %s (objective %r) on a "
                "master where the probe holds a feasible point of objective %.12g. The two "
                "cut budgets disagree about the same MILP; keeping the probe result. This "
                "indicates an invalid cut and should be reported.",
                second.status,
                second.objective,
                probe_obj,
            )
            return replace(probe, node_count=probe.node_count + second.node_count)

    if second.status in _PROBE_FINAL_STATUSES:
        return replace(second, node_count=probe.node_count + second.node_count)

    bounds = [b for b in (probe.bound, second.bound) if b is not None]
    best_bound = max(bounds) if bounds else None

    best = probe
    if probe.objective is None or (
        second.objective is not None and second.objective < probe.objective
    ):
        best = second

    return MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=best.x,
        objective=best.objective,
        bound=best_bound,
        node_count=probe.node_count + second.node_count,
    )


def solve_milp_with_lazy_cuts(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[BoundList] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 1_000_000,
    lazy_callback: Optional[Callable[[np.ndarray], object]] = None,
    node_callback: Optional[Callable[[np.ndarray], object]] = None,
    node_hook_rounds: int = _NODE_HOOK_ROUNDS_DEFAULT,
    node_hook_cut_cap: int = _NODE_HOOK_CUT_CAP_DEFAULT,
    terminate_callback: Optional[Callable[[dict[str, object]], bool]] = None,
    mip_start: Optional[np.ndarray] = None,
) -> MILPResult:
    """Single-tree MILP solve with a lazy-constraint separator, on the Rust simplex.

    The in-house counterpart of :func:`discopt.solvers.gurobi.solve_milp_with_lazy_cuts`
    (issue #1060): ``lazy_callback`` is called with every integer-feasible point the
    search finds, before that point can become the incumbent, and returns an iterable
    of ``(coefficients, rhs)`` rows in ``coefficients @ x <= rhs`` form. An empty
    return (or ``None``) accepts the point; a non-empty return rejects it, adds the
    rows to the shared relaxation, and puts the node **back in the search** — a veto
    is not a proof that the box is empty, so the node is re-queued rather than
    fathomed.

    ``node_callback`` (#1141) is the *fractional*-node counterpart, the in-house
    equivalent of Gurobi's MIPNODE user cuts: it is called with each fractional node
    relaxation solution and returns rows in the same ``coefficients @ x <= rhs``
    form, which must be **globally valid** (for a convex MINLP, the first-order
    linearization ``g(x̄) + ∇g(x̄)·(x − x̄) ≤ 0`` of a violated constraint is a
    supporting hyperplane and therefore is). The rows fold into the shared
    relaxation and the node re-solves against them, so a node costs one LP plus one
    gradient evaluation instead of a full NLP. ``node_hook_rounds`` caps the
    separation rounds one node may run (``0`` disables the hook) and
    ``node_hook_cut_cap`` caps the rows it may add over the whole solve. Unlike a
    lazy veto -- which is mandatory, since it is the only thing keeping a point out
    of the incumbent -- a fractional separation only tightens a relaxation, so
    exhausting either budget is benign and never touches certification.

    ``terminate_callback`` (#1141) is consulted at the driver's per-iteration
    checkpoint -- the in-house analogue of the HiGHS master's per-restart
    check-in. It is handed the snapshot shape the other backends pass
    (``dual_bound``, ``restarts``, ``lazy_cuts``), and returning true stops the
    search, keeping the incumbent and the bound and reporting the result
    **uncertified**, exactly as the driver's own time limit does. ``restarts`` is
    always ``0``, and that is a fact rather than a placeholder: this is a true
    single tree, so there is nothing to restart.

    This used to be refused, on the grounds that "the driver enforces
    ``time_limit`` itself and has no callback-termination hook". The first half is
    true and irrelevant; the second was stale. The driver has had a per-iteration
    checkpoint carrying ``incumbent``/``bound``/``gap``/``elapsed`` with a ``Stop``
    control since the interactive debugger landed, already exposed to Python as
    ``debug_hook``, so this is a composition rather than a new capability. The
    refusal was what stopped #1141's convex-MINLP route from targeting a
    discopt-native master: the #1066 route progress guard installs a
    ``termination_hook``, the refusal raised, and the whole route fell back to the
    spatial path.

    ``callback_stats`` reports ``mipsol_calls`` (separator invocations),
    ``lazy_cuts`` (rows the separator returned), ``driver_lazy_calls`` (the same
    count as the driver saw it) and ``lazy_requeues`` (vetoed nodes put back in
    the search). ``mipsol_calls == 0`` means the separator never saw a point — NOT
    that it accepted everything (CLAUDE.md §6). With a ``node_callback`` attached it
    also reports ``mipnode_calls`` (fractional separator invocations, counted on
    this side), ``node_cuts`` (rows it returned) and ``driver_node_cuts`` (rows the
    driver actually folded in, after dedup and the cap). ``mipnode_calls == 0`` is
    the same kind of signal: the fractional separator never ran, which is NOT the
    same as it having found nothing to cut.
    """
    try:
        from discopt._rust import solve_milp_lazy_csc_py
    except ImportError as err:  # pragma: no cover - exercised via the selector
        raise SimplexBackendUnavailable(
            "discopt._rust.solve_milp_lazy_csc_py is unavailable; build the Rust extension"
        ) from err

    if lazy_callback is None:
        raise ValueError(
            "solve_milp_with_lazy_cuts requires lazy_callback; use solve_milp for a "
            "plain MILP solve"
        )
    if node_callback is not None and (node_hook_rounds <= 0 or node_hook_cut_cap <= 0):
        # A separator wired in with a zero budget can never fire, and would report
        # ``mipnode_calls == 0`` -- indistinguishable from a separator that ran and
        # found nothing (CLAUDE.md §6). Refuse rather than silently dropping the
        # caller's cut strategy.
        raise ValueError(
            "node_callback was supplied with node_hook_rounds="
            f"{node_hook_rounds}/node_hook_cut_cap={node_hook_cut_cap}: a zero budget "
            "means the separator can never fire, which is indistinguishable from one "
            "that found nothing. Pass positive budgets, or omit node_callback."
        )

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    std = _marshal_std_form(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)
    m = std.m

    # Separator invocations and vetoes, counted on this side too so the caller's
    # anti-vacuity check does not depend on the binding's bookkeeping alone.
    stats: dict[str, object] = {
        "mipsol_calls": 0,
        "mipnode_calls": 0,
        "lazy_cuts": 0,
        "node_cuts": 0,
        "terminated": False,
        "terminate_context": None,
    }

    def _separate(x_full: np.ndarray) -> list[tuple[np.ndarray, float]]:
        stats["mipsol_calls"] = int(cast(int, stats["mipsol_calls"])) + 1
        out = _rows_from(lazy_callback, x_full, "lazy_callback")
        stats["lazy_cuts"] = int(cast(int, stats["lazy_cuts"])) + len(out)
        return out

    def _rows_from(callback, x_full: np.ndarray, kind: str) -> list[tuple[np.ndarray, float]]:
        """Marshal one separator return into the driver's row list.

        Shared by the integer-feasible and fractional separators so the two cannot
        drift in what they accept: the callback sees the master's own structural
        vector (the driver works in standard form and hands over
        ``[structural | slacks]``, so the slacks are trimmed rather than leaking a
        layout the caller never declared), and every row is checked against the
        structural width.
        """
        x_arr = np.asarray(x_full, dtype=np.float64).ravel()
        if x_arr.shape[0] < n:
            raise ValueError(
                f"driver returned a {x_arr.shape[0]}-vector for a master with {n} "
                "structural columns"
            )
        rows = callback(x_arr[:n])
        if rows is None:
            return []
        if not isinstance(rows, Iterable):
            raise TypeError(
                f"{kind} must return None or an iterable of (coefficients, rhs) "
                f"rows, got {type(rows).__name__}"
            )
        out: list[tuple[np.ndarray, float]] = []
        for coeffs, rhs in rows:
            row = np.asarray(coeffs, dtype=np.float64).ravel()
            if row.shape[0] != n:
                raise ValueError(
                    f"{kind} returned a row with {row.shape[0]} coefficients "
                    f"but the master has {n} variables"
                )
            out.append((row, float(rhs)))
        return out

    def _separate_node(x_full: np.ndarray) -> list[tuple[np.ndarray, float]]:
        assert node_callback is not None  # guarded by `_node_hook` below
        stats["mipnode_calls"] = int(cast(int, stats["mipnode_calls"])) + 1
        out = _rows_from(node_callback, x_full, "node_callback")
        stats["node_cuts"] = int(cast(int, stats["node_cuts"])) + len(out)
        return out

    _node_hook = None if node_callback is None else _separate_node

    # The driver has ONE hook slot and the interactive debugger already owns it,
    # so a caller's ``terminate_callback`` COMPOSES with it rather than replacing
    # it: silently dropping either would be the "callback that can never fire"
    # this function refuses everywhere else. Either one voting to stop stops the
    # search, and with neither attached the slot stays ``None`` so the search is
    # bit-for-bit the unhooked one.
    from discopt import debug as _debug

    _debug_hook = _debug.rust_hook()
    _term_state: dict[str, object] = {"calls": 0, "terminated": False}

    def _checkin(state: dict) -> bool:
        stop = False
        if _debug_hook is not None:
            stop = bool(_debug_hook(state))
        if terminate_callback is not None and str(state.get("checkpoint")) == "after_select":
            # One checkpoint of the five: this is a budget decision, and asking it
            # at every checkpoint would multiply its cost and make its own call
            # count meaningless as a signal.
            _term_state["calls"] = int(cast(int, _term_state["calls"])) + 1
            if bool(
                terminate_callback(
                    {
                        "dual_bound": state.get("bound"),
                        "restarts": 0,
                        "lazy_cuts": stats["lazy_cuts"],
                    }
                )
            ):
                _term_state["terminated"] = True
                stop = True
        return stop

    _checkin_arg = None if (_debug_hook is None and terminate_callback is None) else _checkin

    # A caller-supplied start is a plain incumbent candidate: the driver validates
    # it against the constraints AND offers it to the separator before seeding, so
    # an infeasible or lazily-excluded start cannot prune the true optimum.
    #
    # The seed is a **structural** point of length ``n_struct``: that is what
    # ``validate_seed_incumbent`` checks (`if seed.len() != ns { return None }`)
    # and it derives the slack activity itself from the row residuals. This used
    # to pad the seed to ``n + m`` with zero slacks, which is the standard-form
    # layout of every *other* array here but the wrong length for the seed -- so
    # the validator dropped it on the length test and every ``mip_start`` on this
    # path was silently ignored (#1060). Rejection there is deliberately silent
    # (a seed that cannot be proven feasible must never prune the optimum), which
    # is exactly why a marshaling mistake could not announce itself. A wrong
    # length is a caller bug, not an unverifiable point, so it is refused here
    # instead of being quietly dropped downstream.
    seed = None
    if mip_start is not None:
        seed_arr = np.asarray(mip_start, dtype=np.float64).ravel()
        if seed_arr.shape[0] != n:
            raise ValueError(
                f"mip_start has {seed_arr.shape[0]} entries but the master has {n} "
                "structural variables; supply a point over the structural columns "
                "only (the driver derives the slacks from the row residuals)"
            )
        seed = np.ascontiguousarray(seed_arr)

    t0 = time.perf_counter()
    (
        status,
        x_full,
        obj,
        bound,
        nodes,
        _iters,
        lazy_calls,
        lazy_requeues,
        node_calls,
        node_cuts_added,
    ) = solve_milp_lazy_csc_py(
        std.c,
        m,
        n + m,
        std.col_ptr,
        std.row_idx,
        std.vals,
        std.b,
        std.lb,
        std.ub,
        std.int_cols,
        n,
        _separate,
        0.0,  # obj_const
        int(max_nodes),
        float(gap_tolerance),
        time_limit_s=None if time_limit is None else max(0.0, float(time_limit)),
        initial_incumbent=seed,
        debug_hook=_checkin_arg,
        node_callback=_node_hook,
        node_hook_rounds=int(node_hook_rounds) if _node_hook is not None else 0,
        node_hook_cut_cap=int(node_hook_cut_cap) if _node_hook is not None else 0,
        **std.lp_kwargs,
    )
    wall_time = time.perf_counter() - t0
    # The binding counts calls the driver actually made; this side counts calls it
    # actually served. A mismatch means a call was lost between the two and the
    # separator's veto may not have been applied — report the driver's count and
    # let the discrepancy surface rather than papering over it.
    stats["driver_lazy_calls"] = int(lazy_calls)
    stats["lazy_requeues"] = int(lazy_requeues)
    # Reported unconditionally so a caller can tell "the check-in ran and never
    # asked to stop" from "the check-in never ran" -- the same anti-vacuity rule
    # the separator counters follow (CLAUDE.md §6).
    stats["terminate_calls"] = int(cast(int, _term_state["calls"]))
    stats["terminated"] = bool(_term_state["terminated"])
    # Same two-sided bookkeeping for the fractional separator: this side counts
    # the calls it served, the driver counts the rows it actually folded in (after
    # dedup and the cap). A `node_cuts` far above `driver_node_cuts` means the
    # separator is re-deriving rows the relaxation already carries.
    stats["driver_node_calls"] = int(node_calls)
    stats["driver_node_cuts"] = int(node_cuts_added)

    if status == "infeasible":
        return MILPResult(
            status=SolveStatus.INFEASIBLE,
            node_count=int(nodes),
            wall_time=wall_time,
            callback_stats=stats,
        )
    if status == "unbounded":
        return MILPResult(
            status=SolveStatus.UNBOUNDED,
            node_count=int(nodes),
            wall_time=wall_time,
            callback_stats=stats,
        )

    x_struct = np.asarray(x_full, dtype=np.float64)[:n]
    if status == "optimal":
        # Optimal WITHIN ``gap_tolerance``: the dual bound is the engine's, and the
        # gap that goes with it is the real one, not a hardcoded 0.0 (which is what
        # made an OA caller read a tolerance-wide interval as a closed certificate).
        # See :func:`_certified_bound`.
        certified = _certified_bound(obj, bound)
        return MILPResult(
            status=SolveStatus.OPTIMAL,
            x=x_struct,
            objective=float(obj),
            bound=certified,
            gap=abs(float(obj) - certified) / max(1.0, abs(float(obj))),
            node_count=int(nodes),
            wall_time=wall_time,
            callback_stats=stats,
        )
    return MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=x_struct,
        objective=float(obj) if np.isfinite(obj) else None,
        bound=float(bound) if np.isfinite(bound) else None,
        node_count=int(nodes),
        wall_time=wall_time,
        callback_stats=stats,
    )


def solve_lp_warm_std(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    bounds: Optional[BoundList],
    in_basis: Optional[tuple[np.ndarray, np.ndarray]] = None,
    *,
    return_cert: bool = False,
    time_limit: Optional[float] = None,
):
    """Warm-startable **pure-LP** solve of ``min c^T x s.t. A_ub x <= b_ub, bounds``.

    Marshals the ``A_ub x <= b_ub`` form into standard form ``[A_ub | I] z = b_ub``
    (one explicit slack per row, structural columns first) and calls the Rust
    ``solve_lp_warm_py``. ``in_basis`` is a ``(col_status, basic_vars)`` pair from a
    previous solve of a *prefix* of the same column set (rows since appended); Rust
    extends it by making the appended slacks basic and dual-simplex re-optimizes.

    Returns ``(result, out_basis)``. ``out_basis`` is the final ``(col_status,
    basic_vars)`` to thread into the next re-solve (``None`` when the LP is not
    optimal). ``result`` is ``None`` for ``iter_limit``/``numerical`` exits so the
    caller can fall back to a cold/HiGHS path. Soundness: the dual simplex
    converges to the LP optimum exactly as a cold solve (a bad basis is ignored
    inside Rust), so the returned objective/bound is unchanged — only the speed is.

    ``time_limit`` bounds this one LP in wall-clock seconds (``None`` = unbounded,
    the historical behaviour). It reaches ``SimplexOptions::deadline``, which the
    dual and primal pivot loops poll, so a stalling LP yields a limit exit instead of
    running past a budget its caller already computed. Without it every per-node LP
    budget on this path was silently dropped: on nvs24 one node LP ran **47 s**
    against the 0.2 s its caller passed (59 494 degenerate dual pivots, Bland never
    activated), turning a 3.9 s solve budget into 53 s. Soundness is untouched — a
    deadline exit returns ``None`` (an ``iter_limit``-class result) exactly as a
    pivot-cap exit does, never a bound.

    When ``return_cert`` is set, returns ``(result, out_basis, cert)`` with a
    :class:`LpWarmCert` built from the simplex's own duals / Farkas ray: a
    rigorous safe lower bound on an ``optimal`` solve, and an independently
    verified infeasibility proof on an ``infeasible`` one — both without a second
    external solve (issue #356).
    """
    from discopt._rust import solve_lp_warm_csc_py

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    if A_ub is not None and b_ub is not None and np.size(b_ub) > 0:
        a_struct = (
            sp.csc_matrix(A_ub)
            if sp.issparse(A_ub)
            else sp.csc_matrix(np.asarray(A_ub, dtype=np.float64).reshape(-1, n))
        )
        b_vec = np.asarray(b_ub, dtype=np.float64).ravel()
    else:
        a_struct = sp.csc_matrix((0, n), dtype=np.float64)
        b_vec = np.zeros(0, dtype=np.float64)
    m = a_struct.shape[0]

    # Standard form ``[A_ub | I_m] z = b`` (one slack per row) assembled directly in
    # CSC, with the slack identity left implicit-sparse — the lifted relaxations are
    # ~0.3% dense, so the old dense ``a_std`` (np.zeros((m, n+m)) + np.eye(m)) was a
    # 431MB / O(m^2) allocation per solve that the Rust side then re-scanned. The
    # sparse matrix flows straight to the CSC-native simplex and the safe-bound /
    # Farkas helpers (both accept a SciPy sparse matrix). (issue #356)
    if m > 0:
        a_std = sp.hstack(
            [a_struct, sp.identity(m, format="csc", dtype=np.float64)], format="csc"
        ).tocsc()
    else:
        a_std = a_struct.tocsc()

    lb, ub = _marshal_col_bounds(bounds, n)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, _INF)])
    c_std = np.concatenate([c_arr, np.zeros(m)])

    # #928: a COLD solve carrying a finite deadline starts the DUAL simplex from
    # the sign-matched slack basis when that basis is dual-feasible, because only
    # the dual loop has an anytime (monotone, bankable) lower bound to return when
    # the deadline fires — the cold primal proves nothing usable mid-run. An
    # ineligible LP (an open bound on a selected side) keeps the primal path.
    #
    # ``lp_cold_dual_start`` (default OFF) extends the SAME start to a cold solve
    # with no deadline, for a different reason: speed. On equality-rich lifted
    # relaxations the cold primal grinds — on the RLT-on QPLIB_1157 root LP it
    # exhausted ``max_iter`` after >150 s where this start returns the identical
    # optimum in 6.2 s. See ``SolverTuning.lp_cold_dual_start`` for the 8-LP table.
    # With the flag off, a deadline-free solve stays bit-identical to before.
    if in_basis is None and (
        (time_limit is not None and np.isfinite(time_limit)) or _cold_dual_start_enabled()
    ):
        in_basis = _dual_start_slack_basis(c_arr, lb, ub, m)

    cs0 = None if in_basis is None else np.ascontiguousarray(in_basis[0], dtype=np.int8)
    bv0 = None if in_basis is None else np.ascontiguousarray(in_basis[1], dtype=np.int64)

    status, x_full, obj, _iters, cs, bv, dual, ray = solve_lp_warm_csc_py(
        np.ascontiguousarray(c_std),
        m,
        n + m,
        np.ascontiguousarray(a_std.indptr, dtype=np.int64),
        np.ascontiguousarray(a_std.indices, dtype=np.int64),
        np.ascontiguousarray(a_std.data, dtype=np.float64),
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        cs0,
        bv0,
        time_limit_s=time_limit,
    )

    def _result_basis_cert():
        x_struct = np.asarray(x_full, dtype=np.float64)[:n]
        if status == "optimal":
            # Rigorous safe lower bound from the simplex's own row duals (sound at
            # any conditioning — never above the true optimum), reported as the
            # ``bound`` so a caller can fathom on it without an independent solve.
            # ``objective`` stays the raw vertex value; ``bound`` is the certified
            # one (the safe bound, clamped to never exceed the raw value, since a
            # well-conditioned raw value <= safe bound is itself sound and tighter).
            safe = _safe_lp_lower_bound(dual, c_std, a_std, b_vec, lb_std, ub_std)
            bound = float(obj) if safe is None else min(float(obj), float(safe))
            return (
                MILPResult(
                    status=SolveStatus.OPTIMAL,
                    x=x_struct,
                    objective=float(obj),
                    bound=bound,
                    node_count=0,
                ),
                (np.asarray(cs), np.asarray(bv)),
                LpWarmCert(
                    safe_bound=(None if safe is None else float(safe)),
                    farkas_certified=False,
                    # Additive marginals (cert:T2.4a): row duals ``y`` and the final
                    # column status. Reduced costs ``d = c - A^T y`` are derived by
                    # the consumer from ``dual``; exposing the raw dual keeps this a
                    # pure plumbing change (no new math here).
                    dual=np.asarray(dual, dtype=np.float64) if dual is not None else None,
                    col_status=np.asarray(cs) if cs is not None else None,
                ),
            )
        if status == "infeasible":
            certified = _farkas_certified_std(dual, a_std, b_vec, lb_std, ub_std)
            return (
                MILPResult(status=SolveStatus.INFEASIBLE, node_count=0),
                None,
                LpWarmCert(safe_bound=None, farkas_certified=certified),
            )
        if status == "unbounded":
            return (
                MILPResult(status=SolveStatus.UNBOUNDED, node_count=0),
                None,
                LpWarmCert(safe_bound=None, farkas_certified=False),
            )
        # iter_limit / numerical: no clean optimum — signal fallback (result None).
        # But if the engine exported a dual candidate from the broken basis (#517),
        # carry a Neumaier–Shcherbina safe lower bound in the cert. It is valid for
        # ANY multiplier vector, so a drifted-basis dual only loosens it — never
        # lifts it above the optimum. The caller (behind a default-OFF flag) uses it
        # only as a last-resort floor when nothing else produced a bound.
        safe = None
        if dual is not None and np.size(dual):
            safe = _safe_lp_lower_bound(dual, c_std, a_std, b_vec, lb_std, ub_std)
        # #671: on a numerical break-down, optionally recover a *tight* bound by
        # re-solving a few RHS-regularized neighbours and keeping the tightest NS
        # bound their duals imply (max over the sweep — always a valid lower bound,
        # so never looser than candidate A above, never unsound). Flag default-OFF;
        # in-house simplex only; the NS bound is evaluated against the ORIGINAL
        # ``b_vec`` for every tau.
        from discopt.solver_tuning import current as _tuning_current

        if _tuning_current().lp_iterative_refinement:
            refined = _refined_safe_bound_regularized(
                solve_lp_warm_csc_py,
                c_std,
                a_std,
                b_vec,
                lb_std,
                ub_std,
                m,
                n,
                time_limit=time_limit,
            )
            if refined is not None and (safe is None or refined > safe):
                safe = refined
        return (
            None,
            None,
            LpWarmCert(
                safe_bound=(None if safe is None else float(safe)),
                farkas_certified=False,
            ),
        )

    result, out_basis, cert = _result_basis_cert()
    if return_cert:
        return result, out_basis, cert
    return result, out_basis
