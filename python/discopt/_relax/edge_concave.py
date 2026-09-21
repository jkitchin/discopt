"""Edge-concave (vertex-polyhedral) relaxation of quadratic blocks.

A function ``f`` is *edge-concave* on a box when it is concave along each
coordinate direction (``d^2 f / d x_i^2 <= 0`` for every ``i``). Tardella's
theorem: the convex envelope of an edge-concave function over a box is
*vertex-polyhedral* — it is the lower convex hull of the ``2^n`` box-vertex
values. Symmetrically, an *edge-convex* function (``d^2 f / d x_i^2 >= 0``) has
a vertex-polyhedral *concave* envelope. (Multilinear products are the special
edge-concave case already handled exactly by ``multilinear_separation``; this
module targets the other practical class — quadratics with negative/positive
square coefficients and bilinear coupling, the ANTIGONE edge-concave family.)

For a quadratic block ``q(x) = sum_i a_i x_i^2 + sum_{i<j} b_ij x_i x_j +
sum_i c_i x_i + const`` whose square coefficients are sign-definite, the builder
already lifts every ``x_i^2`` and ``x_i x_j`` to an auxiliary column. The
vertex-hull supporting hyperplane ``A.x + B`` of ``q`` over the box then yields a
valid cut *directly on those auxiliaries*:

    sum_i a_i w_ii + sum_{i<j} b_ij w_ij + sum_i c_i x_i + const  >=  A.x + B
        (edge-concave: a_i <= 0 -> underestimator)

which is sound because ``w_ii = x_i^2`` and ``w_ij = x_i x_j`` at every true
point, so the left side equals ``q(x) >= A.x + B``. No lifting or model
reformulation is required — the cut references existing columns.

Soundness rests on edge-concavity: a function that is *not* edge-concave can have
a vertex-hull "underestimator" that cuts off true points, so detection
(sign-definite square coefficients) is mandatory. It is exact for a quadratic
*symbolically* — the Hessian is constant — but the square coefficients reach this
module as a floating-point accumulation over the expanded polynomial, and a sign read
off a cancelling sum can be wrong (#1397). Detection therefore declares a sign only
when the accumulated coefficient clears the round-off bound of its own addends, and
declines the block otherwise: a cut whose side cannot be justified is not emitted.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from itertools import product

import numpy as np

logger = logging.getLogger(__name__)


def _separation_lp_solver():
    """Return the LP solver for the vertex-hull separation LP.

    Phase-D lever (``perf-d1``): route the edge-concave separation LP through the
    in-house pure-Rust warm simplex (``lp_simplex.solve_lp``) instead of a cold
    POUNCE IPM solve per call, controlled by ``DISCOPT_SEPARATION_LP_SIMPLEX``
    (default ``"1"`` — ON). The off-switch (``"0"``) restores the POUNCE path.

    Soundness is independent of which backend is used: only the dual *slope* ``A``
    is consumed and the intercept ``B`` is recomputed to the exact validity
    boundary over the box vertices, so the derived cut bounds ``q`` everywhere for
    ANY slope (see the module + :func:`separate_edge_concave_quadratic` docstrings).
    The two backends can disagree on the slope on a *degenerate* vertex-hull LP
    (the IPM returns an analytic-center dual, the simplex a vertex dual), so this
    routing is not byte-for-byte identical to POUNCE — the derived cut can differ
    (both sound). It is validated node-neutral on the cert baseline before shipping.
    """
    use_simplex = os.environ.get("DISCOPT_SEPARATION_LP_SIMPLEX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if use_simplex:
        try:
            from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE, solve_lp

            if SIMPLEX_AVAILABLE:
                return solve_lp
        except ImportError:
            pass
    from discopt.solvers.lp_pounce import solve_lp

    return solve_lp


@dataclass(frozen=True)
class EdgeConcaveQuadratic:
    """A detected edge-concave (or edge-convex) quadratic block.

    ``var_idxs`` are the flat original-variable indices (sorted). ``sq`` maps
    ``i -> coeff of x_i^2``; ``bilin`` maps ``(i, j)`` with ``i < j`` to the
    coeff of ``x_i x_j``; ``lin`` maps ``i -> coeff of x_i``. ``sense`` is
    ``"under"`` for edge-concave (convex underestimator cut) or ``"over"`` for
    edge-convex (concave overestimator cut).
    """

    var_idxs: tuple[int, ...]
    sq: dict[int, float]
    bilin: dict[tuple[int, int], float]
    lin: dict[int, float]
    const: float
    sense: str


#: Safety factor on the round-off bound for an accumulated square coefficient. The
#: separation makes the exact value immaterial: for a coefficient assembled without
#: cancellation the bound is ~1e-14 relative and the sign reads cleanly, while the
#: cancelling case that motivated this (#1397) leaves a residue ~1e6 times *below* the
#: bound. This is a round-off bound, not a tuned tolerance — it carries the units of
#: the addends, which is the whole point.
_SIGN_SAFETY = 64.0


def _definite_sign(total: float, addends: list[float]) -> int:
    """Sign of ``total`` when it is larger than the error of its own accumulation.

    Returns ``+1``/``-1`` when the sign is determined and ``0`` when it is not. The
    yardstick is ``n * eps * sum |addend|``: the error inherited from forming the
    addends upstream (the polynomial expansion multiplies coefficients) is bounded by
    the magnitudes that went in, not by the magnitude that came out. A fixed absolute
    threshold cannot do this job, because the residue of a cancelling sum grows with
    the terms that cancelled while the threshold does not — measured in #1397, a true
    curvature of -3.95e-10 accumulates to +1.49e-09 from addends of magnitude 5.2e7,
    clearing a 1e-9 threshold with the wrong sign.
    """
    if not math.isfinite(total):
        return 0
    scale = math.fsum(abs(a) for a in addends)
    tol = _SIGN_SAFETY * (len(addends) + 1) * float(np.finfo(float).eps) * scale
    if total > tol:
        return 1
    if total < -tol:
        return -1
    return 0


def collect_edge_concave_quadratics(model, *, max_factors: int = 12) -> list[EdgeConcaveQuadratic]:
    """Find edge-concave / edge-convex quadratic blocks in objective + constraints.

    A block qualifies when every monomial has degree <= 2, it has at least one
    bilinear cross term (genuinely coupled — separable quadratics are already
    tight term by term), it spans 2..``max_factors`` variables, and all
    ``x_i^2`` coefficients are sign-definite (all <= 0 with one < 0 ->
    edge-concave; all >= 0 with one > 0 -> edge-convex).

    Sign-definiteness is decided by :func:`_definite_sign`, per coefficient and
    relative to that coefficient's own accumulation, because ``sense`` selects the
    direction of the emitted inequality and a sign read off a cancelling sum can
    invert it (#1397). A variable carrying no square term at all is an exact zero and
    is compatible with either sense; a variable whose accumulated coefficient does not
    clear its round-off bound has no determined sign and disqualifies the block.
    """
    from discopt._relax.milp_relaxation import _expr_to_polynomial
    from discopt._relax.term_classifier import distribute_products

    bodies = []
    if getattr(model, "_objective", None) is not None:
        bodies.append(model._objective.expression)
    for c in model._constraints:
        bodies.append(c.body)

    blocks: list[EdgeConcaveQuadratic] = []
    seen: set = set()
    for body in bodies:
        try:
            poly = _expr_to_polynomial(distribute_products(body), model)
        except Exception as exc:  # noqa: BLE001 - a body we cannot polynomialize is skipped
            # Capability-disabling: every skipped body is an edge-concave block the
            # relaxation never gets, so "edge-concave found nothing" can be an
            # artifact of the extractor rather than of the model.
            logger.debug("edge-concave polynomialization failed: %s: %s", type(exc).__name__, exc)
            continue
        if poly is None:
            continue
        const, terms = poly
        # Accumulate every coefficient group as a LIST, not a running sum. The square
        # coefficients decide ``sense``, which decides the *direction* of the emitted
        # inequality, so a sign read off a cancelling running sum can invert the cut
        # (#1397). Keeping the addends lets the sum be correctly rounded (``math.fsum``)
        # and — the part that matters — lets the sign be tested against the round-off
        # bound of its own accumulation rather than against a fixed constant.
        sq_terms: dict[int, list[float]] = {}
        bilin_terms: dict[tuple[int, int], list[float]] = {}
        lin_terms: dict[int, list[float]] = {}
        const_terms: list[float] = [const]
        varset: set[int] = set()
        ok = True
        for coeff, mono in terms:
            d = len(mono)
            if d == 0:
                const_terms.append(float(coeff))
            elif d == 1:
                lin_terms.setdefault(mono[0], []).append(float(coeff))
                varset.add(mono[0])
            elif d == 2:
                i, j = int(mono[0]), int(mono[1])
                if i == j:
                    sq_terms.setdefault(i, []).append(float(coeff))
                else:
                    key = (min(i, j), max(i, j))
                    bilin_terms.setdefault(key, []).append(float(coeff))
                varset.update((i, j))
            else:
                ok = False
                break
        if not ok or not bilin_terms or not (2 <= len(varset) <= max_factors):
            continue
        const = math.fsum(const_terms)
        sq = {i: math.fsum(v) for i, v in sq_terms.items()}
        bilin = {k: math.fsum(v) for k, v in bilin_terms.items()}
        lin = {i: math.fsum(v) for i, v in lin_terms.items()}
        # Sign-definiteness, decided per entry against that entry's own accumulation
        # scale. A variable absent from ``sq_terms`` has an exactly-zero square
        # coefficient and is compatible with either sense; a variable whose accumulated
        # coefficient does not clear its round-off bound has an *undetermined* sign and
        # is compatible with neither, because the vertex hull is a valid bound only on
        # the side the true curvature actually takes. Refusing the block there is the
        # sound choice: a cut we cannot justify is not emitted (CLAUDE.md §3).
        signs: list[int] = []
        undetermined = False
        for i in varset:
            if i not in sq_terms:
                signs.append(0)
                continue
            s = _definite_sign(sq[i], sq_terms[i])
            if s == 0:
                undetermined = True
                break
            signs.append(s)
        if undetermined:
            logger.debug(
                "edge-concave: square coefficient sign undetermined within round-off; "
                "block over %s skipped",
                sorted(varset),
            )
            continue
        if all(s <= 0 for s in signs) and any(s < 0 for s in signs):
            sense = "under"
        elif all(s >= 0 for s in signs) and any(s > 0 for s in signs):
            sense = "over"
        else:
            continue
        dedup_key = (
            tuple(sorted(varset)),
            tuple(sorted(sq.items())),
            tuple(sorted(bilin.items())),
            tuple(sorted(lin.items())),
            round(const, 12),
            sense,
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        blocks.append(
            EdgeConcaveQuadratic(tuple(sorted(varset)), sq, bilin, lin, float(const), sense)
        )
    return blocks


def _quad_values(block: EdgeConcaveQuadratic, verts: np.ndarray, idx: dict) -> np.ndarray:
    """Evaluate ``q`` at each (full-width) vertex row of ``verts``."""
    vals = np.full(verts.shape[0], block.const, dtype=np.float64)
    for i, coeff in block.sq.items():
        vals += coeff * verts[:, idx[i]] ** 2
    for (i, j), coeff in block.bilin.items():
        vals += coeff * verts[:, idx[i]] * verts[:, idx[j]]
    for i, coeff in block.lin.items():
        vals += coeff * verts[:, idx[i]]
    return vals


def _quad_vertex_values(block: EdgeConcaveQuadratic, lb: np.ndarray, ub: np.ndarray, idx: dict):
    """Evaluate ``q`` at every box vertex of ``block.var_idxs``."""
    n = len(block.var_idxs)
    verts = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in range(n)])))
    return verts, _quad_values(block, verts, idx)


def separate_edge_concave_quadratic(
    block: EdgeConcaveQuadratic,
    lb: np.ndarray,
    ub: np.ndarray,
    x_star: np.ndarray,
    q_star: float,
    *,
    tol: float = 1e-6,
):
    """Return ``(A, B)`` of a violated vertex-hull cut, or ``None``.

    ``A`` is indexed over ``block.var_idxs``; the cut is ``q(x) >= A.x + B``
    (``sense="under"``) or ``q(x) <= A.x + B`` (``sense="over"``). ``A.v + B``
    bounds ``q(v)`` at every box vertex (and hence ``q`` everywhere, by
    edge-concavity), so the cut is sound; it is returned only when ``q_star``
    violates it.

    The vertex-hull LP is solved with the in-house pure-Rust warm simplex by
    default (Phase-D lever ``perf-d1``; ``DISCOPT_SEPARATION_LP_SIMPLEX=0`` restores
    the POUNCE IPM — see :func:`_separation_lp_solver`). Only the dual *slope* ``A``
    is taken from the LP; the intercept ``B`` is recomputed to the exact validity
    boundary over the vertices (``minᵥ(q(v)−A·v)`` under / ``maxᵥ`` over), so by
    edge-concavity the cut bounds ``q`` everywhere for ANY slope — robust to the
    backend's dual / sign / scale. ``None`` if no backend is available or the LP
    did not converge.
    """
    from discopt.solvers import SolveStatus

    solve_lp = _separation_lp_solver()

    n = len(block.var_idxs)
    if n < 2 or not (np.all(np.isfinite(lb[:n])) and np.all(np.isfinite(ub[:n]))):
        return None
    idx = {v: k for k, v in enumerate(block.var_idxs)}
    xs = np.clip(np.asarray(x_star, dtype=np.float64), lb[:n], ub[:n])
    maximize = block.sense == "over"

    # Pinned dims (branching / FBBT set ``lb == ub`` exactly): the ``2^n`` vertex
    # enumeration emits duplicate vertex columns and a redundant equality row, so
    # the vertex-hull LP is degenerate and the Rust simplex cycles to its pivot
    # cap and falls back to the POUNCE IPM. Drop the pinned positions from the
    # enumeration so the LP is non-degenerate. The cut is unchanged in effect: a
    # pinned var is fixed at that constant in the relaxation LP too, so any slope
    # on it is absorbed into the intercept with no change to the feasible region
    # or bound; the returned slope ``A`` is 0 on pinned dims and the intercept —
    # recomputed as ``min/max_v(q(v) − A·v)`` over the vertices — is unaffected
    # because the pinned dims contribute a constant to ``q(v)`` and 0 to ``A·v``.
    pinned = lb[:n] == ub[:n]
    if pinned.any():
        fidx = np.nonzero(~pinned)[0]  # free positions (0..n-1)
        if fidx.size == 0:
            # Fully pinned: q is constant on the box — nothing to separate.
            return None
        pin_pos = np.nonzero(pinned)[0]
        free_verts = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in fidx])))
        m = free_verts.shape[0]
        # Full-width vertices (pinned columns held at their constant) so ``q(v)``
        # evaluates correctly; only the free rows enter the LP.
        verts_full = np.empty((m, n), dtype=np.float64)
        verts_full[:, fidx] = free_verts
        verts_full[:, pin_pos] = lb[pin_pos]
        vals = _quad_values(block, verts_full, idx)
        a_eq = np.vstack([free_verts.T, np.ones(m)])
        b_eq = np.append(xs[fidx], 1.0)
        c = -vals if maximize else vals
        try:
            res = solve_lp(c, A_eq=a_eq, b_eq=b_eq, bounds=[(0.0, np.inf)] * m)
        except ImportError:  # pragma: no cover - POUNCE is a core dependency
            return None
        if res.status != SolveStatus.OPTIMAL or res.dual_values is None:
            return None
        duals = np.asarray(res.dual_values, dtype=np.float64)
        if duals.shape[0] != fidx.size + 1 or not np.all(np.isfinite(duals)):
            return None
        A_free = -duals[: fidx.size] if maximize else duals[: fidx.size]
        A = np.zeros(n, dtype=np.float64)
        A[fidx] = A_free
        if not np.all(np.isfinite(A)):
            return None
        resid = vals - verts_full @ A  # q(v) − A·v
        if maximize:
            B = float(np.max(resid))
            if q_star <= float(A @ xs + B) + tol:
                return None
        else:
            B = float(np.min(resid))
            if q_star >= float(A @ xs + B) - tol:
                return None
        return A, B

    verts, vals = _quad_vertex_values(block, lb, ub, idx)
    m = verts.shape[0]
    a_eq = np.vstack([verts.T, np.ones(m)])
    b_eq = np.append(xs, 1.0)
    c = -vals if maximize else vals
    try:
        res = solve_lp(c, A_eq=a_eq, b_eq=b_eq, bounds=[(0.0, np.inf)] * m)
    except ImportError:  # pragma: no cover - POUNCE is a core dependency
        return None
    if res.status != SolveStatus.OPTIMAL or res.dual_values is None:
        return None
    duals = np.asarray(res.dual_values, dtype=np.float64)
    if duals.shape[0] != n + 1 or not np.all(np.isfinite(duals)):
        return None
    A = -duals[:n] if maximize else duals[:n]
    if not np.all(np.isfinite(A)):
        return None
    # Recompute the intercept to the exact validity boundary over the vertices so
    # the cut bounds q at every vertex (hence everywhere, by edge-concavity)
    # regardless of POUNCE's reported intercept/scale.
    resid = vals - verts @ A  # q(v) − A·v
    if maximize:  # overestimator: A·v + B >= q(v)  ->  B = maxᵥ(q(v)−A·v)
        B = float(np.max(resid))
        if q_star <= float(A @ xs + B) + tol:
            return None
    else:  # underestimator: A·v + B <= q(v)  ->  B = minᵥ(q(v)−A·v)
        B = float(np.min(resid))
        if q_star >= float(A @ xs + B) - tol:
            return None
    return A, B
