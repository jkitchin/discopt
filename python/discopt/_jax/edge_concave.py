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
(sign-definite square coefficients) is mandatory and is exact for a quadratic
(constant Hessian).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


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


def collect_edge_concave_quadratics(model, *, max_factors: int = 12) -> list[EdgeConcaveQuadratic]:
    """Find edge-concave / edge-convex quadratic blocks in objective + constraints.

    A block qualifies when every monomial has degree <= 2, it has at least one
    bilinear cross term (genuinely coupled — separable quadratics are already
    tight term by term), it spans 2..``max_factors`` variables, and all
    ``x_i^2`` coefficients are sign-definite (all <= 0 with one < 0 ->
    edge-concave; all >= 0 with one > 0 -> edge-convex).
    """
    from discopt._jax.milp_relaxation import _expr_to_polynomial
    from discopt._jax.term_classifier import distribute_products

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
        except Exception:
            continue
        if poly is None:
            continue
        const, terms = poly
        sq: dict[int, float] = {}
        bilin: dict[tuple[int, int], float] = {}
        lin: dict[int, float] = {}
        varset: set[int] = set()
        ok = True
        for coeff, mono in terms:
            d = len(mono)
            if d == 0:
                const += float(coeff)
            elif d == 1:
                lin[mono[0]] = lin.get(mono[0], 0.0) + float(coeff)
                varset.add(mono[0])
            elif d == 2:
                i, j = int(mono[0]), int(mono[1])
                if i == j:
                    sq[i] = sq.get(i, 0.0) + float(coeff)
                else:
                    key = (min(i, j), max(i, j))
                    bilin[key] = bilin.get(key, 0.0) + float(coeff)
                varset.update((i, j))
            else:
                ok = False
                break
        if not ok or not bilin or not (2 <= len(varset) <= max_factors):
            continue
        diag = [sq.get(i, 0.0) for i in varset]
        if all(v <= 1e-12 for v in diag) and any(v < -1e-9 for v in diag):
            sense = "under"
        elif all(v >= -1e-12 for v in diag) and any(v > 1e-9 for v in diag):
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


def _quad_vertex_values(block: EdgeConcaveQuadratic, lb: np.ndarray, ub: np.ndarray, idx: dict):
    """Evaluate ``q`` at every box vertex of ``block.var_idxs``."""
    n = len(block.var_idxs)
    verts = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in range(n)])))
    vals = np.full(verts.shape[0], block.const, dtype=np.float64)
    for i, coeff in block.sq.items():
        vals += coeff * verts[:, idx[i]] ** 2
    for (i, j), coeff in block.bilin.items():
        vals += coeff * verts[:, idx[i]] * verts[:, idx[j]]
    for i, coeff in block.lin.items():
        vals += coeff * verts[:, idx[i]]
    return verts, vals


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

    The vertex-hull LP is solved with the pure-Rust POUNCE IPM (issue #356 — no
    SciPy/HiGHS). Only the dual *slope* ``A`` is taken from POUNCE; the intercept
    ``B`` is recomputed to the exact validity boundary over the vertices
    (``minᵥ(q(v)−A·v)`` under / ``maxᵥ`` over), so by edge-concavity the cut
    bounds ``q`` everywhere for ANY slope — robust to POUNCE's analytic-center
    dual / sign / scale. ``None`` if POUNCE is unavailable or did not converge.
    """
    from discopt.solvers import SolveStatus
    from discopt.solvers.lp_pounce import solve_lp

    n = len(block.var_idxs)
    if n < 2 or not (np.all(np.isfinite(lb[:n])) and np.all(np.isfinite(ub[:n]))):
        return None
    idx = {v: k for k, v in enumerate(block.var_idxs)}
    verts, vals = _quad_vertex_values(block, lb, ub, idx)
    m = verts.shape[0]
    xs = np.clip(np.asarray(x_star, dtype=np.float64), lb[:n], ub[:n])
    a_eq = np.vstack([verts.T, np.ones(m)])
    b_eq = np.append(xs, 1.0)
    maximize = block.sense == "over"
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
