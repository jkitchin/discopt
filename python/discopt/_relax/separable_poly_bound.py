"""Rigorous separable lower bound for a polynomial objective over a (possibly
unbounded) box (#1869 part (c)).

A box with an unbounded side defeats both the McCormick relaxation and plain interval
arithmetic on a polynomial whose terms change sign: ``x**6/6 - 1.05*x**4`` over
``[P, inf)`` evaluates to ``inf - inf``. Yet such a polynomial is often *coercive*, and
its value on an outer piece is enormous. ``ex4_1_5`` (``2x^2 - 1.05x^4 + x^6/6 - xy +
y^2``, optimum 0) is the probe: its outer pieces carried no bound at all, so the tree
could never close them.

The bound. Write the objective as ``d + sum_i p_i(x_i) + sum_{i<j} c_ij x_i x_j`` (any
other multivariate monomial abstains). Each cross term is split by
``c x_i x_j >= -(|c|/2)(x_i**2 + x_j**2)`` (AM-GM), which leaves a sum of univariate
polynomials ``q_i``; the bound is ``d + sum_i min_{x_i in [l_i, u_i]} q_i(x_i)``. Each
univariate minimum is computed rigorously, with no root finding: every critical point of
``q`` lies within the Cauchy radius ``R`` of ``q'``, so outside ``[-R, R]`` the
polynomial is monotone and its minimum over ``[l, u]`` is at a finite endpoint or inside
``[l, u] ∩ [-R, R]``, where an interval branch-and-bound with outward rounding gives a
guaranteed lower bound. An unbounded side along which ``q -> -inf`` gives ``-inf``.
"""

from __future__ import annotations

import heapq
import math
import os
import weakref
from typing import Optional

import numpy as np

#: Interval branch-and-bound budget per univariate minimum (leaves popped).
_UNIVARIATE_BB_STEPS = 64
#: Stop refining once the best leaf is this narrow relative to its magnitude.
_UNIVARIATE_BB_REL_WIDTH = 1e-9

#: Per-model separable minorant structure, keyed by the objective's sign.
_Structure = Optional[tuple[float, dict[int, list[float]]]]
_CACHE: "weakref.WeakKeyDictionary[object, dict[float, _Structure]]" = weakref.WeakKeyDictionary()


def separable_poly_bound_enabled() -> bool:
    """``DISCOPT_SEPARABLE_POLY_BOUND`` gate (#1869 (c)). Bound-changing (a tighter
    node bound fathoms more), so gated per CLAUDE.md section 5; read at call time.

    **Graduated (default ON, 2026-10-22).** Panel over the 120 local MINLPLib
    instances (<= 200 vars/cons) that keep an unbounded continuous variable after
    FBBT, ON vs OFF, 30 s, interleaved: cert-clean (240 rows, 0 problems: no bound
    past an optimum, every incumbent passes ``verify_point`` on a fresh model) and
    net-positive -- certified 86 -> 87, none lost (``ex4_1_5`` time limit with no
    bound -> optimal in 159 nodes / 2.5 s; SCIP 2.8 s); node counts identical on all
    86 instances both arms certify, SGM wall 2.99 vs 2.99 s. It only fires where
    plain interval arithmetic gave no finite bound. ``=0`` is the opt-out.
    """
    return os.environ.get("DISCOPT_SEPARABLE_POLY_BOUND", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


# ── outward-rounded scalar interval arithmetic ──


def _down(x: float) -> float:
    return float(np.nextafter(x, -np.inf)) if np.isfinite(x) else x


def _up(x: float) -> float:
    return float(np.nextafter(x, np.inf)) if np.isfinite(x) else x


def _imul(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    prods = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    if any(math.isnan(p) for p in prods):  # 0 * inf
        return (-math.inf, math.inf)
    return (_down(min(prods)), _up(max(prods)))


def _iadd(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    lo, hi = a[0] + b[0], a[1] + b[1]
    if math.isnan(lo) or math.isnan(hi):
        return (-math.inf, math.inf)
    return (_down(lo), _up(hi))


def _horner(coeffs: list[float], x: tuple[float, float]) -> tuple[float, float]:
    """Interval enclosure of ``sum_k coeffs[k] x^k`` (Horner, outward rounded)."""
    acc = (coeffs[-1], coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = _iadd(_imul(acc, x), (c, c))
    return acc


def _deriv(coeffs: list[float]) -> list[float]:
    return [k * coeffs[k] for k in range(1, len(coeffs))] or [0.0]


def _enclose(coeffs: list[float], dcoeffs: list[float], a: float, b: float) -> float:
    """Lower end of an enclosure of the polynomial on ``[a, b]``: the better of the
    natural Horner form and the mean-value form ``q(m) + q'([a,b])([a,b] - m)``."""
    nat = _horner(coeffs, (a, b))[0]
    m = 0.5 * (a + b)
    qm = _horner(coeffs, (m, m))
    dq = _horner(dcoeffs, (a, b))
    mv = _iadd(qm, _imul(dq, (_down(a - m), _up(b - m))))[0]
    return max(nat, mv)


def _univariate_min(coeffs: list[float], lb: float, ub: float) -> float:
    """Rigorous lower bound on ``q(x) = sum_k coeffs[k] x^k`` over ``[lb, ub]``."""
    while len(coeffs) > 1 and coeffs[-1] == 0.0:
        coeffs = coeffs[:-1]
    deg = len(coeffs) - 1
    if deg == 0:
        return coeffs[0]
    lead = coeffs[-1]
    # Asymptotics: an unbounded side along which q decreases without bound.
    if ub == math.inf and lead < 0.0:
        return -math.inf
    if lb == -math.inf and lead * (-1.0) ** deg < 0.0:
        return -math.inf
    dcoeffs = _deriv(coeffs)
    # Cauchy radius of q' (degree deg-1): every real critical point lies inside it.
    dlead = dcoeffs[-1]
    if len(dcoeffs) == 1 or dlead == 0.0:
        radius = 0.0
    else:
        radius = _up(1.0 + max(abs(c / dlead) for c in dcoeffs[:-1]))
    candidates: list[float] = []
    for e in (lb, ub):
        if math.isfinite(e):
            candidates.append(_horner(coeffs, (e, e))[0])
    lo, hi = max(lb, -radius), min(ub, radius)
    if lo <= hi:
        candidates.append(_interval_bb(coeffs, dcoeffs, lo, hi))
    if not candidates:  # [lb, ub] lies beyond R on an unbounded side: monotone, and the
        # asymptotic check above says it increases away from the finite end -- which
        # would be a finite endpoint already listed. Reaching here means both ends are
        # infinite with R < 0, impossible; abstain.
        return -math.inf
    return min(candidates)


def _interval_bb(coeffs: list[float], dcoeffs: list[float], a: float, b: float) -> float:
    """Guaranteed lower bound on the polynomial over a FINITE ``[a, b]``: best-first
    bisection; the answer is the smallest leaf lower bound when the budget runs out."""
    heap = [(_enclose(coeffs, dcoeffs, a, b), a, b)]
    for _ in range(_UNIVARIATE_BB_STEPS):
        lbv, x0, x1 = heapq.heappop(heap)
        if x1 - x0 <= _UNIVARIATE_BB_REL_WIDTH * (1.0 + abs(x0) + abs(x1)):
            heapq.heappush(heap, (lbv, x0, x1))
            break
        mid = 0.5 * (x0 + x1)
        heapq.heappush(heap, (_enclose(coeffs, dcoeffs, x0, mid), x0, mid))
        heapq.heappush(heap, (_enclose(coeffs, dcoeffs, mid, x1), mid, x1))
    return heap[0][0]


# ── structure ──


def _structure(model, sign: float) -> _Structure:
    """``(d, {flat_idx: univariate coeffs})`` -- the separable minorant of
    ``sign * f`` -- or ``None`` when the objective is not such a polynomial. Cached
    per (model, sign)."""
    per_model = _CACHE.setdefault(model, {})
    if sign not in per_model:
        per_model[sign] = _build_structure(model, sign)
    return per_model[sign]


def _build_structure(model, sign: float) -> _Structure:
    from discopt._relax.milp_relaxation import _expr_to_polynomial
    from discopt._relax.term_classifier import distribute_products

    obj = getattr(model, "_objective", None)
    if obj is None:
        return None
    # #1520: no except -- ``_expr_to_polynomial`` abstains by returning ``None``.
    poly = _expr_to_polynomial(distribute_products(obj.expression), model)
    if poly is None:
        return None
    const, terms = poly
    uni: dict[int, list[float]] = {}

    def _add(var: int, power: int, value: float) -> None:
        row = uni.setdefault(var, [0.0])
        row.extend([0.0] * (power + 1 - len(row)))
        row[power] += value

    for coeff, monomial in terms:
        c = sign * float(coeff)
        if not math.isfinite(c):
            return None
        idx = set(monomial)
        if len(idx) == 1:
            _add(int(monomial[0]), len(monomial), c)
        elif len(idx) == 2 and len(monomial) == 2:
            # c x y >= -(|c|/2)(x^2 + y^2): the minorant takes the AM-GM split.
            for v in monomial:
                _add(int(v), 2, -0.5 * abs(c))
        else:
            return None  # a multivariate monomial beyond bilinear: abstain
    if not uni:
        return None
    return sign * float(const), uni


def separable_poly_lower_bound(model, node_lb, node_ub, negate: bool) -> float:
    """Rigorous lower bound on the internal (minimised) objective over the box, or
    ``-inf`` when the objective is not a polynomial of this shape or its minorant is
    unbounded below there. ``negate``: the model maximises, so ``-f`` is bounded."""
    st = _structure(model, -1.0 if negate else 1.0)
    if st is None:
        return -math.inf
    const, uni = st
    total = (const, const)
    for i, coeffs in uni.items():
        m = _univariate_min(list(coeffs), float(node_lb[i]), float(node_ub[i]))
        if not math.isfinite(m):
            return -math.inf
        total = _iadd(total, (m, m))
    return total[0] if math.isfinite(total[0]) else -math.inf
