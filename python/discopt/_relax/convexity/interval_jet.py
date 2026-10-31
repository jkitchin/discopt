"""Univariate third-order interval Taylor jets, for a centred curvature test (#2062).

The interval Hessian of :mod:`.interval_ad` is the *natural* interval extension
of the second derivative: every occurrence of a variable is treated as an
independent copy, so a quotient of two small differences of exponentials --
``(exp(t) - 1) / (exp(0.75 t) - 1)`` -- encloses curvature of either sign on
any box much wider than a hair, although the function is convex everywhere.
Bisection recovers the correlation only linearly in the piece width; #2062
measured 2048-8192 pieces (3-10 s) for two one-variable reactor rows.

A *centred* (mean-value) form removes that loss at second order. For one
variable ``t`` on a piece ``X`` with midpoint ``m``,

    f''(t) in f''(m) + f'''(X) (X - m)        for every t in X,

by the mean-value theorem applied to ``f''``. ``f''(m)`` is evaluated on a
degenerate interval, so it carries only rounding, and the excess width of the
whole enclosure is ``O(w^2)`` instead of ``O(w)`` (Moore 1966, ch. 6;
Neumaier 1990, §2.3). Computing it needs the third derivative, which this
module supplies by propagating the jet ``(f, f', f'', f''')`` of every DAG node
with respect to one scalar variable through outward-rounded interval
arithmetic (Leibniz for products, Faa di Bruno for atoms).

Other variables are held at their box intervals with zero derivative, so the
enclosures are of the *partial* derivatives in ``t`` over the whole box -- the
mean-value step above holds for every fixed value of the other variables.

Soundness: every jet entry is produced by :class:`~.interval.Interval`
arithmetic, which rounds outward; an atom outside the table, a denominator or
``log``/fractional-power argument that is not strictly sign-definite, and an
array-valued leaf all raise :class:`JetAbstain`, which the caller turns into
"no verdict". Abstaining is never a claim about curvature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt._flat_index import flat_index_in_shape
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    Parameter,
    SumOverExpression,
    UnaryOp,
    Variable,
)

from . import interval as iv
from .interval import Interval

#: Normalised Taylor coefficients ``a_k = f^(k)(t) / k!``, ``k = 0..K``, each an
#: interval enclosure over the box (the expansion point ``t`` ranges over it).
Jet = tuple[Interval, ...]

_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


class JetAbstain(ValueError):
    """The jet walker cannot enclose this expression; the caller abstains."""


class _Lanes:
    """Per-piece "undefined here" mask for a batched walk.

    A batched walk evaluates many pieces at once (the interval endpoints are
    arrays, one lane per piece; lanes never mix). An operation undefined on some
    lanes only -- a denominator containing zero, a ``log`` argument that is not
    strictly positive -- substitutes a harmless operand on those lanes so the
    others can proceed, and records them here; the caller discards every
    recorded lane. Nothing computed on a recorded lane is ever used.
    """

    def __init__(self) -> None:
        self.bad: np.ndarray = np.zeros((), dtype=bool)

    def guard(self, x: Interval, ok: np.ndarray, safe: float) -> Interval:
        if bool(np.all(ok)):
            return x
        self.bad = self.bad | ~ok
        return Interval(np.where(ok, x.lo, safe), np.where(ok, x.hi, safe))


_LANES: Optional[_Lanes] = None


def _guard(x: Interval, ok: np.ndarray, safe: float, what: str) -> Interval:
    if _LANES is None:
        if not bool(np.all(ok)):
            raise JetAbstain(what)
        return x
    return _LANES.guard(x, ok, safe)


def _pt(x: float) -> Interval:
    return Interval.point(float(x))


def _const(x: Interval, order: int) -> Jet:
    return (x,) + (_ZERO,) * order


def _scalar_interval(lo, hi) -> Interval:
    return Interval(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))


def _leaf_value(expr: Constant | Parameter) -> Interval:
    arr = np.asarray(expr.value)
    if arr.size != 1:
        raise JetAbstain("array-valued leaf")
    return Interval.point(float(arr.reshape(())))


def _var_value(v: Variable, box: dict, flat: int = 0) -> Interval:
    if v in box:
        bi = box[v]
        return _scalar_interval(
            np.asarray(bi.lo, dtype=np.float64).ravel()[flat],
            np.asarray(bi.hi, dtype=np.float64).ravel()[flat],
        )
    return _scalar_interval(np.asarray(v.lb).ravel()[flat], np.asarray(v.ub).ravel()[flat])


def _add(a: Jet, b: Jet) -> Jet:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: Jet, b: Jet) -> Jet:
    return tuple(x - y for x, y in zip(a, b))


def _neg(a: Jet) -> Jet:
    return tuple(-x for x in a)


def _mul(a: Jet, b: Jet) -> Jet:
    """Cauchy product: ``c_k = sum_j a_j b_{k-j}``."""
    out = []
    for k in range(len(a)):
        acc = a[0] * b[k]
        for j in range(1, k + 1):
            acc = acc + a[j] * b[k - j]
        out.append(acc)
    return tuple(out)


def _div(a: Jet, b: Jet) -> Jet:
    """``c = a / b``: ``c_k = (a_k - sum_{j<k} c_j b_{k-j}) / b_0``."""
    b0 = _guard(b[0], ~b[0].contains_zero(), 1.0, "denominator contains zero")
    out: list[Interval] = []
    for k in range(len(a)):
        acc = a[k]
        for j in range(k):
            acc = acc - out[j] * b[k - j]
        out.append(acc / b0)
    return tuple(out)


def _positive(x: Interval, what: str) -> Interval:
    return _guard(x, x.lo > 0.0, 1.0, f"{what} argument not strictly positive")


def _exp(u: Jet) -> Jet:
    """``c = exp(u)``: ``c_k = (1/k) sum_{j=1..k} j u_j c_{k-j}``."""
    out = [iv.exp(u[0])]
    for k in range(1, len(u)):
        acc = _pt(1) * u[1] * out[k - 1]
        for j in range(2, k + 1):
            acc = acc + _pt(j) * u[j] * out[k - j]
        out.append(acc / _pt(k))
    return tuple(out)


def _log(u: Jet) -> Jet:
    """``c = log(u)``: ``c_k = (u_k - (1/k) sum_{j=1..k-1} j c_j u_{k-j}) / u_0``."""
    u0 = _positive(u[0], "log")
    out = [iv.log(u0)]
    for k in range(1, len(u)):
        acc = _ZERO
        for j in range(1, k):
            acc = acc + _pt(j) * out[j] * u[k - j]
        out.append((u[k] - acc / _pt(k)) / u0)
    return tuple(out)


def _sincos(u: Jet) -> tuple[Jet, Jet]:
    """``s_k = (1/k) sum j u_j c_{k-j}``, ``c_k = -(1/k) sum j u_j s_{k-j}``."""
    s = [iv.sin(u[0])]
    c = [iv.cos(u[0])]
    for k in range(1, len(u)):
        sa = _ZERO
        ca = _ZERO
        for j in range(1, k + 1):
            sa = sa + _pt(j) * u[j] * c[k - j]
            ca = ca + _pt(j) * u[j] * s[k - j]
        s.append(sa / _pt(k))
        c.append(-(ca / _pt(k)))
    return tuple(s), tuple(c)


def _power(u: Jet, p: float) -> Jet:
    order = len(u) - 1
    if p == 0.0:
        return _const(_ONE, order)
    if p == 1.0:
        return u
    if float(p).is_integer():
        k = int(p)
        if k < 0:
            return _div(_const(_ONE, order), _power(u, float(-k)))
        if k == 2:
            return _mul(u, u)
        # Square-and-multiply on jets (every step is a sound Cauchy product).
        result: Optional[Jet] = None
        base = u
        while k:
            if k & 1:
                result = base if result is None else _mul(result, base)
            k >>= 1
            if k:
                base = _mul(base, base)
        assert result is not None
        return result
    # Fractional: ``c_k = 1/(k u_0) sum_{j=0..k-1} (p (k - j) - j) u_{k-j} c_j``.
    u0 = _positive(u[0], "fractional power")
    out = [iv.exp(_pt(p) * iv.log(u0))]
    for k in range(1, len(u)):
        acc = _ZERO
        for j in range(k):
            acc = acc + _pt(p * (k - j) - j) * u[k - j] * out[j]
        out.append(acc / (_pt(k) * u0))
    return tuple(out)


def _function(name: str, u: Jet) -> Jet:
    if name == "exp":
        return _exp(u)
    if name == "log":
        return _log(u)
    if name == "sqrt":
        return _power(u, 0.5)
    if name == "sin":
        return _sincos(u)[0]
    if name == "cos":
        return _sincos(u)[1]
    raise JetAbstain(f"atom {name!r} not in the jet table")


def _walk(expr: Expression, var: Variable, box: dict, cache: dict[int, Jet], order: int) -> Jet:
    key = id(expr)
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = _impl(expr, var, box, cache, order)
    cache[key] = out
    return out


def _seed(val: Interval, order: int) -> Jet:
    return (val, _ONE) + (_ZERO,) * (order - 1)


def _impl(expr: Expression, var: Variable, box: dict, cache: dict[int, Jet], order: int) -> Jet:
    if isinstance(expr, (Constant, Parameter)):
        return _const(_leaf_value(expr), order)
    if isinstance(expr, Variable):
        if expr.size != 1:
            raise JetAbstain("array variable")
        if expr is var:
            # The split variable keeps its lane array (one entry per piece).
            return _seed(box[var] if var in box else _var_value(var, box), order)
        return _const(_var_value(expr, box), order)
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base = expr.base
        if base is var:
            # ``var`` is scalar, so this is ``var`` itself; the scalar walker of
            # :mod:`.interval_ad` has no such case either, so refuse it.
            raise JetAbstain("indexed split variable")
        flat = flat_index_in_shape(expr.index, tuple(base.shape or ()))
        if flat is None:
            raise JetAbstain("index does not name one scalar element")
        return _const(_var_value(base, box, flat), order)
    if isinstance(expr, UnaryOp):
        if expr.op != "neg":
            raise JetAbstain(f"unary {expr.op!r}")
        return _neg(_walk(expr.operand, var, box, cache, order))
    if isinstance(expr, BinaryOp):
        left = _walk(expr.left, var, box, cache, order)
        if expr.op == "**":
            if not isinstance(expr.right, (Constant, Parameter)):
                raise JetAbstain("non-constant exponent")
            raw = np.asarray(expr.right.value)
            if raw.size != 1:
                raise JetAbstain("array exponent")
            p = float(raw.reshape(()))
            if not np.isfinite(p):
                raise JetAbstain("non-finite exponent")
            return _power(left, p)
        right = _walk(expr.right, var, box, cache, order)
        if expr.op == "+":
            return _add(left, right)
        if expr.op == "-":
            return _sub(left, right)
        if expr.op == "*":
            return _mul(left, right)
        if expr.op == "/":
            return _div(left, right)
        raise JetAbstain(f"binary {expr.op!r}")
    if isinstance(expr, FunctionCall):
        if len(expr.args) != 1:
            raise JetAbstain("multi-argument function")
        return _function(expr.func_name, _walk(expr.args[0], var, box, cache, order))
    if isinstance(expr, SumOverExpression):
        if not expr.terms:
            return _const(_ZERO, order)
        acc = _walk(expr.terms[0], var, box, cache, order)
        for t in expr.terms[1:]:
            acc = _add(acc, _walk(t, var, box, cache, order))
        return acc
    raise JetAbstain(f"node {type(expr).__name__} not in the jet table")


def interval_jet(expr: Expression, var: Variable, box: dict, order: int) -> Jet:
    """Enclose the Taylor coefficients ``f^(k)(t)/k!``, ``k = 0..order``, over ``box``.

    ``var`` is the scalar variable ``t``; every other variable is a constant
    ranging over its interval. ``box`` maps variables to :class:`Interval`
    overrides; a variable absent from it ranges over its declared bounds.
    Raises :class:`JetAbstain` when the expression is outside the jet table.
    """
    if var.size != 1 or var.shape != ():
        raise JetAbstain("split variable must be scalar")
    if order < 1:
        raise ValueError("order must be >= 1")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _walk(expr, var, box, {}, order)


#: Terms of the Taylor form of ``f''`` (#2062): ``f''`` is expanded to degree
#: ``TAYLOR_ORDER - 1`` about the piece midpoint with a Lagrange remainder of
#: degree ``TAYLOR_ORDER``, so the jets run to order ``TAYLOR_ORDER + 2``.
#: ``1`` is the mean-value (centred) form.
TAYLOR_ORDER = 4


def second_derivative_enclosure(
    expr: Expression,
    var: Variable,
    box: dict,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    taylor_order: int = TAYLOR_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclosures of ``d2f/dt2`` for ``t = var`` on each piece ``[lo[i], hi[i]]``.

    With ``D = X - m`` (``X = [lo, hi]``, ``m`` its midpoint) and ``K =
    taylor_order``, Taylor's theorem with the Lagrange remainder applied to
    ``f''`` gives, for every ``t`` in ``X`` and some ``xi`` between ``m`` and ``t``,

        f''(t) = sum_{j<K} f^(j+2)(m)/j! (t-m)^j + f^(K+2)(xi)/K! (t-m)^K,

    so ``f''(X)`` is enclosed by the same sum with the coefficients at ``m``
    taken from a jet on the degenerate interval ``[m, m]`` and the remainder
    coefficient from a jet on ``X``. The result is intersected with the natural
    extension ``f''(X)``; both are sound, so the intersection is.

    All pieces are evaluated in one vectorised walk. Returns ``(enc_lo,
    enc_hi)``; a piece with no enclosure -- an operation undefined on it, or a
    non-finite endpoint -- has ``NaN`` in both. Raises :class:`JetAbstain` when
    the expression is outside the jet table (no piece can be enclosed).
    """
    global _LANES
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    if lo.shape != hi.shape or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("piece endpoints must be finite arrays of one shape")
    if np.any(lo > hi):
        raise ValueError("piece with lo > hi")
    K = int(taylor_order)
    if K < 1:
        raise ValueError("taylor_order must be >= 1")
    order = K + 2
    m = 0.5 * (lo + hi)
    X = Interval(lo, hi)
    M = Interval(m, m)
    lanes = _Lanes()
    prev, _LANES = _LANES, lanes
    try:
        jx = interval_jet(expr, var, {**box, var: X}, order)
        jm = interval_jet(expr, var, {**box, var: M}, order)
    finally:
        _LANES = prev
    D = X - M
    # ``f^(j+2)/j! = (j+2)(j+1) a_{j+2}`` in normalised coefficients.
    taylor = _pt(2.0) * jm[2]
    for j in range(1, K):
        taylor = taylor + _pt((j + 2) * (j + 1)) * jm[j + 2] * (D**j)
    taylor = taylor + _pt((K + 2) * (K + 1)) * jx[K + 2] * (D**K)
    natural = _pt(2.0) * jx[2]
    shape = lo.shape
    out_lo = np.broadcast_to(np.maximum(taylor.lo, natural.lo), shape).copy()
    out_hi = np.broadcast_to(np.minimum(taylor.hi, natural.hi), shape).copy()
    bad = np.broadcast_to(lanes.bad, shape) | ~np.isfinite(out_lo) | ~np.isfinite(out_hi)
    if np.any(~bad & (out_lo > out_hi)):
        # Two sound enclosures of one set cannot be disjoint; if they are, an
        # upstream defect is the only explanation, so refuse loudly.
        raise AssertionError("disjoint Taylor and natural enclosures of f''")
    out_lo[bad] = np.nan
    out_hi[bad] = np.nan
    return out_lo, out_hi


__all__ = ["JetAbstain", "TAYLOR_ORDER", "interval_jet", "second_derivative_enclosure"]
