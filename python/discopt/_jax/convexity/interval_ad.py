"""Interval-valued forward-mode automatic differentiation.

Produces a sound enclosure of the gradient and Hessian of a scalar
expression over an input box. Each node carries a triple
``(value, gradient, hessian)`` whose entries are :class:`Interval`
objects; the chain rule is applied with interval arithmetic so that
the resulting matrix ``H`` encloses every pointwise Hessian of the
expression on the box.

This machinery exists to support the sound box-local convexity
certificate. The caller then bounds the minimum eigenvalue of ``H``
using interval Gershgorin (or a tighter test); if that lower bound
is ≥ 0, the expression is convex on the box.

The implementation is pure numpy; no JAX. JAX's autodiff does not
carry interval types, and the per-constraint cost is small enough
that a Python walker is adequate — the big-O concern is the ``n × n``
Hessian matrix, not the arithmetic of traversing the DAG.

Limitations
-----------
Current atom table covers ``+``, ``-``, unary ``neg``, ``*``, ``/``
(constant or strictly-signed denominator), integer powers, ``exp``,
``log``, ``sqrt``. Non-smooth atoms (``abs``, ``max``, ``min``) have
undefined Hessians at kink points and are rejected with an unbounded
Hessian, forcing the certificate to abstain.

References
----------
Moore (1966), *Interval Analysis*, §4 (interval derivatives).
Neumaier (1990), *Interval Methods for Systems of Equations*.
Griewank, Walther (2008), *Evaluating Derivatives*, §3 (forward-mode
automatic differentiation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

from . import interval as iv
from .interval import Interval, _round_down, _round_up

# ──────────────────────────────────────────────────────────────────────
# Data type
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rank1Factor:
    """Sound metadata: this node's Hessian is ``c · v vᵀ``.

    A node carries this when its Hessian is provably rank-1 PSD (or
    NSD, with a sign-bracketed ``c``). The certificate consults it
    for a structural sufficient PSD test that does not depend on the
    entry-wise tightness of the interval matrix — useful on wide
    boxes where the off-diagonal interval blows up but the underlying
    rank-1 structure is intact.

    The ``affine_base_*`` fields are populated when the node arose
    from squaring an expression whose Hessian is identically zero
    (i.e. the base is affine in the variables); a downstream
    division by an affine positive denominator combines them via the
    perspective rule into a tighter rank-1 form.
    """

    c: Interval
    v: Interval
    affine_base_value: Optional[Interval] = None
    affine_base_grad: Optional[Interval] = None


@dataclass(frozen=True)
class IntervalAD:
    """A scalar expression's value, gradient, and Hessian as intervals.

    * ``value`` — scalar interval enclosing ``f(x)`` for ``x`` in the box.
    * ``grad``  — shape-``(n,)`` interval enclosing ``∇f(x)``.
    * ``hess``  — shape-``(n, n)`` symmetric interval enclosing
      ``∇²f(x)``. Symmetry is only enforced downstream by the
      consumer (``hess`` + ``hess.T`` / 2 if needed).
    * ``rank1_factor`` — optional rank-1 metadata; see
      :class:`Rank1Factor`. ``None`` for nodes without a known
      rank-1 structure. Soundness is independent of this field —
      it is metadata that lets the certificate dispatch a tighter
      sufficient PSD test. Any op that does not explicitly preserve
      the factorisation drops the field, so a stale value cannot
      mislead.
    """

    value: Interval
    grad: Interval
    hess: Interval
    rank1_factor: Optional[Rank1Factor] = None


# ──────────────────────────────────────────────────────────────────────
# Flat-variable index map
# ──────────────────────────────────────────────────────────────────────


def _var_offset(var: Variable, model: Model) -> int:
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _flat_size(model: Model) -> int:
    return sum(v.size for v in model._variables)


# ──────────────────────────────────────────────────────────────────────
# Helpers: zeros, outer product, scalar lift
# ──────────────────────────────────────────────────────────────────────


def _zero_grad(n: int) -> Interval:
    z = np.zeros(n, dtype=np.float64)
    return Interval(z, z)


def _zero_hess(n: int) -> Interval:
    z = np.zeros((n, n), dtype=np.float64)
    return Interval(z, z)


def _unit_grad(n: int, slot: int) -> Interval:
    z = np.zeros(n, dtype=np.float64)
    z[slot] = 1.0
    return Interval(z, z)


def _outer(a: Interval, b: Interval) -> Interval:
    """Interval outer product ``a ⊗ b`` with outward rounding.

    When an operand contains unbounded entries (flowing from an
    unsupported-atom abstention), the corner products can hit
    ``0 * inf = NaN``. NaN is sound here: downstream Gershgorin
    already refuses to certify when Hessian entries are non-finite.
    The ``errstate`` below merely suppresses the runtime warning.

    When ``a is b`` the result is ``a aᵀ``, which is symmetric PSD for
    every concrete ``a`` — a property the generic corner-product
    enclosure does not preserve. The self-outer-product specialisation
    tightens the diagonal via the squaring rule (``aᵢ²`` is nonneg and
    bounded between ``0`` and ``max(|loᵢ|, |hiᵢ|)²``) which is
    essential for Hessian enclosures tight enough for Gershgorin to
    certify convexity of compositions like ``exp(x²)``.
    """
    a_lo = a.lo[:, None]
    a_hi = a.hi[:, None]
    b_lo = b.lo[None, :]
    b_hi = b.hi[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        p1 = a_lo * b_lo
        p2 = a_lo * b_hi
        p3 = a_hi * b_lo
        p4 = a_hi * b_hi
    lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    if a is b:
        # Dependency-aware tightening of the diagonal: use the
        # squaring rule to force ``aᵢ² ≥ 0``. Off-diagonal entries
        # stay as general corner products.
        n = a.lo.shape[0]
        zero_in = (a.lo <= 0) & (a.hi >= 0)
        sq_lo = np.where(zero_in, 0.0, np.minimum(a.lo * a.lo, a.hi * a.hi))
        sq_hi = np.maximum(a.lo * a.lo, a.hi * a.hi)
        diag_idx = np.arange(n)
        lo[diag_idx, diag_idx] = sq_lo
        hi[diag_idx, diag_idx] = sq_hi
    return Interval(_round_down(lo), _round_up(hi))


def _scalar_times_array(s: Interval, arr: Interval) -> Interval:
    """``s * arr`` where ``s`` is scalar and ``arr`` has array endpoints."""
    return Interval.point(0.0) if False else _broadcast_product(s, arr)


def _broadcast_product(s: Interval, arr: Interval) -> Interval:
    s_lo = np.asarray(s.lo)
    s_hi = np.asarray(s.hi)
    with np.errstate(over="ignore", invalid="ignore"):
        p1 = s_lo * arr.lo
        p2 = s_lo * arr.hi
        p3 = s_hi * arr.lo
        p4 = s_hi * arr.hi
    lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return Interval(_round_down(lo), _round_up(hi))


def _unbounded_triple(n: int) -> IntervalAD:
    inf = np.float64(np.inf)
    return IntervalAD(
        value=Interval(np.float64(-inf), np.float64(inf)),
        grad=Interval(np.full(n, -inf), np.full(n, inf)),
        hess=Interval(np.full((n, n), -inf), np.full((n, n), inf)),
    )


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def interval_hessian(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
) -> IntervalAD:
    """Interval value, gradient, and Hessian of a scalar expression.

    The returned triple encloses ``(f(x), ∇f(x), ∇²f(x))`` for every
    point ``x`` in the input box. The Hessian enclosure is the
    artifact the convexity certificate consumes.

    Args:
        expr: Scalar expression from :mod:`discopt.modeling.core`.
        model: The model defining the flat variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared
            bounds. Missing variables fall back to ``(v.lb, v.ub)``.

    Raises:
        ValueError: if ``expr`` references array-shaped values that
            the scalar-output AD cannot handle.
    """
    n = _flat_size(model)
    if n == 0:
        raise ValueError("Model has no variables; cannot produce Hessian.")
    box = box or {}
    cache: dict = {}
    return _walk(expr, model, box, cache, n)


# ──────────────────────────────────────────────────────────────────────
# Internal DAG walker
# ──────────────────────────────────────────────────────────────────────


def _walk(expr: Expression, model: Model, box: dict, cache: dict, n: int) -> IntervalAD:
    eid = id(expr)
    if eid in cache:
        return cache[eid]
    out = _impl(expr, model, box, cache, n)
    cache[eid] = out
    return out


def _variable_scalar_value(v: Variable, box: dict) -> Interval:
    """Interval enclosure of a *scalar* variable's value."""
    if v in box:
        return box[v]
    lb = float(np.asarray(v.lb).ravel()[0])
    ub = float(np.asarray(v.ub).ravel()[0])
    return Interval(np.float64(lb), np.float64(ub))


def _indexed_scalar_value(expr: IndexExpression, box: dict) -> Interval:
    v = expr.base
    lb = np.asarray(v.lb).ravel()
    ub = np.asarray(v.ub).ravel()
    # Translate the index into a flat position inside the variable.
    idx = expr.index
    if isinstance(idx, tuple):
        # Only 1-D indexing supported for scalar output.
        if len(idx) == 1:
            idx = idx[0]
        else:
            raise ValueError("Multi-dim indexing unsupported in interval AD")
    if v in box:
        # Box override is shape (size,) when variable is array-valued.
        box_iv = box[v]
        return Interval(np.asarray(box_iv.lo).ravel()[idx], np.asarray(box_iv.hi).ravel()[idx])
    return Interval(np.float64(lb[idx]), np.float64(ub[idx]))


def _impl(expr: Expression, model: Model, box: dict, cache: dict, n: int) -> IntervalAD:
    # --- Leaves -----------------------------------------------------
    if isinstance(expr, Constant):
        v = float(np.asarray(expr.value))
        return IntervalAD(
            value=Interval(np.float64(v), np.float64(v)),
            grad=_zero_grad(n),
            hess=_zero_hess(n),
        )

    if isinstance(expr, Parameter):
        v = float(np.asarray(expr.value))
        return IntervalAD(
            value=Interval(np.float64(v), np.float64(v)),
            grad=_zero_grad(n),
            hess=_zero_hess(n),
        )

    if isinstance(expr, Variable):
        if expr.size != 1:
            raise ValueError(f"Interval Hessian requires scalar variables; got shape {expr.shape}")
        slot = _var_offset(expr, model)
        val = _variable_scalar_value(expr, box)
        return IntervalAD(value=val, grad=_unit_grad(n, slot), hess=_zero_hess(n))

    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        v = expr.base
        raw_idx = expr.index
        if isinstance(raw_idx, tuple):
            if len(raw_idx) != 1:
                raise ValueError("Multi-dim indexing unsupported")
            flat_idx = int(raw_idx[0])
        else:
            flat_idx = int(raw_idx)
        slot = _var_offset(v, model) + flat_idx
        val = _indexed_scalar_value(expr, box)
        return IntervalAD(value=val, grad=_unit_grad(n, slot), hess=_zero_hess(n))

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        child = _walk(expr.operand, model, box, cache, n)
        if expr.op == "neg":
            return IntervalAD(
                value=-child.value,
                grad=-child.grad,
                hess=-child.hess,
            )
        # |x| is non-smooth at 0 — no sound Hessian.
        return _unbounded_triple(n)

    # --- Binary ops -------------------------------------------------
    if isinstance(expr, BinaryOp):
        return _binary(expr, model, box, cache, n)

    # --- Function calls --------------------------------------------
    if isinstance(expr, FunctionCall):
        return _function_call(expr, model, box, cache, n)

    if isinstance(expr, SumExpression):
        return _walk(expr.operand, model, box, cache, n)

    if isinstance(expr, SumOverExpression):
        if not expr.terms:
            return IntervalAD(Interval.point(0.0), _zero_grad(n), _zero_hess(n))
        result = _walk(expr.terms[0], model, box, cache, n)
        for t in expr.terms[1:]:
            other = _walk(t, model, box, cache, n)
            result = IntervalAD(
                value=result.value + other.value,
                grad=result.grad + other.grad,
                hess=result.hess + other.hess,
            )
        return result

    return _unbounded_triple(n)


# ──────────────────────────────────────────────────────────────────────
# Binary-op rules
# ──────────────────────────────────────────────────────────────────────


def _binary(expr: BinaryOp, model: Model, box: dict, cache: dict, n: int) -> IntervalAD:
    left = _walk(expr.left, model, box, cache, n)
    right = _walk(expr.right, model, box, cache, n)

    if expr.op == "+":
        return IntervalAD(
            value=left.value + right.value,
            grad=left.grad + right.grad,
            hess=left.hess + right.hess,
        )

    if expr.op == "-":
        return IntervalAD(
            value=left.value - right.value,
            grad=left.grad - right.grad,
            hess=left.hess - right.hess,
        )

    if expr.op == "*":
        # (f g)'  = g f' + f g'
        # (f g)'' = g f'' + f g'' + f' g'^T + g' f'^T
        fg = left.value * right.value
        grad = _broadcast_product(right.value, left.grad) + _broadcast_product(
            left.value, right.grad
        )
        hess_terms = (
            _broadcast_product(right.value, left.hess)
            + _broadcast_product(left.value, right.hess)
            + _outer(left.grad, right.grad)
            + _outer(right.grad, left.grad)
        )
        # Rank-1 metadata for ``g * g`` (BinaryOp form of squaring)
        # when ``g`` is affine. The cache makes ``left is right`` for
        # any expression node that appears twice as the same instance,
        # so this catches ``x * x`` and any ``e * e`` where the user
        # bound ``e`` to a single Python variable. The Hessian
        # already collapses to ``2 ∇g ∇gᵀ`` exactly here (the two
        # ``_outer`` calls fire self-product tightening), so the
        # metadata claim is sound.
        rank1: Optional[Rank1Factor] = None
        if expr.left is expr.right and _hess_is_exactly_zero(left.hess):
            rank1 = Rank1Factor(
                c=Interval.point(2.0),
                v=left.grad,
                affine_base_value=left.value,
                affine_base_grad=left.grad,
            )
        return IntervalAD(value=fg, grad=grad, hess=hess_terms, rank1_factor=rank1)

    if expr.op == "/":
        return _division(expr, left, right, n)

    if expr.op == "**":
        return _power(expr, left, n)

    return _unbounded_triple(n)


def _division(expr: BinaryOp, left: IntervalAD, right: IntervalAD, n: int) -> IntervalAD:
    """Implement ``f / g`` via the reciprocal chain rule.

    Defined only when ``g`` is strictly sign-determined. Otherwise the
    triple falls to the unbounded enclosure.
    """
    g = right.value
    if g.contains_zero().any():
        return _unbounded_triple(n)

    # Rank-1 fast path: when the numerator is the square of an affine
    # expression and the denominator is itself affine and strictly
    # positive, the quotient's Hessian collapses to a perspective-form
    # rank-1 matrix that the certificate can prove PSD structurally
    # — even on wide boxes where the entry-wise interval enclosure is
    # too loose for Gershgorin.
    if (
        left.rank1_factor is not None
        and left.rank1_factor.affine_base_value is not None
        and left.rank1_factor.affine_base_grad is not None
        and bool(np.all(np.asarray(g.lo) > 0.0))
        and _hess_is_exactly_zero(right.hess)
    ):
        return _rank1_quotient(left, right, n)

    # Reciprocal derivatives:
    #   (1/g)'  = -g' / g^2
    #   (1/g)'' = 2 g' g'^T / g^3 - g'' / g^2
    g2 = g * g
    g3 = g2 * g
    inv_g = Interval.point(1.0) / g
    inv_g2 = Interval.point(1.0) / g2
    inv_g3 = Interval.point(1.0) / g3
    recip_value = inv_g
    recip_grad = _broadcast_product(-inv_g2, right.grad)
    recip_hess = _broadcast_product(
        Interval.point(2.0) * inv_g3, _outer(right.grad, right.grad)
    ) - _broadcast_product(inv_g2, right.hess)

    # f / g = f * (1/g) — apply product rule.
    fg_val = left.value * recip_value
    fg_grad = _broadcast_product(recip_value, left.grad) + _broadcast_product(
        left.value, recip_grad
    )
    fg_hess = (
        _broadcast_product(recip_value, left.hess)
        + _broadcast_product(left.value, recip_hess)
        + _outer(left.grad, recip_grad)
        + _outer(recip_grad, left.grad)
    )
    return IntervalAD(value=fg_val, grad=fg_grad, hess=fg_hess)


def _rank1_quotient(left: IntervalAD, right: IntervalAD, n: int) -> IntervalAD:
    """Specialised ``g² / h`` Hessian when ``g`` and ``h`` are affine.

    For affine ``g`` with gradient ``v_g`` (so ``H_g = 0``) and affine
    ``h`` with gradient ``v_h`` (so ``H_h = 0``) and ``h > 0`` on the
    box, the quotient ``g²/h`` has the exact pointwise Hessian

        ``H = (2/h) · v vᵀ``     where  ``v = v_g − (g/h) · v_h``.

    This is the perspective form: a rank-1 PSD matrix on every point
    of the box. We emit the Hessian via ``_outer(v, v)`` with a
    single :class:`Interval` instance so the self-product tightening
    forces the diagonal to be nonneg, and we attach a
    :class:`Rank1Factor` so the certificate's structural PSD test
    fires regardless of off-diagonal interval blowup.
    """
    assert left.rank1_factor is not None
    assert left.rank1_factor.affine_base_value is not None
    assert left.rank1_factor.affine_base_grad is not None

    g_val = left.rank1_factor.affine_base_value
    v_g = left.rank1_factor.affine_base_grad
    h = right.value
    v_h = right.grad

    # Combined rank-1 vector and coefficient. Each operand is a fresh
    # Interval; the final combined Interval is the single instance we
    # pass to _outer to trigger self-product tightening.
    g_over_h = g_val / h
    v_combined = v_g - _broadcast_product(g_over_h, v_h)
    c_combined = Interval.point(2.0) / h

    outer_vv = _outer(v_combined, v_combined)
    hess = _broadcast_product(c_combined, outer_vv)

    # Value and gradient via the standard reciprocal-product rule —
    # tightness on those is not required for the convexity verdict
    # but they remain sound and consistent with the generic path.
    inv_h = Interval.point(1.0) / h
    inv_h2 = Interval.point(1.0) / (h * h)
    recip_grad = _broadcast_product(-inv_h2, v_h)
    fg_val = left.value * inv_h
    fg_grad = _broadcast_product(inv_h, left.grad) + _broadcast_product(left.value, recip_grad)

    return IntervalAD(
        value=fg_val,
        grad=fg_grad,
        hess=hess,
        rank1_factor=Rank1Factor(c=c_combined, v=v_combined),
    )


def _power(expr: BinaryOp, base: IntervalAD, n_vars: int) -> IntervalAD:
    """``g^p`` for a literal exponent ``p``.

    Supports integer ``p`` on any domain and fractional ``p`` on a
    strictly positive base (through ``exp(p log g)`` composition).
    """
    if not isinstance(expr.right, (Constant, Parameter)):
        return _unbounded_triple(n_vars)
    raw = np.asarray(expr.right.value)
    if raw.ndim != 0:
        return _unbounded_triple(n_vars)
    p = float(raw)

    if np.isclose(p, 0.0):
        return IntervalAD(Interval.point(1.0), _zero_grad(n_vars), _zero_hess(n_vars))
    if np.isclose(p, 1.0):
        return base

    # Integer exponent path — computed without leaving interval arithmetic.
    p_int = int(p)
    if np.isclose(p, float(p_int)):
        return _integer_power(base, p_int, n_vars)

    # Fractional: require strictly positive base; use exp(p log g).
    if np.any(base.value.lo <= 0):
        return _unbounded_triple(n_vars)
    # Build exp(p * log(g)) through the AD machinery by applying log
    # and exp rules to ``base`` directly — equivalent to the general
    # power rule.
    log_g = _apply_log(base, n_vars)
    scaled = IntervalAD(
        value=Interval.point(p) * log_g.value,
        grad=_broadcast_product(Interval.point(p), log_g.grad),
        hess=_broadcast_product(Interval.point(p), log_g.hess),
    )
    return _apply_exp(scaled, n_vars)


_AFFINE_HESS_TOL = 1e-300


def _hess_is_exactly_zero(hess: Interval) -> bool:
    """``True`` iff every entry of the interval Hessian encloses ``0``
    with a magnitude well below any meaningful scale.

    Used to detect "affine in the variables" nodes: when the
    algebraic ``H_g`` is identically zero, the AD walker produces
    an interval Hessian whose entries are within a few ULPs of
    ``0`` (subnormals from outward-rounded additions of exact
    zeros). The tolerance ``1e-300`` is six orders of magnitude
    above the smallest subnormal and still ~270 orders of magnitude
    below any plausible nonzero Hessian entry, so it cannot
    misclassify a real (even tiny) curvature term as affine.

    The check is sufficient-only: a false ``False`` just causes the
    rank-1 fast path to be skipped, falling through to the existing
    Gershgorin path.
    """
    return bool(
        np.all(np.abs(hess.lo) <= _AFFINE_HESS_TOL) and np.all(np.abs(hess.hi) <= _AFFINE_HESS_TOL)
    )


def _integer_power(base: IntervalAD, p: int, n: int) -> IntervalAD:
    """Direct chain rule for ``g^p`` with integer ``p``.

    * value   : ``g^p``
    * gradient: ``p g^{p-1} ∇g``
    * hessian : ``p g^{p-1} H_g + p(p-1) g^{p-2} (∇g ⊗ ∇g)``

    Works for any sign of ``g`` when ``p`` is a positive integer;
    negative integer ``p`` goes through the reciprocal path.
    """
    if p < 0:
        # g^(-k) = (g^k)^-1 via the generic reciprocal chain rule.
        return _reciprocal_power(base, -p, n)
    g = base.value
    g_pm1 = g ** (p - 1)
    g_pm2 = g ** (p - 2) if p >= 2 else Interval.point(0.0)
    coeff1 = Interval.point(float(p)) * g_pm1
    coeff2 = Interval.point(float(p * (p - 1))) * g_pm2
    value = g**p
    grad = _broadcast_product(coeff1, base.grad)
    hess = _broadcast_product(coeff1, base.hess) + _broadcast_product(
        coeff2, _outer(base.grad, base.grad)
    )
    # Rank-1 metadata: when p == 2 and H_g is identically zero (g is
    # affine), the second-order term ``p g^{p-1} H_g`` vanishes and
    # the Hessian collapses exactly to ``2 · ∇g ∇gᵀ``. Soundness is
    # independent of this field; it is consumed only as a tighter
    # sufficient test by the certificate.
    rank1: Optional[Rank1Factor] = None
    if p == 2 and _hess_is_exactly_zero(base.hess):
        rank1 = Rank1Factor(
            c=Interval.point(2.0),
            v=base.grad,
            affine_base_value=base.value,
            affine_base_grad=base.grad,
        )
    return IntervalAD(value=value, grad=grad, hess=hess, rank1_factor=rank1)


def _reciprocal_power(base: IntervalAD, k: int, n: int) -> IntervalAD:
    """``g^(-k) = 1 / g^k`` via the reciprocal rule."""
    if base.value.contains_zero().any():
        return _unbounded_triple(n)
    gk = _integer_power(base, k, n)
    # Reciprocal of gk.
    g = gk.value
    g2 = g * g
    g3 = g2 * g
    value = Interval.point(1.0) / g
    grad = _broadcast_product(-(Interval.point(1.0) / g2), gk.grad)
    hess = _broadcast_product(
        Interval.point(2.0) / g3, _outer(gk.grad, gk.grad)
    ) - _broadcast_product(Interval.point(1.0) / g2, gk.hess)
    return IntervalAD(value=value, grad=grad, hess=hess)


# ──────────────────────────────────────────────────────────────────────
# Function-call rules
# ──────────────────────────────────────────────────────────────────────


def _function_call(expr: FunctionCall, model: Model, box: dict, cache: dict, n: int) -> IntervalAD:
    if len(expr.args) != 1:
        return _unbounded_triple(n)
    arg = _walk(expr.args[0], model, box, cache, n)
    name = expr.func_name
    if name == "exp":
        return _apply_exp(arg, n)
    if name == "log":
        return _apply_log(arg, n)
    if name == "sqrt":
        # sqrt = x^0.5 on the positive domain; reuse the fractional path.
        if np.any(arg.value.lo < 0):
            return _unbounded_triple(n)
        half = IntervalAD(
            value=iv.sqrt(arg.value),
            grad=arg.grad,  # will be overwritten; dummy
            hess=arg.hess,
        )
        # Apply g^0.5 chain rule directly: f = g^0.5, f' = 0.5 g^-0.5 g',
        # f'' = 0.5 g^-0.5 H_g - 0.25 g^-1.5 (∇g ∇g^T).
        sqrt_g = iv.sqrt(arg.value)
        inv_sqrt_g = Interval.point(1.0) / sqrt_g
        inv_sqrt_g3 = inv_sqrt_g * inv_sqrt_g * inv_sqrt_g
        coeff1 = Interval.point(0.5) * inv_sqrt_g
        coeff2 = Interval.point(-0.25) * inv_sqrt_g3
        grad = _broadcast_product(coeff1, arg.grad)
        hess = _broadcast_product(coeff1, arg.hess) + _broadcast_product(
            coeff2, _outer(arg.grad, arg.grad)
        )
        _ = half  # kept to make the intent above explicit
        return IntervalAD(value=sqrt_g, grad=grad, hess=hess)
    # Other atoms (trig, abs, cosh, ...) are unsupported by the v1
    # certificate; return unbounded to force abstention.
    return _unbounded_triple(n)


def _apply_exp(arg: IntervalAD, n: int) -> IntervalAD:
    """Chain rule through ``exp``.

    * value    : ``exp(g)``
    * gradient : ``exp(g) ∇g``
    * hessian  : ``exp(g) (H_g + ∇g ∇g^T)``
    """
    e = iv.exp(arg.value)
    grad = _broadcast_product(e, arg.grad)
    hess = _broadcast_product(e, arg.hess + _outer(arg.grad, arg.grad))
    return IntervalAD(value=e, grad=grad, hess=hess)


def _apply_log(arg: IntervalAD, n: int) -> IntervalAD:
    """Chain rule through ``log``.

    * value    : ``log(g)``
    * gradient : ``(1/g) ∇g``
    * hessian  : ``(1/g) H_g - (1/g²) (∇g ∇g^T)``
    """
    if np.any(arg.value.lo <= 0):
        return _unbounded_triple(n)
    g = arg.value
    inv_g = Interval.point(1.0) / g
    inv_g2 = inv_g * inv_g
    value = iv.log(g)
    grad = _broadcast_product(inv_g, arg.grad)
    hess = _broadcast_product(inv_g, arg.hess) - _broadcast_product(
        inv_g2, _outer(arg.grad, arg.grad)
    )
    return IntervalAD(value=value, grad=grad, hess=hess)


__all__ = ["IntervalAD", "Rank1Factor", "interval_hessian"]
