"""Posynomial and monomial recognition for geometric programming.

A *monomial* is a single term ``c * prod_j x_j^{a_j}`` with a strictly
positive coefficient ``c > 0``, real exponents ``a_j``, over strictly
positive variables ``x_j > 0``. A *posynomial* is a sum of monomials.

Posynomials are **not** convex in ``x`` (``x*y`` is indefinite on the
positive orthant), so the DCP walker in :mod:`rules` correctly leaves
them at ``UNKNOWN``. But under the substitution ``y_j = log(x_j)`` a
posynomial becomes ``sum_i c_i * exp(a_i^T y)`` — a sum of exponentials,
hence convex in ``y``. That log-space convexity is what the geometric
programming pipeline in :mod:`discopt.gp` exploits.

This module is the recogniser: it parses an :class:`Expression` into a
:class:`PosynomialForm` (a normalised sum of :class:`Monomial` terms over
the model's flat scalar-variable indexing) when — and only when — every
posynomial precondition holds. It returns ``None`` otherwise. The
recogniser is deliberately conservative: a returned form is always a
genuine posynomial on the strictly-positive box.

Preconditions enforced (any failure -> ``None``):

* The expression flattens into a sum of monomials.
* Every monomial coefficient is strictly positive (no signomials).
* Every variable leaf has a strictly positive declared lower bound.
* Every exponent is a real *constant* (a :class:`Constant` /
  :class:`Parameter` scalar), never another expression.

References: Boyd & Vandenberghe Ch. 4.5; Agrawal et al. 2019 (Disciplined
Geometric Programming).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    SumOverExpression,
    UnaryOp,
    Variable,
)

# Tolerance for treating an extracted coefficient as non-positive.
_POS_TOL = 1e-12


@dataclass
class Monomial:
    """A single monomial ``coeff * prod_j x_j^{exponents[j]}``.

    ``coeff`` is strictly positive in any monomial that survives into a
    :class:`PosynomialForm`. ``exponents`` maps a *flat scalar-variable
    offset* (the index of the variable in the model's concatenated
    variable vector) to its real exponent. A variable absent from the
    map has exponent zero. An empty map is a positive constant.
    """

    coeff: float
    exponents: dict[int, float] = field(default_factory=dict)


@dataclass
class PosynomialForm:
    """A posynomial as a normalised list of :class:`Monomial` terms."""

    monomials: list[Monomial]

    @property
    def is_monomial(self) -> bool:
        """True when the posynomial is a single monomial."""
        return len(self.monomials) == 1

    def variable_offsets(self) -> set[int]:
        """Flat scalar offsets of every variable appearing with nonzero exponent."""
        offsets: set[int] = set()
        for mono in self.monomials:
            for off, exp in mono.exponents.items():
                if abs(exp) > _POS_TOL:
                    offsets.add(off)
        return offsets


# ──────────────────────────────────────────────────────────────────────
# Flat scalar-offset helpers (kept local so the module is self-contained)
# ──────────────────────────────────────────────────────────────────────


def _var_offset(model: Model, target: Variable) -> Optional[int]:
    offset = 0
    for v in model._variables:
        if v is target or v.name == target.name:
            return offset
        offset += v.size
    return None


def _leaf_offset_and_lb(expr: Expression, model: Model) -> Optional[tuple[int, float]]:
    """Return ``(flat_offset, lower_bound)`` for a scalar variable leaf.

    Handles a bare size-1 :class:`Variable` and an integer-indexed entry of
    a 1-D :class:`Variable`. Returns ``None`` for anything else (multi-d
    indexing, non-variable bases, out-of-range indices).
    """
    if isinstance(expr, Variable):
        if expr.size != 1:
            return None
        offset = _var_offset(model, expr)
        if offset is None:
            return None
        return offset, float(np.asarray(expr.lb).min())
    if isinstance(expr, IndexExpression):
        base = expr.base
        if not isinstance(base, Variable):
            return None
        if len(base.shape) != 1:
            return None
        idx = expr.index
        if isinstance(idx, (tuple, list)):
            return None
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= base.size:
            return None
        base_offset = _var_offset(model, base)
        if base_offset is None:
            return None
        lb = float(np.asarray(base.lb).reshape(-1)[idx])
        return base_offset + idx, lb
    return None


def _const_scalar(expr: Expression) -> Optional[float]:
    """Return the scalar value of a constant/parameter leaf, else ``None``."""
    if isinstance(expr, (Constant, Parameter)):
        val = np.asarray(expr.value)
        if val.ndim == 0:
            return float(val)
    return None


def _flatten_sum_terms(expr: Expression, scale: float, out: list[tuple[float, Expression]]) -> None:
    """Flatten a +/-/neg tree into ``(signed_scale, leaf_term)`` pairs."""
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_sum_terms(expr.left, scale, out)
        _flatten_sum_terms(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_sum_terms(expr.left, scale, out)
        _flatten_sum_terms(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _flatten_sum_terms(expr.operand, -scale, out)
        return
    out.append((scale, expr))


def _merge_into(dst: dict[int, float], src: dict[int, float], sign: float) -> None:
    for off, exp in src.items():
        dst[off] = dst.get(off, 0.0) + sign * exp


# ──────────────────────────────────────────────────────────────────────
# Monomial parsing
# ──────────────────────────────────────────────────────────────────────


def _parse_monomial(expr: Expression, model: Model) -> Optional[Monomial]:
    """Parse ``expr`` into a single :class:`Monomial`, or ``None``.

    Recognises constants, scalar variable leaves (with ``lb > 0``),
    products, quotients, integer/real powers, and ``sqrt``. A ``+``/``-``
    node (a genuine sum) is *not* a monomial and yields ``None`` here; the
    sum is handled one level up by :func:`is_posynomial`.

    The coefficient may be negative at this level (e.g. via ``neg``); sign
    is validated when the monomial is assembled into a posynomial.
    """
    # Constant / parameter scalar -> coefficient-only monomial.
    const = _const_scalar(expr)
    if const is not None:
        return Monomial(const, {})

    # Variable leaf (bare scalar or indexed 1-D entry); require lb > 0.
    leaf = _leaf_offset_and_lb(expr, model)
    if leaf is not None:
        offset, lb = leaf
        if lb <= 0.0:
            return None
        return Monomial(1.0, {offset: 1.0})

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _parse_monomial(expr.operand, model)
        if inner is None:
            return None
        return Monomial(-inner.coeff, inner.exponents)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left = _parse_monomial(expr.left, model)
        right = _parse_monomial(expr.right, model)
        if left is None or right is None:
            return None
        merged: dict[int, float] = dict(left.exponents)
        _merge_into(merged, right.exponents, 1.0)
        return Monomial(left.coeff * right.coeff, merged)

    if isinstance(expr, BinaryOp) and expr.op == "/":
        left = _parse_monomial(expr.left, model)
        right = _parse_monomial(expr.right, model)
        if left is None or right is None:
            return None
        if abs(right.coeff) <= _POS_TOL:
            return None
        merged = dict(left.exponents)
        _merge_into(merged, right.exponents, -1.0)
        return Monomial(left.coeff / right.coeff, merged)

    if isinstance(expr, BinaryOp) and expr.op == "**":
        exponent = _const_scalar(expr.right)
        if exponent is None:
            return None
        base = _parse_monomial(expr.left, model)
        if base is None:
            return None
        # A non-integer power of a negative-coefficient base is not real.
        if base.coeff < 0.0 and not float(exponent).is_integer():
            return None
        new_coeff = float(np.power(base.coeff, exponent))
        if not np.isfinite(new_coeff):
            return None
        return Monomial(new_coeff, {off: e * exponent for off, e in base.exponents.items()})

    if isinstance(expr, FunctionCall) and expr.func_name == "sqrt" and len(expr.args) == 1:
        base = _parse_monomial(expr.args[0], model)
        if base is None or base.coeff < 0.0:
            return None
        return Monomial(
            float(np.sqrt(base.coeff)),
            {off: 0.5 * e for off, e in base.exponents.items()},
        )

    return None


def is_posynomial(expr: Expression, model: Model) -> Optional[PosynomialForm]:
    """Return a :class:`PosynomialForm` if ``expr`` is a posynomial, else ``None``.

    See the module docstring for the exact preconditions. The recogniser
    is sound: a non-``None`` return is a genuine posynomial on the
    strictly-positive box declared by the model's variable bounds.
    """
    # ``SumOverExpression`` (vectorised reductions) are not handled here; the
    # caller works with the scalar expansion. Reject defensively.
    if isinstance(expr, SumOverExpression):
        return None

    terms: list[tuple[float, Expression]] = []
    _flatten_sum_terms(expr, 1.0, terms)

    monomials: list[Monomial] = []
    for scale, term in terms:
        mono = _parse_monomial(term, model)
        if mono is None:
            return None
        coeff = mono.coeff * scale
        if coeff <= _POS_TOL:
            # Non-positive coefficient -> signomial, not a posynomial.
            return None
        monomials.append(Monomial(coeff, mono.exponents))

    if not monomials:
        return None
    return PosynomialForm(monomials)


def is_monomial(expr: Expression, model: Model) -> Optional[Monomial]:
    """Return the single :class:`Monomial` if ``expr`` is one, else ``None``."""
    form = is_posynomial(expr, model)
    if form is not None and form.is_monomial:
        return form.monomials[0]
    return None


__all__ = [
    "Monomial",
    "PosynomialForm",
    "is_monomial",
    "is_posynomial",
]
