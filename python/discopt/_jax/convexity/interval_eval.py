"""Interval-valued evaluator for modeling expressions.

Walks a ``discopt.modeling`` expression DAG with :class:`Interval`
variable values and returns a sound interval enclosure of the
expression's value over the input box. Used as a building block by
the interval-AD Hessian propagator underlying the box-local
convexity certificate.

The evaluator trusts the underlying interval arithmetic primitives
for soundness: each supported atom composes them into a correctly
rounded enclosure. Atoms not in the supported set return an
unbounded interval ``[-inf, +inf]`` so downstream consumers refuse to
certify rather than produce a wrong answer.

References
----------
Moore (1966), *Interval Analysis*.
Neumaier (1990), *Interval Methods for Systems of Equations*.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

from . import interval as iv
from .interval import Interval

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def evaluate_interval(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
    _cache: Optional[dict] = None,
) -> Interval:
    """Return an interval enclosure of ``expr`` over ``box``.

    Args:
        expr: The expression to evaluate.
        model: The :class:`~discopt.modeling.core.Model` the expression
            references. Variables are looked up by object identity.
        box: Optional dict ``{Variable: Interval}`` that overrides the
            variable's declared bounds. When ``None`` or a variable is
            missing, the declared ``(lb, ub)`` of the variable is used.
        _cache: Memoization dict keyed by ``id(expr)``.

    Returns:
        An :class:`Interval` that contains ``expr(x)`` for every ``x``
        consistent with the variable box.
    """
    if _cache is None:
        _cache = {}
    return _eval(expr, model, box or {}, _cache)


# ──────────────────────────────────────────────────────────────────────
# Internal dispatch
# ──────────────────────────────────────────────────────────────────────


def _variable_interval(v: Variable, box: dict) -> Interval:
    """Interval enclosure for a (possibly array-shaped) variable."""
    if v in box:
        return box[v]
    lb = np.asarray(v.lb, dtype=np.float64)
    ub = np.asarray(v.ub, dtype=np.float64)
    return Interval(lb, ub)


def _indexed_interval(expr: IndexExpression, box: dict) -> Optional[Interval]:
    """Interval enclosure for ``var[idx]``; ``None`` when the pattern
    is not a direct index of a Variable (caller falls back to a
    general DAG walk)."""
    if not isinstance(expr.base, Variable):
        return None
    base_iv = _variable_interval(expr.base, box)
    try:
        lo = base_iv.lo[expr.index]
        hi = base_iv.hi[expr.index]
    except (IndexError, TypeError):
        return None
    return Interval(np.asarray(lo), np.asarray(hi))


def _eval(expr: Expression, model: Model, box: dict, cache: dict) -> Interval:
    eid = id(expr)
    if eid in cache:
        return cache[eid]
    result = _eval_impl(expr, model, box, cache)
    cache[eid] = result
    return result


def _eval_impl(expr: Expression, model: Model, box: dict, cache: dict) -> Interval:
    # --- Leaves -----------------------------------------------------
    if isinstance(expr, Constant):
        v = np.asarray(expr.value, dtype=np.float64)
        return Interval(v, v)
    if isinstance(expr, Parameter):
        v = np.asarray(expr.value, dtype=np.float64)
        return Interval(v, v)
    if isinstance(expr, Variable):
        return _variable_interval(expr, box)
    if isinstance(expr, IndexExpression):
        idx_iv = _indexed_interval(expr, box)
        if idx_iv is not None:
            return idx_iv
        # Fallback: recurse into the base expression (non-variable base).
        base = _eval(expr.base, model, box, cache)
        return Interval(np.asarray(base.lo[expr.index]), np.asarray(base.hi[expr.index]))

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        child = _eval(expr.operand, model, box, cache)
        if expr.op == "neg":
            return -child
        if expr.op == "abs":
            return iv.absolute(child)
        return _unbounded(child.lo.shape)

    # --- Binary ops -------------------------------------------------
    if isinstance(expr, BinaryOp):
        left = _eval(expr.left, model, box, cache)
        right = _eval(expr.right, model, box, cache)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "**":
            return _eval_power(expr, left, right)
        return _unbounded(left.lo.shape)

    # --- Function calls --------------------------------------------
    if isinstance(expr, FunctionCall):
        return _eval_function_call(expr, model, box, cache)

    # --- Aggregations ----------------------------------------------
    if isinstance(expr, SumExpression):
        return _eval(expr.operand, model, box, cache)

    if isinstance(expr, SumOverExpression):
        if not expr.terms:
            return Interval(np.zeros(()), np.zeros(()))
        total = _eval(expr.terms[0], model, box, cache)
        for t in expr.terms[1:]:
            total = total + _eval(t, model, box, cache)
        return total

    if isinstance(expr, MatMulExpression):
        return _eval_matmul(expr, model, box, cache)

    return _unbounded(())


def _unbounded(shape) -> Interval:
    return Interval(
        np.full(shape, -np.inf, dtype=np.float64),
        np.full(shape, np.inf, dtype=np.float64),
    )


def _eval_power(expr: BinaryOp, left: Interval, right: Interval) -> Interval:
    """Handle ``base ** exponent`` — constant exponent only for v1."""
    # Require a concrete scalar exponent so we know whether integer or
    # fractional rules apply.
    if not isinstance(expr.right, (Constant, Parameter)):
        return _unbounded(left.lo.shape)
    raw = np.asarray(expr.right.value)
    if raw.ndim != 0:
        return _unbounded(left.lo.shape)
    n = float(raw)
    n_int = int(n)
    if np.isclose(n, float(n_int)):
        return left**n_int
    # Fractional: base must be nonneg; use exp(n log(x)).
    if np.any(left.lo < 0):
        return _unbounded(left.lo.shape)
    return iv.exp(Interval.point(n) * iv.log(left))


def _eval_function_call(expr: FunctionCall, model: Model, box: dict, cache: dict) -> Interval:
    if not expr.args:
        return _unbounded(())
    args = [_eval(a, model, box, cache) for a in expr.args]

    if expr.func_name == "max" and len(args) >= 2:
        lo = args[0].lo
        hi = args[0].hi
        for a in args[1:]:
            lo = np.maximum(lo, a.lo)
            hi = np.maximum(hi, a.hi)
        return Interval(lo, hi)
    if expr.func_name == "min" and len(args) >= 2:
        lo = args[0].lo
        hi = args[0].hi
        for a in args[1:]:
            lo = np.minimum(lo, a.lo)
            hi = np.minimum(hi, a.hi)
        return Interval(lo, hi)

    if len(args) != 1:
        return _unbounded(args[0].lo.shape)

    arg = args[0]
    name = expr.func_name
    if name == "exp":
        return iv.exp(arg)
    if name == "log":
        return iv.log(arg)
    if name == "log2":
        return iv.log(arg) / Interval.point(float(np.log(2.0)))
    if name == "log10":
        return iv.log(arg) / Interval.point(float(np.log(10.0)))
    if name == "sqrt":
        return iv.sqrt(arg)
    if name == "abs":
        return iv.absolute(arg)
    if name == "sin":
        return iv.sin(arg)
    if name == "cos":
        return iv.cos(arg)
    if name == "tan":
        return iv.tan(arg)
    if name == "sinh":
        return iv.sinh(arg)
    if name == "cosh":
        return iv.cosh(arg)
    if name == "tanh":
        return iv.tanh(arg)
    # Unsupported atoms return an unbounded enclosure; the certificate
    # will refuse to prove convexity for expressions that hit this
    # path, preserving soundness.
    return _unbounded(arg.lo.shape)


def _eval_matmul(expr: MatMulExpression, model: Model, box: dict, cache: dict) -> Interval:
    """Interval matrix–vector or matrix–matrix product.

    ``discopt`` uses ``MatMulExpression`` primarily for constant-matrix
    times variable-vector; the handling below covers that case plus
    the symmetric one.
    """
    left = _eval(expr.left, model, box, cache)
    right = _eval(expr.right, model, box, cache)

    # For a matmul of interval matrices A (m × k) by B (k × n) the
    # enclosure is formed from the interval dot products. We express
    # the dot product as the sum of element-wise interval products,
    # which the :class:`Interval` operators already propagate soundly.
    A_lo, A_hi = np.asarray(left.lo), np.asarray(left.hi)
    B_lo, B_hi = np.asarray(right.lo), np.asarray(right.hi)
    if A_lo.ndim == 2 and B_lo.ndim == 1:
        # (m, k) @ (k,) → (m,)
        m, k = A_lo.shape
        lo = np.zeros(m, dtype=np.float64)
        hi = np.zeros(m, dtype=np.float64)
        for i in range(m):
            row_lo = A_lo[i]
            row_hi = A_hi[i]
            prods_lo = np.minimum(
                np.minimum(row_lo * B_lo, row_lo * B_hi),
                np.minimum(row_hi * B_lo, row_hi * B_hi),
            )
            prods_hi = np.maximum(
                np.maximum(row_lo * B_lo, row_lo * B_hi),
                np.maximum(row_hi * B_lo, row_hi * B_hi),
            )
            lo[i] = prods_lo.sum()
            hi[i] = prods_hi.sum()
        return Interval(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))
    # Other shapes fall through as unbounded for now — not needed by
    # any expression the convexity certificate currently targets.
    return _unbounded(A_lo.shape[:-1] if A_lo.ndim >= 1 else ())


__all__ = ["evaluate_interval"]
