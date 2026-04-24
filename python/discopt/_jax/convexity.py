"""
Convexity Detection for Expression DAGs.

Walks the expression DAG and classifies each (sub)expression as one of:
  CONVEX, CONCAVE, AFFINE, or UNKNOWN

using standard composition rules from convex analysis. Results are cached
on the expression objects.

Key composition rules implemented:
  - Constant/Variable/Parameter: AFFINE
  - sum/neg of convex  : CONVEX; sum/neg of concave : CONCAVE
  - const * expr       : preserves curvature if const >= 0, flips if < 0
  - convex(affine)     : CONVEX  (composition rule)
  - concave(affine)    : CONCAVE
  - exp(convex)        : CONVEX  (exp is convex and nondecreasing on the reals)
  - log(concave)       : CONCAVE (log is concave and nondecreasing)
  - x**2               : CONVEX  (for any x)
  - x**p (p>=1, x>=0)  : CONVEX  (on the nonneg reals)
  - bilinear x*y       : UNKNOWN (nonconvex in general)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
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


class Curvature(Enum):
    """Curvature classification of an expression."""

    AFFINE = "affine"
    CONVEX = "convex"
    CONCAVE = "concave"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OACutConvexity:
    """Convexity information relevant to globally valid OA cuts."""

    objective_is_convex: bool
    constraint_mask: list[bool]


def _combine_sum(a: Curvature, b: Curvature) -> Curvature:
    """Curvature of a + b."""
    if a == Curvature.AFFINE:
        return b
    if b == Curvature.AFFINE:
        return a
    if a == b:
        return a  # convex + convex = convex, etc.
    return Curvature.UNKNOWN


def _negate(c: Curvature) -> Curvature:
    """Curvature of -expr given curvature of expr."""
    if c == Curvature.CONVEX:
        return Curvature.CONCAVE
    if c == Curvature.CONCAVE:
        return Curvature.CONVEX
    return c  # AFFINE stays AFFINE, UNKNOWN stays UNKNOWN


def _scale(c: Curvature, sign: int) -> Curvature:
    """Curvature of k * expr where sign = sign(k).

    sign: +1, -1, or 0 (constant zero).
    """
    if sign == 0 or c == Curvature.AFFINE:
        return Curvature.AFFINE
    if sign > 0:
        return c
    return _negate(c)


def _is_nonneg_domain(expr: Expression, model: Model) -> bool:
    """Check if expression always evaluates to nonneg values (lb >= 0)."""
    if isinstance(expr, Constant):
        return bool(np.all(expr.value >= 0))
    if isinstance(expr, Variable):
        return bool(np.all(expr.lb >= 0))
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            idx = expr.index
            if isinstance(idx, int):
                return bool(expr.base.lb.flat[idx] >= 0)
            if isinstance(idx, tuple) and len(idx) == 1:
                return bool(expr.base.lb.flat[idx[0]] >= 0)
        return False
    if isinstance(expr, Parameter):
        return bool(np.all(expr.value >= 0))
    return False


def classify_expr(
    expr: Expression,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> Curvature:
    """Classify the curvature of an expression.

    Args:
        expr: Expression to classify.
        model: Model context (used for variable bound lookups).
        _cache: Internal memoization dict (keyed by expression id).

    Returns:
        Curvature enum value.
    """
    if _cache is None:
        _cache = {}

    eid = id(expr)
    if eid in _cache:
        return _cache[eid]  # type: ignore[no-any-return]

    result = _classify_impl(expr, model, _cache)
    _cache[eid] = result
    return result


def _classify_impl(
    expr: Expression,
    model: Optional[Model],
    cache: dict,
) -> Curvature:
    """Internal recursive classification."""

    # --- Leaves ---
    if isinstance(expr, (Constant, Parameter)):
        return Curvature.AFFINE

    if isinstance(expr, Variable):
        return Curvature.AFFINE

    if isinstance(expr, IndexExpression):
        base_curv = classify_expr(expr.base, model, cache)
        return base_curv  # indexing preserves curvature

    # --- Unary ops ---
    if isinstance(expr, UnaryOp):
        child = classify_expr(expr.operand, model, cache)
        if expr.op == "neg":
            return _negate(child)
        if expr.op == "abs":
            # |x| is convex when x is affine
            if child == Curvature.AFFINE:
                return Curvature.CONVEX
            return Curvature.UNKNOWN
        return Curvature.UNKNOWN

    # --- Binary ops ---
    if isinstance(expr, BinaryOp):
        left = classify_expr(expr.left, model, cache)
        right = classify_expr(expr.right, model, cache)

        if expr.op == "+":
            return _combine_sum(left, right)

        if expr.op == "-":
            return _combine_sum(left, _negate(right))

        if expr.op == "*":
            # const * expr
            if isinstance(expr.left, (Constant, Parameter)):
                val = np.asarray(expr.left.value)
                if val.ndim == 0:
                    s = 1 if float(val) >= 0 else -1
                    return _scale(right, s)
            if isinstance(expr.right, (Constant, Parameter)):
                val = np.asarray(expr.right.value)
                if val.ndim == 0:
                    s = 1 if float(val) >= 0 else -1
                    return _scale(left, s)
            # affine * affine is quadratic form — only convex if
            # both are the same variable (x^2). Otherwise unknown.
            return Curvature.UNKNOWN

        if expr.op == "/":
            # expr / const
            if isinstance(expr.right, (Constant, Parameter)):
                val = np.asarray(expr.right.value)
                if val.ndim == 0 and abs(float(val)) > 1e-30:
                    s = 1 if float(val) > 0 else -1
                    return _scale(left, s)
            return Curvature.UNKNOWN

        if expr.op == "**":
            # x^n where n is constant
            if isinstance(expr.right, (Constant, Parameter)):
                n_val = np.asarray(expr.right.value)
                if n_val.ndim == 0:
                    n = float(n_val)
                    n_int = int(n)
                    base = classify_expr(expr.left, model, cache)

                    # x^1 = x — preserves curvature
                    if np.isclose(n, 1.0):
                        return base

                    # x^0 = constant
                    if np.isclose(n, 0.0):
                        return Curvature.AFFINE

                    # x^2: convex for all x (affine base) or
                    # convex of convex requires monotonicity — x^2 is
                    # not monotone, so only valid for affine base.
                    if np.isclose(n, 2.0):
                        if base == Curvature.AFFINE:
                            return Curvature.CONVEX
                        return Curvature.UNKNOWN

                    # Even integer power >=2: convex on all of R
                    # when composed with affine
                    if np.isclose(n, float(n_int)) and n_int % 2 == 0 and n_int >= 2:
                        if base == Curvature.AFFINE:
                            return Curvature.CONVEX
                        return Curvature.UNKNOWN

                    # Odd integer power >= 3: convex on [0, inf),
                    # concave on (-inf, 0] — only convex when
                    # base is affine and nonneg
                    if np.isclose(n, float(n_int)) and n_int % 2 == 1 and n_int >= 3:
                        if base == Curvature.AFFINE:
                            if model is not None and _is_nonneg_domain(expr.left, model):
                                return Curvature.CONVEX
                        return Curvature.UNKNOWN

                    # Fractional: 0 < n < 1 and nonneg domain: concave
                    if 0 < n < 1:
                        if base == Curvature.AFFINE:
                            if model is not None and _is_nonneg_domain(expr.left, model):
                                return Curvature.CONCAVE
                        return Curvature.UNKNOWN

                    # n > 1 (non-integer), nonneg domain: convex
                    if n > 1:
                        if base == Curvature.AFFINE:
                            if model is not None and _is_nonneg_domain(expr.left, model):
                                return Curvature.CONVEX
                        return Curvature.UNKNOWN

            return Curvature.UNKNOWN

        return Curvature.UNKNOWN

    # --- Function calls ---
    if isinstance(expr, FunctionCall):
        name = expr.func_name
        if len(expr.args) == 1:
            arg_curv = classify_expr(expr.args[0], model, cache)

            # exp(x): convex & nondecreasing
            # exp(convex) = convex, exp(affine) = convex
            if name == "exp":
                if arg_curv in (Curvature.AFFINE, Curvature.CONVEX):
                    return Curvature.CONVEX
                return Curvature.UNKNOWN

            # log(x): concave & nondecreasing
            # log(concave) = concave, log(affine) = concave
            if name in ("log", "log2", "log10"):
                if arg_curv in (Curvature.AFFINE, Curvature.CONCAVE):
                    return Curvature.CONCAVE
                return Curvature.UNKNOWN

            # sqrt(x): concave & nondecreasing on nonneg
            if name == "sqrt":
                if arg_curv in (Curvature.AFFINE, Curvature.CONCAVE):
                    return Curvature.CONCAVE
                return Curvature.UNKNOWN

            # abs(x): convex (nondecreasing for x>0, nonincreasing for x<0
            # — satisfies DCP when composed with affine)
            if name == "abs":
                if arg_curv == Curvature.AFFINE:
                    return Curvature.CONVEX
                return Curvature.UNKNOWN

            # cosh(x): convex & even function
            if name == "cosh":
                if arg_curv == Curvature.AFFINE:
                    return Curvature.CONVEX
                return Curvature.UNKNOWN

            # sin, cos, tan, sinh, tanh, asin, acos, atan:
            # not globally convex or concave
            return Curvature.UNKNOWN

        # Multi-arg functions: unknown
        return Curvature.UNKNOWN

    # --- Sum expressions ---
    if isinstance(expr, SumExpression):
        child = classify_expr(expr.operand, model, cache)
        return child  # sum preserves curvature

    if isinstance(expr, SumOverExpression):
        result = Curvature.AFFINE
        for t in expr.terms:
            t_curv = classify_expr(t, model, cache)
            result = _combine_sum(result, t_curv)
            if result == Curvature.UNKNOWN:
                return Curvature.UNKNOWN
        return result

    # --- MatMul ---
    if isinstance(expr, MatMulExpression):
        # A @ x where A is constant: affine
        left = classify_expr(expr.left, model, cache)
        right = classify_expr(expr.right, model, cache)
        if isinstance(expr.left, (Constant, Parameter)):
            return right  # const @ expr preserves curvature
        if isinstance(expr.right, (Constant, Parameter)):
            return left  # expr @ const preserves curvature
        return Curvature.UNKNOWN

    return Curvature.UNKNOWN


def classify_constraint(
    constraint: Constraint,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> bool:
    """Check if a constraint is convex.

    A constraint g(x) <= 0 is convex when g is convex.
    A constraint g(x) >= 0 is convex when g is concave (i.e., -g convex).
    A constraint g(x) == 0 is convex only when g is affine.

    Returns:
        True if the constraint is convex, False otherwise.
    """
    if _cache is None:
        _cache = {}

    curv = classify_expr(constraint.body, model, _cache)

    if constraint.sense == "<=":
        # body <= rhs means body - rhs <= 0
        # Convex if body is convex
        return curv in (Curvature.CONVEX, Curvature.AFFINE)
    elif constraint.sense == ">=":
        # body >= rhs means rhs - body <= 0
        # Convex if body is concave (so -body is convex)
        return curv in (Curvature.CONCAVE, Curvature.AFFINE)
    elif constraint.sense == "==":
        # Equality: convex only if affine
        return curv == Curvature.AFFINE
    return False


def classify_oa_cut_convexity(model: Model) -> OACutConvexity:
    """Return the convexity information needed to generate sound OA cuts."""
    cache: dict = {}

    obj_convex = True
    if model._objective is not None:
        from discopt.modeling.core import ObjectiveSense

        obj_curv = classify_expr(model._objective.expression, model, cache)
        if model._objective.sense == ObjectiveSense.MINIMIZE:
            obj_convex = obj_curv in (Curvature.CONVEX, Curvature.AFFINE)
        else:
            # Maximize f  ≡ Minimize -f; convex if f is concave
            obj_convex = obj_curv in (Curvature.CONCAVE, Curvature.AFFINE)

    constraint_mask = []
    for c in model._constraints:
        if isinstance(c, Constraint):
            constraint_mask.append(classify_constraint(c, model, cache))
        else:
            constraint_mask.append(False)

    return OACutConvexity(
        objective_is_convex=obj_convex,
        constraint_mask=constraint_mask,
    )


def classify_model(model: Model) -> tuple[bool, list[bool]]:
    """Classify a model's convexity.

    Returns:
        (is_convex, constraint_convexity_mask)
        - is_convex: True if objective is convex and all constraints are convex
        - constraint_convexity_mask: per-constraint convexity flags
    """
    oa_convexity = classify_oa_cut_convexity(model)
    all_convex = oa_convexity.objective_is_convex and all(oa_convexity.constraint_mask)

    return all_convex, oa_convexity.constraint_mask
