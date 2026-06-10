"""Least-squares structure detection and Gauss-Newton Hessian construction.

The dense ``jax.hessian`` that :class:`~discopt._jax.nlp_evaluator.NLPEvaluator`
builds for the objective compiles in time that is *super-linear* in the size of
the objective expression graph (see issue #98). For objectives written as an
explicit sum of squared residuals — collocation parameter estimation, spectral
bilinear least squares ``sum((C @ S - D) ** 2)``, data fitting — that one-time
XLA compile can dominate (or entirely block) an otherwise tiny solve, even
though the runtime Hessian *evaluation* is instant.

This module provides an opt-in escape hatch. Given an objective that is a
non-negative-weighted sum of squares ``f(x) = Σ_k w_k · r_k(x)²``, it extracts
the residual sub-expressions ``r_k`` and builds the **Gauss-Newton** objective
Hessian ``H ≈ 2 Jᵀ J`` where ``J = ∂r/∂x``. The Gauss-Newton Hessian:

* compiles ~1000× faster (only first derivatives of the residuals are needed —
  no dense second-derivative graph is ever materialized),
* is always positive semidefinite (a desirable property for an SQP/IPM step),
* and equals the true Hessian at a zero-residual solution.

It drops the ``Σ_k 2 w_k r_k ∇²r_k`` curvature term, so it is an *approximation*
of the true objective Hessian away from the solution. It therefore changes the
Newton step (and hence the iteration path / count) but not the KKT point a
correctly-converging solver lands on. Because of this it is **opt-in**:
``Model.solve(gauss_newton=True)``.

Detection is deliberately conservative: any expression node that is not a
recognized square / sum-of-squares form makes :func:`extract_residuals` return
``None``, and the evaluator falls back to the exact dense Hessian. We never
silently substitute an approximate Hessian for an objective we did not prove to
be a sum of squares.
"""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


def _scalar_const(expr: Expression) -> float | None:
    """Return the scalar value of ``expr`` if it is a 0-d/size-1 Constant."""
    if isinstance(expr, Constant) and expr.value.size == 1:
        return float(expr.value.reshape(()))
    return None


def _split_const_factor(node: BinaryOp) -> tuple[float | None, Expression | None]:
    """Split ``c * sub`` / ``sub * c`` into ``(c, sub)`` for a scalar constant c.

    Returns ``(None, None)`` when neither operand is a scalar constant, so the
    caller can fall through to the ``a * a`` square check.
    """
    c = _scalar_const(node.left)
    if c is not None:
        return c, node.right
    c = _scalar_const(node.right)
    if c is not None:
        return c, node.left
    return None, None


def _expr_equal(a: Expression, b: Expression) -> bool:
    """Structural equality of two expression DAGs.

    Used to recognize ``a * a`` written as two structurally-identical but
    distinct Python objects, e.g. ``(C @ S - D) * (C @ S - D)``. Variables and
    Parameters compare by identity (the same modeling object); Constants by
    value; composite nodes recurse. ``Expression.__eq__`` is overloaded to build
    a Constraint, so equality must never use ``==`` on expressions.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    # Variables and Parameters are leaves identified by object identity; if the
    # ``a is b`` check above failed they are different model objects.
    if isinstance(a, (Variable, Parameter)):
        return False
    if isinstance(a, Constant) and isinstance(b, Constant):
        return a.value.shape == b.value.shape and bool(np.array_equal(a.value, b.value))
    if isinstance(a, IndexExpression) and isinstance(b, IndexExpression):
        return _index_equal(a.index, b.index) and _expr_equal(a.base, b.base)
    if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
        return a.op == b.op and _expr_equal(a.left, b.left) and _expr_equal(a.right, b.right)
    if isinstance(a, UnaryOp) and isinstance(b, UnaryOp):
        return a.op == b.op and _expr_equal(a.operand, b.operand)
    if isinstance(a, FunctionCall) and isinstance(b, FunctionCall):
        return (
            a.func_name == b.func_name
            and len(a.args) == len(b.args)
            and all(_expr_equal(x, y) for x, y in zip(a.args, b.args))
        )
    if isinstance(a, MatMulExpression) and isinstance(b, MatMulExpression):
        return _expr_equal(a.left, b.left) and _expr_equal(a.right, b.right)
    if isinstance(a, SumExpression) and isinstance(b, SumExpression):
        return a.axis == b.axis and _expr_equal(a.operand, b.operand)
    if isinstance(a, SumOverExpression) and isinstance(b, SumOverExpression):
        return len(a.terms) == len(b.terms) and all(
            _expr_equal(x, y) for x, y in zip(a.terms, b.terms)
        )
    return False


def _index_equal(ia, ib) -> bool:
    """Compare two ``IndexExpression`` indices (ints, slices, tuples, arrays)."""
    if type(ia) is not type(ib):
        # int vs np.integer etc. — fall back to value comparison below.
        try:
            return bool(np.array_equal(np.asarray(ia, dtype=object), np.asarray(ib, dtype=object)))
        except Exception:
            return False
    if isinstance(ia, slice):
        return bool(ia == ib)
    if isinstance(ia, tuple):
        return len(ia) == len(ib) and all(_index_equal(x, y) for x, y in zip(ia, ib))
    try:
        return bool(np.array_equal(ia, ib))
    except Exception:
        return bool(ia == ib)


def _scale_residuals(residuals: list[Expression], scale: float) -> list[Expression]:
    """Multiply each residual by ``scale`` (used to fold in a sqrt weight)."""
    if scale == 1.0:
        return residuals
    factor = Constant(scale)
    return [BinaryOp("*", factor, r) for r in residuals]


def extract_residuals(expr: Expression) -> list[Expression] | None:
    """Extract residual sub-expressions if ``expr`` is a sum of squares.

    Returns a list of residual expressions ``[r_0, r_1, ...]`` (each possibly
    array-valued — ravel and concatenate to form the residual vector) such that
    ``expr == Σ_k ravel(r_k)²``, or ``None`` if ``expr`` is not a recognized
    non-negative-weighted sum of squares.

    Recognized forms (composable through ``+`` and indexed ``Σ`` sums):

    * ``base ** 2``                          → residual ``base``
    * ``a * a`` (structurally equal factors) → residual ``a``
    * ``c * <square>`` with scalar ``c ≥ 0`` → residual ``√c · base``
    * ``sum(<elementwise square array>)``    → raveled residuals
    """
    # Additive split: Σ of sums of squares is a sum of squares.
    if isinstance(expr, BinaryOp) and expr.op == "+":
        left = extract_residuals(expr.left)
        if left is None:
            return None
        right = extract_residuals(expr.right)
        if right is None:
            return None
        return left + right

    if isinstance(expr, BinaryOp) and expr.op == "*":
        # Non-negative scalar weight: c · g  (residuals of g, scaled by √c).
        c, sub = _split_const_factor(expr)
        if c is not None and sub is not None:
            if c < 0.0:
                return None
            inner = extract_residuals(sub)
            if inner is None:
                return None
            return _scale_residuals(inner, float(np.sqrt(c)))
        # Square written as a product of identical factors: a · a.
        if _expr_equal(expr.left, expr.right):
            return [expr.left]
        return None

    if isinstance(expr, BinaryOp) and expr.op == "**":
        if _scalar_const(expr.right) == 2.0:
            return [expr.left]
        return None

    # Full reduction sum(operand): a sum of squares iff operand is an
    # elementwise sum of squares (same recursion, array-valued).
    if isinstance(expr, SumExpression):
        if expr.axis is not None:
            return None
        return extract_residuals(expr.operand)

    # Indexed summation Σ_i term_i: each term must be a sum of squares.
    if isinstance(expr, SumOverExpression):
        out: list[Expression] = []
        for term in expr.terms:
            res = extract_residuals(term)
            if res is None:
                return None
            out += res
        return out

    return None
