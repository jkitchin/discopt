"""Shared sign-tracking worst-case substitution for robust reformulations.

Both box and polyhedral formulations need to traverse the expression DAG
while tracking the effective sign of each uncertain parameter to determine
whether the upper or lower bound is the worst case.  This module provides
that traversal as a single reusable function.

The sign starts at +1 at the tree root and flips through ``-`` (subtraction)
and unary negation nodes.  Multiplication by a negative constant also flips
the sign.

.. note::

   The sign determination for constant multipliers uses
   ``np.sign(np.sum(value))``, which collapses a vector to a single scalar
   sign.  This is correct when all components share the same sign (the
   common case for cost coefficients, demands, etc.) but can be inaccurate
   for mixed-sign constant vectors.  A future release may add per-component
   sign tracking for full element-wise correctness.
"""

from __future__ import annotations

import warnings

import numpy as np


def sign_tracking_substitute(
    expr,
    wc_lower: dict[str, np.ndarray],
    wc_upper: dict[str, np.ndarray],
    param_names: set[str],
    maximize: bool,
    sign: int,
):
    """Recursively replace uncertain Parameters with worst-case constants.

    Parameters
    ----------
    expr : Expression
        Sub-expression to rewrite.
    wc_lower : dict[str, ndarray]
        Maps parameter names to their worst-case lower bound arrays.
    wc_upper : dict[str, ndarray]
        Maps parameter names to their worst-case upper bound arrays.
    param_names : set[str]
        Names of uncertain parameters to substitute.
    maximize : bool
        Whether we are maximising (True) or minimising (False) the root
        expression.
    sign : int
        Accumulated sign (+1 or -1) of the current sub-expression relative
        to the root.  Flips at subtraction and unary negation.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter) and expr.name in param_names:
        # Effective direction: if (maximize XOR sign<0), use upper bound.
        use_upper = maximize ^ (sign < 0)
        wc_value = wc_upper[expr.name] if use_upper else wc_lower[expr.name]
        return Constant(wc_value)

    if isinstance(expr, Constant):
        return expr

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign
            )
        elif expr.op == "-":
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign * -1
            )
        elif expr.op == "*":
            # Propagate sign through multiplication only when one side is a
            # constant (the most common case: scalar * variable).
            if isinstance(expr.left, Constant):
                left_sign = int(np.sign(np.sum(expr.left.value)))
                nl = expr.left
                nr = sign_tracking_substitute(
                    expr.right,
                    wc_lower,
                    wc_upper,
                    param_names,
                    maximize,
                    sign * (left_sign or 1),
                )
            elif isinstance(expr.right, Constant):
                right_sign = int(np.sign(np.sum(expr.right.value)))
                nl = sign_tracking_substitute(
                    expr.left,
                    wc_lower,
                    wc_upper,
                    param_names,
                    maximize,
                    sign * (right_sign or 1),
                )
                nr = expr.right
            else:
                # Neither side is a Constant node.  This could be:
                # (a) Parameter * Variable: the Parameter will be substituted
                #     by the recursive call, which is fine.
                # (b) True bilinear: two non-constant, non-parameter expressions
                #     both involving uncertain parameters.  This case cannot be
                #     handled correctly by sign tracking alone.
                # We warn only for case (b): both sides contain uncertain params.
                left_has_param = _contains_uncertain_param(expr.left, param_names)
                right_has_param = _contains_uncertain_param(expr.right, param_names)
                if left_has_param and right_has_param:
                    warnings.warn(
                        "Bilinear term detected in robust reformulation: "
                        "both sides of a multiplication contain uncertain "
                        "parameters.  The sign-tracking substitution may "
                        "not produce a conservative robust counterpart.  "
                        "Consider reformulating to separate uncertain "
                        "parameters from decision variables.",
                        stacklevel=4,
                    )
                nl = sign_tracking_substitute(
                    expr.left, wc_lower, wc_upper, param_names, maximize, sign
                )
                nr = sign_tracking_substitute(
                    expr.right, wc_lower, wc_upper, param_names, maximize, sign
                )
        else:
            # Division, power: propagate sign through without flip.
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign
            )
        return BinaryOp(expr.op, nl, nr)

    if isinstance(expr, UnaryOp):
        child_sign = sign * -1 if expr.op == "neg" else sign
        return UnaryOp(
            expr.op,
            sign_tracking_substitute(
                expr.operand, wc_lower, wc_upper, param_names, maximize, child_sign
            ),
        )

    if isinstance(expr, FunctionCall):
        new_args = [
            sign_tracking_substitute(a, wc_lower, wc_upper, param_names, maximize, sign)
            for a in expr.args
        ]
        return FunctionCall(expr.func_name, *new_args)

    if isinstance(expr, MatMulExpression):
        nl = sign_tracking_substitute(expr.left, wc_lower, wc_upper, param_names, maximize, sign)
        nr = sign_tracking_substitute(expr.right, wc_lower, wc_upper, param_names, maximize, sign)
        return MatMulExpression(nl, nr)

    if isinstance(expr, IndexExpression):
        nb = sign_tracking_substitute(expr.base, wc_lower, wc_upper, param_names, maximize, sign)
        return IndexExpression(nb, expr.index)

    if isinstance(expr, SumExpression):
        new_operand = sign_tracking_substitute(
            expr.operand, wc_lower, wc_upper, param_names, maximize, sign
        )
        return SumExpression(new_operand, expr.axis)

    if isinstance(expr, SumOverExpression):
        new_terms = [
            sign_tracking_substitute(t, wc_lower, wc_upper, param_names, maximize, sign)
            for t in expr.terms
        ]
        return SumOverExpression(new_terms)

    # Variable, unknown node: return as-is.
    return expr


def substitute_param(expr, param_name: str, replacement):
    """Replace every occurrence of ``Parameter(param_name)`` with *replacement*.

    This is the parameter analogue of ``_substitute_var`` in affine_policy.py.
    Used by the coefficient-extraction approach for bilinear robust reformulation.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter):
        return replacement if expr.name == param_name else expr
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            substitute_param(expr.left, param_name, replacement),
            substitute_param(expr.right, param_name, replacement),
        )
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, substitute_param(expr.operand, param_name, replacement))
    if isinstance(expr, MatMulExpression):
        return MatMulExpression(
            substitute_param(expr.left, param_name, replacement),
            substitute_param(expr.right, param_name, replacement),
        )
    if isinstance(expr, FunctionCall):
        return FunctionCall(
            expr.func_name,
            *[substitute_param(a, param_name, replacement) for a in expr.args],
        )
    if isinstance(expr, IndexExpression):
        return IndexExpression(substitute_param(expr.base, param_name, replacement), expr.index)
    if isinstance(expr, SumExpression):
        return SumExpression(substitute_param(expr.operand, param_name, replacement), expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([substitute_param(t, param_name, replacement) for t in expr.terms])
    return expr


def _contains_uncertain_param(expr, param_names: set[str]) -> bool:
    """Check whether *expr* contains any Parameter whose name is in *param_names*."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter):
        return expr.name in param_names
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _contains_uncertain_param(expr.left, param_names) or _contains_uncertain_param(
            expr.right, param_names
        )
    if isinstance(expr, UnaryOp):
        return _contains_uncertain_param(expr.operand, param_names)
    if isinstance(expr, FunctionCall):
        return any(_contains_uncertain_param(a, param_names) for a in expr.args)
    if isinstance(expr, IndexExpression):
        return _contains_uncertain_param(expr.base, param_names)
    if isinstance(expr, SumExpression):
        return _contains_uncertain_param(expr.operand, param_names)
    if isinstance(expr, SumOverExpression):
        return any(_contains_uncertain_param(t, param_names) for t in expr.terms)
    return False
